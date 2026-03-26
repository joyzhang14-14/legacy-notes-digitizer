"""
Step 3: 线稿增强 — 手绘插图描实线
用法：
  python src/enhance_lines.py               # 全部页面
  python src/enhance_lines.py --test 3      # 前3页
  python src/enhance_lines.py --pages 1-5   # 指定范围（支持 "1-5" / "2,4" / "3"）
  python src/enhance_lines.py --force       # 强制重新处理
  python src/enhance_lines.py --mode strengthen|vectorize|both
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cairosvg
import cv2
import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).parent.parent
CONFIG_PATH = ROOT / "config.yaml"

# ─────────────────────────── 工具函数 ───────────────────────────

def parse_pages_arg(arg: str, total: int) -> list[int]:
    pages = set()
    for part in arg.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            pages.update(range(int(start), int(end) + 1))
        else:
            pages.add(int(part))
    return sorted(p for p in pages if 1 <= p <= total)


def ssim_cv(gray1: np.ndarray, gray2: np.ndarray) -> float:
    """用 OpenCV/numpy 计算 SSIM（结构相似性）"""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = gray1.astype(np.float64)
    img2 = gray2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel)
    mu1 = cv2.filter2D(img1, -1, window, borderType=cv2.BORDER_REFLECT)
    mu2 = cv2.filter2D(img2, -1, window, borderType=cv2.BORDER_REFLECT)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    s1 = cv2.filter2D(img1 ** 2, -1, window, borderType=cv2.BORDER_REFLECT) - mu1_sq
    s2 = cv2.filter2D(img2 ** 2, -1, window, borderType=cv2.BORDER_REFLECT) - mu2_sq
    s12 = cv2.filter2D(img1 * img2, -1, window, borderType=cv2.BORDER_REFLECT) - mu1_mu2
    num = (2 * mu1_mu2 + C1) * (2 * s12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (s1 + s2 + C2)
    return float(np.mean(num / (den + 1e-10)))


def clamp_bbox(x, y, w, h, img_w, img_h, pad=0) -> tuple[int, int, int, int]:
    """扩展 padding 并裁剪到图像边界，返回 (x1,y1,x2,y2)"""
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(img_w, x + w + pad)
    y2 = min(img_h, y + h + pad)
    return x1, y1, x2, y2


# ─────────────────────────── 版本A：加深原线条 ───────────────────────────

def separate_lines(img_bgr: np.ndarray) -> np.ndarray:
    """返回线条 mask（线条=255，背景=0）"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=25, C=8,
    )
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close)
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_open)
    return binary


def separate_colored_lines(img_bgr: np.ndarray) -> dict[str, np.ndarray]:
    """分别提取黑色、红色、蓝色线条 mask"""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    black = cv2.inRange(hsv, (0, 0, 0), (180, 80, 120))
    red1 = cv2.inRange(hsv, (0, 60, 60), (10, 255, 255))
    red2 = cv2.inRange(hsv, (160, 60, 60), (180, 255, 255))
    red = cv2.bitwise_or(red1, red2)
    blue = cv2.inRange(hsv, (90, 60, 60), (130, 255, 255))
    return {"black": black, "red": red, "blue": blue}


def _darken_by_mask(img_bgr: np.ndarray, mask: np.ndarray,
                    thickness_add: int, contrast_boost: float) -> np.ndarray:
    """对 mask 区域做加粗+加深，返回修改后图像（不改变原图）"""
    result = img_bgr.copy()
    if thickness_add > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * thickness_add + 1, 2 * thickness_add + 1),
        )
        thick_mask = cv2.dilate(mask, k)
    else:
        thick_mask = mask

    darkened = result.astype(np.float32)
    darkened[thick_mask > 0] *= (1.0 / max(contrast_boost, 1e-5))
    darkened = np.clip(darkened, 0, 255).astype(np.uint8)
    mask_3ch = cv2.cvtColor(thick_mask, cv2.COLOR_GRAY2BGR)
    return np.where(mask_3ch > 0, darkened, result)


def strengthen_region(crop_bgr: np.ndarray, cfg_s: dict) -> np.ndarray:
    """
    版本A：对裁剪区域加深线条。
    分色处理：保留各色线条原色，分别加深。
    """
    thickness = cfg_s.get("line_thickness_add", 1)
    boost = cfg_s.get("contrast_boost", 1.5)
    result = crop_bgr.copy()
    color_masks = separate_colored_lines(crop_bgr)
    for color, mask in color_masks.items():
        if mask.sum() == 0:
            continue
        result = _darken_by_mask(result, mask, thickness, boost)
    return result


# ─────────────────────────── 版本B：矢量化重绘 ───────────────────────────

def prepare_for_vectorize(img_bgr: np.ndarray, min_area: int = 20) -> np.ndarray:
    """得到干净的二值化线条图（线条=255，背景=0）"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # 去除细小噪点（面积小于 min_area 的连通域）
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    clean = np.zeros_like(binary)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == i] = 255
    return clean


def detect_dashed_lines(binary_img: np.ndarray) -> list[dict]:
    """
    用 HoughLinesP 检测虚线区域。
    返回 [{"bbox": [x,y,w,h], "angle_deg": float}, ...]
    """
    lines = cv2.HoughLinesP(
        binary_img, 1, np.pi / 180,
        threshold=15, minLineLength=5, maxLineGap=3,
    )
    if lines is None:
        return []

    # 按角度分组，找间距规律的同向线段簇
    segments = [line[0] for line in lines]

    def angle(seg):
        x1, y1, x2, y2 = seg
        return np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180

    # 简单方案：同向线段（角度差<10°）且间距<15px的一组，判定为虚线
    used = [False] * len(segments)
    dashed_regions = []
    for i, seg_i in enumerate(segments):
        if used[i]:
            continue
        group = [seg_i]
        ai = angle(seg_i)
        for j, seg_j in enumerate(segments):
            if i == j or used[j]:
                continue
            if abs(angle(seg_j) - ai) < 10:
                # 计算端点距离
                cx_i = (seg_i[0] + seg_i[2]) / 2
                cy_i = (seg_i[1] + seg_i[3]) / 2
                cx_j = (seg_j[0] + seg_j[2]) / 2
                cy_j = (seg_j[1] + seg_j[3]) / 2
                if np.hypot(cx_i - cx_j, cy_i - cy_j) < 15:
                    group.append(seg_j)
                    used[j] = True
        if len(group) >= 3:  # 至少3段才算虚线
            xs = [p for seg in group for p in (seg[0], seg[2])]
            ys = [p for seg in group for p in (seg[1], seg[3])]
            x1, y1 = min(xs), min(ys)
            x2, y2 = max(xs), max(ys)
            dashed_regions.append({
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "angle_deg": float(ai),
            })
        used[i] = True
    return dashed_regions


def vectorize_with_potrace(binary_img: np.ndarray, svg_out: Path,
                            turdsize: int, alphamax: float,
                            opttolerance: float) -> None:
    """将二值化图（线条=255）用 potrace 转为 SVG"""
    potrace_bin = shutil.which("potrace") or "/opt/homebrew/bin/potrace"
    if not Path(potrace_bin).exists():
        raise RuntimeError("potrace 未找到，请运行: brew install potrace")

    with tempfile.NamedTemporaryFile(suffix=".pbm", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # potrace 需要黑色(0)作为前景，PIL "1" 模式中 0=黑色
        inverted = 255 - binary_img
        pil_img = Image.fromarray(inverted).convert("1")
        pil_img.save(tmp_path)

        subprocess.run(
            [potrace_bin, tmp_path, "-s",
             "-o", str(svg_out),
             "-t", str(turdsize),
             "-a", str(alphamax),
             "-O", str(opttolerance),
             "--flat"],
            check=True, capture_output=True,
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def postprocess_svg(svg_path: Path, stroke_width: float,
                    stroke_color: str, dashed_regions: list[dict]) -> None:
    """SVG 后处理：设置线条样式，标记虚线区域"""
    # potrace SVG 默认不带 xmlns 前缀，用正则注入也可，但 ET 解析更稳
    tree = ET.parse(str(svg_path))
    root = tree.getroot()

    # 统一 namespace
    ns_uri = "http://www.w3.org/2000/svg"
    ET.register_namespace("", ns_uri)
    ns = {"svg": ns_uri}

    # 获取 SVG 画布尺寸（用于虚线区域的几何判断）
    try:
        svg_w = float(root.attrib.get("width", "0").rstrip("pt").rstrip("px"))
        svg_h = float(root.attrib.get("height", "0").rstrip("pt").rstrip("px"))
    except ValueError:
        svg_w = svg_h = 0

    for path_el in root.findall(f".//{{{ns_uri}}}path"):
        path_el.set("fill", "none")
        path_el.set("stroke", stroke_color)
        path_el.set("stroke-width", str(stroke_width))
        path_el.set("stroke-linecap", "round")
        path_el.set("stroke-linejoin", "round")

        # 简单判断：path 的 d 属性中点坐标是否落在虚线区域
        if dashed_regions and svg_w > 0:
            d_attr = path_el.get("d", "")
            coords = re.findall(r"[-\d.]+", d_attr)
            if len(coords) >= 2:
                px = float(coords[0])
                py = float(coords[1])
                for dr in dashed_regions:
                    bx, by, bw, bh = dr["bbox"]
                    if bx <= px <= bx + bw and by <= py <= by + bh:
                        path_el.set("stroke-dasharray", "6,4")
                        break

    tree.write(str(svg_path), xml_declaration=True, encoding="utf-8")


def render_svg_to_png(svg_path: Path, png_path: Path,
                       width: int, height: int) -> None:
    """用 cairosvg 将 SVG 渲染为 PNG"""
    cairosvg.svg2png(
        url=str(svg_path),
        write_to=str(png_path),
        output_width=width,
        output_height=height,
    )


def vectorize_region(crop_bgr: np.ndarray, cfg_v: dict,
                     svg_out: Path) -> np.ndarray:
    """
    版本B：矢量化流程。
    返回白底黑线的 PNG（numpy array, BGR）。
    """
    binary = prepare_for_vectorize(crop_bgr)

    dashed = []
    if cfg_v.get("detect_dashed", True):
        dashed = detect_dashed_lines(binary)

    vectorize_with_potrace(
        binary, svg_out,
        turdsize=cfg_v.get("potrace_turdsize", 5),
        alphamax=cfg_v.get("potrace_alphamax", 1.0),
        opttolerance=cfg_v.get("potrace_opttolerance", 0.2),
    )

    postprocess_svg(
        svg_out,
        stroke_width=cfg_v.get("stroke_width", 1.5),
        stroke_color=cfg_v.get("stroke_color", "#1a1a1a"),
        dashed_regions=dashed,
    )

    h, w = crop_bgr.shape[:2]
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_png = tmp.name
    try:
        render_svg_to_png(svg_out, Path(tmp_png), w, h)
        rendered = cv2.imread(tmp_png)
        if rendered is None:
            # 渲染失败时退回空白图
            rendered = np.full_like(crop_bgr, 255)
    finally:
        Path(tmp_png).unlink(missing_ok=True)

    return rendered


# ─────────────────────────── 质量评估 ───────────────────────────

def _quality_label(score: float) -> str:
    if score >= 0.70:
        return "good"
    if score >= 0.45:
        return "fair"
    return "poor"


def assess_strengthen_quality(orig_gray: np.ndarray,
                               result_gray: np.ndarray) -> str:
    """基于 SSIM 评估加深版本的质量（太低说明改动过大或噪点严重）"""
    if orig_gray.shape != result_gray.shape:
        result_gray = cv2.resize(result_gray, (orig_gray.shape[1], orig_gray.shape[0]))
    score = ssim_cv(orig_gray, result_gray)
    return _quality_label(score)


def assess_vectorize_quality(orig_gray: np.ndarray,
                              binary: np.ndarray,
                              rendered_gray: np.ndarray) -> str:
    """
    用两个指标评估矢量化质量：
    1. 连通域数量比（矢量化前后）
    2. SSIM（结构相似性）
    """
    if orig_gray.shape != rendered_gray.shape:
        rendered_gray = cv2.resize(rendered_gray,
                                   (orig_gray.shape[1], orig_gray.shape[0]))
    _, binary_orig = cv2.threshold(orig_gray, 0, 255,
                                   cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n_orig, *_ = cv2.connectedComponentsWithStats(binary_orig)

    render_gray = cv2.cvtColor(rendered_gray if rendered_gray.ndim == 3
                               else np.stack([rendered_gray] * 3, axis=-1),
                               cv2.COLOR_BGR2GRAY) \
        if rendered_gray.ndim == 3 else rendered_gray
    _, binary_vec = cv2.threshold(render_gray, 200, 255, cv2.THRESH_BINARY_INV)
    n_vec, *_ = cv2.connectedComponentsWithStats(binary_vec)

    ratio = (n_vec / max(n_orig, 1))
    conn_score = 1.0 if 0.4 <= ratio <= 2.5 else 0.3

    ssim_score = ssim_cv(orig_gray, render_gray if render_gray.ndim == 2
                         else cv2.cvtColor(render_gray, cv2.COLOR_BGR2GRAY))
    # 综合得分
    combined = 0.4 * conn_score + 0.6 * max(ssim_score, 0)
    return _quality_label(combined)


# ─────────────────────────── 单页处理 ───────────────────────────

def process_page(page_num: int, cfg: dict, pages_dir: Path,
                 data_dir: Path, out_dir: Path,
                 force: bool, mode: str) -> str:
    """
    处理单页线稿增强。
    返回: "skipped" | "ok" | "error"
    """
    json_path = data_dir / f"page_{page_num:03d}.json"
    if not json_path.exists():
        return "skipped"  # 没有分析数据，跳过

    page_data = json.loads(json_path.read_text(encoding="utf-8"))

    if page_data["page_type"] == "text_only":
        return "skipped"

    orig_path = pages_dir / f"page_{page_num:03d}.png"
    if not orig_path.exists():
        return "skipped"

    # 断点续跑检查（两个版本都存在才跳过）
    out_strengthen = out_dir / f"page_{page_num:03d}_strengthen.png"
    out_vectorized_png = out_dir / f"page_{page_num:03d}_vectorized.png"
    out_meta = out_dir / f"page_{page_num:03d}_meta.json"

    need_strengthen = mode in ("strengthen", "both") and \
        cfg["line_enhancement"]["strengthen"]["enabled"]
    need_vectorize = mode in ("vectorize", "both") and \
        cfg["line_enhancement"]["vectorize"]["enabled"]

    if not force:
        s_done = (not need_strengthen) or out_strengthen.exists()
        v_done = (not need_vectorize) or out_vectorized_png.exists()
        if s_done and v_done:
            return "skipped"

    orig_bgr = cv2.imread(str(orig_path))
    if orig_bgr is None:
        raise RuntimeError(f"无法读取 {orig_path}")

    img_h, img_w = orig_bgr.shape[:2]
    cfg_s = cfg["line_enhancement"]["strengthen"]
    cfg_v = cfg["line_enhancement"]["vectorize"]
    CROP_PAD = 20

    # illustration_only：整页作为一个大插图
    illustrations = page_data.get("illustration_regions", [])
    if page_data["page_type"] == "illustration_only" and not illustrations:
        illustrations = [{"id": "i_full", "bbox": [0, 0, img_w, img_h],
                          "type": "other", "description": "整页插图"}]

    result_a = orig_bgr.copy()  # strengthen 结果（逐区域写回）
    result_b = orig_bgr.copy()  # vectorize 结果（逐区域写回白底线稿）
    meta_illustrations = []

    for illust in illustrations:
        iid = illust["id"]
        bx, by, bw, bh = illust["bbox"]
        x1, y1, x2, y2 = clamp_bbox(bx, by, bw, bh, img_w, img_h, pad=CROP_PAD)
        crop = orig_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        str_quality = "n/a"
        vec_quality = "n/a"

        # ── 版本A：加深原线条 ──
        if need_strengthen and (force or not out_strengthen.exists()):
            strengthened = strengthen_region(crop, cfg_s)
            result_a[y1:y2, x1:x2] = strengthened
            s_gray = cv2.cvtColor(strengthened, cv2.COLOR_BGR2GRAY)
            str_quality = assess_strengthen_quality(crop_gray, s_gray)

        # ── 版本B：矢量化重绘 ──
        if need_vectorize and (force or not out_vectorized_png.exists()):
            svg_path = out_dir / f"page_{page_num:03d}_{iid}_vectorized.svg"
            try:
                rendered = vectorize_region(crop, cfg_v, svg_path)
                # 将矢量化结果写回（白底），先把区域填白再贴线条
                result_b[y1:y2, x1:x2] = cv2.resize(
                    rendered, (x2 - x1, y2 - y1)
                )
                r_gray = cv2.cvtColor(
                    cv2.resize(rendered, (x2 - x1, y2 - y1)),
                    cv2.COLOR_BGR2GRAY,
                )
                binary = prepare_for_vectorize(crop)
                vec_quality = assess_vectorize_quality(crop_gray, binary, r_gray)
            except Exception as e:
                vec_quality = "poor"
                # 矢量化失败时该区域退回原图
                pass

        # 推荐版本
        quality_rank = {"good": 2, "fair": 1, "poor": 0, "n/a": -1}
        if quality_rank.get(vec_quality, -1) >= quality_rank.get(str_quality, -1):
            recommended = "vectorize"
        else:
            recommended = "strengthen"

        meta_illustrations.append({
            "id": iid,
            "strengthen_quality": str_quality,
            "vectorize_quality": vec_quality,
            "recommended_version": recommended,
        })

    # 保存输出
    if need_strengthen and (force or not out_strengthen.exists()):
        cv2.imwrite(str(out_strengthen), result_a)

    if need_vectorize and (force or not out_vectorized_png.exists()):
        cv2.imwrite(str(out_vectorized_png), result_b)
        # 整页 SVG：仅当只有一个插图区域时保留有意义的整页 svg，否则分区域 svg 已保留
        if len(illustrations) == 1:
            single_svg = out_dir / f"page_{page_num:03d}_{illustrations[0]['id']}_vectorized.svg"
            target_svg = out_dir / f"page_{page_num:03d}_vectorized.svg"
            if single_svg.exists() and not target_svg.exists():
                shutil.copy(single_svg, target_svg)

    meta = {"page_number": page_num, "illustrations": meta_illustrations}
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    return "ok"


# ─────────────────────────── 主流程 ───────────────────────────

def main():
    parser = argparse.ArgumentParser(description="线稿增强：加深原线条 / 矢量化重绘")
    parser.add_argument("--force", action="store_true", help="强制重新处理（覆盖已有文件）")
    parser.add_argument("--test", type=int, metavar="N", help="只处理前 N 页")
    parser.add_argument("--pages", type=str, metavar="RANGE",
                        help="指定页码范围，如 '1-5' 或 '1,3,5'")
    parser.add_argument("--mode", choices=["strengthen", "vectorize", "both"],
                        default="both", help="处理模式（默认 both）")
    args = parser.parse_args()

    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    inter_dir = ROOT / cfg["paths"]["intermediate_dir"]
    pages_dir = inter_dir / "pages"
    data_dir = inter_dir / "page_data"
    out_dir = inter_dir / "enhanced_lines"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 确定要处理的页码
    all_jsons = sorted(data_dir.glob("page_*.json"))
    total = len(all_jsons)
    if total == 0:
        print(f"[错误] {data_dir} 中没有 JSON 文件，请先运行 analyze.py")
        sys.exit(1)

    if args.pages:
        target_nums = parse_pages_arg(args.pages, total)
    elif args.test:
        target_nums = list(range(1, args.test + 1))
        print(f"[测试模式] 只处理前 {args.test} 页")
    else:
        target_nums = list(range(1, total + 1))

    print(f"共 {total} 页 JSON，本次处理 {len(target_nums)} 页，模式={args.mode}")

    # 统计各页 page_type，方便预报
    text_only_count = 0
    for n in target_nums:
        p = data_dir / f"page_{n:03d}.json"
        if p.exists():
            d = json.loads(p.read_text())
            if d.get("page_type") == "text_only":
                text_only_count += 1
    print(f"  其中 text_only（自动跳过）：{text_only_count} 页，"
          f"需增强：{len(target_nums) - text_only_count} 页")

    start = time.time()
    results = {}
    done = 0
    PROGRESS_INTERVAL = 10

    def task(num):
        try:
            status = process_page(num, cfg, pages_dir, data_dir,
                                  out_dir, args.force, args.mode)
            return num, status, None
        except Exception:
            return num, "error", traceback.format_exc()

    with ThreadPoolExecutor(max_workers=4) as executor:
        future_map = {executor.submit(task, n): n for n in target_nums}
        for future in as_completed(future_map):
            num, status, err = future.result()
            results[num] = status
            done += 1
            if err:
                print(f"  [错误] page_{num:03d}: {err}")
            if done % PROGRESS_INTERVAL == 0 or done == len(target_nums):
                elapsed = time.time() - start
                print(f"  [{done}/{len(target_nums)}] page_{num:03d} → {status}"
                      f"  耗时: {elapsed:.1f}s")

    ok = sum(1 for s in results.values() if s == "ok")
    skip = sum(1 for s in results.values() if s == "skipped")
    err = sum(1 for s in results.values() if s == "error")

    print("\n" + "=" * 50)
    print("线稿增强完成")
    print(f"  成功处理：{ok} 页")
    print(f"  跳过    ：{skip} 页（text_only 或已存在）")
    print(f"  失败    ：{err} 页")
    print(f"  总耗时  ：{time.time() - start:.1f} 秒")
    print(f"  输出目录：{out_dir}")


if __name__ == "__main__":
    main()
