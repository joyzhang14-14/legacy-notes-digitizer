"""
Phase 2a: 插图通道 — 裁剪 + 线稿增强 + 矢量化
用法：
  python src/illustration.py                        # 处理全部页面
  python src/illustration.py --pages 1-5            # 处理第1-5页
  python src/illustration.py --force                # 强制重新处理
  python src/illustration.py --mode strengthen      # 只做加深版
  python src/illustration.py --mode vectorize       # 只做矢量化版
  python src/illustration.py --mode both            # 全部版本（默认）
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image

# ─────────────────────────── 路径 ───────────────────────────

ROOT = Path(__file__).parent.parent
PAGES_DIR = ROOT / "intermediate" / "pages"
STRUCTURE_DIR = ROOT / "intermediate" / "structure"
OUT_DIR = ROOT / "intermediate" / "illustrations"
CONFIG_PATH = ROOT / "config.yaml"

MAX_WORKERS = 4
CROP_PADDING = 30

# ─────────────────────────── 配置加载 ───────────────────────────

def load_config() -> dict:
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    return cfg.get("line_enhancement", {})


# ─────────────────────────── 裁剪 ───────────────────────────

def crop_illustration(img_bgr: np.ndarray, bbox: list, padding: int = CROP_PADDING):
    """
    从原图裁剪插图区域（带 padding）。
    返回 (crop_bgr, offset)，offset=(x1, y1) 用于坐标还原。
    """
    x, y, w, h = bbox
    img_h, img_w = img_bgr.shape[:2]
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(img_w, x + w + padding)
    y2 = min(img_h, y + h + padding)
    return img_bgr[y1:y2, x1:x2].copy(), (x1, y1)


# ─────────────────────────── 颜色提取 ───────────────────────────

def extract_color_mask(hsv: np.ndarray, ranges: dict) -> np.ndarray:
    """
    从 HSV 图中提取指定颜色的 mask。
    ranges 格式:
      单一H范围: {'h': (lo, hi), 's': (lo, hi), 'v': (lo, hi)}
      多H范围(红):{'h': [(lo1,hi1),(lo2,hi2)], 's':..., 'v':...}
    """
    h_range = ranges["h"]
    s_lo, s_hi = ranges["s"]
    v_lo, v_hi = ranges["v"]

    if isinstance(h_range[0], (list, tuple)):
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for h_lo, h_hi in h_range:
            part = cv2.inRange(
                hsv,
                np.array([h_lo, s_lo, v_lo]),
                np.array([h_hi, s_hi, v_hi]),
            )
            mask = cv2.bitwise_or(mask, part)
        return mask
    else:
        h_lo, h_hi = h_range
        return cv2.inRange(
            hsv,
            np.array([h_lo, s_lo, v_lo]),
            np.array([h_hi, s_hi, v_hi]),
        )


# ─────────────────────────── 版本A：加深线条 ───────────────────────────

_COLOR_RANGES = {
    "black": {"h": (0, 180), "s": (0, 80),  "v": (0, 130)},
    "red":   {"h": [(0, 12), (160, 180)], "s": (50, 255), "v": (50, 255)},
    "blue":  {"h": (90, 135), "s": (50, 255), "v": (50, 255)},
}


def strengthen_illustration(crop_bgr: np.ndarray, cfg: dict) -> np.ndarray:
    """
    加深版：只加深已有线条，不改变线条走向，背景保持原样。
    """
    darken_factor = cfg.get("contrast_boost", 1.8)
    thickness_add = int(cfg.get("line_thickness_add", 1))
    denoise_k = max(2, int(cfg.get("denoise_kernel", 2)))

    result = crop_bgr.astype(np.float64)
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

    for color_name, ranges in _COLOR_RANGES.items():
        color_mask = extract_color_mask(hsv, ranges)
        if color_mask.sum() == 0:
            continue

        # 在颜色 mask 内做自适应阈值，提取线条
        gray_masked = cv2.bitwise_and(gray, color_mask)
        refined = cv2.adaptiveThreshold(
            gray_masked, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 25, 8,
        )
        refined = cv2.bitwise_and(refined, color_mask)

        # 去噪（开运算，去除孤立小点）
        k_open = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (denoise_k, denoise_k)
        )
        refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, k_open)

        # 膨胀加粗
        if thickness_add > 0:
            sz = 2 * thickness_add + 1
            k_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (sz, sz))
            refined = cv2.dilate(refined, k_dilate)

        # 在结果图上加深线条区域（RGB 三通道）
        mask_bool = refined.astype(bool)
        mask_3ch = np.stack([mask_bool] * 3, axis=-1)
        result[mask_3ch] = result[mask_3ch] / darken_factor

    return np.clip(result, 0, 255).astype(np.uint8)


# ─────────────────────────── SVG 后处理 ───────────────────────────

def postprocess_svg(svg_path: str, stroke_width: float, stroke_color: str) -> None:
    """替换 potrace 默认黑色填充为描边样式，设置线宽和颜色。"""
    text = Path(svg_path).read_text(encoding="utf-8")
    # potrace 输出的 path 元素默认有 fill="black"，改为描边
    text = re.sub(r'fill="[^"]*"', f'fill="none"', text)
    text = re.sub(
        r'(<path\b)',
        f'<path stroke="{stroke_color}" stroke-width="{stroke_width}" fill="none"',
        text,
        count=1,
    )
    # 更稳妥：给所有 path 都加上描边属性
    text = re.sub(
        r'(<path\b(?![^>]*stroke=))',
        f'\\1 stroke="{stroke_color}" stroke-width="{stroke_width}"',
        text,
    )
    Path(svg_path).write_text(text, encoding="utf-8")


# ─────────────────────────── 版本B：矢量化 ───────────────────────────

def vectorize_illustration(
    crop_bgr: np.ndarray,
    out_dir: Path,
    region_id: str,
    cfg: dict,
) -> tuple[str, str] | None:
    """
    矢量化版：binary → potrace → SVG → PNG。
    返回 (svg_path, png_path)，失败返回 None。
    """
    min_area = int(cfg.get("potrace_turdsize", 5))
    alphamax = float(cfg.get("potrace_alphamax", 1.0))
    opttol = float(cfg.get("potrace_opttolerance", 0.2))
    stroke_width = float(cfg.get("stroke_width", 1.5))
    stroke_color = cfg.get("stroke_color", "#1a1a1a")

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 21, 6,
    )

    # 去除面积小于 min_area 的噪点
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    clean = np.zeros_like(binary)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == i] = 255

    # 保存为 PBM（potrace 输入，黑色=前景，需反转）
    pbm_img = Image.fromarray(255 - clean)
    with tempfile.NamedTemporaryFile(suffix=".pbm", delete=False) as f:
        pbm_path = f.name
    pbm_img.save(pbm_path)

    svg_path = str(out_dir / f"{region_id}_vectorized.svg")
    png_path = str(out_dir / f"{region_id}_vectorized.png")

    try:
        subprocess.run(
            [
                "potrace", pbm_path,
                "-s",
                "-o", svg_path,
                "-t", str(min_area),
                "-a", str(alphamax),
                "-O", str(opttol),
            ],
            check=True,
            capture_output=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        os.unlink(pbm_path)
        return None  # potrace 未安装或失败
    finally:
        if os.path.exists(pbm_path):
            os.unlink(pbm_path)

    postprocess_svg(svg_path, stroke_width, stroke_color)

    # SVG → PNG
    try:
        import cairosvg
        cairosvg.svg2png(
            url=svg_path,
            write_to=png_path,
            output_width=crop_bgr.shape[1],
            output_height=crop_bgr.shape[0],
        )
    except Exception:
        # cairosvg 失败时生成空白 PNG 作为占位
        Image.fromarray(np.ones_like(crop_bgr) * 255).save(png_path)

    return svg_path, png_path


# ─────────────────────────── Inpainting ───────────────────────────

def inpaint_overlapping_text(
    crop_bgr: np.ndarray,
    overlapping_texts: list,
    offset: tuple[int, int],
) -> np.ndarray:
    """擦除插图上叠加的手写文字（按 structure JSON 中的 overlapping_text 坐标）。"""
    if not overlapping_texts:
        return crop_bgr

    ox, oy = offset
    mask = np.zeros(crop_bgr.shape[:2], dtype=np.uint8)

    for item in overlapping_texts:
        pos = item.get("position", [0, 0])
        tx, ty = int(pos[0]) - ox, int(pos[1]) - oy
        content = item.get("content", "")
        char_count = max(1, len(content))
        text_w = char_count * 35
        text_h = 40
        x1 = max(0, tx - 5)
        y1 = max(0, ty - 5)
        x2 = min(crop_bgr.shape[1], tx + text_w + 5)
        y2 = min(crop_bgr.shape[0], ty + text_h + 5)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

    # 膨胀确保完全覆盖笔画
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.dilate(mask, k)

    return cv2.inpaint(crop_bgr, mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)


# ─────────────────────────── 质量评分 ───────────────────────────

def compute_ssim_approx(a: np.ndarray, b: np.ndarray) -> float:
    """轻量级 SSIM 近似（不依赖 skimage），基于均值/方差/协方差。"""
    fa = a.astype(np.float64)
    fb = b.astype(np.float64)
    mu_a, mu_b = fa.mean(), fb.mean()
    sig_a = fa.std()
    sig_b = fb.std()
    sig_ab = ((fa - mu_a) * (fb - mu_b)).mean()
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    num = (2 * mu_a * mu_b + C1) * (2 * sig_ab + C2)
    den = (mu_a ** 2 + mu_b ** 2 + C1) * (sig_a ** 2 + sig_b ** 2 + C2)
    return float(np.clip(num / (den + 1e-8), 0.0, 1.0))


# ─────────────────────────── 单区域处理 ───────────────────────────

def process_region(
    page_num: int,
    region: dict,
    img_bgr: np.ndarray,
    page_out_dir: Path,
    mode: str,
    strengthen_cfg: dict,
    vectorize_cfg: dict,
    force: bool,
) -> dict | None:
    """
    处理单个 ILLUSTRATION 区域，生成所有版本文件，返回 meta dict。
    """
    region_id = region["id"]
    meta_path = page_out_dir / f"{region_id}_meta.json"

    if not force and meta_path.exists():
        return None  # 已处理，跳过

    bbox = region["bbox"]
    overlapping_texts = region.get("overlapping_text", [])
    illus_type = region.get("illustration_type", "other")

    crop, offset = crop_illustration(img_bgr, bbox)
    crop_h, crop_w = crop.shape[:2]

    versions: dict[str, str] = {}
    quality_scores: dict[str, float] = {}

    # ── 版本 original ──
    orig_path = page_out_dir / f"{region_id}_original.png"
    cv2.imwrite(str(orig_path), crop)
    versions["original"] = orig_path.name

    # ── Inpainting（如果有叠加文字）──
    inpainted_crop = inpaint_overlapping_text(crop, overlapping_texts, offset)
    if overlapping_texts:
        inp_path = page_out_dir / f"{region_id}_inpainted.png"
        cv2.imwrite(str(inp_path), inpainted_crop)
        versions["inpainted"] = inp_path.name

    # ── 版本 strengthened ──
    if mode in ("strengthen", "both"):
        strengthened = strengthen_illustration(inpainted_crop, strengthen_cfg)
        str_path = page_out_dir / f"{region_id}_strengthened.png"
        cv2.imwrite(str(str_path), strengthened)
        versions["strengthened"] = str_path.name

        orig_gray = cv2.cvtColor(inpainted_crop, cv2.COLOR_BGR2GRAY)
        str_gray = cv2.cvtColor(strengthened, cv2.COLOR_BGR2GRAY)
        quality_scores["strengthen_ssim"] = round(
            compute_ssim_approx(orig_gray, str_gray), 3
        )

    # ── 版本 vectorized ──
    vec_result = None
    if mode in ("vectorize", "both"):
        vec_result = vectorize_illustration(
            inpainted_crop, page_out_dir, region_id, vectorize_cfg
        )
        if vec_result:
            svg_path, vec_png_path = vec_result
            versions["vectorized_svg"] = Path(svg_path).name
            versions["vectorized_png"] = Path(vec_png_path).name

            vec_img = cv2.imread(vec_png_path, cv2.IMREAD_GRAYSCALE)
            if vec_img is not None:
                orig_gray = cv2.cvtColor(inpainted_crop, cv2.COLOR_BGR2GRAY)
                vec_img_rs = cv2.resize(vec_img, (orig_gray.shape[1], orig_gray.shape[0]))
                quality_scores["vectorize_ssim"] = round(
                    compute_ssim_approx(orig_gray, vec_img_rs), 3
                )

    # ── 推荐版本 ──
    vec_ssim = quality_scores.get("vectorize_ssim", 0.0)
    if "vectorized_png" in versions and vec_ssim > 0.75:
        recommended = "vectorized"
    elif "strengthened" in versions:
        recommended = "strengthened"
    else:
        recommended = "original"

    meta = {
        "region_id": region_id,
        "page_number": page_num,
        "original_bbox": bbox,
        "crop_offset": list(offset),
        "crop_size": [crop_w, crop_h],
        "illustration_type": illus_type,
        "has_overlapping_text": bool(overlapping_texts),
        "overlapping_text_count": len(overlapping_texts),
        "versions": versions,
        "quality_scores": quality_scores,
        "recommended_version": recommended,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


# ─────────────────────────── 单页处理 ───────────────────────────

def process_page(
    page_num: int,
    mode: str,
    strengthen_cfg: dict,
    vectorize_cfg: dict,
    force: bool,
) -> tuple[int, str, dict]:
    """
    处理单页所有 ILLUSTRATION（和 LABEL_SYSTEM）区域。
    返回 (page_num, status, stats)
    """
    struct_path = STRUCTURE_DIR / f"page_{page_num:03d}.json"
    img_path = PAGES_DIR / f"page_{page_num:03d}.png"
    page_label = f"page_{page_num:03d}"

    if not struct_path.exists():
        return page_num, "no_struct", {}
    if not img_path.exists():
        return page_num, "no_image", {}

    struct = json.loads(struct_path.read_text(encoding="utf-8"))
    regions = struct.get("regions", [])

    target_types = {"ILLUSTRATION", "LABEL_SYSTEM"}
    target_regions = [r for r in regions if r.get("type") in target_types]

    if not target_regions:
        return page_num, "skip_text_only", {}

    # 检查是否已全部处理（非 --force 时）
    page_out_dir = OUT_DIR / page_label
    if not force:
        all_done = all(
            (page_out_dir / f"{r['id']}_meta.json").exists()
            for r in target_regions
        )
        if all_done:
            return page_num, "skip", {}

    page_out_dir.mkdir(parents=True, exist_ok=True)

    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        return page_num, "error", {"msg": "图片读取失败"}

    t0 = time.time()
    processed = 0
    skipped = 0

    for region in target_regions:
        rtype = region.get("type")
        if rtype == "LABEL_SYSTEM":
            # LABEL_SYSTEM 只保存原始裁剪（Phase 3 用于引线重建）
            rid = region["id"]
            meta_path = page_out_dir / f"{rid}_meta.json"
            if not force and meta_path.exists():
                skipped += 1
                continue
            crop, offset = crop_illustration(img_bgr, region["bbox"])
            orig_path = page_out_dir / f"{rid}_original.png"
            cv2.imwrite(str(orig_path), crop)
            meta = {
                "region_id": rid,
                "page_number": page_num,
                "original_bbox": region["bbox"],
                "crop_offset": list(offset),
                "crop_size": [crop.shape[1], crop.shape[0]],
                "illustration_type": "label_system",
                "versions": {"original": orig_path.name},
            }
            meta_path.write_text(
                json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            processed += 1
            continue

        # ILLUSTRATION
        try:
            meta = process_region(
                page_num, region, img_bgr,
                page_out_dir, mode,
                strengthen_cfg, vectorize_cfg,
                force,
            )
            if meta is None:
                skipped += 1
            else:
                processed += 1
        except Exception:
            tb = traceback.format_exc()
            (page_out_dir / f"{region['id']}.error.txt").write_text(
                tb, encoding="utf-8"
            )

    elapsed = time.time() - t0
    illus_count = sum(1 for r in target_regions if r.get("type") == "ILLUSTRATION")
    return page_num, "ok", {
        "illustrations": illus_count,
        "processed": processed,
        "skipped": skipped,
        "elapsed": elapsed,
    }


# ─────────────────────────── 页码范围解析 ───────────────────────────

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


# ─────────────────────────── 主流程 ───────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 2a: 插图通道处理")
    parser.add_argument("--pages", type=str, default=None,
                        help='指定页码范围，如 "1-5" / "2,4,6" / "3"')
    parser.add_argument("--force", action="store_true",
                        help="强制重新处理（覆盖已有输出）")
    parser.add_argument("--mode", choices=["strengthen", "vectorize", "both"],
                        default="both", help="生成版本（默认 both）")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg = load_config()
    strengthen_cfg = cfg.get("strengthen", {})
    vectorize_cfg = cfg.get("vectorize", {})

    # ── 收集所有 structure JSON ──
    all_structs = sorted(STRUCTURE_DIR.glob("page_*.json"))
    if not all_structs:
        print(f"[错误] {STRUCTURE_DIR} 中没有找到结构 JSON，请先运行 structure.py")
        sys.exit(1)

    all_nums = [int(p.stem.split("_")[1]) for p in all_structs]
    total = len(all_nums)
    print(f"[Phase 2a] 共找到 {total} 页结构数据")

    if args.pages:
        target_nums = parse_pages_arg(args.pages, max(all_nums))
        target_nums = [n for n in target_nums if n in all_nums]
    else:
        target_nums = all_nums

    print(f"[Phase 2a] 待处理 {len(target_nums)} 页  模式: {args.mode}  并发: {MAX_WORKERS}\n")

    t_start = time.time()
    results: dict[int, tuple[str, dict]] = {}

    def task_fn(page_num: int):
        return process_page(
            page_num, args.mode,
            strengthen_cfg, vectorize_cfg,
            args.force,
        )

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {executor.submit(task_fn, n): n for n in target_nums}
        for future in as_completed(future_map):
            pn, status, stats = future.result()
            results[pn] = (status, stats)
            key = f"page_{pn:03d}"

            if status == "ok":
                print(f"[{key}] ✓  插图 {stats['illustrations']}  "
                      f"处理 {stats['processed']}  "
                      f"跳过 {stats['skipped']}  "
                      f"{stats['elapsed']:.1f}s")
            elif status in ("skip", "skip_text_only"):
                pass  # 静默跳过
            elif status == "no_struct":
                print(f"[{key}] 无结构 JSON，跳过")
            elif status == "no_image":
                print(f"[{key}] 无原始图片，跳过")

    # ── 汇总 ──
    elapsed_total = time.time() - t_start
    ok = [n for n, (s, _) in results.items() if s == "ok"]
    skipped_pages = [n for n, (s, _) in results.items() if s in ("skip", "skip_text_only")]
    total_illus = sum(
        stats.get("illustrations", 0)
        for _, (s, stats) in results.items()
        if s == "ok"
    )

    print()
    print("─" * 45)
    print(f"[Phase 2a] 汇总")
    print(f"  成功页数  : {len(ok)}")
    print(f"  跳过页数  : {len(skipped_pages)}")
    print(f"  插图总数  : {total_illus}")
    print(f"  总耗时    : {elapsed_total:.1f}s")
    print(f"  输出目录  : {OUT_DIR}")
    print("─" * 45)


if __name__ == "__main__":
    main()
