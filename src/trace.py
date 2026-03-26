"""
Phase 3: 文字擦除 → 颜色分离 → 马克笔检测 → Trace SVG
用法：
  python src/trace.py               # 处理全部页面
  python src/trace.py --pages 1-3   # 处理第1-3页
  python src/trace.py --force       # 强制覆盖已有结果
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import traceback
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# ─────────────────────────── 路径 ───────────────────────────

ROOT = Path(__file__).parent.parent
PAGES_DIR = ROOT / "intermediate" / "pages"
STRUCTURE_DIR = ROOT / "intermediate" / "structure"
OCR_DIR = ROOT / "intermediate" / "text_ocr"
OUT_DIR = ROOT / "intermediate" / "traces"

MAX_WORKERS = 4

COLOR_MAP = {
    "black":  "#1a1a1a",
    "red":    "#CC1414",
    "blue":   "#1933B8",
    "yellow": "#D4A017",
}

# ─────────────────────────── Step A: 擦除文字 ───────────────────────────

def erase_text_regions(img_bgr: np.ndarray, structure: dict) -> np.ndarray:
    """用白色像素块精准擦除文字区域（逐行擦除）。"""
    result = img_bgr.copy()
    h, w = result.shape[:2]

    for region in structure.get("regions", []):
        if region.get("type") not in ("TEXT_BLOCK", "PAGE_META"):
            continue

        lines = region.get("lines", [])
        if lines:
            for line in lines:
                y = line.get("y_px", 0)
                x = line.get("x_px", 0)
                char_h = line.get("char_height_px", 30)
                n_chars = line.get("estimated_chars", 20)
                char_w = line.get("char_width_px", char_h)
                line_w = n_chars * char_w
                pad = 5

                y1 = max(0, y - pad)
                y2 = min(h, y + char_h + pad)
                x1 = max(0, x - pad)
                x2 = min(w, x + int(line_w) + pad)
                result[y1:y2, x1:x2] = 255
        else:
            # 没有行信息时退回到整个 bbox
            bbox = region.get("bbox", [])
            if len(bbox) == 4:
                bx, by, bw, bh = bbox
                pad = 5
                y1 = max(0, by - pad)
                y2 = min(h, by + bh + pad)
                x1 = max(0, bx - pad)
                x2 = min(w, bx + bw + pad)
                result[y1:y2, x1:x2] = 255

    return result


# ─────────────────────────── Step B: 颜色分离 ───────────────────────────

def separate_colors(img_bgr: np.ndarray) -> tuple[dict, tuple]:
    """在 HSV 空间按颜色分离，返回 (channels_dict, bg_hsv)。"""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    border = np.concatenate([
        hsv[:20, :, :].reshape(-1, 3),
        hsv[-20:, :, :].reshape(-1, 3),
        hsv[:, :20, :].reshape(-1, 3),
        hsv[:, -20:, :].reshape(-1, 3),
    ])
    bg_h = float(np.median(border[:, 0]))
    bg_s = float(np.median(border[:, 1]))
    bg_v = float(np.median(border[:, 2]))

    channels = {}

    channels["black"] = cv2.inRange(hsv, (0, 0, 0), (180, 60, 120))

    red1 = cv2.inRange(hsv, (0, 50, 50), (12, 255, 255))
    red2 = cv2.inRange(hsv, (160, 50, 50), (180, 255, 255))
    channels["red"] = red1 | red2

    channels["blue"] = cv2.inRange(hsv, (90, 50, 50), (135, 255, 255))

    yellow_raw = cv2.inRange(hsv, (18, 50, 50), (38, 255, 255))
    if 15 < bg_h < 40:
        yellow_sat_diff = hsv[:, :, 1].astype(int) - int(bg_s)
        yellow_significant = (yellow_sat_diff > 30).astype(np.uint8) * 255
        channels["yellow"] = cv2.bitwise_and(yellow_raw, yellow_significant)
    else:
        channels["yellow"] = yellow_raw

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    for color in channels:
        channels[color] = cv2.morphologyEx(channels[color], cv2.MORPH_OPEN, kernel)

    return channels, (bg_h, bg_s, bg_v)


# ─────────────────────────── Step C: 马克笔检测 ───────────────────────────

def detect_marker_pen(color_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """返回 (normal_mask, marker_mask)。"""
    if color_mask.sum() == 0:
        return color_mask, np.zeros_like(color_mask)

    dist = cv2.distanceTransform(color_mask, cv2.DIST_L2, 5)
    fg_distances = dist[color_mask > 0]
    if len(fg_distances) == 0:
        return color_mask, np.zeros_like(color_mask)

    median_width = float(np.median(fg_distances)) * 2
    marker_threshold = median_width * 2.5

    marker_core = (dist > marker_threshold / 2).astype(np.uint8) * 255
    marker_core = cv2.bitwise_and(marker_core, color_mask)

    if marker_core.sum() > 0:
        marker_width = max(1, int(np.median(dist[marker_core > 0]) * 2))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (marker_width, marker_width))
        marker_mask = cv2.dilate(marker_core, k)
        marker_mask = cv2.bitwise_and(marker_mask, color_mask)
    else:
        marker_mask = np.zeros_like(color_mask)

    normal_mask = cv2.bitwise_and(color_mask, cv2.bitwise_not(marker_mask))
    return normal_mask, marker_mask


def check_transparency(black_mask: np.ndarray, color_mask: np.ndarray) -> float:
    """彩色线条覆盖黑色线条超过 10% 时返回 0.5（半透明）。"""
    color_area = float(color_mask.sum()) / 255
    if color_area == 0:
        return 1.0
    overlap = cv2.bitwise_and(black_mask, color_mask)
    overlap_area = float(overlap.sum()) / 255
    return 0.5 if (overlap_area / color_area) > 0.1 else 1.0


# ─────────────────────────── Step D: Trace → SVG ───────────────────────────

def postprocess_svg(svg_path: str, color_name: str, is_marker: bool, opacity: float) -> None:
    """后处理 SVG：圆头 + 颜色 + 透明度。"""
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    tree = ET.parse(svg_path)
    root = tree.getroot()
    ns = {"svg": "http://www.w3.org/2000/svg"}

    color_hex = COLOR_MAP.get(color_name, "#1a1a1a")
    stroke_width = "3.0" if is_marker else "1.2"

    for path in root.findall(".//svg:path", ns):
        path.set("fill", "none")
        path.set("stroke", color_hex)
        path.set("stroke-width", stroke_width)
        path.set("stroke-linecap", "round")
        path.set("stroke-linejoin", "round")
        if opacity < 1.0:
            path.set("opacity", str(opacity))

    tree.write(svg_path, xml_declaration=True, encoding="utf-8")


def trace_to_svg(
    mask: np.ndarray,
    out_path: Path,
    color_name: str,
    is_marker: bool = False,
    opacity: float = 1.0,
    turdsize: int = 15,
    alphamax: float = 1.0,
) -> bool:
    """将二值 mask trace 为 SVG，保存到 out_path。返回是否成功。"""
    if mask.sum() == 0:
        return False

    pbm_img = Image.fromarray(255 - mask)
    with tempfile.NamedTemporaryFile(suffix=".pbm", delete=False) as f:
        pbm_img.save(f.name)
        pbm_path = f.name

    svg_tmp = pbm_path.replace(".pbm", ".svg")
    try:
        subprocess.run(
            ["potrace", pbm_path, "-s", "-o", svg_tmp,
             "-t", str(turdsize), "-a", str(alphamax), "-O", "0.2"],
            check=True, capture_output=True,
        )
    except FileNotFoundError:
        print("  [警告] potrace 未安装（brew install potrace），跳过矢量化")
        os.unlink(pbm_path)
        return False
    except subprocess.CalledProcessError as e:
        os.unlink(pbm_path)
        return False
    finally:
        if os.path.exists(pbm_path):
            os.unlink(pbm_path)

    postprocess_svg(svg_tmp, color_name, is_marker, opacity)
    Path(svg_tmp).rename(out_path)
    return True


# ─────────────────────────── 单页处理 ───────────────────────────

def process_page(page_num: int, force: bool) -> tuple[int, str, dict]:
    """返回 (page_num, status, stats)。"""
    page_key = f"page_{page_num:03d}"
    img_path = PAGES_DIR / f"{page_key}.png"
    struct_path = STRUCTURE_DIR / f"{page_key}.json"
    out_page_dir = OUT_DIR / page_key
    meta_path = out_page_dir / "meta.json"
    err_path = out_page_dir / "error.txt"

    if not force and meta_path.exists():
        return page_num, "skip", {}
    if not img_path.exists():
        return page_num, "no_image", {}

    out_page_dir.mkdir(parents=True, exist_ok=True)

    try:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise RuntimeError("cv2.imread 返回 None")

        structure = {}
        if struct_path.exists():
            structure = json.loads(struct_path.read_text(encoding="utf-8"))

        # Step A
        erased = erase_text_regions(img_bgr, structure)
        cv2.imwrite(str(out_page_dir / "erased.png"), erased)

        # Step B
        channels, bg_hsv = separate_colors(erased)

        # Step C + D
        black_mask = channels.get("black", np.zeros(img_bgr.shape[:2], dtype=np.uint8))
        meta = {"page_number": page_num, "bg_hsv": list(bg_hsv), "channels": {}}
        svg_count = 0

        for color_name, mask in channels.items():
            if mask.sum() == 0:
                continue

            normal_mask, marker_mask = detect_marker_pen(mask)

            if color_name != "black":
                normal_opacity = check_transparency(black_mask, normal_mask)
                marker_opacity = check_transparency(black_mask, marker_mask)
            else:
                normal_opacity = marker_opacity = 1.0

            ch_stats = {
                "normal_px": int(normal_mask.sum() // 255),
                "marker_px": int(marker_mask.sum() // 255),
                "normal_opacity": normal_opacity,
                "marker_opacity": marker_opacity,
            }

            if normal_mask.sum() > 0:
                ok = trace_to_svg(normal_mask, out_page_dir / f"{color_name}_normal.svg",
                                  color_name, is_marker=False, opacity=normal_opacity)
                if ok:
                    svg_count += 1

            if marker_mask.sum() > 0:
                ok = trace_to_svg(marker_mask, out_page_dir / f"{color_name}_marker.svg",
                                  color_name, is_marker=True, opacity=marker_opacity)
                if ok:
                    svg_count += 1
                    ch_stats["marker_detected"] = True

            meta["channels"][color_name] = ch_stats

        meta["svg_count"] = svg_count
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        active_colors = [c for c, m in channels.items() if m.sum() > 0]
        marker_colors = [c for c in active_colors
                         if meta["channels"].get(c, {}).get("marker_detected")]

        return page_num, "ok", {
            "colors": active_colors,
            "markers": marker_colors,
            "svgs": svg_count,
        }

    except Exception:
        err_path.write_text(traceback.format_exc(), encoding="utf-8")
        return page_num, "error", {}


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
    parser = argparse.ArgumentParser(description="Phase 3: 文字擦除 + 颜色分离 + Trace SVG")
    parser.add_argument("--pages", type=str, default=None,
                        help='指定页码范围，如 "1-5" / "2,4,6" / "3"')
    parser.add_argument("--force", action="store_true",
                        help="强制覆盖已有结果")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_pages = sorted(PAGES_DIR.glob("page_*.png"))
    if not all_pages:
        print(f"[错误] {PAGES_DIR} 中没有找到图片，请先运行 orientation.py")
        sys.exit(1)

    all_nums = [int(p.stem.split("_")[1]) for p in all_pages]
    print(f"[Phase 3] 共找到 {len(all_pages)} 页")

    if args.pages:
        target_nums = parse_pages_arg(args.pages, max(all_nums))
        target_nums = [n for n in target_nums if n in all_nums]
    else:
        target_nums = all_nums

    if not args.force:
        pending = [n for n in target_nums
                   if not (OUT_DIR / f"page_{n:03d}" / "meta.json").exists()]
        skipped = len(target_nums) - len(pending)
        if skipped:
            print(f"[Phase 3] 跳过已处理 {skipped} 页（使用 --force 强制重新处理）")
    else:
        pending = target_nums

    if not pending:
        print("[Phase 3] 所有页面已处理完毕。")
        return

    print(f"[Phase 3] 待处理 {len(pending)} 页，并发数 {MAX_WORKERS}\n")

    results = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {executor.submit(process_page, n, args.force): n for n in pending}
        for future in as_completed(future_map):
            pn, status, stats = future.result()
            results[pn] = (status, stats)
            key = f"page_{pn:03d}"
            if status == "ok":
                markers = stats.get("markers", [])
                marker_str = f"  马克笔: {markers}" if markers else ""
                print(f"[{key}] ✓  颜色: {stats['colors']}  SVG: {stats['svgs']}{marker_str}")
            elif status == "error":
                print(f"[{key}] ✗  失败，详见 {OUT_DIR}/{key}/error.txt")
            elif status in ("no_image", "no_struct"):
                print(f"[{key}] -  跳过（{status}）")

    ok_list = [n for n, (s, _) in results.items() if s == "ok"]
    err_list = [n for n, (s, _) in results.items() if s == "error"]
    total_svgs = sum(stats.get("svgs", 0) for _, (s, stats) in results.items() if s == "ok")

    print()
    print("─" * 45)
    print(f"[Phase 3] 汇总")
    print(f"  成功    : {len(ok_list)} 页")
    if err_list:
        print(f"  失败    : {len(err_list)} 页 → {[f'page_{n:03d}' for n in err_list]}")
    print(f"  SVG 总数: {total_svgs}")
    print(f"  输出目录: {OUT_DIR}")
    print("─" * 45)


if __name__ == "__main__":
    main()
