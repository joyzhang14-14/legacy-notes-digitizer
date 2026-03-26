"""
Step 1: 扫描件预处理 — PDF拆页与图像增强
用法：
  python preprocess.py            # 处理全部页面
  python preprocess.py --test 3  # 只处理前3页（测试用）
  python preprocess.py --force   # 强制重新处理（覆盖已有文件）
"""

import argparse
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import yaml
from pdf2image import convert_from_path
from PIL import Image


# ─────────────────────────── 配置加载 ───────────────────────────

def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


# ─────────────────────────── 文件收集 ───────────────────────────

def collect_input_files(input_dir: Path) -> list[Path]:
    """收集所有 PDF 和图片，按文件名排序"""
    exts = {".pdf", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    files = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in exts])
    return files


# ─────────────────────────── PDF / 图片 → 页列表 ───────────────────────────

def load_pages(file_path: Path, dpi: int) -> list[Image.Image]:
    """将单个文件展开成 PIL Image 列表（PDF 多页、图片单页）"""
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        pages = convert_from_path(str(file_path), dpi=dpi)
        return pages
    else:
        img = Image.open(file_path).convert("RGB")
        return [img]


# ─────────────────────────── 图像增强流水线 ───────────────────────────

def enhance_image(pil_img: Image.Image, cfg: dict) -> Image.Image:
    """
    生成高对比度版本（仅用于 OCR），流程：
    1. 灰度化
    2. CLAHE 自适应直方图均衡化
    3. 非锐化掩模（Unsharp Mask）
    4. 自适应阈值二值化
    5. 形态学开运算去噪
    """
    pre = cfg["preprocessing"]

    # 1. 灰度
    img_np = np.array(pil_img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # 亮度自适应：偏暗时加大 clipLimit
    mean_brightness = gray.mean()
    clip_limit = pre["clahe_clip_limit"]
    if mean_brightness < 100:
        clip_limit = min(clip_limit * 1.5, 5.0)

    # 2. CLAHE
    tile = pre["clahe_tile_size"]
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    equalized = clahe.apply(gray)

    # 3. 非锐化掩模（高斯模糊差值实现）
    blurred = cv2.GaussianBlur(equalized, (0, 0), sigmaX=2)
    sharpened = cv2.addWeighted(equalized, 1.5, blurred, -0.5, 0)

    # 4. 自适应阈值二值化
    block = pre["adaptive_block_size"]
    c_val = pre["adaptive_c"]
    binary = cv2.adaptiveThreshold(
        sharpened, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block, c_val
    )

    # 5. 形态学开运算去噪
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return Image.fromarray(cleaned)


# ─────────────────────────── 单页处理任务 ───────────────────────────

def process_single_page(
    pil_img: Image.Image,
    page_num: int,
    pages_dir: Path,
    enhanced_dir: Path,
    cfg: dict,
    force: bool,
) -> bool:
    """处理并保存单页，返回是否实际执行了处理"""
    filename = f"page_{page_num:03d}.png"
    out_page = pages_dir / filename
    out_enhanced = enhanced_dir / filename

    # 断点续跑：两个文件都存在才跳过
    if not force and out_page.exists() and out_enhanced.exists():
        return False

    # 保存原始版本
    if force or not out_page.exists():
        pil_img.save(str(out_page), format="PNG")

    # 生成并保存高对比度版本
    if force or not out_enhanced.exists():
        enhanced = enhance_image(pil_img, cfg)
        enhanced.save(str(out_enhanced), format="PNG")

    return True


# ─────────────────────────── 主流程 ───────────────────────────

def main():
    parser = argparse.ArgumentParser(description="扫描件预处理：PDF拆页 + 图像增强")
    parser.add_argument("--force", action="store_true", help="强制重新处理（覆盖已有文件）")
    parser.add_argument("--test", type=int, metavar="N", help="只处理前 N 页（测试用）")
    args = parser.parse_args()

    cfg = load_config()
    base = Path(".")
    input_dir = base / cfg["paths"]["input_dir"]
    inter_dir = base / cfg["paths"]["intermediate_dir"]
    pages_dir = inter_dir / "pages"
    enhanced_dir = inter_dir / "enhanced_ocr"

    pages_dir.mkdir(parents=True, exist_ok=True)
    enhanced_dir.mkdir(parents=True, exist_ok=True)

    # 收集文件
    files = collect_input_files(input_dir)
    if not files:
        print(f"[错误] {input_dir} 中没有找到 PDF 或图片文件")
        sys.exit(1)

    print(f"找到 {len(files)} 个输入文件：{[f.name for f in files]}")

    # 展开所有页面
    dpi = cfg["preprocessing"]["dpi"]
    all_pages: list[tuple[int, Image.Image]] = []
    page_counter = 1
    for file_path in files:
        print(f"  读取: {file_path.name} ...")
        pages = load_pages(file_path, dpi)
        for img in pages:
            all_pages.append((page_counter, img))
            page_counter += 1

    total = len(all_pages)
    print(f"共 {total} 页")

    # --test 限制页数
    if args.test is not None:
        limit = args.test
        all_pages = all_pages[:limit]
        print(f"[测试模式] 只处理前 {limit} 页")

    # 并行处理
    start_time = time.time()
    success = 0
    failed = 0
    skipped = 0
    MAX_WORKERS = 4
    PROGRESS_INTERVAL = 10

    def task(item):
        num, img = item
        try:
            processed = process_single_page(img, num, pages_dir, enhanced_dir, cfg, args.force)
            return num, processed, None
        except Exception as e:
            return num, False, traceback.format_exc()

    process_total = len(all_pages)
    done_count = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(task, item): item[0] for item in all_pages}
        for future in as_completed(futures):
            num, processed, err = future.result()
            done_count += 1

            if err:
                print(f"  [错误] page_{num:03d}: {err}")
                failed += 1
            elif processed:
                success += 1
            else:
                skipped += 1

            if done_count % PROGRESS_INTERVAL == 0 or done_count == process_total:
                print(f"  [{done_count}/{process_total}] Processing page_{num:03d}.png ...")

    elapsed = time.time() - start_time
    print("\n" + "=" * 50)
    print(f"处理完成")
    print(f"  总页数：{process_total}")
    print(f"  成功  ：{success}")
    print(f"  跳过  ：{skipped}（已存在，使用 --force 可重新处理）")
    print(f"  失败  ：{failed}")
    print(f"  耗时  ：{elapsed:.1f} 秒")
    print(f"  输出  ：{pages_dir}  /  {enhanced_dir}")


if __name__ == "__main__":
    main()
