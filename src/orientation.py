"""
Phase 0: 方向检测与自动旋转
用法：
  python src/orientation.py               # 处理全部页面
  python src/orientation.py --pages 1-5   # 处理第1-5页
  python src/orientation.py --force       # 强制重新检测（覆盖已有记录）
  python src/orientation.py --dpi 300     # 指定拆分DPI（默认300）
"""

import argparse
import base64
import glob
import io
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic
from PIL import Image

# ─────────────────────────── API Key ───────────────────────────

_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not _API_KEY:
    print("[错误] 未找到 ANTHROPIC_API_KEY 环境变量，请先 export ANTHROPIC_API_KEY=...")
    sys.exit(1)

# ─────────────────────────── 路径 ───────────────────────────

ROOT = Path(__file__).parent.parent
INPUT_DIR = ROOT / "input"
PAGES_DIR = ROOT / "intermediate" / "pages"
LOG_PATH = ROOT / "intermediate" / "orientation_log.json"

MODEL = "claude-opus-4-20250514"
MAX_SIDE = 800
MAX_CONCURRENT = 3

# ─────────────────────────── Prompt ───────────────────────────

ORIENTATION_PROMPT = """\
你看到的是一页手写中文教案的扫描件。这份教案来自中央美术学院服装设计专业。

请判断这张图片的方向是否正确。

判断依据：
- 中文文字应该是从左到右、从上到下书写
- 标题通常在页面顶部
- 页码通常在页面顶部右侧或底部
- 如果有竖排文字，整体页面仍应是正向的

请返回需要旋转的角度（顺时针）：
- 0: 方向正确，不需要旋转
- 90: 需要顺时针旋转90度
- 180: 需要旋转180度
- 270: 需要顺时针旋转270度（即逆时针90度）

只返回一个数字，不要其他内容：
0 或 90 或 180 或 270\
"""

# ─────────────────────────── 页面拆分 ───────────────────────────

def extract_pages(input_dir: Path, output_dir: Path, dpi: int = 300) -> int:
    """将 input_dir 中所有 PDF/图片拆分为单页 PNG，返回总页数。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(
        glob.glob(str(input_dir / "*")),
        key=lambda f: os.path.basename(f)
    )
    page_num = 0
    for fpath in files:
        ext = os.path.splitext(fpath)[1].lower()
        if ext == ".pdf":
            try:
                from pdf2image import convert_from_path
            except ImportError:
                print("[错误] 缺少 pdf2image，请运行: pip install pdf2image")
                sys.exit(1)
            images = convert_from_path(fpath, dpi=dpi)
            for img in images:
                page_num += 1
                out_path = output_dir / f"page_{page_num:03d}.png"
                img.save(out_path)
                print(f"  [拆分] {os.path.basename(fpath)} → {out_path.name}")
        elif ext in (".jpg", ".jpeg", ".png", ".tiff", ".tif"):
            page_num += 1
            out_path = output_dir / f"page_{page_num:03d}.png"
            img = Image.open(fpath)
            img.save(out_path)
            print(f"  [拆分] {os.path.basename(fpath)} → {out_path.name}")
    return page_num

# ─────────────────────────── 方向检测 ───────────────────────────

def detect_orientation(image_path: Path, client: anthropic.Anthropic) -> int:
    """调用 Claude Vision 检测图片需要顺时针旋转的角度（0/90/180/270）。"""
    img = Image.open(image_path)
    ratio = min(MAX_SIDE / img.width, MAX_SIDE / img.height)
    if ratio < 1:
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.standard_b64encode(buf.getvalue()).decode()

    response = client.messages.create(
        model=MODEL,
        max_tokens=16,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": b64,
                    },
                },
                {"type": "text", "text": ORIENTATION_PROMPT},
            ],
        }],
    )
    raw = response.content[0].text.strip()
    angle = int(raw)
    if angle not in (0, 90, 180, 270):
        raise ValueError(f"意外的角度值: {raw!r}")
    return angle


def rotate_if_needed(image_path: Path, angle: int) -> None:
    """如需旋转则原地旋转保存（PIL rotate 是逆时针，取反得顺时针）。"""
    if angle == 0:
        return
    img = Image.open(image_path)
    rotated = img.rotate(-angle, expand=True)
    rotated.save(image_path)

# ─────────────────────────── 范围解析 ───────────────────────────

def parse_pages_arg(pages_str: str, total: int) -> list[int]:
    """解析 --pages 参数，返回 1-based 页码列表。支持 "1-5" / "2,4,6" / "3"。"""
    result = set()
    for part in pages_str.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            result.update(range(int(lo), int(hi) + 1))
        else:
            result.add(int(part))
    return sorted(p for p in result if 1 <= p <= total)

# ─────────────────────────── 主流程 ───────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 0: 方向检测与自动旋转")
    parser.add_argument("--pages", type=str, default=None,
                        help='处理指定页，格式: "1-5" / "2,4,6" / "3"')
    parser.add_argument("--force", action="store_true",
                        help="强制重新检测（覆盖已有记录）")
    parser.add_argument("--dpi", type=int, default=300,
                        help="PDF 拆分 DPI（默认 300）")
    args = parser.parse_args()

    # ── 1. 拆分 PDF/图片（如果 pages 目录为空） ──
    existing = sorted(PAGES_DIR.glob("page_*.png"))
    if not existing:
        print("[Phase 0] 正在拆分输入文件...")
        total = extract_pages(INPUT_DIR, PAGES_DIR, dpi=args.dpi)
        print(f"[Phase 0] 拆分完成，共 {total} 页\n")
        existing = sorted(PAGES_DIR.glob("page_*.png"))
    else:
        print(f"[Phase 0] 已找到 {len(existing)} 页（跳过拆分，如需重新拆分请删除 intermediate/pages/）\n")

    # ── 2. 确定待处理页码 ──
    all_page_nums = [
        int(p.stem.split("_")[1]) for p in existing
    ]
    if args.pages:
        target_nums = parse_pages_arg(args.pages, max(all_page_nums))
        target_nums = [n for n in target_nums if n in all_page_nums]
    else:
        target_nums = all_page_nums

    # ── 3. 读取已有日志 ──
    log: dict[str, dict] = {}
    if LOG_PATH.exists():
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            log = json.load(f)

    # ── 4. 跳过已处理（除非 --force） ──
    if not args.force:
        pending = [n for n in target_nums if f"page_{n:03d}" not in log]
        skipped = len(target_nums) - len(pending)
        if skipped:
            print(f"[Phase 0] 跳过已处理 {skipped} 页（使用 --force 强制重新检测）")
    else:
        pending = target_nums

    if not pending:
        print("[Phase 0] 所有页面已处理完毕。")
        _print_summary(log)
        return

    print(f"[Phase 0] 待检测 {len(pending)} 页，并发数 {MAX_CONCURRENT}\n")

    # ── 5. 并发检测方向 ──
    client = anthropic.Anthropic(api_key=_API_KEY)

    def process_page(page_num: int) -> tuple[int, int, str]:
        """返回 (page_num, angle, status_msg)"""
        key = f"page_{page_num:03d}"
        img_path = PAGES_DIR / f"{key}.png"
        angle = detect_orientation(img_path, client)
        rotate_if_needed(img_path, angle)
        if angle == 0:
            msg = f"[{key}] 方向正确，无需旋转"
        else:
            msg = f"[{key}] 检测到旋转 {angle}° → 已纠正"
        return page_num, angle, msg

    results: dict[int, int] = {}
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as executor:
        futures = {executor.submit(process_page, n): n for n in pending}
        for future in as_completed(futures):
            page_num = futures[future]
            try:
                pn, angle, msg = future.result()
                print(msg)
                results[pn] = angle
            except Exception as e:
                key = f"page_{page_num:03d}"
                print(f"[{key}] 检测失败: {e}")
                results[page_num] = -1  # 标记失败

    # ── 6. 更新日志 ──
    for pn, angle in results.items():
        key = f"page_{pn:03d}"
        log[key] = {"angle": angle, "status": "ok" if angle >= 0 else "error"}

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    # ── 7. 汇总 ──
    print()
    _print_summary(log)


def _print_summary(log: dict):
    entries = list(log.values())
    total = len(entries)
    ok_entries = [e for e in entries if e.get("status") == "ok"]
    rotated = [e for e in ok_entries if e["angle"] != 0]

    angle_dist: dict[int, int] = {}
    for e in ok_entries:
        a = e["angle"]
        angle_dist[a] = angle_dist.get(a, 0) + 1

    print("─" * 40)
    print(f"[Phase 0] 汇总")
    print(f"  总页数:    {total}")
    print(f"  需旋转:    {len(rotated)}")
    print(f"  角度分布:")
    for angle in sorted(angle_dist):
        label = "正确" if angle == 0 else f"旋转{angle}°"
        print(f"    {label}: {angle_dist[angle]} 页")
    errors = total - len(ok_entries)
    if errors:
        print(f"  检测失败:  {errors} 页")
    print("─" * 40)


if __name__ == "__main__":
    main()
