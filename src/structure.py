"""
Phase 1: 结构扫描 — 单轮 Sonnet Vision 分析页面布局
用法：
  python src/structure.py               # 处理全部页面
  python src/structure.py --pages 1-5   # 处理第1-5页（支持 "1-5" / "2,4,6" / "3"）
  python src/structure.py --force       # 强制重新分析（覆盖已有 JSON）
"""

import argparse
import base64
import io
import json
import os
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from PIL import Image

# ─────────────────────────── 环境变量 ───────────────────────────

load_dotenv()

_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not _API_KEY:
    print("[错误] 未找到 ANTHROPIC_API_KEY，请在 .env 中配置或 export ANTHROPIC_API_KEY=...")
    sys.exit(1)

# ─────────────────────────── 路径 ───────────────────────────

ROOT = Path(__file__).parent.parent
PAGES_DIR = ROOT / "intermediate" / "pages"
OUT_DIR = ROOT / "intermediate" / "structure"

MODEL = "claude-sonnet-4-20250514"
MAX_CONCURRENT = 3
RATE_LIMIT_WAIT = 30   # Sonnet 恢复快，等 30s 够了

# ─────────────────────────── Prompt ───────────────────────────

STRUCTURE_PROMPT = """\
你是一位文档结构分析师。请分析这页手写中文教案的布局结构。
这是中央美术学院服装设计教授的手写教案。

## 重要：不需要识别文字内容，只需要标注位置和区域类型。

### 任务

1. 将页面划分为以下区域类型：
   - TEXT_BLOCK: 文字段落（标题、正文、注释）
   - ILLUSTRATION: 手绘插图（骨骼图、版型图、示意图）
   - LABEL_SYSTEM: 标注系统（引线 + 标签文字，围绕插图的）
   - DIMENSION: 尺寸标注（数字 + 箭头）
   - PAGE_META: 页码、页眉

2. 对每个 TEXT_BLOCK，估算：
   - 区域内有几行文字
   - 每行的 y 坐标（距页面顶部，像素）
   - 每行的 x 起始坐标（像素）
   - 每行的字符高度（像素）—— 标题约 40-60px，正文约 25-35px，注释约 15-25px
   - 每行的大致字符数
   - 颜色：black / red / blue
   - 是否是标题

3. 对每个 ILLUSTRATION，标注：
   - 类型: skeleton / pattern / diagram / figure / other
   - 图中是否有文字叠加在上面
   - 线条粗细是否均匀（用于后续马克笔检测）

## 输出格式（严格 JSON，无其他内容）：
{
  "page_number": N,
  "page_width_px": W,
  "page_height_px": H,
  "regions": [
    {
      "id": "r1",
      "type": "TEXT_BLOCK",
      "bbox": [x, y, w, h],
      "lines": [
        {
          "y_px": 50,
          "x_px": 30,
          "char_height_px": 35,
          "estimated_chars": 15,
          "color": "black",
          "is_title": true
        }
      ]
    },
    {
      "id": "r2",
      "type": "ILLUSTRATION",
      "bbox": [x, y, w, h],
      "illustration_type": "pattern",
      "has_overlapping_text": false,
      "has_varied_line_thickness": true
    }
  ]
}\
"""

# ─────────────────────────── 图片工具 ───────────────────────────

def prepare_image_for_api(
    image_path: Path,
    max_bytes: int = 4_500_000,
    max_side: int = 4000,
) -> tuple[str, str]:
    """确保图片不超过 API 限制，返回 (base64_str, media_type)。"""
    file_size = os.path.getsize(image_path)
    img = Image.open(image_path)

    if file_size <= max_bytes and max(img.size) <= max_side:
        with open(image_path, "rb") as f:
            return base64.standard_b64encode(f.read()).decode(), "image/png"

    ratio = min(max_side / max(img.size), 1.0)
    if ratio < 1:
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    if buf.tell() <= max_bytes:
        return base64.standard_b64encode(buf.getvalue()).decode(), "image/png"

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.standard_b64encode(buf.getvalue()).decode(), "image/jpeg"


# ─────────────────────────── JSON 解析 ───────────────────────────

def parse_json_response(text: str) -> dict:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return json.loads(cleaned)


# ─────────────────────────── API 调用（带重试）───────────────────────────

def _call_with_retry(fn, label: str):
    """执行 fn()，处理 429 和 JSON 解析失败，最多重试 2 次。"""
    for attempt in range(1, 3):
        try:
            return fn(), None
        except anthropic.RateLimitError:
            print(f"  [429] {label} 触发 rate limit，等待 {RATE_LIMIT_WAIT}s 后重试...")
            time.sleep(RATE_LIMIT_WAIT)
        except json.JSONDecodeError as e:
            if attempt == 1:
                print(f"  [JSON错误] {label} 解析失败，重试一次...")
                time.sleep(2)
            else:
                return None, f"json_decode: {e}"
        except Exception as e:
            if attempt == 1:
                print(f"  [错误] {label} 第1次失败: {e}，重试...")
                time.sleep(5)
            else:
                return None, traceback.format_exc()
    return None, "超过最大重试次数"


def call_structure(
    client: anthropic.Anthropic,
    img_b64: str,
    media_type: str,
    page_num: int,
) -> dict:
    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": img_b64,
                    },
                },
                {"type": "text", "text": STRUCTURE_PROMPT},
            ],
        }],
    )
    result = parse_json_response(response.content[0].text)
    result["page_number"] = page_num
    return result


# ─────────────────────────── 单页处理 ───────────────────────────

def analyze_page(
    client: anthropic.Anthropic,
    page_num: int,
    img_path: Path,
    out_dir: Path,
    force: bool,
) -> tuple[int, str, dict | None]:
    """返回 (page_num, status, stats)，status: "ok" | "skip" | "error"。"""
    out_json = out_dir / f"page_{page_num:03d}.json"
    out_err = out_dir / f"page_{page_num:03d}.error.txt"

    if not force and out_json.exists():
        return page_num, "skip", None

    t_start = time.time()
    page_label = f"page_{page_num:03d}"

    try:
        img_b64, media_type = prepare_image_for_api(img_path)
    except Exception as e:
        out_err.write_text(f"图片读取失败: {e}", encoding="utf-8")
        return page_num, "error", None

    result, err = _call_with_retry(
        lambda: call_structure(client, img_b64, media_type, page_num),
        page_label,
    )
    if result is None:
        out_err.write_text(str(err), encoding="utf-8")
        return page_num, "error", None

    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    elapsed = time.time() - t_start
    regions = result.get("regions", [])
    by_type: dict[str, int] = {}
    for r in regions:
        t = r.get("type", "UNKNOWN")
        by_type[t] = by_type.get(t, 0) + 1

    return page_num, "ok", {"regions": len(regions), "by_type": by_type, "elapsed": elapsed}


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
    parser = argparse.ArgumentParser(description="Phase 1: 单轮 Sonnet 结构扫描")
    parser.add_argument("--pages", type=str, default=None,
                        help='指定页码范围，如 "1-5" / "2,4,6" / "3"')
    parser.add_argument("--force", action="store_true",
                        help="强制重新分析（覆盖已有 JSON）")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_pages = sorted(PAGES_DIR.glob("page_*.png"))
    if not all_pages:
        print(f"[错误] {PAGES_DIR} 中没有找到图片，请先运行 orientation.py")
        sys.exit(1)

    all_nums = [int(p.stem.split("_")[1]) for p in all_pages]
    print(f"[Phase 1] 共找到 {len(all_pages)} 页")

    if args.pages:
        target_nums = parse_pages_arg(args.pages, max(all_nums))
        target_nums = [n for n in target_nums if n in all_nums]
    else:
        target_nums = all_nums

    if not args.force:
        pending = [n for n in target_nums
                   if not (OUT_DIR / f"page_{n:03d}.json").exists()]
        skipped = len(target_nums) - len(pending)
        if skipped:
            print(f"[Phase 1] 跳过已处理 {skipped} 页（使用 --force 强制重新分析）")
    else:
        pending = target_nums

    if not pending:
        print("[Phase 1] 所有页面已处理完毕。")
        return

    print(f"[Phase 1] 待分析 {len(pending)} 页，并发数 {MAX_CONCURRENT}\n")

    client = anthropic.Anthropic(api_key=_API_KEY, base_url="https://api.anthropic.com")
    t_start = time.time()
    results: dict[int, tuple[str, dict | None]] = {}

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as executor:
        future_map = {
            executor.submit(analyze_page, client, n,
                            PAGES_DIR / f"page_{n:03d}.png", OUT_DIR, args.force): n
            for n in pending
        }
        for future in as_completed(future_map):
            pn, status, stats = future.result()
            results[pn] = (status, stats)
            key = f"page_{pn:03d}"
            if status == "ok" and stats:
                by_type_str = "  ".join(
                    f"{t}×{c}" for t, c in sorted(stats["by_type"].items())
                )
                print(f"[{key}] ✓  {stats['regions']} 个区域  ({by_type_str})  {stats['elapsed']:.1f}s")
            elif status == "error":
                print(f"[{key}] ✗  分析失败，详见 {OUT_DIR}/{key}.error.txt")

    elapsed_total = time.time() - t_start
    ok_list = [n for n, (s, _) in results.items() if s == "ok"]
    err_list = [n for n, (s, _) in results.items() if s == "error"]

    total_by_type: dict[str, int] = {}
    for _, (s, stats) in results.items():
        if s == "ok" and stats:
            for t, c in stats["by_type"].items():
                total_by_type[t] = total_by_type.get(t, 0) + c

    print()
    print("─" * 45)
    print(f"[Phase 1] 汇总")
    print(f"  成功    : {len(ok_list)} 页")
    if err_list:
        print(f"  失败    : {len(err_list)} 页 → {[f'page_{n:03d}' for n in err_list]}")
    print(f"  区域类型分布:")
    for t, c in sorted(total_by_type.items()):
        print(f"    {t}: {c}")
    print(f"  总耗时  : {elapsed_total:.1f}s")
    print(f"  输出目录: {OUT_DIR}")
    print("─" * 45)


if __name__ == "__main__":
    main()
