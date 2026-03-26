"""
Phase 2: 整页 OCR — Gemini 识别 + Sonnet 位置重映射
用法：
  python src/ocr.py                          # 处理全部页面
  python src/ocr.py --pages 1-5             # 处理第1-5页
  python src/ocr.py --force                 # 强制覆盖已有结果
  python src/ocr.py --enhancement heavy     # 指定增强级别（auto/light/medium/heavy/extreme）
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
import cv2
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

# ─────────────────────────── 环境变量 ───────────────────────────

_ROOT = Path(__file__).parent.parent
load_dotenv(_ROOT / ".env", override=True)

_GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
_ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

if not _GEMINI_API_KEY:
    print("[错误] 未找到 GEMINI_API_KEY，请在 .env 文件中设置")
    sys.exit(1)
if not _ANTHROPIC_API_KEY:
    print("[错误] 未找到 ANTHROPIC_API_KEY，请在 .env 文件中设置")
    sys.exit(1)

# ─────────────────────────── 路径 & 常量 ───────────────────────────

PAGES_DIR = _ROOT / "intermediate" / "pages"
STRUCTURE_DIR = _ROOT / "intermediate" / "structure"
OUT_DIR = _ROOT / "intermediate" / "text_ocr"

GEMINI_MODEL = "gemini-2.5-pro"
CLAUDE_MODEL = "claude-sonnet-4-20250514"

MAX_CONCURRENT = 2
GEMINI_INTERVAL = 1.0
CLAUDE_INTERVAL = 1.0
RATE_LIMIT_WAIT = 30
MAX_RETRIES = 3

# ─────────────────────────── Step A: 整页增强 ───────────────────────────

def enhance_whole_page(img_bgr: np.ndarray) -> tuple[dict, str]:
    """整页增强，返回 (versions_dict, auto_best_key)。"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(gray.mean())
    kernel_sharp = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

    versions = {"original": gray}

    clahe1 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    versions["light"] = clahe1.apply(gray)

    clahe2 = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(6, 6))
    enhanced2 = clahe2.apply(gray)
    versions["medium"] = np.clip(
        cv2.filter2D(enhanced2, -1, kernel_sharp), 0, 255
    ).astype(np.uint8)

    clahe3 = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(4, 4))
    enhanced3 = clahe3.apply(gray)
    lut = np.array([int(((i / 255.0) ** 0.5) * 255) for i in range(256)], dtype=np.uint8)
    brightened = cv2.LUT(enhanced3, lut)
    versions["heavy"] = np.clip(
        cv2.filter2D(brightened, -1, kernel_sharp), 0, 255
    ).astype(np.uint8)

    inverted = 255 - gray
    clahe4 = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(4, 4))
    versions["extreme"] = 255 - clahe4.apply(inverted)

    if mean_brightness > 210:
        best = "extreme"
    elif mean_brightness > 180:
        best = "heavy"
    elif mean_brightness > 150:
        best = "medium"
    else:
        best = "light"

    return versions, best


# ─────────────────────────── 日文过滤 ───────────────────────────

_JP_PATTERN = re.compile(r"[\u3040-\u309F\u30A0-\u30FF]")


def filter_japanese(text: str) -> tuple[str, int]:
    chars = _JP_PATTERN.findall(text)
    if not chars:
        return text, 0
    for ch in set(chars):
        text = text.replace(ch, f"[?{ch}?]")
    return text, len(chars)


# ─────────────────────────── JSON 解析 ───────────────────────────

def parse_json_response(text: str) -> dict:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return json.loads(cleaned)


# ─────────────────────────── Step B: Gemini 整页 OCR ───────────────────────────

ROUND1_PROMPT = """\
请识别这张手写中文教案扫描件中的所有文字。

关键规则：
1. 这是中国中央美术学院服装设计教授的教案，语言只有中文+英文+数字
2. 绝对没有日文。如果看到像日文的字符，一定是手写中文，请猜测最接近的中文
3. 不确定的字用 [?] 标记
4. 保持原始排版结构（标题、段落、换行）
5. 红色文字用 <red>文字</red> 标记
6. 蓝色文字用 <blue>文字</blue> 标记
7. 插图区域不要尝试识别，用 [插图] 标记
8. 尺寸标注原样保留（如 H/4+10.5, W/2, 3~4cm）
9. 按从上到下、从左到右的阅读顺序输出

请返回纯文字结果，保持排版结构。不要返回 JSON，直接返回文字。\
"""

ROUND2_PROMPT = """\
我提供了同一页教案的 3 个不同对比度版本。请综合所有版本识别文字。
某些字可能在某个版本中更清晰。

规则同上：
1. 中国中央美术学院服装设计教案，只有中文+英文+数字
2. 绝对没有日文
3. 涉及领域：人体解剖、骨骼结构、服装裁剪、版型设计、缝制工艺
4. 不确定的字用 [?X?] 标记（X 是最佳猜测）
5. 红色用 <red></red>，蓝色用 <blue></blue>
6. 插图区域用 [插图] 标记

返回纯文字，保持排版结构。\
"""


def gemini_ocr(client: genai.Client, image_arrays: list, prompt: str) -> str:
    """调用 Gemini Vision，image_arrays 为 numpy 灰度/BGR 数组列表。"""
    parts = []
    for arr in image_arrays:
        pil = Image.fromarray(arr) if arr.ndim == 2 else Image.fromarray(
            cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        )
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        parts.append(types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png"))
    parts.append(types.Part.from_text(text=prompt))

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=parts,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=8192,
                ),
            )
            return response.text
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                if attempt == MAX_RETRIES:
                    raise
                print(f"  [Gemini 429] 等待 {RATE_LIMIT_WAIT}s 后重试 ({attempt}/{MAX_RETRIES})...")
                time.sleep(RATE_LIMIT_WAIT)
            else:
                if attempt == MAX_RETRIES:
                    raise
                print(f"  [Gemini 错误] {e}，5s 后重试...")
                time.sleep(5)
    raise RuntimeError("Gemini OCR 所有重试均失败")


def gemini_whole_page_ocr(
    client: genai.Client,
    versions: dict,
    best_level: str,
) -> tuple[str, str]:
    """整页两轮 OCR，返回 (round1_text, round2_text)。"""
    # Round 1: 最佳增强版
    time.sleep(GEMINI_INTERVAL)
    round1_text = gemini_ocr(client, [versions[best_level]], ROUND1_PROMPT)

    # Round 2: 3 个不同增强版本
    r2_images = [versions[lv] for lv in ("light", "medium", "heavy") if lv in versions]
    time.sleep(GEMINI_INTERVAL)
    round2_text = gemini_ocr(client, r2_images, ROUND2_PROMPT)

    return round1_text, round2_text


# ─────────────────────────── Step C: Sonnet 位置重映射 ───────────────────────────

REMAP_PROMPT_TEMPLATE = """\
你是一个文档排版专家。我有两份数据需要你合并：

## 数据1: 页面结构（来自 Phase 1 的位置编码）
{structure_json}

## 数据2: OCR 识别结果（来自 Gemini 的两轮识别）
### Gemini Round 1:
{round1_text}

### Gemini Round 2:
{round2_text}

## 任务
将 Gemini 识别出的文字内容，按照语义和位置对应关系，
绑定到结构数据中每个 TEXT_BLOCK 区域的每一行上。

规则：
1. 结构数据中的 TEXT_BLOCK 有 N 行，Gemini 文字也大约有 N 行，按顺序逐行匹配
2. 如果两轮 Gemini 结果有差异，选择更合理的那个
3. 只做明显的错别字修正，不确定的保留 Gemini 的原始结果
4. 这是服装设计教案，常见术语：胸省、肩省、腰省、肋省、裁片、缝份、
   原型、省道、版型、领片、袖片、前片、后片、
   胸围线、腰围线、臀围线、下摆线、肩线等
5. 如果出现日文假名，替换为最接近的中文字

## 输出格式（严格 JSON，无其他内容）：
{{
  "page_number": {page_number},
  "text_regions": [
    {{
      "region_id": "r1",
      "bbox": [x, y, w, h],
      "lines": [
        {{
          "line_num": 1,
          "content": "最终文字",
          "y_px": 50,
          "x_px": 30,
          "char_height_px": 35,
          "color": "black",
          "font_level": "title|subtitle|body|annotation|label",
          "confidence": 0.9
        }}
      ]
    }}
  ],
  "corrections": [
    {{"original": "错字", "corrected": "正确", "reason": "理由"}}
  ],
  "page_confidence": 0.85
}}\
"""


def prepare_image_b64(image_path: Path, max_bytes: int = 4_500_000, max_side: int = 4000) -> tuple[str, str]:
    """为 Claude API 准备 base64 图片，返回 (b64, media_type)。"""
    file_size = os.path.getsize(image_path)
    img = Image.open(image_path)

    if file_size <= max_bytes and max(img.size) <= max_side:
        with open(image_path, "rb") as f:
            return base64.standard_b64encode(f.read()).decode(), "image/png"

    ratio = min(max_side / max(img.size), 1.0)
    if ratio < 1:
        img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    if buf.tell() <= max_bytes:
        return base64.standard_b64encode(buf.getvalue()).decode(), "image/png"

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.standard_b64encode(buf.getvalue()).decode(), "image/jpeg"


def remap_positions(
    claude_client: anthropic.Anthropic,
    structure: dict,
    round1_text: str,
    round2_text: str,
    img_path: Path,
) -> dict:
    """用 Sonnet 把 Gemini 文字绑定到位置编码，同时发送原始图供验证。"""
    prompt = REMAP_PROMPT_TEMPLATE.format(
        structure_json=json.dumps(structure, ensure_ascii=False, indent=2),
        round1_text=round1_text,
        round2_text=round2_text,
        page_number=structure.get("page_number", 0),
    )
    img_b64, img_mime = prepare_image_b64(img_path)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            time.sleep(CLAUDE_INTERVAL)
            response = claude_client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=4096,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": img_mime,
                                "data": img_b64,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }],
            )
            return parse_json_response(response.content[0].text)
        except anthropic.RateLimitError:
            if attempt == MAX_RETRIES:
                raise
            print(f"  [Claude 429] 等待 {RATE_LIMIT_WAIT}s 后重试 ({attempt}/{MAX_RETRIES})...")
            time.sleep(RATE_LIMIT_WAIT)
        except json.JSONDecodeError as e:
            if attempt == MAX_RETRIES:
                raise
            print(f"  [Claude JSON错误] {e}，重试...")
            time.sleep(2)
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise
            print(f"  [Claude 错误] {e}，5s 后重试...")
            time.sleep(5)
    raise RuntimeError("Sonnet 位置重映射所有重试均失败")


# ─────────────────────────── 单页处理 ───────────────────────────

def process_page(
    page_num: int,
    gemini_client: genai.Client,
    claude_client: anthropic.Anthropic,
    force: bool,
    forced_level: str | None,
) -> tuple[int, str, dict]:
    """处理单页，返回 (page_num, status, stats)。"""
    struct_path = STRUCTURE_DIR / f"page_{page_num:03d}.json"
    img_path = PAGES_DIR / f"page_{page_num:03d}.png"
    out_json = OUT_DIR / f"page_{page_num:03d}.json"
    out_err = OUT_DIR / f"page_{page_num:03d}.error.txt"

    if not force and out_json.exists():
        return page_num, "skip", {}
    if not struct_path.exists():
        return page_num, "no_struct", {}
    if not img_path.exists():
        return page_num, "no_image", {}

    structure = json.loads(struct_path.read_text(encoding="utf-8"))
    text_blocks = [r for r in structure.get("regions", []) if r.get("type") == "TEXT_BLOCK"]
    if not text_blocks:
        return page_num, "skip_no_text", {}

    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        out_err.write_text("图片读取失败", encoding="utf-8")
        return page_num, "error", {}

    t0 = time.time()
    page_label = f"page_{page_num:03d}"

    try:
        # Step A: 整页增强
        versions, auto_best = enhance_whole_page(img_bgr)
        level = forced_level if forced_level else auto_best
        print(f"  [{page_label}] 增强级别: {level}（亮度均值: {img_bgr.mean():.0f}）")

        # Step B: Gemini 整页 OCR
        round1_text, round2_text = gemini_whole_page_ocr(gemini_client, versions, level)

        # 过滤日文
        round1_text, jp1 = filter_japanese(round1_text)
        round2_text, jp2 = filter_japanese(round2_text)
        total_jp = jp1 + jp2
        if total_jp:
            print(f"  [{page_label}] ⚠ 检测到 {total_jp} 处日文字符，已标记")

        # Step C: Sonnet 位置重映射
        result = remap_positions(claude_client, structure, round1_text, round2_text, img_path)
        result["page_number"] = page_num

        out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

        elapsed = time.time() - t0
        n_regions = len(result.get("text_regions", []))
        n_lines = sum(len(r.get("lines", [])) for r in result.get("text_regions", []))
        conf = result.get("page_confidence", 0.0)
        corrections = len(result.get("corrections", []))

        return page_num, "ok", {
            "regions": n_regions,
            "lines": n_lines,
            "confidence": conf,
            "corrections": corrections,
            "japanese": total_jp,
            "elapsed": elapsed,
        }

    except Exception as e:
        out_err.write_text(traceback.format_exc(), encoding="utf-8")
        return page_num, "error", {"msg": str(e)}


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
    parser = argparse.ArgumentParser(description="Phase 2: 整页 OCR + 位置重映射")
    parser.add_argument("--pages", type=str, default=None,
                        help='指定页码范围，如 "1-5" / "2,4,6" / "3"')
    parser.add_argument("--force", action="store_true",
                        help="强制覆盖已有结果")
    parser.add_argument("--enhancement", type=str, default=None,
                        choices=["auto", "light", "medium", "heavy", "extreme"],
                        help="指定增强级别（默认 auto）")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_pages = sorted(PAGES_DIR.glob("page_*.png"))
    if not all_pages:
        print(f"[错误] {PAGES_DIR} 中没有找到图片，请先运行 orientation.py")
        sys.exit(1)

    all_nums = [int(p.stem.split("_")[1]) for p in all_pages]
    print(f"[Phase 2] 共找到 {len(all_pages)} 页")

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
            print(f"[Phase 2] 跳过已处理 {skipped} 页（使用 --force 强制重新处理）")
    else:
        pending = target_nums

    if not pending:
        print("[Phase 2] 所有页面已处理完毕。")
        return

    forced_level = None if args.enhancement in (None, "auto") else args.enhancement
    print(f"[Phase 2] 待处理 {len(pending)} 页，并发数 {MAX_CONCURRENT}\n")

    gemini_client = genai.Client(api_key=_GEMINI_API_KEY)
    claude_client = anthropic.Anthropic(api_key=_ANTHROPIC_API_KEY, base_url="https://api.anthropic.com")

    t_start = time.time()
    results: dict[int, tuple[str, dict]] = {}

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as executor:
        future_map = {
            executor.submit(
                process_page, n, gemini_client, claude_client, args.force, forced_level
            ): n
            for n in pending
        }
        for future in as_completed(future_map):
            pn, status, stats = future.result()
            results[pn] = (status, stats)
            key = f"page_{pn:03d}"
            if status == "ok":
                print(
                    f"[{key}] ✓  {stats['regions']}区域 {stats['lines']}行  "
                    f"置信度:{stats['confidence']:.2f}  "
                    f"修正:{stats['corrections']}  {stats['elapsed']:.1f}s"
                )
            elif status in ("skip", "skip_no_text", "skip_illus_only"):
                pass
            elif status == "no_struct":
                print(f"[{key}] ⚠  缺少结构文件，请先运行 structure.py")
            else:
                print(f"[{key}] ✗  {status}，详见 {OUT_DIR}/{key}.error.txt")

    elapsed_total = time.time() - t_start
    ok_list = [n for n, (s, _) in results.items() if s == "ok"]
    err_list = [n for n, (s, _) in results.items() if s == "error"]
    all_confs = [stats["confidence"] for _, (s, stats) in results.items()
                 if s == "ok" and stats.get("confidence")]
    total_jp = sum(stats.get("japanese", 0) for _, (s, stats) in results.items() if s == "ok")

    print()
    print("─" * 50)
    print(f"[Phase 2] 汇总")
    print(f"  成功    : {len(ok_list)} 页")
    if err_list:
        print(f"  失败    : {[f'page_{n:03d}' for n in err_list]}")
    if all_confs:
        print(f"  平均置信度: {sum(all_confs)/len(all_confs):.3f}")
    if total_jp:
        print(f"  ⚠ 日文字符: {total_jp} 处，需人工确认")
    print(f"  总耗时  : {elapsed_total:.1f}s")
    print(f"  输出目录: {OUT_DIR}")
    print("─" * 50)


if __name__ == "__main__":
    main()
