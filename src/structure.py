"""
Phase 1: 结构扫描 — 双轮 Opus Vision 分析页面布局
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
from PIL import Image

# ─────────────────────────── API Key ───────────────────────────

_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not _API_KEY:
    print("[错误] 未找到 ANTHROPIC_API_KEY 环境变量，请先 export ANTHROPIC_API_KEY=...")
    sys.exit(1)

# ─────────────────────────── 路径 ───────────────────────────

ROOT = Path(__file__).parent.parent
PAGES_DIR = ROOT / "intermediate" / "pages"
OUT_DIR = ROOT / "intermediate" / "structure"

MODEL = "claude-opus-4-20250514"
MAX_CONCURRENT = 2
REQUEST_INTERVAL = 2.0      # 每次请求前等待秒数
RATE_LIMIT_WAIT = 120       # 429 错误后等待秒数
MAX_IMAGE_LONG_SIDE = 4000  # 超过此尺寸才缩放（节省API调用量）
MAX_IMAGE_BYTES = 20 * 1024 * 1024  # 20MB

# ─────────────────────────── Prompts ───────────────────────────

ROUND1_PROMPT = """\
你是一位专业的文档结构分析师。请分析这张手写中文教案的扫描件。
这是中央美术学院服装设计专业教授的手写教案，内容涉及：
- 人体解剖/骨骼结构
- 服装版型图（裁片、省道、缝份）
- 服装设计原理和制作工艺

## 重要：这份教案只包含中文和少量英文/数字。绝对不包含日文。如果你看到像日文的内容，那一定是手写中文，请按中文理解。

## 任务：全局布局分析

### 1. 页面方向确认
确认页面方向是否正确（文字从左到右、从上到下）。

### 2. 区域分类
将页面划分为以下类型的区域：
- ILLUSTRATION: 手绘插图（骨骼图、版型图、示意图等）
- TEXT_BLOCK: 连续文字段落（标题、正文、说明）
- LABEL_SYSTEM: 标注系统（引线 + 标签文字，通常围绕插图）
- DIMENSION: 尺寸标注（数字 + 单位 + 引线/箭头）
- PAGE_META: 页码、页眉等

### 3. 每个区域的属性
对每个区域返回：
- 类型
- 边界框 [x, y, width, height]（像素坐标）
- 置信度
- 简要内容描述

### 4. 区域关系
标注哪些区域之间有关联（如：标注系统指向哪个插图）

## 输出格式（严格JSON，无其他内容）
{
  "page_number": <int>,
  "page_width_px": <int>,
  "page_height_px": <int>,
  "orientation_confirmed": true,
  "regions": [
    {
      "id": "r1",
      "type": "ILLUSTRATION | TEXT_BLOCK | LABEL_SYSTEM | DIMENSION | PAGE_META",
      "bbox": [x, y, width, height],
      "description": "简要描述",
      "confidence": 0.95,
      "related_to": ["r2", "r3"]
    }
  ],
  "layout_summary": "页面布局总结"
}\
"""

ROUND2_PROMPT_TEMPLATE = """\
你是一位专业的文档数字化专家。我已经完成了这页教案的全局布局分析。
现在需要你对每个区域做精细分析。

这是中央美术学院服装设计教授的手写教案。
语言：中文（绝对没有日文！看起来像日文的一定是手写中文）。
涉及领域：人体解剖、服装裁剪、版型设计、缝制工艺。

以下是第一轮识别出的区域布局：
{round1_json}

## 任务：对每个区域做精细分析

### 对 TEXT_BLOCK 区域：
返回该区域内每一行文字的：
- 行号
- 行的 y 坐标（像素）
- 文字内容（初步识别，允许不确定的字用 ? 标记）
- 字号级别: title / subtitle / body / annotation
- 颜色: black / red / blue
- 书写方向: horizontal / vertical
- 行内是否有加粗或下划线

### 额外要求：精确位置和字体大小估算

对每行文字，还需要返回以下字段：
- y_px: 该行顶部距离页面顶部的像素值
- x_px: 该行左侧起始位置距离页面左侧的像素值
- char_height_px: 单个字符的高度（像素）—— 这非常重要，直接影响最终排版
- char_width_px: 单个字符的平均宽度（像素）
- line_spacing_px: 行间距（当前行底部到下一行顶部的距离，最后一行填 0）

估算方法：
- 标题通常 40-60px 高
- 正文通常 25-35px 高
- 注释/标注通常 15-25px 高
- 通过比较同一行中不同字的大小来获得平均值

为什么重要：
这些位置信息将直接用于最终文档的排版。
如果字体大小不准确，文字和配图的相对位置会偏移。
尤其是插图旁边的标注文字，位置必须和原始教案完全一致。

### 对 ILLUSTRATION 区域：
返回：
- 插图类型: skeleton(骨骼图) / pattern(版型图) / diagram(示意图) / figure(人体图) / other
- 图中的主要视觉元素列表
- 图中是否有文字叠加（如果有，列出叠加的文字和位置）
- 图的线条特征: solid(实线) / dashed(虚线) / mixed(混合) / pencil(铅笔淡线)
- 图中的颜色: 列出用到的颜色

### 对 LABEL_SYSTEM 区域：
返回每条标注：
- 标签文字内容
- 标签文字颜色
- 标签文字位置 [x, y]
- 引线起点（文字侧）[x, y]
- 引线终点（插图侧）[x, y]
- 引线类型: line / arrow / bracket
- 指向的插图区域 ID

### 对 DIMENSION 区域：
返回：
- 数值（如 "3~4", "0.5", "H/4+10.5"）
- 位置 [x, y]
- 关联的引线/箭头端点

## 输出格式（严格JSON，无其他内容）
{{
  "page_number": <int>,
  "detailed_regions": [
    {{
      "id": "r1",
      "type": "TEXT_BLOCK",
      "lines": [
        {{
          "line_num": 1,
          "y_position": 120,
          "content": "胸省的裁制",
          "uncertain_chars": [],
          "font_level": "title",
          "color": "black",
          "direction": "horizontal",
          "has_emphasis": true,
          "y_px": 120,
          "x_px": 45,
          "char_height_px": 48,
          "char_width_px": 46,
          "line_spacing_px": 12
        }}
      ]
    }},
    {{
      "id": "r2",
      "type": "ILLUSTRATION",
      "illustration_type": "skeleton",
      "visual_elements": ["正面人体骨骼", "肋骨", "脊柱", "骨盆"],
      "overlapping_text": [
        {{"content": "胸大", "position": [100, 200], "color": "red"}}
      ],
      "line_style": "mixed",
      "colors_used": ["black", "red"]
    }},
    {{
      "id": "r3",
      "type": "LABEL_SYSTEM",
      "labels": [
        {{
          "text": "腓肠",
          "color": "red",
          "text_position": [85, 920],
          "line_start": [120, 915],
          "line_end": [160, 850],
          "line_type": "arrow",
          "target_illustration": "r2"
        }}
      ]
    }},
    {{
      "id": "r4",
      "type": "DIMENSION",
      "values": [
        {{"value": "3~4", "position": [340, 560], "endpoints": [[320, 555], [380, 565]]}}
      ]
    }}
  ]
}}\
"""

# ─────────────────────────── 图片工具 ───────────────────────────

def encode_image(path: Path) -> str:
    """读取图片，必要时缩放（超 20MB 或长边 > 4000px），返回 base64 字符串。"""
    img = Image.open(path)
    orig_w, orig_h = img.size
    long_side = max(orig_w, orig_h)

    needs_resize = False
    if long_side > MAX_IMAGE_LONG_SIDE:
        needs_resize = True
        ratio = MAX_IMAGE_LONG_SIDE / long_side
    else:
        # 先检查文件大小
        raw_bytes = path.read_bytes()
        if len(raw_bytes) > MAX_IMAGE_BYTES:
            needs_resize = True
            ratio = (MAX_IMAGE_LONG_SIDE / long_side)
        else:
            return base64.standard_b64encode(raw_bytes).decode("utf-8")

    new_w = int(orig_w * ratio)
    new_h = int(orig_h * ratio)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode("utf-8")


# ─────────────────────────── JSON 解析 ───────────────────────────

def parse_json_response(text: str) -> dict:
    """去除 Claude 可能包裹的 markdown 代码块标记，然后解析 JSON。"""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return json.loads(cleaned)


# ─────────────────────────── 单轮 API 调用 ───────────────────────────

def call_round1(
    client: anthropic.Anthropic,
    img_b64: str,
    page_num: int,
) -> dict:
    """第一轮：全局布局分析。"""
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
                        "media_type": "image/png",
                        "data": img_b64,
                    },
                },
                {"type": "text", "text": ROUND1_PROMPT},
            ],
        }],
    )
    result = parse_json_response(response.content[0].text)
    # 确保 page_number 正确
    result["page_number"] = page_num
    return result


def call_round2(
    client: anthropic.Anthropic,
    img_b64: str,
    round1_data: dict,
    page_num: int,
) -> dict:
    """第二轮：精细元素分析，附带第一轮结果作为上下文。"""
    round1_json = json.dumps(round1_data, ensure_ascii=False, indent=2)
    prompt = ROUND2_PROMPT_TEMPLATE.format(round1_json=round1_json)

    response = client.messages.create(
        model=MODEL,
        max_tokens=8192,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": img_b64,
                    },
                },
                {"type": "text", "text": prompt},
            ],
        }],
    )
    result = parse_json_response(response.content[0].text)
    result["page_number"] = page_num
    return result


# ─────────────────────────── 合并两轮结果 ───────────────────────────

def merge_results(round1: dict, round2: dict) -> dict:
    """
    以第一轮的 regions 为骨架，将第二轮的详细字段合并进来。
    两轮通过 region id 对应。
    """
    # 建立 id → 详细数据 的索引
    detail_index: dict[str, dict] = {
        r["id"]: r for r in round2.get("detailed_regions", [])
    }

    merged_regions = []
    for region in round1.get("regions", []):
        rid = region["id"]
        detail = detail_index.get(rid, {})
        merged = dict(region)  # 保留第一轮的所有字段

        rtype = region.get("type", "")
        if rtype == "TEXT_BLOCK":
            merged["lines"] = detail.get("lines", [])
        elif rtype == "ILLUSTRATION":
            merged["illustration_type"] = detail.get("illustration_type", "other")
            merged["visual_elements"] = detail.get("visual_elements", [])
            merged["overlapping_text"] = detail.get("overlapping_text", [])
            merged["line_style"] = detail.get("line_style", "")
            merged["colors_used"] = detail.get("colors_used", [])
        elif rtype == "LABEL_SYSTEM":
            merged["labels"] = detail.get("labels", [])
        elif rtype == "DIMENSION":
            merged["values"] = detail.get("values", [])
        # PAGE_META: 不需要额外字段

        merged_regions.append(merged)

    return {
        "page_number": round1.get("page_number"),
        "page_width_px": round1.get("page_width_px"),
        "page_height_px": round1.get("page_height_px"),
        "orientation_confirmed": round1.get("orientation_confirmed", True),
        "layout_summary": round1.get("layout_summary", ""),
        "regions": merged_regions,
    }


# ─────────────────────────── 单页处理 ───────────────────────────

def _api_call_with_retry(fn, label: str):
    """
    执行 fn()，处理 rate limit（等待 120s 重试）和 JSON 解析失败（重试一次）。
    返回 (result_dict, error_str)。
    """
    last_error = None
    raw_text = None

    for attempt in range(1, 3):  # 最多 2 次
        try:
            return fn(), None
        except anthropic.RateLimitError:
            wait = RATE_LIMIT_WAIT
            print(f"  [429] {label} 触发 rate limit，等待 {wait}s 后重试...")
            time.sleep(wait)
            last_error = "rate_limit"
        except json.JSONDecodeError as e:
            last_error = f"json_decode: {e}"
            if attempt == 1:
                print(f"  [JSON错误] {label} 解析失败，重试一次...")
                time.sleep(2)
            else:
                break
        except Exception as e:
            last_error = traceback.format_exc()
            if attempt == 1:
                print(f"  [错误] {label} 第1次失败: {e}，重试...")
                time.sleep(5)
            else:
                break

    return None, last_error


def analyze_page(
    client: anthropic.Anthropic,
    page_num: int,
    img_path: Path,
    out_dir: Path,
    force: bool,
) -> tuple[int, str, dict | None]:
    """
    处理单页，执行双轮分析。
    返回 (page_num, status, stats)
    status: "ok" | "skip" | "error"
    stats: {"regions": N, "by_type": {...}, "elapsed": s}
    """
    out_json = out_dir / f"page_{page_num:03d}.json"
    out_err = out_dir / f"page_{page_num:03d}.error.txt"

    if not force and out_json.exists():
        return page_num, "skip", None

    t_start = time.time()
    page_label = f"page_{page_num:03d}"

    # ── 编码图片 ──
    try:
        img_b64 = encode_image(img_path)
    except Exception as e:
        out_err.write_text(f"图片读取失败: {e}", encoding="utf-8")
        return page_num, "error", None

    # ── 第一轮 ──
    time.sleep(REQUEST_INTERVAL)
    round1, err = _api_call_with_retry(
        lambda: call_round1(client, img_b64, page_num),
        f"{page_label} Round1",
    )
    if round1 is None:
        out_err.write_text(f"Round1失败: {err}", encoding="utf-8")
        return page_num, "error", None

    # ── 第二轮 ──
    time.sleep(REQUEST_INTERVAL)
    round2, err = _api_call_with_retry(
        lambda: call_round2(client, img_b64, round1, page_num),
        f"{page_label} Round2",
    )
    if round2 is None:
        out_err.write_text(f"Round2失败: {err}", encoding="utf-8")
        return page_num, "error", None

    # ── 合并并保存 ──
    merged = merge_results(round1, round2)
    out_json.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")

    elapsed = time.time() - t_start
    regions = merged.get("regions", [])
    by_type: dict[str, int] = {}
    for r in regions:
        t = r.get("type", "UNKNOWN")
        by_type[t] = by_type.get(t, 0) + 1

    return page_num, "ok", {
        "regions": len(regions),
        "by_type": by_type,
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
    parser = argparse.ArgumentParser(description="Phase 1: 双轮结构扫描")
    parser.add_argument("--pages", type=str, default=None,
                        help='指定页码范围，如 "1-5" / "2,4,6" / "3"')
    parser.add_argument("--force", action="store_true",
                        help="强制重新分析（覆盖已有 JSON）")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 收集所有页面 ──
    all_pages = sorted(PAGES_DIR.glob("page_*.png"))
    if not all_pages:
        print(f"[错误] {PAGES_DIR} 中没有找到图片，请先运行 orientation.py")
        sys.exit(1)

    total = len(all_pages)
    all_nums = [int(p.stem.split("_")[1]) for p in all_pages]
    print(f"[Phase 1] 共找到 {total} 页")

    # ── 确定待处理页码 ──
    if args.pages:
        target_nums = parse_pages_arg(args.pages, max(all_nums))
        target_nums = [n for n in target_nums if n in all_nums]
    else:
        target_nums = all_nums

    # ── 跳过已处理 ──
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

    client = anthropic.Anthropic(api_key=_API_KEY)
    t_start = time.time()
    results: dict[int, tuple[str, dict | None]] = {}

    def task_fn(page_num: int):
        img_path = PAGES_DIR / f"page_{page_num:03d}.png"
        return analyze_page(client, page_num, img_path, OUT_DIR, args.force)

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as executor:
        future_map = {executor.submit(task_fn, n): n for n in pending}
        for future in as_completed(future_map):
            pn, status, stats = future.result()
            results[pn] = (status, stats)
            key = f"page_{pn:03d}"

            if status == "ok" and stats:
                by_type_str = "  ".join(
                    f"{t}×{c}" for t, c in sorted(stats["by_type"].items())
                )
                print(f"[{key}] ✓  {stats['regions']} 个区域  "
                      f"({by_type_str})  {stats['elapsed']:.1f}s")
            elif status == "error":
                print(f"[{key}] ✗  分析失败，详见 {key}.error.txt")

    # ── 汇总 ──
    elapsed_total = time.time() - t_start
    ok_list = [n for n, (s, _) in results.items() if s == "ok"]
    err_list = [n for n, (s, _) in results.items() if s == "error"]

    # 各类型汇总
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
