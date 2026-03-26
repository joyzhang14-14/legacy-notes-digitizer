"""
Phase 2b: 文字通道 — 区域增强 + 三轮 Opus OCR + 交叉校验
用法：
  python src/text_ocr.py               # 处理全部页面
  python src/text_ocr.py --pages 1-5   # 处理第1-5页（支持 "1-5" / "2,4,6" / "3"）
  python src/text_ocr.py --force       # 强制重新分析（覆盖已有 JSON）
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
from PIL import Image

# ─────────────────────────── API Key ───────────────────────────

_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not _API_KEY:
    print("[错误] 未找到 ANTHROPIC_API_KEY 环境变量，请先 export ANTHROPIC_API_KEY=...")
    sys.exit(1)

# ─────────────────────────── 路径 ───────────────────────────

ROOT = Path(__file__).parent.parent
PAGES_DIR = ROOT / "intermediate" / "pages"
STRUCTURE_DIR = ROOT / "intermediate" / "structure"
OUT_DIR = ROOT / "intermediate" / "text_ocr"

MODEL = "claude-opus-4-20250514"
MAX_CONCURRENT = 2
REQUEST_INTERVAL = 2.0
RATE_LIMIT_WAIT = 120
MAX_RETRIES = 5
TEXT_CROP_PADDING = 15

# ─────────────────────────── Prompts ───────────────────────────

ROUND1_PROMPT = """\
请准确识别这张图片中的手写中文文字。

重要规则：
1. 这是中文教案，只包含中文、英文字母、数字、数学符号。绝对没有日文。
2. 如果某个字不确定，用 [?X?] 标记（X 是你的最佳猜测）
3. 保持原始的换行和段落结构
4. 红色文字用 <red>文字</red> 标记
5. 蓝色文字用 <blue>文字</blue> 标记
6. 竖排文字按从上到下的顺序转写

只返回识别出的文字，不要其他说明。\
"""

ROUND2_PROMPT = """\
请准确识别这两张图片中的手写中文文字。
第一张是增强对比度版本（文字更清晰），第二张是原始版本（颜色准确）。
请综合两张图进行识别。

重要规则：
1. 这是中央美术学院服装设计教授的手写教案
2. 只包含中文、英文、数字。绝对没有日文。
3. 内容涉及：人体解剖、骨骼结构、服装裁剪、版型设计、缝制工艺
4. 如果某个字不确定，用 [?X?] 标记
5. 红色文字用 <red>文字</red> 标记
6. 蓝色文字用 <blue>文字</blue> 标记
7. 数学公式和尺寸标注原样保留（如 H/4+10.5, W/2, 3~4cm）

只返回识别出的文字，不要其他说明。\
"""

ROUND3_PROMPT_TEMPLATE = """\
你是服装设计和人体解剖学的专家。下面是同一段手写中文文字的两次独立 OCR 结果。
请综合两个版本，结合专业知识，给出最终的准确文字。

## 第一次 OCR 结果：
{round1_text}

## 第二次 OCR 结果：
{round2_text}

## 专业术语参考（如果OCR结果中有相近的词，应该修正为这些术语）：

### 服装裁剪术语：
胸省、肩省、腰省、肋省、袖孔省、领孔省、省道、省尖、省量、
原型、衣身原型、袖子原型、裁片、缝份、放码、打版、
前片、后片、袖片、领片、裁剪、缝制、制图、
经纱、纬纱、斜纱、布纹方向、褶裥、明线、暗线、
前中心线、后中心线、肩线、侧缝线、下摆线、
胸围线、腰围线、臀围线、袖笼线、领围线、
胸宽、背宽、肩宽、袖长、衣长、
贴边、挂面、里布、衬布、粘合衬、
领座、翻领、立领、驳领、青果领、
袖山、袖肥、袖口、连袖、插肩袖、

### 人体解剖术语：
骨骼、脊柱、颈椎、胸椎、腰椎、骶骨、尾骨、
肋骨、胸骨、锁骨、肩胛骨、肱骨、桡骨、尺骨、
骨盆、髂骨、坐骨、耻骨、股骨、胫骨、腓骨、
肌肉、胸大肌、三角肌、斜方肌、背阔肌、
腹直肌、腹外斜肌、臀大肌、股四头肌、腓肠肌、比目鱼肌、
肩关节、肘关节、髋关节、膝关节、
体表标志、颈窝、乳点、肩点、
人体比例、头身比、

### 设计术语：
款式、造型、廓形、轮廓、结构线、装饰线、
省道转移、省道设计、省道变化、
公主线、刀背缝、分割线、
褶裥设计、抽褶、活褶、死褶、
面料、梭织、针织、弹性、悬垂性、
A字型、H型、X型、T型、

### 教学术语：
教学目的、教学内容、教学方法、教学重点、教学难点、
讲授、示范、练习、实验、作业、考核、
课时、学时、教案、大纲、

## 任务：
1. 对比两次 OCR 结果，找出差异
2. 对每处差异，结合上下文语义和专业术语判断哪个更准确
3. 如果两次都不确定的字，用 [?X?] 标记你的最佳猜测
4. 修正明显的语义错误（如 "我穿" 可能是 "裁片"，"异形" 可能是 "造型"）
5. 确保没有日文字符

## 输出格式（严格JSON，无其他内容）：
{{
  "final_text": "最终准确文字（保留换行，红色用<red>标记，蓝色用<blue>标记）",
  "corrections": [
    {{
      "original": "两次OCR中的错误文字",
      "corrected": "修正后的文字",
      "reason": "修正理由"
    }}
  ],
  "uncertain_chars": [
    {{
      "position": "第X行第Y个字",
      "best_guess": "猜测",
      "alternatives": ["备选1", "备选2"]
    }}
  ],
  "confidence": 0.85
}}\
"""

LABEL_PROMPT = """\
请识别这张图片中的标注文字。这是服装设计教案中的插图标注。

图中有红色/黑色的引线从文字标签指向插图的某个部位。
请识别每个标签的文字内容。

规则：
1. 只有中文、英文、数字，没有日文
2. 常见标注词：胸大肌、三角肌、腋外斜肌、腓肠肌、比目鱼肌、
   肩省、胸围线、前中心线、后中心线、等
3. 尺寸标注如：3~4, 0.5, H/4+10.5, W/2+3

返回JSON（严格JSON，无其他内容）：
{
  "labels": [
    {
      "text": "标注文字",
      "color": "red|black|blue",
      "approximate_position": [x, y]
    }
  ],
  "dimensions": [
    {
      "value": "H/4+10.5",
      "approximate_position": [x, y]
    }
  ]
}\
"""

# ─────────────────────────── 日文检测 ───────────────────────────

_JP_PATTERN = re.compile(r"[\u3040-\u309F\u30A0-\u30FF]")


def detect_japanese(text: str) -> list[str]:
    return _JP_PATTERN.findall(text)


def filter_japanese(text: str) -> tuple[str, int]:
    """将日文字符替换为 [?X?] 标记，返回 (过滤后文本, 检测到的日文字符数)。"""
    chars = detect_japanese(text)
    if not chars:
        return text, 0
    for ch in set(chars):
        text = text.replace(ch, f"[?{ch}?]")
    return text, len(chars)


# ─────────────────────────── 图片工具 ───────────────────────────

def encode_array(arr: np.ndarray) -> str:
    """numpy 图片数组 → base64 PNG 字符串。支持灰度和 BGR 输入。"""
    if arr.ndim == 2:
        pil = Image.fromarray(arr)
    else:
        pil = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode("utf-8")


def enhance_text_region(
    img_bgr: np.ndarray, bbox: list, padding: int = TEXT_CROP_PADDING
) -> dict:
    """裁剪文字区域并生成多个增强版本，用于多轮 OCR。"""
    x, y, w, h = bbox
    img_h, img_w = img_bgr.shape[:2]
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(img_w, x + w + padding)
    y2 = min(img_h, y + h + padding)

    crop = img_bgr[y1:y2, x1:x2].copy()
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    mean_brightness = float(gray.mean())
    if mean_brightness < 120:
        clip_limit = 4.0
    elif mean_brightness < 160:
        clip_limit = 3.0
    else:
        clip_limit = 2.0

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)

    kernel_sharp = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel_sharp)
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    binary = cv2.adaptiveThreshold(
        sharpened, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, 5,
    )

    return {
        "enhanced_gray": enhanced,
        "sharpened": sharpened,
        "binary": binary,
        "original_crop": crop,
        "offset": (x1, y1),
    }


# ─────────────────────────── API 调用（含重试）───────────────────────────

def call_api_with_retry(
    client: anthropic.Anthropic,
    messages: list,
    max_tokens: int,
    label: str,
) -> str:
    """
    调用 Claude API，遇到 rate limit 等待 RATE_LIMIT_WAIT 秒后重试，
    最多重试 MAX_RETRIES 次。返回文本响应。
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.messages.create(
                model=MODEL,
                max_tokens=max_tokens,
                messages=messages,
            )
            return resp.content[0].text
        except anthropic.RateLimitError:
            if attempt == MAX_RETRIES:
                raise
            print(f"  [429] {label} 触发 rate limit，等待 {RATE_LIMIT_WAIT}s 后重试 "
                  f"({attempt}/{MAX_RETRIES})...")
            time.sleep(RATE_LIMIT_WAIT)
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise
            print(f"  [错误] {label} 第{attempt}次失败: {e}，5s 后重试...")
            time.sleep(5)
    raise RuntimeError(f"{label} 所有重试均失败")


def parse_json_from_response(text: str) -> dict:
    """去除 markdown 代码块后解析 JSON。"""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return json.loads(cleaned)


# ─────────────────────────── TEXT_BLOCK 三轮 OCR ───────────────────────────

def ocr_text_block(
    client: anthropic.Anthropic,
    img_bgr: np.ndarray,
    region: dict,
    struct_lines: list,
) -> dict:
    """
    对 TEXT_BLOCK 区域执行三轮 OCR，返回完整结果 dict。
    struct_lines: 来自 Phase 1 结构分析的行元数据（font_level、color 等）。
    """
    bbox = region["bbox"]
    region_id = region["id"]
    enhanced = enhance_text_region(img_bgr, bbox)

    sharp_b64 = encode_array(enhanced["sharpened"])
    enh_b64 = encode_array(enhanced["enhanced_gray"])
    orig_b64 = encode_array(enhanced["original_crop"])

    # ── 第一轮：锐化图单图裸识别 ──
    time.sleep(REQUEST_INTERVAL)
    round1_text = call_api_with_retry(
        client,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64",
                 "media_type": "image/png", "data": sharp_b64}},
                {"type": "text", "text": ROUND1_PROMPT},
            ],
        }],
        max_tokens=4096,
        label=f"{region_id} Round1",
    )

    # ── 第二轮：增强 + 原图双图对比识别 ──
    time.sleep(REQUEST_INTERVAL)
    round2_text = call_api_with_retry(
        client,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64",
                 "media_type": "image/png", "data": enh_b64}},
                {"type": "image", "source": {"type": "base64",
                 "media_type": "image/png", "data": orig_b64}},
                {"type": "text", "text": ROUND2_PROMPT},
            ],
        }],
        max_tokens=4096,
        label=f"{region_id} Round2",
    )

    # ── 第三轮：交叉校验 + 专业术语 ──
    round3_prompt = ROUND3_PROMPT_TEMPLATE.format(
        round1_text=round1_text,
        round2_text=round2_text,
    )
    time.sleep(REQUEST_INTERVAL)
    round3_raw = call_api_with_retry(
        client,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64",
                 "media_type": "image/png", "data": orig_b64}},
                {"type": "text", "text": round3_prompt},
            ],
        }],
        max_tokens=4096,
        label=f"{region_id} Round3",
    )

    try:
        round3 = parse_json_from_response(round3_raw)
    except json.JSONDecodeError:
        # 解析失败时降级：取 round2_text 作为 final_text
        round3 = {
            "final_text": round2_text,
            "corrections": [],
            "uncertain_chars": [],
            "confidence": 0.5,
        }

    final_text, jp_count = filter_japanese(round3.get("final_text", ""))
    if jp_count:
        print(f"  [WARNING] {region_id} 检测到 {jp_count} 个日文字符，已标记")

    # 将 final_text 按行分割，与 struct_lines 元数据对齐
    text_lines = [ln for ln in final_text.split("\n")]
    confidence = float(round3.get("confidence", 0.8))

    merged_lines = _merge_lines_with_struct(text_lines, struct_lines, confidence)

    return {
        "region_id": region_id,
        "region_type": "TEXT_BLOCK",
        "bbox": bbox,
        "final_text": final_text,
        "lines": merged_lines,
        "corrections_applied": round3.get("corrections", []),
        "uncertain_chars": round3.get("uncertain_chars", []),
        "overall_confidence": confidence,
        "japanese_chars_detected": jp_count,
        "_raw": {
            "round1": round1_text,
            "round2": round2_text,
            "round3_parsed": round3,
        },
    }


def _merge_lines_with_struct(
    text_lines: list[str],
    struct_lines: list[dict],
    default_confidence: float,
) -> list[dict]:
    """
    将 OCR 文字行与 Phase 1 结构分析的行元数据合并。
    文字内容以 OCR 为准，字号/颜色/方向以结构分析为准（有则用，无则默认）。
    """
    result = []
    for i, content in enumerate(text_lines):
        if not content.strip():
            continue
        struct = struct_lines[i] if i < len(struct_lines) else {}
        result.append({
            "line_num": i + 1,
            "content": content,
            "font_level": struct.get("font_level", "body"),
            "color": struct.get("color", "black"),
            "direction": struct.get("direction", "horizontal"),
            "has_emphasis": struct.get("has_emphasis", False),
            "confidence": default_confidence,
        })
    return result


# ─────────────────────────── LABEL_SYSTEM / DIMENSION OCR ───────────────────────────

def ocr_label_region(
    client: anthropic.Anthropic,
    img_bgr: np.ndarray,
    region: dict,
) -> dict:
    """对 LABEL_SYSTEM 或 DIMENSION 区域做单轮 OCR。"""
    bbox = region["bbox"]
    region_id = region["id"]
    enhanced = enhance_text_region(img_bgr, bbox)
    orig_b64 = encode_array(enhanced["original_crop"])

    time.sleep(REQUEST_INTERVAL)
    raw = call_api_with_retry(
        client,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64",
                 "media_type": "image/png", "data": orig_b64}},
                {"type": "text", "text": LABEL_PROMPT},
            ],
        }],
        max_tokens=1024,
        label=f"{region_id} Labels",
    )

    try:
        parsed = parse_json_from_response(raw)
    except json.JSONDecodeError:
        parsed = {"labels": [], "dimensions": []}

    # 日文过滤
    total_jp = 0
    for item in parsed.get("labels", []):
        filtered, n = filter_japanese(item.get("text", ""))
        item["text"] = filtered
        total_jp += n

    return {
        "region_id": region_id,
        "region_type": region.get("type"),
        "bbox": bbox,
        "labels": parsed.get("labels", []),
        "dimensions": parsed.get("dimensions", []),
        "japanese_chars_detected": total_jp,
    }


# ─────────────────────────── 单页处理 ───────────────────────────

def process_page(
    page_num: int,
    client: anthropic.Anthropic,
    force: bool,
) -> tuple[int, str, dict]:
    """
    处理单页所有文字相关区域，返回 (page_num, status, stats)。
    status: "ok" | "skip" | "no_struct" | "no_image" | "error"
    """
    struct_path = STRUCTURE_DIR / f"page_{page_num:03d}.json"
    img_path = PAGES_DIR / f"page_{page_num:03d}.png"
    out_json = OUT_DIR / f"page_{page_num:03d}.json"
    out_raw = OUT_DIR / f"page_{page_num:03d}_raw.json"
    out_err = OUT_DIR / f"page_{page_num:03d}.error.txt"
    page_label = f"page_{page_num:03d}"

    if not force and out_json.exists():
        return page_num, "skip", {}
    if not struct_path.exists():
        return page_num, "no_struct", {}
    if not img_path.exists():
        return page_num, "no_image", {}

    struct = json.loads(struct_path.read_text(encoding="utf-8"))
    regions = struct.get("regions", [])

    # 按类型分类
    text_blocks = [r for r in regions if r.get("type") == "TEXT_BLOCK"]
    label_regions = [r for r in regions if r.get("type") in ("LABEL_SYSTEM", "DIMENSION")]

    if not text_blocks and not label_regions:
        # 纯插图页面
        return page_num, "skip_illus_only", {}

    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        return page_num, "error", {"msg": "图片读取失败"}

    t0 = time.time()
    text_results = []
    label_results = []
    total_jp = 0
    total_corrections = 0
    all_confidences = []
    raw_records = []

    try:
        # ── TEXT_BLOCK 三轮 OCR ──
        for region in text_blocks:
            struct_lines = region.get("lines", [])
            result = ocr_text_block(client, img_bgr, region, struct_lines)

            raw_records.append({
                "region_id": result["region_id"],
                "round1": result["_raw"]["round1"],
                "round2": result["_raw"]["round2"],
                "round3": result["_raw"]["round3_parsed"],
            })

            # 去掉 _raw（不写入最终 JSON）
            final_entry = {k: v for k, v in result.items() if k != "_raw"}
            text_results.append(final_entry)

            total_jp += result["japanese_chars_detected"]
            total_corrections += len(result["corrections_applied"])
            all_confidences.append(result["overall_confidence"])

        # ── LABEL_SYSTEM / DIMENSION 单轮 OCR ──
        for region in label_regions:
            result = ocr_label_region(client, img_bgr, region)
            label_results.append(result)
            total_jp += result["japanese_chars_detected"]

    except Exception:
        tb = traceback.format_exc()
        out_err.write_text(tb, encoding="utf-8")
        return page_num, "error", {"msg": tb[:200]}

    avg_conf = sum(all_confidences) / len(all_confidences) if all_confidences else None

    # ── 组装最终 JSON ──
    final = {
        "page_number": page_num,
        "text_regions": text_results,
        "label_regions": [r for r in label_results if r["region_type"] == "LABEL_SYSTEM"],
        "dimension_regions": [r for r in label_results if r["region_type"] == "DIMENSION"],
        "japanese_chars_detected": total_jp,
        "average_confidence": round(avg_conf, 3) if avg_conf is not None else None,
    }
    out_json.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── 保存原始三轮结果（调试用）──
    out_raw.write_text(
        json.dumps({"page_number": page_num, "raw": raw_records},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    elapsed = time.time() - t0
    return page_num, "ok", {
        "text_blocks": len(text_blocks),
        "label_regions": len(label_regions),
        "corrections": total_corrections,
        "japanese": total_jp,
        "avg_confidence": avg_conf,
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
    parser = argparse.ArgumentParser(description="Phase 2b: 文字通道 OCR")
    parser.add_argument("--pages", type=str, default=None,
                        help='指定页码范围，如 "1-5" / "2,4,6" / "3"')
    parser.add_argument("--force", action="store_true",
                        help="强制重新分析（覆盖已有 JSON）")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 收集所有 structure JSON ──
    all_structs = sorted(STRUCTURE_DIR.glob("page_*.json"))
    if not all_structs:
        print(f"[错误] {STRUCTURE_DIR} 中没有找到结构 JSON，请先运行 structure.py")
        sys.exit(1)

    all_nums = [int(p.stem.split("_")[1]) for p in all_structs]
    total = len(all_nums)
    print(f"[Phase 2b] 共找到 {total} 页结构数据")

    if args.pages:
        target_nums = parse_pages_arg(args.pages, max(all_nums))
        target_nums = [n for n in target_nums if n in all_nums]
    else:
        target_nums = all_nums

    if not args.force:
        pending = [n for n in target_nums
                   if not (OUT_DIR / f"page_{n:03d}.json").exists()]
        skipped_pre = len(target_nums) - len(pending)
        if skipped_pre:
            print(f"[Phase 2b] 跳过已处理 {skipped_pre} 页（使用 --force 强制重新分析）")
    else:
        pending = target_nums

    if not pending:
        print("[Phase 2b] 所有页面已处理完毕。")
        return

    print(f"[Phase 2b] 待处理 {len(pending)} 页，并发数 {MAX_CONCURRENT}\n")

    client = anthropic.Anthropic(api_key=_API_KEY)
    t_start = time.time()
    results: dict[int, tuple[str, dict]] = {}

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as executor:
        future_map = {
            executor.submit(process_page, n, client, args.force): n
            for n in pending
        }
        for future in as_completed(future_map):
            pn, status, stats = future.result()
            results[pn] = (status, stats)
            key = f"page_{pn:03d}"

            if status == "ok":
                conf_str = (f"{stats['avg_confidence']:.2f}"
                            if stats.get("avg_confidence") else "N/A")
                jp_warn = f"  ⚠ {stats['japanese']}个日文字" if stats["japanese"] else ""
                print(f"[{key}] ✓  "
                      f"文字块:{stats['text_blocks']}  "
                      f"标注:{stats['label_regions']}  "
                      f"修正:{stats['corrections']}  "
                      f"置信度:{conf_str}  "
                      f"{stats['elapsed']:.1f}s"
                      f"{jp_warn}")
            elif status in ("skip", "skip_illus_only"):
                pass
            else:
                print(f"[{key}] ✗  {status}")

    # ── 汇总 ──
    elapsed_total = time.time() - t_start
    ok_list = [n for n, (s, _) in results.items() if s == "ok"]
    err_list = [n for n, (s, _) in results.items() if s == "error"]

    all_confs = [
        stats["avg_confidence"]
        for _, (s, stats) in results.items()
        if s == "ok" and stats.get("avg_confidence") is not None
    ]
    global_avg_conf = sum(all_confs) / len(all_confs) if all_confs else None

    total_corrections = sum(
        stats.get("corrections", 0)
        for _, (s, stats) in results.items() if s == "ok"
    )
    total_jp_chars = sum(
        stats.get("japanese", 0)
        for _, (s, stats) in results.items() if s == "ok"
    )

    # 按置信度排名（最低的前5页需要人工校对）
    conf_ranking = sorted(
        [(n, stats["avg_confidence"])
         for n, (s, stats) in results.items()
         if s == "ok" and stats.get("avg_confidence") is not None],
        key=lambda x: x[1],
    )

    print()
    print("─" * 50)
    print(f"[Phase 2b] 汇总")
    print(f"  成功    : {len(ok_list)} 页")
    if err_list:
        print(f"  失败    : {len(err_list)} 页 → {[f'page_{n:03d}' for n in err_list]}")
    if global_avg_conf is not None:
        print(f"  平均置信度 : {global_avg_conf:.3f}")
    print(f"  总修正数  : {total_corrections}")
    if total_jp_chars:
        print(f"  ⚠ 日文字符: {total_jp_chars} 处，需人工确认")
    if conf_ranking[:5]:
        print(f"  低置信度页面（需校对）:")
        for n, c in conf_ranking[:5]:
            print(f"    page_{n:03d}: {c:.3f}")
    print(f"  总耗时  : {elapsed_total:.1f}s")
    print(f"  输出目录: {OUT_DIR}")
    print("─" * 50)


if __name__ == "__main__":
    main()
