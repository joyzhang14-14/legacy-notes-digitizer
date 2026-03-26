"""
Phase 2b: 文字通道 — Gemini×2 OCR + Claude 交叉校验 + 精确位置编码
用法：
  python src/text_ocr.py                           # 处理全部页面
  python src/text_ocr.py --pages 10-20             # 处理第10-20页
  python src/text_ocr.py --force                   # 强制覆盖已有结果
  python src/text_ocr.py --enhancement heavy       # 指定增强级别
  python src/text_ocr.py --corrections corrections.json  # 应用人工修正
"""

import argparse
import io
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

import anthropic
import cv2
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

# ─────────────────────────── 环境变量 ───────────────────────────

# 优先加载项目根目录的 .env 文件
_ROOT = Path(__file__).parent.parent
load_dotenv(_ROOT / ".env")

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
CLAUDE_MODEL = "claude-opus-4-20250514"

MAX_CONCURRENT = 1        # Gemini 免费额度并发限制，保守设为 1
GEMINI_INTERVAL = 3.0     # Gemini 请求间隔（秒）
CLAUDE_INTERVAL = 2.0     # Claude 请求间隔（秒）
RATE_LIMIT_WAIT = 60
MAX_RETRIES = 4
TEXT_CROP_PADDING = 15

# ─────────��───────────────── Gemini 客户端 ───────────────────────────

def create_gemini_client() -> genai.Client:
    """创建 Gemini 客户端。"""
    return genai.Client(api_key=_GEMINI_API_KEY)


def gemini_ocr(client: genai.Client, image_arrays: list, prompt: str) -> str:
    """
    调用 Gemini Vision 进行 OCR。
    image_arrays: numpy 图片数组列表（BGR 或灰度）。
    返回模型原始文本。
    """
    parts = []
    for arr in image_arrays:
        if arr.ndim == 2:
            pil = Image.fromarray(arr)
        else:
            pil = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        parts.append(
            types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png")
        )
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


# ─────────────────────────── 多级对比度增强 ───────────────────────────

def multi_level_enhance(crop_bgr: np.ndarray) -> tuple[dict, str]:
    """
    生成 5 个不同增强级别的版本（灰度）。
    返回 (versions_dict, auto_best_key)。
    版本键：original / light / medium / heavy / extreme
    """
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(gray.mean())
    kernel_sharp = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

    # Level 0: 原始灰度
    versions = {"original": gray}

    # Level 1: 轻度增强（正常页面）
    clahe1 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    versions["light"] = clahe1.apply(gray)

    # Level 2: 中度增强（偏暗页面）
    clahe2 = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(6, 6))
    enhanced2 = clahe2.apply(gray)
    versions["medium"] = np.clip(
        cv2.filter2D(enhanced2, -1, kernel_sharp), 0, 255
    ).astype(np.uint8)

    # Level 3: 重度增强（很暗的页面）
    clahe3 = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(4, 4))
    enhanced3 = clahe3.apply(gray)
    lut = np.array([
        int(((i / 255.0) ** 0.5) * 255) for i in range(256)
    ], dtype=np.uint8)
    brightened = cv2.LUT(enhanced3, lut)
    versions["heavy"] = np.clip(
        cv2.filter2D(brightened, -1, kernel_sharp), 0, 255
    ).astype(np.uint8)

    # Level 4: 极端增强（曝光拉满 / 几乎全白）
    inverted = 255 - gray
    clahe4 = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(4, 4))
    versions["extreme"] = 255 - clahe4.apply(inverted)

    # 自动选择最佳级别
    if mean_brightness > 220:
        best_key = "extreme"
    elif mean_brightness > 190:
        best_key = "heavy"
    elif mean_brightness > 160:
        best_key = "medium"
    else:
        best_key = "light"

    return versions, best_key


def crop_region(img_bgr: np.ndarray, bbox: list, padding: int = TEXT_CROP_PADDING) -> np.ndarray:
    """裁剪 bbox 区域，带 padding。"""
    x, y, w, h = bbox
    img_h, img_w = img_bgr.shape[:2]
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(img_w, x + w + padding)
    y2 = min(img_h, y + h + padding)
    return img_bgr[y1:y2, x1:x2].copy()


# ─────────────────────────── 日文字符过滤 ───────────────────────────

_JP_PATTERN = re.compile(r"[\u3040-\u309F\u30A0-\u30FF]")


def filter_japanese(text: str) -> tuple[str, int]:
    """将日文字符替换为 [?X?] 标记，返回 (过滤后文本, 检测到的字符数)。"""
    chars = _JP_PATTERN.findall(text)
    if not chars:
        return text, 0
    for ch in set(chars):
        text = text.replace(ch, f"[?{ch}?]")
    return text, len(chars)


# ─────────────────────────── JSON 解析工具 ───────────────────────────

def parse_json_response(text: str) -> dict:
    """去除 markdown 代码块后解析 JSON。"""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return json.loads(cleaned)


# ─────────────────────────── Round 1 & 2 Prompts ───────────────────────────

ROUND1_PROMPT = """\
请准确识别这张图片中的手写中文文字。

关键规则：
1. 这是中国中央美术学院服装设计教授的手写教案
2. 语言：只有中文、英文字母、阿拉伯数字、数学符号
3. 绝对不包含日文（平假名、片假名）。如果你看到像日文的字符，那一定是手写中文，请猜测最接近的中文字
4. 如果某个字完全无法辨认，用 [?] 标记
5. 保持原始换行结构
6. 红色文字标记为 <red>文字</red>
7. 蓝色文字标记为 <blue>文字</blue>

请同时返回每行文字的位置信息（估算像素坐标即可）。

输出格式（严格JSON，无其他内容）：
{
  "lines": [
    {
      "line_num": 1,
      "content": "识别出的文字",
      "y_px": 50,
      "x_px": 30,
      "char_height_px": 35,
      "color": "black",
      "direction": "horizontal",
      "is_title": false
    }
  ]
}\
"""

ROUND2_PROMPT = """\
我提供了同一段手写文字的4个不同对比度版本（从原始到重度增强）。
请综合所有版本进行最准确的识别。
某些字可能在某个增强级别下更清晰——请逐字比较各版本。

规则：
1. 这是中国中央美术学院服装设计教授的手写教案
2. 只有中文+英文+数字，绝对没有日文
3. 内容涉及：服装裁剪、省道设计、人体结构、缝制工艺
4. 不确定的字用 [?X?] 标记（X是最佳猜测）
5. 红色文字用 <red>文字</red> 标记
6. 蓝色文字用 <blue>文字</blue> 标记

输出格式（严格JSON，无其他内容）：
{
  "lines": [
    {
      "line_num": 1,
      "content": "识别出的文字",
      "y_px": 50,
      "x_px": 30,
      "char_height_px": 35,
      "color": "black",
      "direction": "horizontal",
      "is_title": false
    }
  ]
}\
"""

# ─────────────────────────── 自适应 OCR（Round 1 带重试） ───────────────────────────

_LEVEL_ORDER = ["light", "medium", "heavy", "extreme"]
_UNCERTAIN_RATIO_THRESHOLD = 0.3


def adaptive_gemini_round1(
    client: genai.Client,
    versions: dict,
    auto_best: str,
    region_id: str,
) -> tuple[dict, str]:
    """
    Round 1：从自动推荐的增强级别开始，若不确定字符超过 30% 则自动升级。
    返回 (parsed_json, level_used)。
    """
    start_idx = _LEVEL_ORDER.index(auto_best) if auto_best in _LEVEL_ORDER else 0
    last_parsed = None

    for level in _LEVEL_ORDER[start_idx:]:
        time.sleep(GEMINI_INTERVAL)
        raw = gemini_ocr(client, [versions[level]], ROUND1_PROMPT)

        try:
            parsed = parse_json_response(raw)
        except (json.JSONDecodeError, ValueError):
            print(f"  [Round1/{level}] JSON 解析失败，升级...")
            last_parsed = {"lines": []}
            continue

        lines = parsed.get("lines", [])
        total_chars = sum(len(ln.get("content", "")) for ln in lines)
        uncertain = sum(
            ln.get("content", "").count("[?") for ln in lines
        )

        if total_chars == 0:
            print(f"  [Round1/{level}] 未识别到文字，升级...")
            last_parsed = parsed
            continue

        ratio = uncertain / total_chars
        if ratio > _UNCERTAIN_RATIO_THRESHOLD:
            print(f"  [Round1/{level}] 不确定率 {ratio:.0%}，升级...")
            last_parsed = parsed
            continue

        print(f"  [Round1/{level}] {total_chars}字，不确定率 {ratio:.0%} ✓")
        return parsed, level

    print(f"  [WARNING] {region_id} 所有增强级别均无法达到满意质量，使用最后结果")
    return last_parsed or {"lines": []}, _LEVEL_ORDER[-1]


def gemini_round2(
    client: genai.Client,
    versions: dict,
    region_id: str,
) -> dict:
    """
    Round 2：同时发送 4 张图（original/light/medium/heavy），综合识别。
    返回 parsed_json。
    """
    images = [
        versions["original"],
        versions["light"],
        versions["medium"],
        versions["heavy"],
    ]
    time.sleep(GEMINI_INTERVAL)
    raw = gemini_ocr(client, images, ROUND2_PROMPT)

    try:
        return parse_json_response(raw)
    except (json.JSONDecodeError, ValueError):
        print(f"  [Round2] {region_id} JSON 解析失败，返回空结果")
        return {"lines": []}


# ─────────────────────────── Round 3 Prompt & Claude 校验 ───────────────────────────

ROUND3_PROMPT_TEMPLATE = """\
你是服装设计和人体解剖学的专家。
下面是同一段手写中文教案的两次独立 OCR 结果（由 Gemini 识别）。
我也提供了原始图片供你参考。

## Gemini 第一次识别结果：
{round1_json}

## Gemini 第二次识别结果：
{round2_json}

## 专业术语参考词典：

### 服装裁剪/设计：
胸省、肩省、腰省、肋省、袖孔省、领孔省、省道、省尖、省量、
原型、衣身原型、裁片、缝份、放码、打版、制图、
前片、后片、袖片、领片、裁剪、缝制、
胸围线、腰围线、臀围线、袖笼线、领围线、
贴边、挂面、里布、衬布、粘合衬、
公主线、刀背缝、分割线、面料、
省道转移、省道设计、省道变化、
A字型、H型、X型、T型、
款式图、效果图、结构图、样板、

### 人体解剖：
骨骼、脊柱、颈椎、胸椎、腰椎、骶骨、尾骨、
肋骨、胸骨、锁骨、肩胛骨、肱骨、桡骨、尺骨、
骨盆、髂骨、股骨、胫骨、腓骨、
胸大肌、三角肌、斜方肌、背阔肌、
腹直肌、腹外斜肌、臀大肌、股四头肌、腓肠肌、比目鱼肌、
体表标志、颈窝、乳点、肩点、人体比例、

### 教学术语：
教学目的、教学内容、教学方法、教学重点、教学难点、
讲授、示范、练习、作业、考核、课时、教案、大纲、
任课教师、教学安排、预备、

## 任务：
1. 对比两次结果，找出所有差异
2. 对每处差异，结合上下文语义和专业术语判断正确文字
3. 修正明显的语义错误：
   - 如 "我穿" 可能是 "裁片" 或 "裁剪"
   - "异形" 可能是 "造型"
   - 如果出现日文字符，替换为最接近的中文
4. 保留位置信息（y_px, x_px, char_height_px），两轮不一致时取平均值

## 输出格式（严格JSON，无其他内容）：
{{
  "lines": [
    {{
      "line_num": 1,
      "content": "最终准确文字",
      "y_px": 50,
      "x_px": 30,
      "char_height_px": 35,
      "char_width_px": 30,
      "line_height_px": 45,
      "color": "black",
      "direction": "horizontal",
      "is_title": false,
      "font_level": "title|subtitle|body|annotation|label",
      "confidence": 0.92
    }}
  ],
  "corrections": [
    {{"original": "原始错误", "corrected": "修正后", "reason": "理由"}}
  ],
  "page_confidence": 0.88
}}\
"""

LABEL_PROMPT = """\
请识别这张图片中的标注文字。这是服装设计教案中的插图标注。

图中有引线从文字标签指向插图的某个部位。请识别每个标签的文字内容。

规则：
1. 只有中文、英文、数字，没有日文
2. 常见标注词：胸大肌、三角肌、腓肠肌、比目鱼肌、肩省、胸围线、前中心线、后中心线等
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


def claude_round3(
    claude_client: anthropic.Anthropic,
    round1_parsed: dict,
    round2_parsed: dict,
    orig_crop: np.ndarray,
    region_id: str,
) -> dict:
    """
    Round 3：将 Gemini 两轮结果发给 Claude Opus，做语义校验 + 专业术语修复。
    返回 parsed_json（含 lines / corrections / page_confidence）。
    """
    import base64

    # 将原图编码为 base64 供 Claude 参考
    pil = Image.fromarray(cv2.cvtColor(orig_crop, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    orig_b64 = base64.standard_b64encode(buf.getvalue()).decode("utf-8")

    round3_prompt = ROUND3_PROMPT_TEMPLATE.format(
        round1_json=json.dumps(round1_parsed, ensure_ascii=False, indent=2),
        round2_json=json.dumps(round2_parsed, ensure_ascii=False, indent=2),
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            time.sleep(CLAUDE_INTERVAL)
            resp = claude_client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=8192,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": orig_b64,
                            },
                        },
                        {"type": "text", "text": round3_prompt},
                    ],
                }],
            )
            raw = resp.content[0].text
            return parse_json_response(raw)
        except anthropic.RateLimitError:
            if attempt == MAX_RETRIES:
                raise
            print(f"  [Claude 429] 等待 {RATE_LIMIT_WAIT}s 后重试 ({attempt}/{MAX_RETRIES})...")
            time.sleep(RATE_LIMIT_WAIT)
        except (json.JSONDecodeError, ValueError):
            print(f"  [Round3] {region_id} JSON 解析失败，降级使用 Round2 结果")
            return {
                "lines": round2_parsed.get("lines", []),
                "corrections": [],
                "page_confidence": 0.6,
            }
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise
            print(f"  [Claude 错误] {e}，5s 后重试...")
            time.sleep(5)

    raise RuntimeError(f"Claude Round3 {region_id} 所有重试均失败")


# ─────────────────────────── corrections 工作流 ───────────────────────────

def build_correction_map(corrections: dict) -> dict:
    """
    从用户的手动修正中学习字符级映射规则。
    corrections 格式：{"page_3": {"line_1": {"original": "...", "corrected": "..."}}}
    返回：{"错误字": "正确字"}（至少出现 2 次才自动应用）
    """
    char_map: dict[str, dict[str, int]] = {}
    for page_fixes in corrections.values():
        for fix in page_fixes.values():
            orig = fix.get("original", "")
            corr = fix.get("corrected", "")
            for o, c in zip(orig, corr):
                if o != c:
                    char_map.setdefault(o, {})
                    char_map[o][c] = char_map[o].get(c, 0) + 1

    return {
        o: max(cmap, key=cmap.get)
        for o, cmap in char_map.items()
        if max(cmap.values()) >= 2
    }


def apply_corrections(lines: list, corrections_map: dict) -> list:
    """将字符映射应用到所有行的 content 字段。"""
    if not corrections_map:
        return lines
    result = []
    for ln in lines:
        content = ln.get("content", "")
        for wrong, right in corrections_map.items():
            content = content.replace(wrong, right)
        result.append({**ln, "content": content})
    return result


# ─────────────────────────── 单区域 OCR ───────────────────────────

def ocr_text_block(
    gemini_client: genai.Client,
    claude_client: anthropic.Anthropic,
    img_bgr: np.ndarray,
    region: dict,
    forced_level: str | None,
) -> dict:
    """
    对 TEXT_BLOCK 区域执行三轮 OCR，返回完整结果 dict。
    forced_level: 如果指定则跳过自动选择，直接用该增强级别。
    """
    bbox = region["bbox"]
    region_id = region["id"]
    crop = crop_region(img_bgr, bbox)
    versions, auto_best = multi_level_enhance(crop)

    level = forced_level if forced_level else auto_best

    # Round 1：自适应 Gemini 识别
    print(f"  [{region_id}] Round1 (自动级别={level})...")
    round1_parsed, level_used = adaptive_gemini_round1(
        gemini_client, versions, level, region_id
    )

    # Round 2：4图综合识别
    print(f"  [{region_id}] Round2 (4图对比)...")
    round2_parsed = gemini_round2(gemini_client, versions, region_id)

    # Round 3：Claude 交叉校验
    print(f"  [{region_id}] Round3 (Claude 校验)...")
    round3_parsed = claude_round3(
        claude_client, round1_parsed, round2_parsed, crop, region_id
    )

    # 日文过滤
    lines = round3_parsed.get("lines", [])
    total_jp = 0
    filtered_lines = []
    for ln in lines:
        content, jp_count = filter_japanese(ln.get("content", ""))
        total_jp += jp_count
        filtered_lines.append({**ln, "content": content})

    if total_jp:
        print(f"  [WARNING] {region_id} 检测到 {total_jp} 个日文字符，已标记")

    return {
        "region_id": region_id,
        "region_type": "TEXT_BLOCK",
        "bbox": bbox,
        "enhancement_level_used": level_used,
        "lines": filtered_lines,
        "corrections_applied": round3_parsed.get("corrections", []),
        "overall_confidence": round3_parsed.get("page_confidence", 0.8),
        "japanese_chars_detected": total_jp,
        "_raw": {
            "round1": round1_parsed,
            "round2": round2_parsed,
            "round3": round3_parsed,
        },
    }


def ocr_label_region(
    gemini_client: genai.Client,
    img_bgr: np.ndarray,
    region: dict,
) -> dict:
    """对 LABEL_SYSTEM 或 DIMENSION 区域做单轮 Gemini OCR。"""
    bbox = region["bbox"]
    region_id = region["id"]
    crop = crop_region(img_bgr, bbox)
    versions, _ = multi_level_enhance(crop)

    time.sleep(GEMINI_INTERVAL)
    raw = gemini_ocr(gemini_client, [versions["light"]], LABEL_PROMPT)

    try:
        parsed = parse_json_response(raw)
    except (json.JSONDecodeError, ValueError):
        parsed = {"labels": [], "dimensions": []}

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
    gemini_client: genai.Client,
    claude_client: anthropic.Anthropic,
    force: bool,
    forced_level: str | None,
    corrections_map: dict,
) -> tuple[int, str, dict]:
    """处理单页，返回 (page_num, status, stats)。"""
    struct_path = STRUCTURE_DIR / f"page_{page_num:03d}.json"
    img_path = PAGES_DIR / f"page_{page_num:03d}.png"
    out_json = OUT_DIR / f"page_{page_num:03d}.json"
    out_raw = OUT_DIR / f"page_{page_num:03d}_raw.json"
    out_err = OUT_DIR / f"page_{page_num:03d}.error.txt"

    if not force and out_json.exists():
        return page_num, "skip", {}
    if not struct_path.exists():
        return page_num, "no_struct", {}
    if not img_path.exists():
        return page_num, "no_image", {}

    struct = json.loads(struct_path.read_text(encoding="utf-8"))
    regions = struct.get("regions", [])
    text_blocks = [r for r in regions if r.get("type") == "TEXT_BLOCK"]
    label_regions = [r for r in regions if r.get("type") in ("LABEL_SYSTEM", "DIMENSION")]

    if not text_blocks and not label_regions:
        return page_num, "skip_illus_only", {}

    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        return page_num, "error", {"msg": "图片读取失败"}

    t0 = time.time()
    text_results, label_results = [], []
    total_jp = total_corrections = 0
    all_confidences = []
    raw_records = []

    try:
        for region in text_blocks:
            result = ocr_text_block(
                gemini_client, claude_client, img_bgr, region, forced_level
            )
            # 应用用��修正映射
            result["lines"] = apply_corrections(result["lines"], corrections_map)

            raw_records.append({
                "region_id": result["region_id"],
                "round1": result["_raw"]["round1"],
                "round2": result["_raw"]["round2"],
                "round3": result["_raw"]["round3"],
            })
            final_entry = {k: v for k, v in result.items() if k != "_raw"}
            text_results.append(final_entry)

            total_jp += result["japanese_chars_detected"]
            total_corrections += len(result["corrections_applied"])
            all_confidences.append(result["overall_confidence"])

        for region in label_regions:
            result = ocr_label_region(gemini_client, img_bgr, region)
            label_results.append(result)
            total_jp += result["japanese_chars_detected"]

    except Exception:
        tb = traceback.format_exc()
        out_err.write_text(tb, encoding="utf-8")
        return page_num, "error", {"msg": tb[:200]}

    avg_conf = sum(all_confidences) / len(all_confidences) if all_confidences else None

    final = {
        "page_number": page_num,
        "ocr_engine": f"{GEMINI_MODEL} + {CLAUDE_MODEL}",
        "text_regions": text_results,
        "label_regions": [r for r in label_results if r["region_type"] == "LABEL_SYSTEM"],
        "dimension_regions": [r for r in label_results if r["region_type"] == "DIMENSION"],
        "japanese_chars_detected": total_jp,
        "average_confidence": round(avg_conf, 3) if avg_conf is not None else None,
    }
    out_json.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    out_raw.write_text(
        json.dumps({"page_number": page_num, "raw": raw_records}, ensure_ascii=False, indent=2),
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

def parse_pages_arg(arg: str, max_page: int) -> list[int]:
    pages = set()
    for part in arg.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            pages.update(range(int(start), int(end) + 1))
        else:
            pages.add(int(part))
    return sorted(p for p in pages if 1 <= p <= max_page)


# ─────────────────────────── 主流程 ───────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 2b: Gemini×2 OCR + Claude 交叉校验")
    parser.add_argument("--pages", type=str, default="10-20",
                        help='指定页码范围，如 "10-20" / "2,4,6" / "3"（默认 10-20）')
    parser.add_argument("--force", action="store_true",
                        help="强制重新分析（覆盖已有 JSON）")
    parser.add_argument(
        "--enhancement",
        choices=["auto", "light", "medium", "heavy", "extreme"],
        default="auto",
        help="指定增强级别（默认 auto，根据亮度自动选择）",
    )
    parser.add_argument("--corrections", type=str, default=None,
                        help="人工修正文件路径（corrections.json）")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 加载人工修正
    corrections_map: dict = {}
    if args.corrections:
        corr_path = Path(args.corrections)
        if corr_path.exists():
            corrections_raw = json.loads(corr_path.read_text(encoding="utf-8"))
            corrections_map = build_correction_map(corrections_raw)
            print(f"[修正] 从 {corr_path.name} 加载了 {len(corrections_map)} 条字符映射")
        else:
            print(f"[警告] 修正文件不存在：{corr_path}")

    forced_level = None if args.enhancement == "auto" else args.enhancement

    # 收集所有结构 JSON
    all_structs = sorted(STRUCTURE_DIR.glob("page_*.json"))
    if not all_structs:
        print(f"[错误] {STRUCTURE_DIR} 中没有结构 JSON，请先运行 structure.py")
        sys.exit(1)

    all_nums = [int(p.stem.split("_")[1]) for p in all_structs]
    print(f"[Phase 2b] 共找到 {len(all_nums)} 页结构数据")

    target_nums = parse_pages_arg(args.pages, max(all_nums))
    target_nums = [n for n in target_nums if n in all_nums]

    if not args.force:
        pending = [n for n in target_nums
                   if not (OUT_DIR / f"page_{n:03d}.json").exists()]
        skipped_pre = len(target_nums) - len(pending)
        if skipped_pre:
            print(f"[Phase 2b] 跳过已处理 {skipped_pre} 页（--force 可强制重跑）")
    else:
        pending = target_nums

    if not pending:
        print("[Phase 2b] 所有指定页面已处理完毕。")
        return

    print(f"[Phase 2b] 待处理 {len(pending)} 页，增强模式={args.enhancement}\n")

    gemini_client = create_gemini_client()
    claude_client = anthropic.Anthropic(api_key=_ANTHROPIC_API_KEY)

    t_start = time.time()
    results: dict[int, tuple[str, dict]] = {}

    # 顺序处理（Gemini 免费额度有并发限制）
    for n in pending:
        pn, status, stats = process_page(
            n, gemini_client, claude_client, args.force, forced_level, corrections_map
        )
        results[pn] = (status, stats)
        key = f"page_{pn:03d}"

        if status == "ok":
            conf_str = f"{stats['avg_confidence']:.2f}" if stats.get("avg_confidence") else "N/A"
            jp_warn = f"  ⚠ {stats['japanese']}个日文字" if stats["japanese"] else ""
            print(f"[{key}] ✓  文字块:{stats['text_blocks']}  "
                  f"标注:{stats['label_regions']}  修正:{stats['corrections']}  "
                  f"置信度:{conf_str}  {stats['elapsed']:.1f}s{jp_warn}")
        elif status not in ("skip", "skip_illus_only"):
            print(f"[{key}] ✗  {status}")

    # 汇总
    elapsed_total = time.time() - t_start
    ok_list = [n for n, (s, _) in results.items() if s == "ok"]
    err_list = [n for n, (s, _) in results.items() if s == "error"]
    all_confs = [
        stats["avg_confidence"]
        for _, (s, stats) in results.items()
        if s == "ok" and stats.get("avg_confidence") is not None
    ]
    global_avg_conf = sum(all_confs) / len(all_confs) if all_confs else None
    total_jp_chars = sum(
        stats.get("japanese", 0)
        for _, (s, stats) in results.items() if s == "ok"
    )
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
        print(f"  失败    : {[f'page_{n:03d}' for n in err_list]}")
    if global_avg_conf is not None:
        print(f"  平均置信度 : {global_avg_conf:.3f}")
    if total_jp_chars:
        print(f"  ⚠ 日文字符: {total_jp_chars} 处，需人工确认")
    if conf_ranking[:5]:
        print("  低置信度页面（需校对）:")
        for n, c in conf_ranking[:5]:
            print(f"    page_{n:03d}: {c:.3f}")
    print(f"  总耗时  : {elapsed_total:.1f}s")
    print(f"  输出目录: {OUT_DIR}")
    print("─" * 50)


if __name__ == "__main__":
    main()

