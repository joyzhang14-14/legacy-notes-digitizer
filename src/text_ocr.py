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
