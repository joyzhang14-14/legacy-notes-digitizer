"""
Step 2: Claude Vision 区域分割与文字识别
用法：
  python src/analyze.py               # 处理全部页面
  python src/analyze.py --test 3      # 只处理前3页
  python src/analyze.py --pages 1-5   # 处理第1-5页（支持 "1-5" / "2,4,6" / "3"）
  python src/analyze.py --force       # 强制重新分析（覆盖已有 JSON）
  python src/analyze.py --max-concurrent 2
"""

import argparse
import base64
import json
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import os

import anthropic
import yaml
from PIL import Image

# ─────────────────────────── API Key（显式从系统环境变量读取）───────────────────────────

_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not _API_KEY:
    print("[错误] 未找到 ANTHROPIC_API_KEY 环境变量，请先 export ANTHROPIC_API_KEY=...")
    sys.exit(1)

# ─────────────────────────── 路径（相对于项目根目录运行）───────────────────────────

ROOT = Path(__file__).parent.parent
CONFIG_PATH = ROOT / "config.yaml"

# ─────────────────────────── Vision Prompt ───────────────────────────

VISION_PROMPT = """你是一位专业的文档数字化专家，擅长分析手写中文教案。

我提供了同一页教案的两个版本：
- 第一张图：原始扫描件（颜色完整）
- 第二张图：高对比度处理版本（文字更清晰）

请综合两张图进行分析。这是中央美术学院服装设计专业的教案，内容涉及服装裁剪、打版、制图等。

## 任务

### 1. 判断页面类型
- text_only: 纯文字页面
- illustration_only: 纯插图页面（如整页服装版型图）
- mixed: 文字和插图混合

### 2. 识别所有文字区域
对每个文字块返回：
- bbox: 边界框 [x, y, width, height]（像素坐标，基于原始图片尺寸）
- content: 准确的文字内容（手写中文识别）
- font_level: "title" / "subtitle" / "body" / "annotation" / "label"
  - title: 页面大标题
  - subtitle: 小标题/序号标题（如 "① 胸省的种类和名称"）
  - body: 正文段落
  - annotation: 旁注/注释（通常字较小）
  - label: 图片上的标注文字（如 "肩省"、"胸围线"、"3～4"）
- color: "black" / "red" / "blue"（从原始图判断）
- writing_direction: "horizontal" / "vertical"
- confidence: 0.0-1.0 你对识别结果的置信度

### 3. 识别所有插图/图形区域
对每个插图返回：
- bbox: 边界框 [x, y, width, height]
- type: "pattern"(服装版型图) / "diagram"(示意图) / "sketch"(草图) / "other"
- description: 简短描述

### 4. 注意事项
- 标注文字（label）是写在插图旁边或上方的简短文字（如"肩省""A""B"），
  它们虽然靠近插图，但属于文字区域，需要被数字化替换
- bbox 坐标不要重叠——如果文字和插图靠得很近，bbox 之间要留有间隙
- 箭头、引线如果连接文字和插图，归入插图区域
- 有些页面可能有页码、页眉，也归入文字区域
- 注意页面可能是横向放置的，请据实识别方向

## 输出格式
严格返回以下 JSON，不要包含任何其他文字、不要包含 markdown 代码块标记：
{
  "page_type": "text_only|illustration_only|mixed",
  "page_width_px": <int>,
  "page_height_px": <int>,
  "text_regions": [
    {
      "id": "t1",
      "bbox": [x, y, w, h],
      "content": "识别的文字",
      "font_level": "title|subtitle|body|annotation|label",
      "color": "black|red|blue",
      "writing_direction": "horizontal|vertical",
      "confidence": 0.95
    }
  ],
  "illustration_regions": [
    {
      "id": "i1",
      "bbox": [x, y, w, h],
      "type": "pattern|diagram|sketch|other",
      "description": "描述"
    }
  ],
  "notes": "页面整体备注（可选）"
}"""

# ─────────────────────────── 服装专业术语表 ───────────────────────────

FASHION_TERMS = [
    "胸省", "肩省", "腰省", "肋省", "袖孔省", "领孔省",
    "胸围线", "腰围线", "臀围线", "原型", "省道", "裁片", "缝份",
    "前片", "后片", "袖片", "领片", "裁剪", "缝制", "打版", "放码",
    "经纱", "纬纱", "斜纱", "褶裥", "前中心线", "后中心线",
    "肩线", "侧缝线", "下摆线",
]


def _edit_distance(a: str, b: str) -> int:
    """Levenshtein 编辑距离"""
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for ch_a in a:
        curr = [prev[0] + 1]
        for j, ch_b in enumerate(b):
            curr.append(min(prev[j] + (0 if ch_a == ch_b else 1),
                            curr[-1] + 1, prev[j + 1] + 1))
        prev = curr
    return prev[-1]


def correct_terms(text: str) -> str:
    """对文字内容中的专业术语做模糊匹配纠正（编辑距离≤1）"""
    for term in FASHION_TERMS:
        # 在文本中找相同长度的候选子串
        for i in range(len(text) - len(term) + 1):
            candidate = text[i:i + len(term)]
            if candidate != term and _edit_distance(candidate, term) <= 1:
                text = text[:i] + term + text[i + len(term):]
                break
    return text


# ─────────────────────────── 后处理校验 ───────────────────────────

def clip_bbox(bbox: list, w: int, h: int) -> list:
    """将 bbox 裁剪到页面范围内"""
    x, y, bw, bh = bbox
    x = max(0, min(x, w))
    y = max(0, min(y, h))
    bw = max(0, min(bw, w - x))
    bh = max(0, min(bh, h - y))
    return [x, y, bw, bh]


def iou_area(a: list, b: list) -> int:
    """返回两个 bbox 的重叠面积"""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    return ix * iy


def shrink_text_bbox(tbbox: list, ibbox: list) -> list:
    """文字 bbox 与插图 bbox 有重叠时，向内收缩文字 bbox"""
    tx, ty, tw, th = tbbox
    ix, iy, iw, ih = ibbox
    # 四个方向上的重叠量
    overlap_right = (tx + tw) - ix
    overlap_bottom = (ty + th) - iy
    overlap_left = (ix + iw) - tx
    overlap_top = (iy + ih) - ty

    # 找最小收缩方向
    candidates = []
    if overlap_right > 0:
        candidates.append(("right", overlap_right))
    if overlap_bottom > 0:
        candidates.append(("bottom", overlap_bottom))
    if overlap_left > 0:
        candidates.append(("left", overlap_left))
    if overlap_top > 0:
        candidates.append(("top", overlap_top))

    if not candidates:
        return tbbox

    direction, amount = min(candidates, key=lambda c: c[1])
    margin = 2  # 留2px间距

    if direction == "right":
        tw = max(0, tw - amount - margin)
    elif direction == "bottom":
        th = max(0, th - amount - margin)
    elif direction == "left":
        tx = tx + amount + margin
        tw = max(0, tw - amount - margin)
    elif direction == "top":
        ty = ty + amount + margin
        th = max(0, th - amount - margin)

    return [tx, ty, tw, th]


def postprocess(data: dict, page_w: int, page_h: int) -> dict:
    """对 API 返回的 JSON 做后处理：bbox校验、重叠处理、文字清洗"""
    # 用 API 返回的尺寸（优先），否则用实际尺寸
    w = data.get("page_width_px") or page_w
    h = data.get("page_height_px") or page_h
    data["page_width_px"] = w
    data["page_height_px"] = h

    # 1. 裁剪越界 bbox
    for t in data.get("text_regions", []):
        t["bbox"] = clip_bbox(t["bbox"], w, h)
    for i in data.get("illustration_regions", []):
        i["bbox"] = clip_bbox(i["bbox"], w, h)

    # 2. 解决文字区域与插图区域的重叠（收缩文字）
    for t in data.get("text_regions", []):
        for ill in data.get("illustration_regions", []):
            if iou_area(t["bbox"], ill["bbox"]) > 0:
                t["bbox"] = shrink_text_bbox(t["bbox"], ill["bbox"])

    # 3. 文字内容清洗 + 术语纠正
    for t in data.get("text_regions", []):
        content = t.get("content", "")
        content = content.strip()
        content = re.sub(r" {2,}", " ", content)
        content = correct_terms(content)
        t["content"] = content

    return data


# ─────────────────────────── 图片缩放（超 token 时用）───────────────────────────

def resize_image_bytes(path: Path, scale: float) -> bytes:
    """将图片按比例缩小后返回 PNG bytes"""
    img = Image.open(path)
    new_w = int(img.width * scale)
    new_h = int(img.height * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    from io import BytesIO
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def read_image_b64(path: Path, scale: float = 1.0) -> str:
    if scale == 1.0:
        data = path.read_bytes()
    else:
        data = resize_image_bytes(path, scale)
    return base64.standard_b64encode(data).decode("utf-8")


# ─────────────────────────── API 调用 ───────────────────────────

def call_api(client: anthropic.Anthropic, orig_path: Path, enh_path: Path,
             model: str, max_tokens: int, scale: float = 1.0) -> dict:
    """调用 Claude Vision API，返回解析后的 JSON dict"""
    orig_b64 = read_image_b64(orig_path, scale)
    enh_b64 = read_image_b64(enh_path, scale)

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64",
                                              "media_type": "image/png",
                                              "data": orig_b64}},
                {"type": "image", "source": {"type": "base64",
                                              "media_type": "image/png",
                                              "data": enh_b64}},
                {"type": "text", "text": VISION_PROMPT},
            ],
        }],
    )

    raw_text = response.content[0].text
    # 去除 Claude 有时会包裹的 markdown 代码块标记
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw_text.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return json.loads(cleaned)


def analyze_page(
    client: anthropic.Anthropic,
    page_num: int,
    orig_path: Path,
    enh_path: Path,
    out_dir: Path,
    cfg: dict,
    force: bool,
    request_interval: float,
) -> tuple[int, str, float | None]:
    """
    处理单页。返回 (page_num, status, avg_confidence)
    status: "ok" | "skip" | "error"
    """
    out_json = out_dir / f"page_{page_num:03d}.json"
    out_err = out_dir / f"page_{page_num:03d}.error.txt"

    if not force and out_json.exists():
        return page_num, "skip", None

    analysis_cfg = cfg["analysis"]
    model = analysis_cfg["model"]
    max_tokens = analysis_cfg.get("max_tokens", 4096)
    max_retries = analysis_cfg["max_retries"]
    retry_delay = analysis_cfg["retry_delay"]

    time.sleep(request_interval)  # 请求间隔

    # 获取原始图片尺寸（用于后处理校验）
    with Image.open(orig_path) as img:
        page_w, page_h = img.size

    last_error = None
    raw_text = None

    for attempt in range(1, max_retries + 1):
        try:
            scale = 1.0 if attempt == 1 else 0.7  # 第2次起缩小图片
            data = call_api(client, orig_path, enh_path, model, max_tokens, scale)
            data = postprocess(data, page_w, page_h)
            out_json.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

            confidences = [t.get("confidence", 0) for t in data.get("text_regions", [])]
            avg_conf = sum(confidences) / len(confidences) if confidences else None
            return page_num, "ok", avg_conf

        except anthropic.RateLimitError:
            print(f"  [rate limit] page_{page_num:03d}，等待 {retry_delay}s 后重试 ({attempt}/{max_retries})...")
            time.sleep(retry_delay)
            last_error = "rate_limit"

        except json.JSONDecodeError as e:
            # JSON 解析失败：保存原始响应
            if raw_text:
                out_err.write_text(raw_text, encoding="utf-8")
            last_error = f"json_decode: {e}"
            break  # JSON 错误不重试

        except Exception as e:
            last_error = traceback.format_exc()
            if attempt < max_retries:
                print(f"  [错误] page_{page_num:03d} 第{attempt}次失败: {e}，重试...")
                time.sleep(5)
            else:
                break

    # 所有重试均失败
    out_err.write_text(str(last_error), encoding="utf-8")
    return page_num, "error", None


# ─────────────────────────── 页码范围解析 ───────────────────────────

def parse_pages_arg(arg: str, total: int) -> list[int]:
    """
    解析 --pages 参数：
      "1-5"   → [1,2,3,4,5]
      "1,3,5" → [1,3,5]
      "3"     → [3]
    """
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
    parser = argparse.ArgumentParser(description="Claude Vision 页面分析")
    parser.add_argument("--force", action="store_true", help="强制重新分析（覆盖已有 JSON）")
    parser.add_argument("--test", type=int, metavar="N", help="只处理前 N 页（测试用）")
    parser.add_argument("--pages", type=str, metavar="RANGE",
                        help="指定页码范围，如 '1-5' 或 '1,3,5'")
    parser.add_argument("--max-concurrent", type=int, metavar="N",
                        help="最大并发请求数（覆盖 config.yaml）")
    args = parser.parse_args()

    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    inter_dir = ROOT / cfg["paths"]["intermediate_dir"]
    pages_dir = inter_dir / "pages"
    enhanced_dir = inter_dir / "enhanced_ocr"
    out_dir = inter_dir / "page_data"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 收集所有已有 pages
    all_page_files = sorted(pages_dir.glob("page_*.png"))
    if not all_page_files:
        print(f"[错误] {pages_dir} 中没有找到图片，请先运行 preprocess.py")
        sys.exit(1)

    total = len(all_page_files)
    print(f"共找到 {total} 页")

    # 确定要处理的页码列表
    if args.pages:
        target_nums = parse_pages_arg(args.pages, total)
    elif args.test:
        target_nums = list(range(1, args.test + 1))
        print(f"[测试模式] 只处理前 {args.test} 页")
    else:
        target_nums = list(range(1, total + 1))

    # 过滤出实际存在的文件
    tasks: list[tuple[int, Path, Path]] = []
    for num in target_nums:
        orig = pages_dir / f"page_{num:03d}.png"
        enh = enhanced_dir / f"page_{num:03d}.png"
        if not orig.exists():
            print(f"  [跳过] page_{num:03d}.png 不存在")
            continue
        if not enh.exists():
            print(f"  [警告] page_{num:03d}.png 无增强版，使用原始版替代")
            enh = orig
        tasks.append((num, orig, enh))

    analysis_cfg = cfg["analysis"]
    max_concurrent = args.max_concurrent or analysis_cfg["max_concurrent"]
    request_interval = analysis_cfg["request_interval"]

    client = anthropic.Anthropic(api_key=_API_KEY)

    # 并发处理
    start_time = time.time()
    results: dict[int, tuple[str, float | None]] = {}
    done_count = 0
    PROGRESS_INTERVAL = 5

    def task_fn(item):
        num, orig, enh = item
        return analyze_page(client, num, orig, enh, out_dir, cfg, args.force, request_interval)

    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        future_map = {executor.submit(task_fn, t): t[0] for t in tasks}
        for future in as_completed(future_map):
            num, status, avg_conf = future.result()
            results[num] = (status, avg_conf)
            done_count += 1

            if done_count % PROGRESS_INTERVAL == 0 or done_count == len(tasks):
                elapsed = time.time() - start_time
                print(f"  [{done_count}/{len(tasks)}] page_{num:03d} → {status}"
                      f"  累计耗时: {elapsed:.1f}s")

    # ─── 汇总 ───
    elapsed_total = time.time() - start_time
    ok_list = [n for n, (s, _) in results.items() if s == "ok"]
    skip_list = [n for n, (s, _) in results.items() if s == "skip"]
    err_list = [n for n, (s, _) in results.items() if s == "error"]

    all_confs = [c for _, (s, c) in results.items() if s == "ok" and c is not None]
    avg_conf_total = sum(all_confs) / len(all_confs) if all_confs else None

    low_conf_threshold = cfg.get("qa", {}).get("low_confidence_threshold", 0.7)
    low_conf_pages = []
    for num in ok_list:
        _, conf = results[num]
        if conf is not None and conf < low_conf_threshold:
            low_conf_pages.append(f"page_{num:03d}(conf={conf:.2f})")

    print("\n" + "=" * 55)
    print("分析完成")
    print(f"  处理页数：{len(tasks)}")
    print(f"  成功    ：{len(ok_list)}")
    print(f"  跳过    ：{len(skip_list)}（已存在，用 --force 重新分析）")
    print(f"  失败    ：{len(err_list)}{' → ' + str(err_list) if err_list else ''}")
    print(f"  平均置信度：{avg_conf_total:.2f}" if avg_conf_total is not None else "  平均置信度：N/A")
    if low_conf_pages:
        print(f"  低置信度页面（< {low_conf_threshold}）：{low_conf_pages}")
    print(f"  总耗时  ：{elapsed_total:.1f} 秒")
    print(f"  输出目录：{out_dir}")


if __name__ == "__main__":
    main()
