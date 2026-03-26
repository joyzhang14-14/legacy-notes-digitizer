```xml
<task>
  <title>Phase 2b: 文字通道 — 区域增强 + 三轮 Opus OCR + 交叉校验</title>
  <role>
    你是一位手写中文OCR专家，精通服装设计/裁剪/人体解剖专业术语。
    对每个文字区域进行多轮识别和校验，追求最高准确度。
  </role>

  <config>
    <input_pages_dir>./intermediate/pages/</input_pages_dir>
    <input_structure_dir>./intermediate/structure/</input_structure_dir>
    <output_dir>./intermediate/text_ocr/</output_dir>
    <model>claude-opus-4-20250514</model>
  </config>

  <critical_principle>
    文字通道的核心原则：最大化 OCR 准确度。
    - 只对文字区域做对比度增强（不影响插图）
    - 三轮独立 OCR + 投票机制
    - 第三轮带专业术语上下文引导
    - 绝对不允许出现日文（教案只有中文+英文+数字）
  </critical_principle>

  <goal>
    编写 src/text_ocr.py 脚本：
    1. 从 structure JSON 中找到所有 TEXT_BLOCK、LABEL_SYSTEM、DIMENSION 区域
    2. 对每个文字区域单独裁剪并增强对比度
    3. 执行三轮 Opus OCR
    4. 交叉校验 + 专业术语校正
    5. 输出最终文字结果
  </goal>

  <!-- ========== 文字区域增强 ========== -->
  <step name="文字区域单独增强">
    <description>
      只对文字区域做高对比度增强。每个文字区域单独裁剪后处理，
      不影响插图区域的原始对比度。
    </description>

    ```python
    def enhance_text_region(original_img, bbox, padding=15):
        """裁剪文字区域并增强对比度"""
        x, y, w, h = bbox
        img_h, img_w = original_img.shape[:2]
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img_w, x + w + padding)
        y2 = min(img_h, y + h + padding)

        crop = original_img[y1:y2, x1:x2].copy()

        # 转灰度
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # 检测整体亮度，自适应调整参数
        mean_brightness = gray.mean()
        if mean_brightness < 120:
            clip_limit = 4.0  # 偏暗页面，增强更强
        elif mean_brightness < 160:
            clip_limit = 3.0
        else:
            clip_limit = 2.0

        # CLAHE 自适应直方图均衡化
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(4, 4))
        enhanced = clahe.apply(gray)

        # 锐化
        kernel_sharp = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel_sharp)

        # 自适应阈值二值化（可选，用于辅助识别）
        binary = cv2.adaptiveThreshold(
            sharpened, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 15, 5
        )

        return {
            'enhanced_gray': enhanced,
            'sharpened': sharpened,
            'binary': binary,
            'original_crop': crop,
            'offset': (x1, y1)
        }
    ```
  </step>

  <!-- ========== 三轮 OCR ========== -->
  <step name="三轮 Opus OCR">
    <description>
      对每个文字区域执行三轮独立 OCR，使用不同的提示策略：
      - 第一轮：无上下文裸识别（baseline）
      - 第二轮：带增强图 + 原始图对比识别
      - 第三轮：带前两轮结果 + 专业术语词典 + 上下文校验
    </description>

    <round1_prompt>
      <![CDATA[
请准确识别这张图片中的手写中文文字。

重要规则：
1. 这是中文教案，只包含中文、英文字母、数字、数学符号。绝对没有日文。
2. 如果某个字不确定，用 [?X?] 标记（X 是你的最佳猜测）
3. 保持原始的换行和段落结构
4. 红色文字用 <red>文字</red> 标记
5. 蓝色文字用 <blue>文字</blue> 标记
6. 竖排文字按从上到下的顺序转写

只返回识别出的文字，不要其他说明。
      ]]>
    </round1_prompt>

    <round2_prompt>
      <![CDATA[
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

只返回识别出的文字，不要其他说明。
      ]]>
    </round2_prompt>

    <round3_prompt>
      <![CDATA[
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

## 输出格式（严格JSON）：
{
  "final_text": "最终准确文字（保留换行，红色用<red>标记，蓝色用<blue>标记）",
  "corrections": [
    {
      "original": "两次OCR中的错误文字",
      "corrected": "修正后的文字",
      "reason": "修正理由"
    }
  ],
  "uncertain_chars": [
    {
      "position": "第X行第Y个字",
      "best_guess": "猜测",
      "alternatives": ["备选1", "备选2"]
    }
  ],
  "confidence": 0.85
}
      ]]>
    </round3_prompt>

    <implementation>
      ```python
      import anthropic, base64, json, io
      from PIL import Image

      client = anthropic.Anthropic()

      def ocr_text_region(original_crop, enhanced_images, region_info):
          """对一个文字区域执行三轮OCR"""

          # 编码图片
          def encode_img(img_array):
              pil = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                                    if len(img_array.shape)==3
                                    else img_array)
              buf = io.BytesIO()
              pil.save(buf, format='PNG')
              return base64.standard_b64encode(buf.getvalue()).decode()

          # ---- 第一轮：裸识别 ----
          r1 = client.messages.create(
              model="claude-opus-4-20250514",
              max_tokens=4096,
              messages=[{
                  "role": "user",
                  "content": [
                      {"type": "image", "source": {"type": "base64",
                       "media_type": "image/png",
                       "data": encode_img(enhanced_images['sharpened'])}},
                      {"type": "text", "text": ROUND1_PROMPT}
                  ]
              }]
          )
          round1_text = r1.content[0].text

          # ---- 第二轮：双图对比识别 ----
          r2 = client.messages.create(
              model="claude-opus-4-20250514",
              max_tokens=4096,
              messages=[{
                  "role": "user",
                  "content": [
                      {"type": "image", "source": {"type": "base64",
                       "media_type": "image/png",
                       "data": encode_img(enhanced_images['enhanced_gray'])}},
                      {"type": "image", "source": {"type": "base64",
                       "media_type": "image/png",
                       "data": encode_img(original_crop)}},
                      {"type": "text", "text": ROUND2_PROMPT}
                  ]
              }]
          )
          round2_text = r2.content[0].text

          # ---- 第三轮：交叉校验 + 专业术语 ----
          round3_prompt = ROUND3_PROMPT.format(
              round1_text=round1_text,
              round2_text=round2_text
          )
          r3 = client.messages.create(
              model="claude-opus-4-20250514",
              max_tokens=4096,
              messages=[{
                  "role": "user",
                  "content": [
                      {"type": "image", "source": {"type": "base64",
                       "media_type": "image/png",
                       "data": encode_img(original_crop)}},
                      {"type": "text", "text": round3_prompt}
                  ]
              }]
          )
          result = json.loads(r3.content[0].text)

          return {
              'round1_raw': round1_text,
              'round2_raw': round2_text,
              'final': result,
              'region_id': region_info['id'],
              'region_type': region_info['type'],
              'bbox': region_info['bbox']
          }
      ```
    </implementation>
  </step>

  <!-- ========== 标注文字和尺寸的特殊处理 ========== -->
  <step name="LABEL_SYSTEM 和 DIMENSION 的处理">
    <description>
      标注文字（LABEL_SYSTEM）和尺寸标注（DIMENSION）通常是短文本，
      不需要三轮 OCR，但需要精确的位置信息。
    </description>

    <label_prompt>
      <![CDATA[
请识别这张图片中的标注文字。这是服装设计教案中的插图标注。

图中有红色/黑色的引线从文字标签指向插图的某个部位。
请识别每个标签的文字内容。

规则：
1. 只有中文、英文、数字，没有日文
2. 常见标注词：胸大肌、三角肌、腋外斜肌、腓肠肌、比目鱼肌、
   肩省、胸围线、前中心线、后中心线、等
3. 尺寸标注如：3~4, 0.5, H/4+10.5, W/2+3

返回JSON：
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
}
      ]]>
    </label_prompt>
  </step>

  <!-- ========== 日文过滤 ========== -->
  <step name="日文字符检测与过滤">
    ```python
    import re

    def contains_japanese(text):
        """检测文本中是否包含日文字符"""
        # 平假名: \u3040-\u309F
        # 片假名: \u30A0-\u30FF
        japanese_pattern = re.compile(r'[\u3040-\u309F\u30A0-\u30FF]')
        matches = japanese_pattern.findall(text)
        return len(matches) > 0, matches

    def filter_japanese(text):
        """将误识别为日文的字符替换为 [?] 标记"""
        has_jp, chars = contains_japanese(text)
        if has_jp:
            print(f"  WARNING: 检测到 {len(chars)} 个日文字符，已标记为不确定")
            for ch in set(chars):
                text = text.replace(ch, f'[?{ch}?]')
        return text
    ```
  </step>

  <!-- ========== 输出结构 ========== -->
  <output_structure>
    intermediate/text_ocr/
      page_010.json     ← 最终OCR结果
      page_010_raw.json ← 三轮原始结果（用于调试和人工校对）

    page_010.json 格式：
    {
      "page_number": 10,
      "text_regions": [
        {
          "region_id": "r1",
          "region_type": "TEXT_BLOCK",
          "bbox": [50, 30, 400, 200],
          "final_text": "骨骼\n\n一、对服装的裁制也越到人体必须。\n\n服装→美→舒适→功能性",
          "lines": [
            {
              "line_num": 1,
              "content": "骨骼",
              "font_level": "title",
              "color": "black",
              "direction": "horizontal",
              "has_emphasis": true,
              "confidence": 0.95
            },
            {
              "line_num": 2,
              "content": "一、对服装的裁制也越到人体必须。",
              "font_level": "body",
              "color": "black",
              "direction": "horizontal",
              "has_emphasis": false,
              "confidence": 0.82
            }
          ],
          "corrections_applied": [
            {"original": "异形", "corrected": "造型", "reason": "服装设计常用术语"}
          ],
          "overall_confidence": 0.88
        }
      ],
      "label_regions": [
        {
          "region_id": "r3",
          "labels": [
            {"text": "胸大肌", "color": "red", "position": [180, 620]},
            {"text": "三角肌", "color": "red", "position": [220, 570]}
          ]
        }
      ],
      "dimension_regions": [
        {
          "region_id": "r4",
          "dimensions": [
            {"value": "3~4", "position": [340, 560]}
          ]
        }
      ],
      "japanese_chars_detected": 0,
      "average_confidence": 0.85
    }
  </output_structure>

  <batch_optimization>
    <item>每个文字区域需要 3 次 API 调用，因此并发要控制: max_workers=2</item>
    <item>每次请求间隔 2 秒（Opus rate limit 严格）</item>
    <item>429 自动等待 120 秒重试，最多 5 次</item>
    <item>跳过已有 JSON 的页面（--force 强制）</item>
    <item>支持 --pages 10-20 指定范围</item>
    <item>每页打印: 区域数、平均置信度、修正数、日文字符数</item>
    <item>末尾汇总: 总修正数、总不确定字数、平均置信度排名</item>
  </batch_optimization>

  <dependencies>
    anthropic, opencv-python-headless, numpy, Pillow
  </dependencies>

  <cli>
    python src/text_ocr.py [--pages 10-20] [--force]
  </cli>
</task>
```
