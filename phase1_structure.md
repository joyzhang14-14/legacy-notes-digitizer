```xml
<task>
  <title>Phase 1: 结构扫描 — 在原始图上识别布局（双轮 Opus）</title>
  <role>
    你是一个文档结构分析专家。在完全不修改原始图像的前提下，
    识别页面上每个元素的精确位置、类型、和空间关系。
    输出的位置编码将作为后续所有处理步骤的唯一坐标参考。
  </role>

  <config>
    <input_pages_dir>./intermediate/pages/</input_pages_dir>
    <output_data_dir>./intermediate/structure/</output_data_dir>
    <model>claude-opus-4-20250514</model>
  </config>

  <critical_principle>
    绝对不对图像做任何增强、二值化、对比度调整。
    所有分析都基于原始扫描件，保留完整的灰度和颜色信息。
    这一步只做"看"和"标注"，不做"改"。
  </critical_principle>

  <goal>
    编写 src/structure.py 脚本：
    对每一页执行两轮 Opus Vision 扫描：
    - 第一轮：全局布局分析（识别大区域：文字块、插图、标注系统）
    - 第二轮：精细元素分析（每个文字块的精确内容预览、每条引线的路径）
    两轮结果合并为一个完整的结构 JSON。
  </goal>

  <!-- ========== 第一轮：全局布局 ========== -->
  <round id="1" name="全局布局分析">
    <purpose>
      识别页面的宏观结构：哪些区域是插图、哪些是文字、
      它们的相对位置关系如何。
    </purpose>

    <vision_prompt>
      <![CDATA[
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
}
      ]]>
    </vision_prompt>
  </round>

  <!-- ========== 第二轮：精细元素分析 ========== -->
  <round id="2" name="精细元素分析">
    <purpose>
      基于第一轮的布局结果，对每个区域进行精细分析：
      - TEXT_BLOCK → 逐行识别文字内容（预览，不是最终OCR）
      - ILLUSTRATION → 描述图中细节、标注所有组成元素
      - LABEL_SYSTEM → 每条引线的起终点、关联的文字标签
      - DIMENSION → 具体数值和位置
    </purpose>

    <vision_prompt>
      <![CDATA[
你是一位专业的文档数字化专家。我已经完成了这页教案的全局布局分析。
现在需要你对每个区域做精细分析。

这是中央美术学院服装设计教授的手写教案。
语言：中文（绝对没有日文！看起来像日文的一定是手写中文）。
涉及领域：人体解剖、服装裁剪、版型设计、缝制工艺。

以下是第一轮识别出的区域布局：
{round1_result_json}

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
{
  "page_number": <int>,
  "detailed_regions": [
    {
      "id": "r1",
      "type": "TEXT_BLOCK",
      "lines": [
        {
          "line_num": 1,
          "y_position": 120,
          "content": "胸省的裁制",
          "uncertain_chars": [],
          "font_level": "title",
          "color": "black",
          "direction": "horizontal",
          "has_emphasis": true
        }
      ]
    },
    {
      "id": "r2",
      "type": "ILLUSTRATION",
      "illustration_type": "skeleton",
      "visual_elements": ["正面人体骨骼", "肋骨", "脊柱", "骨盆"],
      "overlapping_text": [
        {"content": "胸大", "position": [x, y], "color": "red"}
      ],
      "line_style": "mixed",
      "colors_used": ["black", "red"]
    },
    {
      "id": "r3",
      "type": "LABEL_SYSTEM",
      "labels": [
        {
          "text": "腓肠",
          "color": "red",
          "text_position": [85, 920],
          "line_start": [120, 915],
          "line_end": [160, 850],
          "line_type": "arrow",
          "target_illustration": "r2"
        }
      ]
    },
    {
      "id": "r4",
      "type": "DIMENSION",
      "values": [
        {"value": "3~4", "position": [340, 560], "endpoints": [[320, 555], [380, 565]]}
      ]
    }
  ]
}
      ]]>
    </vision_prompt>
  </round>

  <!-- ========== 合并两轮结果 ========== -->
  <merge_strategy>
    合并第一轮和第二轮的结果为最终结构 JSON：
    intermediate/structure/page_NNN.json

    最终 JSON 结构：
    {
      "page_number": N,
      "page_width_px": W,
      "page_height_px": H,
      "layout_summary": "...",
      "regions": [
        {
          "id": "r1",
          "type": "TEXT_BLOCK",
          "bbox": [x, y, w, h],
          "confidence": 0.9,
          "lines": [...],       // 来自第二轮
          "related_to": [...]
        },
        {
          "id": "r2",
          "type": "ILLUSTRATION",
          "bbox": [x, y, w, h],
          "illustration_type": "skeleton",
          "visual_elements": [...],
          "overlapping_text": [...],
          "line_style": "mixed",
          "colors_used": ["black", "red"]
        },
        ...
      ]
    }
  </merge_strategy>

  <implementation_notes>
    <note>
      两轮都发送原始图片（不是增强版），但第二轮同时发送第一轮的 JSON 结果
      作为文字上下文，帮助 Opus 理解已知的布局信息。
    </note>
    <note>
      图片发送时使用原始分辨率（不缩小），Opus 需要看清每一笔细节。
      如果图片超过 API 限制（20MB），缩小到长边 4000px。
    </note>
    <note>
      如果某页分析失败（JSON解析错误），保存原始响应并重试一次。
    </note>
  </implementation_notes>

  <batch_optimization>
    <item>并发2个请求（Opus rate limit 较严格）</item>
    <item>每次请求间隔 2 秒</item>
    <item>支持 --pages 10-20 指定范围</item>
    <item>支持 --force 强制重新分析</item>
    <item>跳过已有 JSON 的页面</item>
    <item>每页打印: 区域数量、各类型数量、耗时</item>
    <item>429 rate limit 自动等待 120 秒重试</item>
  </batch_optimization>

  <dependencies>
    anthropic, Pillow, numpy
  </dependencies>

  <cli>
    python src/structure.py [--pages 10-20] [--force]
  </cli>
</task>
```
