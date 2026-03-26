```xml
<task>
  <title>Phase 3: 从白纸创建全新文档 — PDF(5层) + PPT + 纯文字</title>
  <role>
    你是一个文档合成工程师。从空白页面开始，用位置编码精确放置
    增强后的插图和数字化文字，创建全新的电子版文档。
    不在扫描件上叠加 — 完全重建。
  </role>

  <config>
    <input_pages_dir>./intermediate/pages/</input_pages_dir>
    <input_structure_dir>./intermediate/structure/</input_structure_dir>
    <input_illustrations_dir>./intermediate/illustrations/</input_illustrations_dir>
    <input_text_dir>./intermediate/text_ocr/</input_text_dir>
    <output_dir>./output/</output_dir>
    <config_file>./config.yaml</config_file>
  </config>

  <critical_principle>
    1. 从白纸开始 — 不用扫描件做背景（扫描件只作为可切换的参考图层）
    2. 字体必须嵌入 — 使用开源思源字体（pip 安装），确保任何电脑都能打开
    3. 位置编码精确还原 — 每个元素的位置严格按照 structure JSON 的坐标放置
  </critical_principle>

  <goal>
    编写 src/build.py 脚本，输出：
    1. output/教案_数字版.pdf — 5层可切换PDF
    2. output/教案_数字版.pptx — 可编辑PPT
    3. output/教案_纯文字.md — 纯文字Markdown
    4. output/校对报告.md — QA报告
  </goal>

  <!-- ========== 字体方案（最重要） ========== -->
  <font_solution>
    <description>
      上一版的 PDF 全白问题就是因为字体没嵌入。
      这次使用 pip 可安装的开源字体，彻底解决跨平台问题。
    </description>

    <setup>
      ```python
      # 在 requirements.txt 中添加:
      # fontools  (不需要，reportlab 自带 TTFont 注册)

      # 使用思源字体（Google Noto CJK）
      # 方案1: 用 pip 安装 noto fonts
      # pip install noto-fonts-cjk  （如果可用）

      # 方案2: 直接下载字体文件到项目目录
      import os, urllib.request

      FONT_DIR = os.path.join(os.path.dirname(__file__), '..', 'fonts')
      os.makedirs(FONT_DIR, exist_ok=True)

      FONT_URLS = {
          'NotoSansSC-Regular': 'https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf',
          'NotoSansSC-Bold': 'https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Bold.otf',
          'NotoSerifSC-Regular': 'https://github.com/googlefonts/noto-cjk/raw/main/Serif/OTF/SimplifiedChinese/NotoSerifCJKsc-Regular.otf',
          'NotoSerifSC-Bold': 'https://github.com/googlefonts/noto-cjk/raw/main/Serif/OTF/SimplifiedChinese/NotoSerifCJKsc-Bold.otf',
      }

      def download_fonts():
          for name, url in FONT_URLS.items():
              path = os.path.join(FONT_DIR, f'{name}.otf')
              if not os.path.exists(path):
                  print(f'Downloading {name}...')
                  urllib.request.urlretrieve(url, path)
                  print(f'  ✓ Saved to {path}')

      def register_fonts():
          """注册字体到 reportlab"""
          from reportlab.pdfbase import pdfmetrics
          from reportlab.pdfbase.ttfonts import TTFont

          download_fonts()  # 确保字体已下载

          font_map = {}
          for name in FONT_URLS:
              path = os.path.join(FONT_DIR, f'{name}.otf')
              try:
                  pdfmetrics.registerFont(TTFont(name, path))
                  font_map[name] = path
                  print(f'  ✓ Font registered: {name}')
              except Exception as e:
                  print(f'  ✗ Failed to register {name}: {e}')
          return font_map
      ```
    </setup>

    <mapping>
      font_level → 字体映射:
      - title      → NotoSansSC-Bold, 28-36pt
      - subtitle   → NotoSansSC-Bold, 20-24pt
      - body       → NotoSerifSC-Regular, 14-18pt
      - annotation → NotoSerifSC-Regular, 10-14pt
      - label      → NotoSansSC-Regular, 10-14pt
      - dimension  → NotoSansSC-Regular, 10-12pt

      color → RGB:
      - black → (0, 0, 0)
      - red   → (0.80, 0.08, 0.08)
      - blue  → (0.10, 0.20, 0.72)
    </mapping>

    <pptx_fonts>
      python-pptx 需要系统字体或嵌入字体。
      使用同样的 Noto Sans SC / Noto Serif SC。
      如果 Mac 上没安装，脚本自动将下载的 OTF 字体安装到
      ~/Library/Fonts/（当前用户级别，不需要管理员权限）。

      ```python
      import shutil, platform

      def install_fonts_for_pptx():
          if platform.system() == 'Darwin':  # macOS
              user_font_dir = os.path.expanduser('~/Library/Fonts')
              os.makedirs(user_font_dir, exist_ok=True)
              for name in FONT_URLS:
                  src = os.path.join(FONT_DIR, f'{name}.otf')
                  dst = os.path.join(user_font_dir, f'{name}.otf')
                  if not os.path.exists(dst):
                      shutil.copy2(src, dst)
                      print(f'  ✓ Installed font for system: {name}')
          # Linux/Windows 类似处理...
      ```
    </pptx_fonts>
  </font_solution>

  <!-- ========== 输出1: PDF（5层可切换） ========== -->
  <output id="pdf" name="5层可切换PDF">
    <description>
      从白纸创建全新PDF。5个图层全部可独立切换：

      ┌──────────────────────────────────────┐
      │ Layer 5: 标注文字 + 引线（数字版）     │ ← OCG 可切换
      │ Layer 4: 正文文字（数字版）            │ ← OCG 可切换
      │ Layer 3: 尺寸标注（数字版）            │ ← OCG 可切换
      │ Layer 2: 插图（增强版）               │ ← OCG 可切换
      │ Layer 1: 原始扫描件（参考底图）        │ ← OCG 可切换
      │ Layer 0: 白色背景                     │ ← 始终可见
      └──────────────────────────────────────┘

      效果组合示例：
      A) 全开 → 完整的数字版文档（白底+清晰插图+数字文字）
      B) 只开 Layer 1 → 看原始扫描件
      C) 关闭 Layer 1，开其他 → 白底上的纯数字版
      D) 开 Layer 1+2 → 原始扫描件+增强插图叠加
      E) 只开 Layer 2+5 → 只看插图和标注
    </description>

    <implementation>
      ```python
      from reportlab.pdfgen import canvas
      from reportlab.lib.units import inch

      def build_pdf(all_pages, output_path, config):
          c = canvas.Canvas(output_path)

          # 注册字体
          font_map = register_fonts()

          # 创建 5 个 OCG 图层
          layer_scan = c.addOCG("原始扫描件", visible=False)
          layer_illust = c.addOCG("增强插图", visible=True)
          layer_dim = c.addOCG("尺寸标注", visible=True)
          layer_text = c.addOCG("数字化文字", visible=True)
          layer_labels = c.addOCG("标注引线", visible=True)

          for page in all_pages:
              # 页面尺寸：按原始扫描件的宽高比，但设为合理的物理尺寸
              img_w, img_h = page['page_width_px'], page['page_height_px']
              # 假设 300 DPI → 计算物理尺寸（points）
              dpi = 300
              page_w = img_w / dpi * 72
              page_h = img_h / dpi * 72
              c.setPageSize((page_w, page_h))

              # 坐标转换函数
              sx = page_w / img_w
              sy = page_h / img_h
              def px2pdf(x, y, w=None, h=None):
                  """像素坐标 → PDF坐标（左下角为原点）"""
                  px = x * sx
                  py = page_h - y * sy  # Y轴翻转
                  if w is not None and h is not None:
                      pw = w * sx
                      ph = h * sy
                      return (px, py - ph, pw, ph)
                  return (px, py)

              # ---- Layer 0: 白色背景（始终可见） ----
              c.setFillColorRGB(1, 1, 1)
              c.rect(0, 0, page_w, page_h, fill=1, stroke=0)

              # ---- Layer 1: 原始扫描件（默认关闭） ----
              c.beginLayer(layer_scan)
              c.drawImage(page['original_image_path'],
                          0, 0, width=page_w, height=page_h)
              c.endLayer()

              # ---- Layer 2: 增强插图 ----
              c.beginLayer(layer_illust)
              for illust in page.get('illustrations', []):
                  img_path = illust['image_path']  # 增强后的插图PNG
                  ix, iy, iw, ih = px2pdf(*illust['bbox'])
                  c.drawImage(img_path, ix, iy, width=iw, height=ih,
                              preserveAspectRatio=True, mask='auto')
              c.endLayer()

              # ---- Layer 3: 尺寸标注 ----
              c.beginLayer(layer_dim)
              for dim in page.get('dimensions', []):
                  dx, dy = px2pdf(*dim['position'])
                  c.setFont('NotoSansSC-Regular', 10)
                  c.setFillColorRGB(0, 0, 0)
                  c.drawString(dx, dy, dim['value'])
              c.endLayer()

              # ---- Layer 4: 正文文字 ----
              c.beginLayer(layer_text)
              for text_region in page.get('text_regions', []):
                  for line in text_region.get('lines', []):
                      # 确定字体
                      font_cfg = get_font_config(line['font_level'])
                      font_name = font_cfg['name']
                      font_size = font_cfg['size']

                      # 确定颜色
                      cr, cg, cb = get_color_rgb(line.get('color', 'black'))
                      c.setFillColorRGB(cr, cg, cb)
                      c.setFont(font_name, font_size)

                      # 计算位置
                      rx, ry, rw, rh = px2pdf(*text_region['bbox'])
                      line_y = ry + rh - (line['line_num'] * font_size * 1.5)

                      # 绘制文字
                      text_content = line['content']
                      # 处理 <red>...</red> 标记
                      draw_colored_text(c, rx, line_y, text_content,
                                        font_name, font_size, (cr, cg, cb))
              c.endLayer()

              # ---- Layer 5: 标注引线 + 标注文字 ----
              c.beginLayer(layer_labels)
              for label_region in page.get('label_regions', []):
                  for label in label_region.get('labels', []):
                      # 绘制引线
                      if 'line_start' in label and 'line_end' in label:
                          lx1, ly1 = px2pdf(*label['line_start'])
                          lx2, ly2 = px2pdf(*label['line_end'])
                          cr, cg, cb = get_color_rgb(label.get('color', 'red'))
                          c.setStrokeColorRGB(cr, cg, cb)
                          c.setLineWidth(0.8)
                          c.line(lx1, ly1, lx2, ly2)
                          # 箭头
                          draw_arrowhead(c, lx1, ly1, lx2, ly2, size=4)

                      # 绘制标注文字
                      tx, ty = px2pdf(*label['text_position'])
                      c.setFont('NotoSansSC-Regular', 11)
                      c.setFillColorRGB(cr, cg, cb)
                      c.drawString(tx, ty, label['text'])
              c.endLayer()

              c.showPage()

          c.save()
          print(f"✓ PDF saved: {output_path} ({os.path.getsize(output_path)/1024/1024:.1f} MB)")
      ```
    </implementation>
  </output>

  <!-- ========== 输出2: PPT ========== -->
  <output id="pptx" name="可编辑PPT">
    <description>
      PPT 版本，方便后续手动修改文字。
      每张幻灯片从白色背景开始，依次放置插图和文字。
      原始扫描件作为隐藏的备注附图。
    </description>

    ```python
    from pptx import Presentation
    from pptx.util import Inches, Pt, Emu
    from pptx.dml.color import RGBColor
    from pptx.enum.text import PP_ALIGN

    def build_pptx(all_pages, output_path, config):
        prs = Presentation()

        for page in all_pages:
            # 动态设置幻灯片尺寸（匹配原始页面比例）
            img_w, img_h = page['page_width_px'], page['page_height_px']
            aspect = img_w / img_h
            if aspect > 1:  # 横向
                prs.slide_width = Inches(13.33)
                prs.slide_height = Inches(13.33 / aspect)
            else:  # 纵向
                prs.slide_height = Inches(10)
                prs.slide_width = Inches(10 * aspect)

            slide = prs.slides.add_slide(prs.slide_layouts[6])  # 空白布局

            # 坐标转换
            slide_w = prs.slide_width
            slide_h = prs.slide_height
            def px2emu(x, y, w=None, h=None):
                ex = int(x / img_w * slide_w)
                ey = int(y / img_h * slide_h)
                if w and h:
                    ew = int(w / img_w * slide_w)
                    eh = int(h / img_h * slide_h)
                    return (ex, ey, ew, eh)
                return (ex, ey)

            # 放置插图
            for illust in page.get('illustrations', []):
                ix, iy, iw, ih = px2emu(*illust['bbox'])
                slide.shapes.add_picture(
                    illust['image_path'],
                    left=ix, top=iy, width=iw, height=ih
                )

            # 放置文字
            for text_region in page.get('text_regions', []):
                rx, ry, rw, rh = px2emu(*text_region['bbox'])
                txBox = slide.shapes.add_textbox(left=rx, top=ry,
                                                  width=rw, height=rh)
                tf = txBox.text_frame
                tf.word_wrap = True

                for i, line in enumerate(text_region.get('lines', [])):
                    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
                    p.text = line['content']

                    font_cfg = get_font_config(line['font_level'])
                    p.font.name = 'Noto Sans SC'  # PPT使用系统字体名
                    p.font.size = Pt(font_cfg['size'])
                    p.font.bold = font_cfg.get('bold', False)

                    color = line.get('color', 'black')
                    if color == 'red':
                        p.font.color.rgb = RGBColor(204, 20, 20)
                    elif color == 'blue':
                        p.font.color.rgb = RGBColor(25, 51, 184)

            # 备注中记录元信息
            notes = slide.notes_slide
            notes.notes_text_frame.text = (
                f"Page {page['page_number']} | "
                f"Confidence: {page.get('average_confidence', 'N/A')} | "
                f"Illustrations: {len(page.get('illustrations', []))}"
            )

        prs.save(output_path)
    ```
  </output>

  <!-- ========== 输出3: 纯文字 + QA报告 ========== -->
  <output id="text_and_qa" name="纯文字 + 校对报告">
    <markdown_format>
      和之前一样但质量更高：
      - 按页码分隔
      - 标题/小标题保留原始格式
      - 红色文字标注为 **文字**（红色标注）
      - 蓝色文字标注为 **文字**（蓝色标注）
      - 不确定的字用 ⚠️[?猜测?]⚠️ 标记
      - 插图用 [插图: 描述] 占位
      - 尺寸标注原样保留
    </markdown_format>

    <qa_report>
      校对报告包含：
      - 汇总统计（页数、置信度分布、修正次数）
      - 每页的置信度和问题标记
      - 所有 Phase 2b 中做的自动修正列表
      - 所有不确定字符的汇总
      - 检测到的日文字符数（应该为0）
    </qa_report>
  </output>

  <!-- ========== 线稿版本选择 ========== -->
  <line_style_selection>
    命令行参数 --line-style 控制插图使用哪个版本：
    - original: 原始裁剪（不增强）
    - strengthen: 加深线稿（默认）
    - vectorize: 矢量化线稿
    - inpainted: 擦除叠加文字后的干净版 + 数字标注
    - auto: 根据 meta.json 中的质量评分自动选择

    对于有 overlapping_text 的插图，默认使用 inpainted 版本。
  </line_style_selection>

  <batch_optimization>
    <item>纯本地操作，不调用API，可以快速运行</item>
    <item>每10页打印进度</item>
    <item>字体下载只在首次运行时执行</item>
    <item>如果某页数据不完整，跳过并记录到错误日志</item>
  </batch_optimization>

  <dependencies>
    reportlab, python-pptx, Pillow, pyyaml
    字体: 自动下载 Google Noto CJK（约50MB）
  </dependencies>

  <cli>
    python src/build.py [--pages 10-20] [--format pdf,pptx,text]
           [--line-style auto|strengthen|vectorize|inpainted]
           [--force]
  </cli>
</task>
```
