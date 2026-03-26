```xml
<task>
  <title>Phase 0: 方向检测与自动旋转</title>
  <role>你是一个图像预处理工程师。扫描件可能存在 0°/90°/180°/270° 的旋转，需要在所有后续处理之前纠正方向。</role>

  <config>
    <input_dir>./input/</input_dir>
    <output_pages_dir>./intermediate/pages/</output_pages_dir>
    <model>claude-opus-4-20250514</model>
    <dpi>300</dpi>
  </config>

  <goal>
    编写 src/orientation.py 脚本：
    1. 将所有输入PDF/图片拆分为单页PNG（300dpi）
    2. 对每一页调用 Claude Opus Vision 检测正确方向
    3. 如果需要旋转，自动旋转后保存
    4. 输出到 intermediate/pages/page_001.png, page_002.png ...
  </goal>

  <implementation>
    <step name="拆分PDF为单页">
      ```python
      from pdf2image import convert_from_path
      from PIL import Image
      import os, glob, re

      def extract_pages(input_dir, output_dir, dpi=300):
          files = sorted(
              glob.glob(os.path.join(input_dir, '*')),
              key=lambda f: os.path.basename(f)
          )
          page_num = 0
          for fpath in files:
              ext = os.path.splitext(fpath)[1].lower()
              if ext == '.pdf':
                  images = convert_from_path(fpath, dpi=dpi)
                  for img in images:
                      page_num += 1
                      img.save(os.path.join(output_dir, f'page_{page_num:03d}.png'))
              elif ext in ('.jpg', '.jpeg', '.png', '.tiff', '.tif'):
                  page_num += 1
                  img = Image.open(fpath)
                  img.save(os.path.join(output_dir, f'page_{page_num:03d}.png'))
          return page_num
      ```
    </step>

    <step name="AI方向检测">
      对每一页，发送缩小版图片（长边800px以内，节省token）给 Opus，
      让它判断正确方向。

      <vision_prompt>
        <![CDATA[
你看到的是一页手写中文教案的扫描件。这份教案来自中央美术学院服装设计专业。

请判断这张图片的方向是否正确。

判断依据：
- 中文文字应该是从左到右、从上到下书写
- 标题通常在页面顶部
- 页码通常在页面顶部右侧或底部
- 如果有竖排文字，整体页面仍应是正向的

请返回需要旋转的角度（顺时针）：
- 0: 方向正确，不需要旋转
- 90: 需要顺时针旋转90度
- 180: 需要旋转180度
- 270: 需要顺时针旋转270度（即逆时针90度）

只返回一个数字，不要其他内容：
0 或 90 或 180 或 270
        ]]>
      </vision_prompt>

      ```python
      import anthropic, base64
      from PIL import Image
      import io

      client = anthropic.Anthropic()

      def detect_orientation(image_path):
          # 缩小图片节省token
          img = Image.open(image_path)
          max_side = 800
          ratio = min(max_side / img.width, max_side / img.height)
          if ratio < 1:
              img = img.resize((int(img.width * ratio), int(img.height * ratio)),
                               Image.LANCZOS)
          buf = io.BytesIO()
          img.save(buf, format='PNG')
          b64 = base64.standard_b64encode(buf.getvalue()).decode()

          response = client.messages.create(
              model="claude-opus-4-20250514",
              max_tokens=16,
              messages=[{
                  "role": "user",
                  "content": [
                      {"type": "image", "source": {"type": "base64",
                       "media_type": "image/png", "data": b64}},
                      {"type": "text", "text": ORIENTATION_PROMPT}
                  ]
              }]
          )
          angle = int(response.content[0].text.strip())
          assert angle in (0, 90, 180, 270)
          return angle

      def rotate_if_needed(image_path, angle):
          if angle == 0:
              return
          img = Image.open(image_path)
          # PIL rotate 是逆时针，所以要取反
          rotated = img.rotate(-angle, expand=True)
          rotated.save(image_path)
      ```
    </step>
  </implementation>

  <batch_optimization>
    <item>并发3个请求检测方向（图片很小，token少，不容易rate limit）</item>
    <item>跳过已处理的页面（除非 --force）</item>
    <item>每页打印方向检测结果: "[page_003] 检测到旋转 90° → 已纠正"</item>
    <item>保存方向记录到 intermediate/orientation_log.json</item>
    <item>末尾汇总: 总页数, 需旋转数, 各角度分布</item>
  </batch_optimization>

  <dependencies>
    anthropic, pdf2image, Pillow
    系统: poppler-utils
  </dependencies>

  <cli>
    python src/orientation.py [--pages 10-20] [--force] [--dpi 300]
  </cli>
</task>
```
