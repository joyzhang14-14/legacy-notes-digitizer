```xml
<task>
  <title>Phase 2a: 插图通道 — 原始对比度提取 + 线稿增强</title>
  <role>
    你是一个插图修复工程师。从原始扫描件中提取插图区域，
    在保留原始灰度细节的前提下增强线条清晰度。
    不做全局对比度增强 — 只做局部线条加深。
  </role>

  <config>
    <input_pages_dir>./intermediate/pages/</input_pages_dir>
    <input_structure_dir>./intermediate/structure/</input_structure_dir>
    <output_dir>./intermediate/illustrations/</output_dir>
    <config_file>./config.yaml</config_file>
  </config>

  <critical_principle>
    插图通道的核心原则：保留原始对比度和灰度信息。
    不做全局二值化、不做全局CLAHE、不做全局锐化。
    只在线条像素上做局部增强。
    这和文字通道（Phase 2b）形成对比 — 文字通道会大幅提高对比度。
  </critical_principle>

  <goal>
    编写 src/illustration.py 脚本：
    1. 读取 structure JSON，找到所有 ILLUSTRATION 和 LABEL_SYSTEM 区域
    2. 从原始图中裁剪出插图区域（带 padding）
    3. 对每个插图生成三个版本：
       a. original: 原始裁剪（不做任何处理）
       b. strengthened: 线条加深版（保留手绘质感）
       c. vectorized: 矢量化重绘版（干净电脑线条）+ SVG源文件
    4. 处理插图上叠加的文字（inpainting 擦除）
    5. 重建标注引线（从手绘 → 电子版）
    6. 保存所有结果和位置编码
  </goal>

  <!-- ========== 裁剪 ========== -->
  <step name="裁剪插图区域">
    从原始图中按 structure JSON 的 bbox 裁剪。
    四周扩展 30px padding（不要切到插图边缘）。
    记录裁剪偏移量，后续合成时需要还原位置。

    ```python
    def crop_illustration(original_img, bbox, padding=30):
        x, y, w, h = bbox
        img_h, img_w = original_img.shape[:2]
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img_w, x + w + padding)
        y2 = min(img_h, y + h + padding)
        crop = original_img[y1:y2, x1:x2]
        offset = (x1, y1)  # 位置编码：裁剪区域在原图中的偏移
        return crop, offset
    ```
  </step>

  <!-- ========== 版本A：加深原线条 ========== -->
  <step name="版本A — 加深原线条（strengthened）">
    <principle>
      只加深已有的线条，不改变线条走向，不引入新线条。
      背景保持原样（不变白、不变灰）。
    </principle>

    <pipeline>
      1. 颜色分离：分别提取黑色、红色、蓝色通道的线条
      2. 对每种颜色的线条：
         a. 用自适应阈值提取线条mask
         b. 膨胀 1-2 px 加粗
         c. 在原图的对应颜色通道上加深
      3. 合并各颜色通道的结果
    </pipeline>

    ```python
    import cv2
    import numpy as np

    def strengthen_illustration(crop_bgr, config):
        result = crop_bgr.copy().astype(np.float64)
        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)

        color_ranges = {
            'black': {'h': (0, 180), 's': (0, 80), 'v': (0, 130)},
            'red':   {'h': [(0, 12), (160, 180)], 's': (50, 255), 'v': (50, 255)},
            'blue':  {'h': (90, 135), 's': (50, 255), 'v': (50, 255)},
        }

        darken_factor = config.get('contrast_boost', 1.8)
        thickness_add = config.get('line_thickness_add', 1)

        for color_name, ranges in color_ranges.items():
            # 提取该颜色的线条 mask
            mask = extract_color_mask(hsv, ranges)

            if mask.sum() == 0:
                continue

            # 自适应阈值细化线条（去除背景噪声）
            gray_masked = cv2.bitwise_and(
                cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY), mask
            )
            refined = cv2.adaptiveThreshold(
                gray_masked, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 25, 8
            )
            refined = cv2.bitwise_and(refined, mask)

            # 去噪
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel_open)

            # 膨胀加粗
            if thickness_add > 0:
                kernel_dilate = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (2*thickness_add+1, 2*thickness_add+1)
                )
                refined = cv2.dilate(refined, kernel_dilate)

            # 在原图上加深线条区域
            mask_3ch = cv2.cvtColor(refined, cv2.COLOR_GRAY2BGR).astype(bool)
            result[mask_3ch] = result[mask_3ch] / darken_factor

        return np.clip(result, 0, 255).astype(np.uint8)
    ```
  </step>

  <!-- ========== 版本B：矢量化重绘 ========== -->
  <step name="版本B — 矢量化重绘（vectorized）">
    <pipeline>
      1. 提取线条 → 二值化 → 骨架化
      2. potrace 转 SVG
      3. SVG 后处理（线宽、颜色、虚线检测）
      4. 渲染为 PNG
    </pipeline>

    ```python
    import subprocess, tempfile, os
    from PIL import Image

    def vectorize_illustration(crop_bgr, config):
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

        # 自适应阈值（比全局阈值更适合手绘线稿）
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 21, 6
        )

        # 去噪
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
        min_area = config.get('potrace_turdsize', 15)
        clean = np.zeros_like(binary)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                clean[labels == i] = 255

        # 保存为 PBM（potrace 输入格式）
        # potrace 中黑色是前景，需要反转
        pbm_img = Image.fromarray(255 - clean)
        with tempfile.NamedTemporaryFile(suffix='.pbm', delete=False) as f:
            pbm_img.save(f.name)
            pbm_path = f.name

        svg_path = pbm_path.replace('.pbm', '.svg')

        subprocess.run([
            'potrace', pbm_path,
            '-s',  # SVG output
            '-o', svg_path,
            '-t', str(min_area),
            '-a', str(config.get('potrace_alphamax', 1.0)),
            '-O', str(config.get('potrace_opttolerance', 0.2)),
        ], check=True)

        os.unlink(pbm_path)

        # 后处理 SVG
        postprocess_svg(svg_path,
            stroke_width=config.get('stroke_width', 1.5),
            stroke_color=config.get('stroke_color', '#1a1a1a'))

        # 渲染为 PNG
        import cairosvg
        png_path = svg_path.replace('.svg', '.png')
        cairosvg.svg2png(
            url=svg_path, write_to=png_path,
            output_width=crop_bgr.shape[1],
            output_height=crop_bgr.shape[0]
        )

        return svg_path, png_path
    ```
  </step>

  <!-- ========== Inpainting: 擦除插图上的叠加文字 ========== -->
  <step name="Inpainting — 擦除插图上的手写文字">
    <description>
      structure JSON 中标记了 overlapping_text（叠加在插图上的文字）。
      使用 OpenCV inpainting 擦除这些文字，让插图干净。
      后续 Phase 3 会在干净的插图上方叠加数字化文字。
    </description>

    ```python
    def inpaint_overlapping_text(crop_bgr, overlapping_texts, offset):
        """
        擦除插图上叠加的手写文字。
        overlapping_texts: 从 structure JSON 中获取的叠加文字列表
        offset: 裁剪偏移量，用于坐标转换
        """
        if not overlapping_texts:
            return crop_bgr

        mask = np.zeros(crop_bgr.shape[:2], dtype=np.uint8)
        ox, oy = offset

        for text_info in overlapping_texts:
            tx, ty = text_info['position']
            # 转换为裁剪区域内的坐标
            local_x = tx - ox
            local_y = ty - oy

            # 估算文字区域大小（每个中文字约 30-40px 宽高）
            char_count = len(text_info['content'])
            text_w = char_count * 35
            text_h = 40

            # 创建矩形 mask
            cv2.rectangle(mask,
                (max(0, local_x - 5), max(0, local_y - 5)),
                (min(crop_bgr.shape[1], local_x + text_w + 5),
                 min(crop_bgr.shape[0], local_y + text_h + 5)),
                255, -1)

        # 膨胀 mask 确保完全覆盖笔画
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.dilate(mask, kernel)

        # Inpainting
        result = cv2.inpaint(crop_bgr, mask, inpaintRadius=7,
                             flags=cv2.INPAINT_TELEA)
        return result
    ```
  </step>

  <!-- ========== 输出结构 ========== -->
  <output_structure>
    intermediate/illustrations/
      page_010/
        r2_original.png          ← 原始裁剪
        r2_strengthened.png      ← 加深线稿
        r2_vectorized.png        ← 矢量化线稿
        r2_vectorized.svg        ← 矢量 SVG 源文件
        r2_inpainted.png         ← 擦除叠加文字后的干净版
        r2_meta.json             ← 元数据（位置编码、裁剪偏移、质量评分）
      page_011/
        ...

    r2_meta.json 格式：
    {
      "region_id": "r2",
      "page_number": 10,
      "original_bbox": [100, 200, 400, 500],
      "crop_offset": [70, 170],
      "crop_size": [460, 560],
      "illustration_type": "skeleton",
      "has_overlapping_text": true,
      "overlapping_text_count": 5,
      "versions": {
        "original": "r2_original.png",
        "strengthened": "r2_strengthened.png",
        "vectorized_png": "r2_vectorized.png",
        "vectorized_svg": "r2_vectorized.svg",
        "inpainted": "r2_inpainted.png"
      },
      "quality_scores": {
        "strengthen_ssim": 0.92,
        "vectorize_ssim": 0.78
      },
      "recommended_version": "strengthened"
    }
  </output_structure>

  <batch_optimization>
    <item>纯本地计算，不调用AI API，可以 max_workers=4 并行</item>
    <item>跳过已有输出的插图（--force 强制重新处理）</item>
    <item>text_only 页面自动跳过</item>
    <item>每处理一页打印: 插图数量、各版本耗时</item>
  </batch_optimization>

  <dependencies>
    opencv-python-headless, numpy, Pillow, cairosvg, pyyaml
    系统: potrace
  </dependencies>

  <cli>
    python src/illustration.py [--pages 10-20] [--force]
           [--mode strengthen|vectorize|both]
  </cli>
</task>
```
