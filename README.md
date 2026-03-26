# 教案数字化工具 v3

> 将手写教案扫描件转化为清晰的电子版

## 架构原则（从 9 次失败中总结）

1. **先识别再处理** — 在原始图上扫描结构，不做任何预处理
2. **整页上下文 OCR** — 发送整页增强图给 Gemini，不切碎
3. **快模型做主力** — Sonnet 做结构，Gemini 做 OCR
4. **特殊符号用 trace** — 马克笔、圈重点、箭头等直接 trace 为 SVG
5. **从白纸重建** — 不在扫描件上叠加，全新创建

## 执行流程

```
Phase 0: 拆页+旋转检测
    ↓
Phase 1: Sonnet 结构扫描（位置编码 + 字体大小）
    ↓
Phase 2: 整页增强 → Gemini OCR → 位置重映射
    ↓
Phase 3: 文字擦除 → 颜色分离 → 马克笔检测 → Trace
    ↓
Phase 4: 白纸合成 → PDF(图层) + PPT + 纯文字
```

| Phase | 脚本 | AI模型 | 耗时/页 |
|-------|------|--------|---------|
| Phase 0 | orientation.py | Sonnet | ~5s |
| Phase 1 | structure.py | Sonnet ×1轮 | ~15s |
| Phase 2 | ocr.py | Gemini 2.5 Pro ×2 + Sonnet ×1 | ~30s |
| Phase 3 | trace.py | 无（纯CV） | ~5s |
| Phase 4 | build.py | 无（纯本地） | ~3s |

**100 页预计**: ~1 小时，API 费用 ~$5-8

## 失败记录
详见 [Failed.md](Failed.md) — 9 次失败的详细记录和教训

## 快速开始

```bash
# 环境准备
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # 填入 GEMINI_API_KEY 和 ANTHROPIC_API_KEY

# 放入扫描件
cp ~/Downloads/教案*.pdf ./input/

# 按顺序执行（先测试 3 页）
python3 src/orientation.py --pages 1-3
python3 src/structure.py --pages 1-3
python3 src/ocr.py --pages 1-3
python3 src/trace.py --pages 1-3
python3 src/build.py --pages 1-3

# 满意后跑全部
python3 src/orientation.py && python3 src/structure.py && python3 src/ocr.py && python3 src/trace.py && python3 src/build.py
```
