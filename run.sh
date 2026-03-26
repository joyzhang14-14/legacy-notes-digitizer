#!/bin/bash
# ============================================
# 教案数字化工具 — 一键运行脚本
# ============================================

set -e

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "=========================================="
echo "  教案数字化工具 (Legacy Notes Digitizer)"
echo "=========================================="
echo ""

# 检查环境
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo -e "${RED}错误: 未设置 ANTHROPIC_API_KEY${NC}"
    echo "请运行: export ANTHROPIC_API_KEY=\"sk-ant-你的key\""
    exit 1
fi

# 检查输入文件
INPUT_COUNT=$(find ./input -type f \( -name "*.pdf" -o -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" -o -name "*.tiff" \) 2>/dev/null | wc -l | tr -d ' ')
if [ "$INPUT_COUNT" = "0" ]; then
    echo -e "${RED}错误: input/ 目录中没有找到扫描件${NC}"
    echo "请将 PDF 或图片文件放入 input/ 文件夹"
    exit 1
fi
echo -e "${GREEN}找到 $INPUT_COUNT 个输入文件${NC}"
echo ""

# 解析参数
PAGES=""
LINE_STYLE="auto"
STEPS="1,2,3,4,5"

while [[ $# -gt 0 ]]; do
    case $1 in
        --pages) PAGES="--pages $2"; shift 2 ;;
        --line-style) LINE_STYLE="$2"; shift 2 ;;
        --steps) STEPS="$2"; shift 2 ;;
        --help|-h)
            echo "用法: ./run.sh [选项]"
            echo ""
            echo "选项:"
            echo "  --pages 1-5       只处理指定页码"
            echo "  --line-style X    线稿版本: strengthen / vectorize / auto"
            echo "  --steps 1,2,3     只运行指定步骤"
            echo ""
            echo "示例:"
            echo "  ./run.sh                          # 运行全部"
            echo "  ./run.sh --pages 1-3              # 只处理前3页（测试）"
            echo "  ./run.sh --steps 4,5              # 只重跑合成和QA"
            echo "  ./run.sh --line-style vectorize   # 指定使用矢量线稿"
            exit 0
            ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

# 运行各步骤
run_step() {
    local step=$1
    local name=$2
    local cmd=$3
    echo ""
    echo -e "${YELLOW}━━━ Step $step: $name ━━━${NC}"
    echo "运行: $cmd"
    echo ""
    eval $cmd
    echo -e "${GREEN}✓ Step $step 完成${NC}"
}

if [[ $STEPS == *"1"* ]]; then
    run_step 1 "预处理（拆页+图像增强）" "python src/preprocess.py $PAGES"
fi

if [[ $STEPS == *"2"* ]]; then
    run_step 2 "Claude Vision 分析（调用API）" "python src/analyze.py $PAGES"
fi

if [[ $STEPS == *"3"* ]]; then
    run_step 3 "线稿增强" "python src/enhance_lines.py $PAGES"
fi

if [[ $STEPS == *"4"* ]]; then
    run_step 4 "合成输出" "python src/compose.py $PAGES --line-style $LINE_STYLE"
fi

if [[ $STEPS == *"5"* ]]; then
    run_step 5 "质量检查" "python src/qa.py $PAGES"
fi

echo ""
echo "=========================================="
echo -e "${GREEN}  全部完成！${NC}"
echo "=========================================="
echo ""
echo "输出文件:"
echo "  output/教案_数字版.pdf   — PDF（图层可切换）"
echo "  output/教案_数字版.pptx  — PPT（可编辑）"
echo "  output/教案_纯文字.md    — 纯文字"
echo "  output/校对报告.md       — 校对报告"
echo "  output/comparison/       — 对比图"
echo ""
echo "用 Adobe Acrobat 打开PDF → 图层面板 → 切换显示"
