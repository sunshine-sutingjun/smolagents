#!/bin/bash

# 默认参数配置
CONCURRENCY=1
MODEL_ID="Qwen/Qwen2.5-72B-Instruct-128K"
RUN_NAME="test-run"
SET_TO_RUN="test"
USE_OPEN_MODELS=false
USE_RAW_DATASET=true
ENABLE_TELEMETRY=true

# 显示帮助信息
show_help() {
    echo "用法: $0 [选项]"
    echo "选项:"
    echo "  -c, --concurrency    并发数 (默认: 8)"
    echo "  -m, --model-id       模型ID (默认: o1)"
    echo "  -r, --run-name       运行名称 (必需)"
    echo "  -s, --set-to-run     数据集 (默认: validation)"
    echo "  -o, --open-models    使用开放模型 (默认: false)"
    echo "  -d, --raw-dataset    使用原始数据集 (默认: false)"
    echo "  -p, --enable-telemetry 启用Phoenix遥测 (默认: false)"
    echo "  -h, --help           显示此帮助信息"
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--concurrency)
            CONCURRENCY="$2"
            shift 2
            ;;
        -m|--model-id)
            MODEL_ID="$2"
            shift 2
            ;;
        -r|--run-name)
            RUN_NAME="$2"
            shift 2
            ;;
        -s|--set-to-run)
            SET_TO_RUN="$2"
            shift 2
            ;;
        -o|--open-models)
            USE_OPEN_MODELS=true
            shift
            ;;
        -d|--raw-dataset)
            USE_RAW_DATASET=true
            shift
            ;;
        -p|--enable-telemetry)
            ENABLE_TELEMETRY=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            show_help
            exit 1
            ;;
    esac
done

# 检查必需参数
if [ -z "$RUN_NAME" ]; then
    echo "错误: 必须指定运行名称 (--run-name)"
    show_help
    exit 1
fi

# 构建命令
CMD="python ./examples/open_deep_research/run_gaia.py \
    --concurrency $CONCURRENCY \
    --model-id $MODEL_ID \
    --run-name $RUN_NAME \
    --set-to-run $SET_TO_RUN"

# 添加可选参数
if [ "$USE_OPEN_MODELS" = true ]; then
    CMD="$CMD --use-open-models"
fi

if [ "$USE_RAW_DATASET" = true ]; then
    CMD="$CMD --use-raw-dataset"
fi

if [ "$ENABLE_TELEMETRY" = true ]; then
    CMD="$CMD --enable-telemetry"
fi

# 显示将要执行的命令
echo "执行命令: $CMD"

# 执行命令
eval $CMD
