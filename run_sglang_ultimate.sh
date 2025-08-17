#!/bin/bash
# SGLang 궁극적 실행 스크립트 (모든 해결책 통합)

echo "🚀 SGLang 궁극적 실행 스크립트"
echo "=============================="

MODEL_PATH="${1:-microsoft/DialoGPT-medium}"
PORT="${2:-8000}"

echo "모델: $MODEL_PATH"
echo "포트: $PORT"
echo ""

echo "선택하세요:"
echo "1) CPU 모드 (가장 안정적, 느림)"
echo "2) Docker 모드 (권장, 빠름)"
echo "3) GPU 모드 재시도 (위험)"
echo "4) 문제 해결 가이드 보기"

read -p "선택 (1-4): " choice

case $choice in
    1)
        echo "💻 CPU 모드 실행..."
        python start_sglang_cpu_mode.py "$MODEL_PATH" "$PORT"
        ;;
    2)
        echo "🐳 Docker 모드 실행..."
        bash start_sglang_docker.sh "$MODEL_PATH" "$PORT"
        ;;
    3)
        echo "⚠️ GPU 모드 재시도..."
        export TORCH_MULTIPROCESSING_START_METHOD=spawn
        python start_sglang_cuda_fixed.py --model "$MODEL_PATH" --port "$PORT"
        ;;
    4)
        echo "📖 문제 해결 가이드:"
        cat sglang_troubleshooting_guide.md
        ;;
    *)
        echo "❌ 잘못된 선택"
        exit 1
        ;;
esac
