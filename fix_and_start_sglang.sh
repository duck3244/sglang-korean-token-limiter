#!/bin/bash
# SGLang CUDA 멀티프로세싱 문제 통합 해결 및 시작 스크립트

set -e

echo "🔧 SGLang CUDA 멀티프로세싱 문제 통합 해결"
echo "============================================"

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 1단계: 환경 변수 설정
echo -e "${BLUE}1단계: 환경 변수 설정...${NC}"
export TORCH_MULTIPROCESSING_START_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0
export TOKENIZERS_PARALLELISM=false
export SGLANG_DISABLE_FLASHINFER_WARNING=1

echo "✅ 환경 변수 설정 완료"

# 2단계: 기존 프로세스 정리
echo -e "\n${BLUE}2단계: 기존 프로세스 정리...${NC}"
pkill -f "sglang.*launch_server" 2>/dev/null || true
pkill -f "python.*sglang" 2>/dev/null || true
sleep 2

echo "✅ 프로세스 정리 완료"

# 3단계: CUDA 캐시 정리
echo -e "\n${BLUE}3단계: CUDA 캐시 정리...${NC}"
python -c "
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    print('✅ CUDA 캐시 정리 완료')
else:
    print('💻 CPU 모드')
"

# 4단계: SGLang 서버 시작
echo -e "\n${BLUE}4단계: SGLang 서버 시작 (CUDA 멀티프로세싱 해결)...${NC}"

MODEL_PATH="${1:-microsoft/DialoGPT-medium}"
PORT="${2:-8000}"

echo "모델: $MODEL_PATH"
echo "포트: $PORT"

# Python 스크립트로 서버 시작 (멀티프로세싱 문제 해결)
python start_sglang_cuda_fixed.py --model "$MODEL_PATH" --port "$PORT"
