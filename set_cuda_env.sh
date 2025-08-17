#!/bin/bash
# CUDA 멀티프로세싱 환경 변수 설정

echo "🔧 CUDA 멀티프로세싱 환경 변수 설정"

# 필수 환경 변수 설정
export TORCH_MULTIPROCESSING_START_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0
export TOKENIZERS_PARALLELISM=false
export SGLANG_DISABLE_FLASHINFER_WARNING=1

echo "✅ 환경 변수 설정 완료"
echo "TORCH_MULTIPROCESSING_START_METHOD=$TORCH_MULTIPROCESSING_START_METHOD"
echo "PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
echo "CUDA_LAUNCH_BLOCKING=$CUDA_LAUNCH_BLOCKING"
echo "TOKENIZERS_PARALLELISM=$TOKENIZERS_PARALLELISM"
echo "SGLANG_DISABLE_FLASHINFER_WARNING=$SGLANG_DISABLE_FLASHINFER_WARNING"

# 이 스크립트를 source로 실행하세요:
# source set_cuda_env.sh
