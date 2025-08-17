#!/bin/bash
# SGLang Docker 기반 실행 (CUDA 문제 완전 회피)

echo "🐳 SGLang Docker 기반 실행"
echo "========================="

MODEL_PATH="${1:-microsoft/DialoGPT-medium}"
PORT="${2:-8000}"

echo "모델: $MODEL_PATH"
echo "포트: $PORT"

# Docker 이미지 확인
if ! docker images | grep -q "sglang"; then
    echo "SGLang Docker 이미지 빌드 중..."
    
    # Dockerfile 생성
    cat > Dockerfile.sglang << 'DOCKER_EOF'
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# SGLang 설치
RUN pip install "sglang[all]==0.2.15" --no-cache-dir

# 멀티프로세싱 설정
ENV TORCH_MULTIPROCESSING_START_METHOD=spawn
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
ENV TOKENIZERS_PARALLELISM=false

WORKDIR /app

# 포트 노출
EXPOSE 8000

# 시작 명령어
ENTRYPOINT ["python", "-m", "sglang.launch_server"]
DOCKER_EOF

    # Docker 이미지 빌드
    docker build -f Dockerfile.sglang -t sglang:latest .
    
    if [ $? -eq 0 ]; then
        echo "✅ SGLang Docker 이미지 빌드 완료"
    else
        echo "❌ Docker 이미지 빌드 실패"
        exit 1
    fi
fi

# Docker 컨테이너 실행
echo "SGLang Docker 컨테이너 시작..."

docker run -d \
    --name sglang-korean \
    --gpus all \
    -p $PORT:8000 \
    -e TORCH_MULTIPROCESSING_START_METHOD=spawn \
    -e PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 \
    sglang:latest \
    --model-path "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --mem-fraction-static 0.7 \
    --max-running-requests 4

if [ $? -eq 0 ]; then
    echo "✅ SGLang Docker 컨테이너 시작 완료"
    
    # 컨테이너 준비 대기
    echo "⏳ 컨테이너 준비 대기..."
    for i in {1..60}; do
        if curl -s http://localhost:$PORT/get_model_info > /dev/null 2>&1; then
            echo "✅ SGLang Docker 서버 준비 완료!"
            break
        fi
        sleep 2
    done
    
    echo ""
    echo "🐳 Docker 컨테이너 정보:"
    docker ps | grep sglang-korean
    
    echo ""
    echo "📋 관리 명령어:"
    echo "  로그 확인: docker logs sglang-korean"
    echo "  컨테이너 중지: docker stop sglang-korean"
    echo "  컨테이너 제거: docker rm sglang-korean"
    
else
    echo "❌ Docker 컨테이너 시작 실패"
    exit 1
fi
