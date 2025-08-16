#!/bin/bash
# SGLang 기반 한국어 Token Limiter 패키지 설치 스크립트 (수정된 버전)

set -e

echo "🚀 SGLang 기반 한국어 Token Limiter 설치"
echo "======================================"

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

# 환경 확인
check_environment() {
    echo -e "${BLUE}🔍 환경 확인 중...${NC}"

    # Python 버전 확인 (SGLang은 3.10+ 필요)
    PYTHON_VERSION=$(python --version 2>&1 | cut -d' ' -f2)
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

    echo "Python 버전: $PYTHON_VERSION"

    if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]); then
        echo -e "${RED}❌ SGLang은 Python 3.10+ 필요 (현재: $PYTHON_VERSION)${NC}"
        echo "업그레이드 방법:"
        echo "  conda install python=3.10"
        echo "  또는 최신 Python 설치"
        exit 1
    fi

    echo -e "${GREEN}✅ Python 버전 적합 (SGLang 지원)${NC}"

    # 가상환경 확인
    if [[ "$VIRTUAL_ENV" == "" ]] && [[ "$CONDA_DEFAULT_ENV" == "" ]]; then
        echo -e "${YELLOW}⚠️ 가상환경이 활성화되지 않았습니다${NC}"
        echo "권장: conda activate korean_sglang 또는 source venv/bin/activate"

        read -p "계속 진행하시겠습니까? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        if [[ "$CONDA_DEFAULT_ENV" != "" ]]; then
            echo -e "${GREEN}✅ Conda 환경: $CONDA_DEFAULT_ENV${NC}"
        else
            echo -e "${GREEN}✅ Python venv 환경 활성화됨${NC}"
        fi
    fi

    # GPU 확인
    if command -v nvidia-smi &> /dev/null; then
        GPU_AVAILABLE=true
        echo -e "${GREEN}🎮 GPU 감지됨${NC}"
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
        GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
        echo "GPU: $GPU_NAME"
        echo "메모리: ${GPU_MEMORY}MB"

        # CUDA 버전 확인
        if command -v nvcc &> /dev/null; then
            CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
            echo "CUDA 개발 도구: $CUDA_VERSION"
        else
            echo -e "${YELLOW}⚠️ CUDA 개발 도구가 설치되지 않았습니다 (런타임은 사용 가능)${NC}"
        fi
    else
        GPU_AVAILABLE=false
        echo -e "${YELLOW}⚠️ GPU를 찾을 수 없습니다. CPU 모드로 설치합니다.${NC}"
    fi
}

# 기본 도구 업그레이드
upgrade_basic_tools() {
    echo -e "\n${BLUE}📦 기본 도구 업그레이드${NC}"
    pip install --upgrade pip wheel setuptools
    echo -e "${GREEN}✅ 기본 도구 업그레이드 완료${NC}"
}

# PyTorch 설치 (SGLang 호환성 고려)
install_pytorch() {
    echo -e "\n${BLUE}🔥 PyTorch 설치 (SGLang 호환)${NC}"

    # 기존 PyTorch 확인
    if python -c "import torch" 2>/dev/null; then
        EXISTING_TORCH=$(python -c "import torch; print(torch.__version__)" 2>/dev/null)
        echo "기존 PyTorch 버전: $EXISTING_TORCH"

        # SGLang 호환성 확인
        if python -c "import torch; assert torch.__version__.startswith('2.1') or torch.__version__.startswith('2.2')" 2>/dev/null; then
            echo -e "${GREEN}✅ 기존 PyTorch가 SGLang과 호환됩니다${NC}"

            read -p "기존 PyTorch를 유지하시겠습니까? (Y/n): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Nn]$ ]]; then
                echo "기존 PyTorch를 유지합니다."
                return 0
            fi
        else
            echo -e "${YELLOW}⚠️ 기존 PyTorch가 SGLang과 호환되지 않을 수 있습니다${NC}"
            echo "SGLang 권장 버전으로 업데이트합니다."
        fi
    fi

    if [ "$GPU_AVAILABLE" = true ]; then
        echo "CUDA 버전 PyTorch 설치 중 (SGLang 최적화)..."
        # SGLang이 지원하는 PyTorch 버전
        pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
    else
        echo "CPU 버전 PyTorch 설치 중..."
        pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu
    fi

    # 설치 확인
    python -c "
import torch
print(f'✅ PyTorch {torch.__version__} 설치 완료 (SGLang 호환)')
if torch.cuda.is_available():
    print(f'🎮 CUDA 사용 가능: {torch.cuda.get_device_name()}')
    print(f'🎮 CUDA 버전: {torch.version.cuda}')
    gpu_count = torch.cuda.device_count()
    print(f'🎮 GPU 개수: {gpu_count}')

    # GPU 메모리 확인
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / 1024**3
        print(f'🎮 GPU {i}: {props.name} ({memory_gb:.1f}GB)')
else:
    print('💻 CPU 모드로 실행됩니다')
"
}

# SGLang 설치 (메인)
install_sglang() {
    echo -e "\n${PURPLE}🚀 SGLang 설치 (메인 프레임워크)${NC}"

    if [ "$GPU_AVAILABLE" = true ]; then
        echo "GPU 버전 SGLang 설치 중..."

        # SGLang 코어 패키지 설치
        echo "SGLang 코어 패키지..."
        pip install "sglang[all]==0.2.6"

        # 성능 최적화 패키지들
        echo "SGLang 성능 최적화 패키지..."

        # FlashAttention (중요: SGLang 성능에 핵심)
        pip install flashinfer==0.0.5 --no-build-isolation || echo "⚠️ FlashInfer 설치 실패 (성능에 영향)"
        pip install flash-attn==2.3.6 --no-build-isolation || echo "⚠️ Flash Attention 설치 실패 (대안 사용)"

        # 추가 최적화 라이브러리
        pip install triton==2.1.0 || echo "⚠️ Triton 설치 실패 (선택사항)"
        pip install xformers==0.0.22.post7 || echo "⚠️ xformers 설치 실패 (대안 존재)"

    else
        echo "CPU 버전 SGLang 설치 중..."
        # CPU 버전은 코어 패키지만
        pip install "sglang[all]==0.2.6"
        echo -e "${YELLOW}⚠️ CPU 모드: 성능 최적화 패키지 생략${NC}"
    fi

    # SGLang 설치 검증
    python -c "
try:
    import sglang
    print(f'✅ SGLang {sglang.__version__} 설치 완료')

    # SGLang 기본 기능 테스트
    from sglang import function, system, user, assistant, gen, select
    print('✅ SGLang 핵심 기능 import 성공')

    # 토큰화 테스트
    from sglang.lang.backend.runtime_endpoint import RuntimeEndpoint
    print('✅ SGLang 런타임 엔드포인트 사용 가능')

    # 추가 컴포넌트 확인
    try:
        from sglang.srt.server import launch_server
        print('✅ SGLang 서버 런처 사용 가능')
    except ImportError as e:
        print(f'⚠️ SGLang 서버 런처 일부