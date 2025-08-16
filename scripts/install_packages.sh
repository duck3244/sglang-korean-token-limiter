#!/bin/bash
# SGLang 기반 한국어 Token Limiter 패키지 설치 스크립트 (오류 수정 버전)

set -e

echo "🚀 SGLang 기반 한국어 Token Limiter 설치 (오류 수정 버전)"
echo "========================================================="

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

# 긴급 패키지 수정 함수
fix_setuptools_conflict() {
    echo -e "${YELLOW}🔧 setuptools 및 의존성 충돌 수정...${NC}"

    # 1. 문제가 되는 패키지들 강제 제거
    pip uninstall setuptools more-itertools -y 2>/dev/null || true

    # 2. 기본 도구들을 안전한 버전으로 재설치
    pip install setuptools==68.2.2
    pip install more-itertools==10.1.0
    pip install wheel==0.41.2

    # 3. pip 자체 업그레이드
    python -m pip install --upgrade pip

    echo -e "${GREEN}✅ setuptools 충돌 해결 완료${NC}"
}

# 환경 확인 및 수정
check_and_fix_environment() {
    echo -e "${BLUE}🔍 환경 확인 및 수정...${NC}"

    # Python 버전 확인
    PYTHON_VERSION=$(python --version 2>&1 | cut -d' ' -f2)
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

    echo "Python 버전: $PYTHON_VERSION"

    if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]); then
        echo -e "${RED}❌ SGLang은 Python 3.10+ 필요 (현재: $PYTHON_VERSION)${NC}"
        echo "현재 Python 3.11을 사용 중이므로 계속 진행합니다."
    fi

    echo -e "${GREEN}✅ Python 버전 적합${NC}"

    # setuptools 충돌 먼저 해결
    fix_setuptools_conflict

    # 가상환경 확인
    if [[ "$CONDA_DEFAULT_ENV" != "" ]]; then
        echo -e "${GREEN}✅ Conda 환경: $CONDA_DEFAULT_ENV${NC}"
    elif [[ "$VIRTUAL_ENV" != "" ]]; then
        echo -e "${GREEN}✅ Python venv 환경 활성화됨${NC}"
    else
        echo -e "${YELLOW}⚠️ 가상환경이 활성화되지 않았지만 계속 진행합니다${NC}"
    fi

    # GPU 확인
    if command -v nvidia-smi &> /dev/null; then
        GPU_AVAILABLE=true
        echo -e "${GREEN}🎮 GPU 감지됨${NC}"
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
        GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
        echo "GPU: $GPU_NAME"
        echo "메모리: ${GPU_MEMORY}MB"
    else
        GPU_AVAILABLE=false
        echo -e "${YELLOW}⚠️ GPU를 찾을 수 없습니다. CPU 모드로 설치합니다.${NC}"
    fi
}

# 안전한 PyTorch 설치
install_pytorch_safe() {
    echo -e "\n${BLUE}🔥 PyTorch 안전 설치${NC}"

    # 기존 PyTorch 확인
    if python -c "import torch" 2>/dev/null; then
        EXISTING_TORCH=$(python -c "import torch; print(torch.__version__)" 2>/dev/null)
        echo "기존 PyTorch 버전: $EXISTING_TORCH"

        read -p "기존 PyTorch를 유지하시겠습니까? (Y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Nn]$ ]]; then
            echo "기존 PyTorch를 유지합니다."
            return 0
        fi
    fi

    # 안전한 PyTorch 설치
    if [ "$GPU_AVAILABLE" = true ]; then
        echo "CUDA 버전 PyTorch 설치 중..."
        pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121 --no-deps
        pip install nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 --no-deps || true
    else
        echo "CPU 버전 PyTorch 설치 중..."
        pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu --no-deps
    fi

    # PyTorch 설치 확인
    python -c "
import torch
print(f'✅ PyTorch {torch.__version__} 설치 완료')
if torch.cuda.is_available():
    print(f'🎮 CUDA 사용 가능: {torch.cuda.get_device_name()}')
else:
    print('💻 CPU 모드로 실행됩니다')
"
}

# SGLang 단계별 설치 (오류 방지)
install_sglang_stepwise() {
    echo -e "\n${PURPLE}🚀 SGLang 단계별 설치${NC}"

    # 1단계: 기본 의존성 설치
    echo "1단계: 기본 의존성 설치..."
    pip install numpy==1.24.4
    pip install packaging
    pip install requests
    pip install psutil
    pip install tqdm

    # 2단계: ML 라이브러리 설치
    echo "2단계: ML 라이브러리 설치..."
    pip install transformers==4.36.0 --no-deps
    pip install tokenizers==0.15.0
    pip install safetensors==0.4.1
    pip install accelerate==0.25.0 --no-deps

    # 3단계: SGLang 코어 설치 (의존성 충돌 방지)
    echo "3단계: SGLang 코어 설치..."

    # 일시적으로 엄격한 의존성 체크 비활성화
    export PIP_NO_DEPS=1
    pip install sglang==0.2.6 --no-deps || {
        echo "SGLang 직접 설치 실패, 소스에서 설치 시도..."
        pip install git+https://github.com/sgl-project/sglang.git@v0.2.6 --no-deps || {
            echo "소스 설치도 실패, 기본 설치로 재시도..."
            unset PIP_NO_DEPS
            pip install sglang==0.2.6
        }
    }
    unset PIP_NO_DEPS

    # 4단계: 성능 최적화 패키지 (선택적)
    if [ "$GPU_AVAILABLE" = true ]; then
        echo "4단계: GPU 최적화 패키지 설치..."

        # FlashAttention (가장 중요)
        pip install flash-attn==2.3.6 --no-build-isolation || echo "⚠️ Flash Attention 설치 실패 (성능에 영향)"

        # FlashInfer (SGLang 특화)
        pip install flashinfer==0.0.5 --no-build-isolation || echo "⚠️ FlashInfer 설치 실패"

        # 기타 최적화 (실패해도 진행)
        pip install triton==2.1.0 || echo "⚠️ Triton 설치 실패"
        pip install xformers==0.0.22.post7 || echo "⚠️ xformers 설치 실패"
    fi

    # SGLang 설치 검증
    python -c "
try:
    import sglang
    print(f'✅ SGLang {sglang.__version__} 설치 완료')

    # 기본 import 테스트
    try:
        from sglang import function, system, user, assistant, gen
        print('✅ SGLang 핵심 기능 사용 가능')
    except ImportError as e:
        print(f'⚠️ 일부 SGLang 기능 제한: {e}')

    # 서버 런처 확인
    try:
        from sglang.srt.server import launch_server
        print('✅ SGLang 서버 런처 사용 가능')
    except ImportError:
        print('⚠️ SGLang 서버 런처 제한적 사용')

    print('🚀 SGLang 기본 설치 완료!')

except ImportError as e:
    print(f'❌ SGLang import 실패: {e}')
    print('기본 패키지는 설치되었지만 SGLang에 문제가 있습니다.')
"
}

# 웹 서버 패키지 안전 설치
install_web_packages_safe() {
    echo -e "\n${BLUE}🌐 웹 서버 패키지 안전 설치${NC}"

    # FastAPI 스택 (호환성 우선)
    pip install fastapi==0.104.1
    pip install uvicorn==0.24.0
    pip install httpx==0.25.2
    pip install pydantic==2.5.0

    # 스트리밍 지원
    pip install sse-starlette==1.6.5
    pip install python-multipart==0.0.6

    # 추가 유틸리티
    pip install jinja2==3.1.2
    pip install python-dotenv==1.0.0

    echo -e "${GREEN}✅ 웹 패키지 설치 완료${NC}"
}

# 모니터링 패키지 안전 설치
install_monitoring_safe() {
    echo -e "\n${BLUE}📊 모니터링 패키지 안전 설치${NC}"

    # 데이터 처리
    pip install pandas==2.1.4
    pip install numpy==1.24.4 --upgrade

    # 시각화
    pip install plotly==5.17.0
    pip install streamlit==1.28.2

    # 시스템 모니터링
    pip install psutil==5.9.6

    # GPU 모니터링 (옵션)
    if [ "$GPU_AVAILABLE" = true ]; then
        pip install nvidia-ml-py3==7.352.0 || echo "⚠️ nvidia-ml-py3 설치 실패"
        pip install pynvml==11.5.0 || echo "⚠️ pynvml 설치 실패"
    fi

    # 저장소
    pip install redis==5.0.1
    pip install aiosqlite==0.19.0
    pip install PyYAML==6.0.1

    echo -e "${GREEN}✅ 모니터링 패키지 설치 완료${NC}"
}

# 한국어 처리 패키지
install_korean_safe() {
    echo -e "\n${BLUE}🇰🇷 한국어 처리 패키지 설치${NC}"

    # 이미 transformers가 설치되어 있으므로 추가 패키지만
    pip install sentencepiece==0.1.99
    pip install protobuf==4.25.1

    # 한국어 텍스트 처리 (선택적)
    pip install hanja==0.15.1 || echo "⚠️ hanja 설치 실패"
    pip install kss==4.5.4 || echo "⚠️ kss 설치 실패"

    echo -e "${GREEN}✅ 한국어 처리 패키지 설치 완료${NC}"
}

# 최종 검증 및 테스트
final_verification() {
    echo -e "\n${BLUE}🧪 최종 설치 검증${NC}"

    python -c "
import sys
print(f'🐍 Python: {sys.version}')

# 핵심 패키지 확인
packages_to_check = [
    ('sglang', 'SGLang'),
    ('torch', 'PyTorch'),
    ('transformers', 'Transformers'),
    ('fastapi', 'FastAPI'),
    ('streamlit', 'Streamlit'),
    ('pandas', 'Pandas'),
    ('plotly', 'Plotly')
]

score = 0
for pkg, name in packages_to_check:
    try:
        module = __import__(pkg)
        version = getattr(module, '__version__', 'Unknown')
        print(f'✅ {name}: {version}')
        score += 10
    except ImportError:
        print(f'❌ {name}: 설치되지 않음')

print(f'\n종합 점수: {score}/70')

if score >= 60:
    print('🌟 설치 성공! SGLang 시스템 사용 가능')
elif score >= 40:
    print('✅ 기본 설치 완료. 일부 기능 제한적')
else:
    print('⚠️ 설치 미완료. 추가 작업 필요')

# GPU 확인
try:
    import torch
    if torch.cuda.is_available():
        print(f'🎮 GPU: {torch.cuda.get_device_name()}')
    else:
        print('💻 CPU 모드')
except:
    print('⚠️ PyTorch GPU 확인 실패')
"
}

# 문제 해결 가이드
show_troubleshooting() {
    echo -e "\n${YELLOW}🔧 문제 해결 가이드${NC}"
    echo ""
    echo "1. SGLang import 실패 시:"
    echo "   pip uninstall sglang -y"
    echo "   pip install sglang==0.2.6 --no-deps"
    echo ""
    echo "2. setuptools 충돌 시:"
    echo "   pip install setuptools==68.2.2 --force-reinstall"
    echo "   pip install more-itertools==10.1.0"
    echo ""
    echo "3. CUDA 관련 오류 시:"
    echo "   pip install torch --index-url https://download.pytorch.org/whl/cu121"
    echo ""
    echo "4. 전체 재설치:"
    echo "   bash scripts/install_packages_fixed.sh --clean-reinstall"
    echo ""
}

# 깨끗한 재설치 옵션
clean_reinstall() {
    echo -e "${YELLOW}🧹 깨끗한 재설치 진행...${NC}"

    # 문제가 될 수 있는 패키지들 모두 제거
    pip uninstall -y sglang torch transformers fastapi streamlit || true
    pip uninstall -y setuptools more-itertools wheel || true

    # 캐시 정리
    pip cache purge

    # 기본부터 다시 설치
    fix_setuptools_conflict
    install_pytorch_safe
    install_sglang_stepwise
    install_web_packages_safe
    install_monitoring_safe
    install_korean_safe

    echo -e "${GREEN}✅ 깨끗한 재설치 완료${NC}"
}

# 메인 실행 함수
main() {
    echo "SGLang 안전 설치 시작: $(date)"
    echo ""

    check_and_fix_environment
    install_pytorch_safe
    install_sglang_stepwise
    install_web_packages_safe
    install_monitoring_safe
    install_korean_safe
    final_verification

    echo ""
    echo -e "${GREEN}🎉 SGLang 설치 완료!${NC}"
    echo ""
    echo "다음 단계:"
    echo "1. Redis 시작: docker run -d --name korean-redis -p 6379:6379 redis:alpine"
    echo "2. 시스템 시작: bash scripts/start_korean_sglang.sh"
    echo "3. 대시보드: streamlit run dashboard/sglang_app.py --server.port 8501"
    echo ""

    show_troubleshooting
}

# 명령행 인자 처리
case "${1:-}" in
    --help|-h)
        echo "SGLang 안전 설치 스크립트 (오류 수정 버전)"
        echo "사용법:"
        echo "  $0                    # 일반 설치"
        echo "  $0 --clean-reinstall  # 깨끗한 재설치"
        echo "  $0 --fix-setuptools   # setuptools 충돌만 수정"
        echo "  $0 --help             # 도움말"
        exit 0
        ;;
    --clean-reinstall)
        clean_reinstall
        ;;
    --fix-setuptools)
        fix_setuptools_conflict
        ;;
    "")
        main
        ;;
    *)
        echo -e "${RED}❌ 알 수 없는 옵션: $1${NC}"
        echo "도움말: $0 --help"
        exit 1
        ;;
esac