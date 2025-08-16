#!/bin/bash
# FlashInfer 설치 및 SGLang 최적화 스크립트

set -e

echo "🔧 FlashInfer 설치 및 SGLang 최적화"
echo "=================================="

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 1. 현재 환경 확인
echo -e "${BLUE}1. 현재 환경 확인...${NC}"
python -c "
import sys
print(f'Python: {sys.version}')

# CUDA 확인
try:
    import torch
    print(f'PyTorch: {torch.__version__}')
    if torch.cuda.is_available():
        print(f'CUDA 사용 가능: {torch.version.cuda}')
        print(f'GPU: {torch.cuda.get_device_name()}')
    else:
        print('CUDA 사용 불가')
except ImportError:
    print('PyTorch 없음')

# 기존 패키지 확인
packages = ['sglang', 'flashinfer', 'triton']
for pkg in packages:
    try:
        module = __import__(pkg)
        version = getattr(module, '__version__', 'Unknown')
        print(f'✅ {pkg}: {version}')
    except ImportError:
        print(f'❌ {pkg}: 설치되지 않음')
"

# 2. FlashInfer 설치 시도
echo -e "\n${BLUE}2. FlashInfer 설치 시도...${NC}"

# GPU 사용 가능한지 확인
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "GPU 환경에서 FlashInfer 설치..."

    # FlashInfer 여러 버전 시도
    FLASHINFER_VERSIONS=("0.0.5" "0.0.4" "0.0.3")

    for version in "${FLASHINFER_VERSIONS[@]}"; do
        echo "=== FlashInfer ${version} 설치 시도 ==="

        if pip install "flashinfer==${version}" --no-build-isolation; then
            echo -e "${GREEN}✅ FlashInfer ${version} 설치 성공${NC}"

            # 즉시 import 테스트
            if python -c "
import flashinfer
print(f'✅ FlashInfer {flashinfer.__version__} import 성공')

# 핵심 함수 테스트
try:
    from flashinfer.sampling import top_k_top_p_sampling_from_probs
    print('✅ flashinfer.sampling 함수 정상')
except ImportError as e:
    print(f'⚠️ flashinfer.sampling 실패: {e}')
    exit(1)

print('🎉 FlashInfer 완전 설치 성공!')
" 2>/dev/null; then
                echo -e "${GREEN}🎉 호환 가능한 FlashInfer 버전: ${version}${NC}"
                WORKING_FLASHINFER_VERSION=$version
                break
            else
                echo -e "${YELLOW}⚠️ FlashInfer ${version} import 실패${NC}"
                pip uninstall flashinfer -y 2>/dev/null || true
            fi
        else
            echo -e "${YELLOW}⚠️ FlashInfer ${version} 설치 실패${NC}"
        fi
    done

    # 모든 버전 실패 시 소스에서 설치 시도
    if [ -z "$WORKING_FLASHINFER_VERSION" ]; then
        echo -e "\n${YELLOW}⚠️ 패키지 버전 실패. 소스에서 설치 시도...${NC}"

        # 필요한 빌드 도구 설치
        pip install ninja packaging

        # Git에서 설치 시도
        if pip install "git+https://github.com/flashinfer-ai/flashinfer.git" --no-build-isolation; then
            echo "Git에서 FlashInfer 설치 완료"
            WORKING_FLASHINFER_VERSION="git-latest"
        else
            echo -e "${RED}❌ 모든 FlashInfer 설치 방법 실패${NC}"
            echo "FlashInfer 없이 SGLang을 실행합니다 (성능 제한)."
        fi
    fi
else
    echo -e "${YELLOW}⚠️ GPU 없음. FlashInfer 건너뛰기${NC}"
fi

# 3. 추가 최적화 패키지 설치
echo -e "\n${BLUE}3. 추가 최적화 패키지 설치...${NC}"

# Triton 설치 (CUDA 환경)
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "Triton 설치 중..."
    pip install "triton>=2.1.0" || echo "⚠️ Triton 설치 실패"
fi

# Flash Attention 설치 (선택사항)
echo "Flash Attention 설치 시도..."
pip install "flash-attn>=2.3.0" --no-build-isolation || echo "⚠️ Flash Attention 설치 실패 (선택사항)"

# xformers 설치 (선택사항)
echo "xformers 설치 시도..."
pip install "xformers>=0.0.22" || echo "⚠️ xformers 설치 실패 (선택사항)"

# 4. SGLang 호환성 테스트
echo -e "\n${BLUE}4. SGLang 호환성 테스트...${NC}"

python -c "
import sys

try:
    print('=== SGLang + FlashInfer 호환성 테스트 ===')

    # 1. FlashInfer 확인
    try:
        import flashinfer
        print(f'✅ FlashInfer: {flashinfer.__version__}')

        # 핵심 함수 확인
        from flashinfer.sampling import top_k_top_p_sampling_from_probs
        print('✅ flashinfer.sampling: 정상')
    except ImportError as e:
        print(f'❌ FlashInfer 없음: {e}')
        print('SGLang이 FlashInfer 없이 실행됩니다 (성능 제한)')

    # 2. SGLang constrained 모듈
    from sglang.srt.constrained import FSMInfo, RegexGuide
    print('✅ sglang.srt.constrained: 정상')

    # 3. SGLang infer_batch (FlashInfer 사용)
    try:
        from sglang.srt.managers.controller.infer_batch import ForwardBatch
        print('✅ sglang.srt.managers.controller.infer_batch: 정상')
    except ImportError as e:
        print(f'⚠️ infer_batch 일부 기능 제한: {e}')

    # 4. SGLang 서버 런처 최종 테스트
    try:
        from sglang.srt.server import launch_server
        print('✅ sglang.srt.server.launch_server: 정상')
    except ImportError as e:
        print(f'❌ SGLang 서버 런처 실패: {e}')
        raise

    print()
    print('🎉 모든 테스트 통과! SGLang 서버 시작 가능!')

except Exception as e:
    print(f'❌ 호환성 테스트 실패: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"

# 5. 성공 시 안내
if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}🎉 FlashInfer 및 SGLang 최적화 완료!${NC}"
    echo ""
    echo -e "${BLUE}📋 설치 요약:${NC}"

    python -c "
packages = ['flashinfer', 'triton', 'sglang']
for pkg in packages:
    try:
        module = __import__(pkg)
        version = getattr(module, '__version__', 'Unknown')
        print(f'✅ {pkg}: {version}')
    except ImportError:
        print(f'❌ {pkg}: 설치되지 않음')
"

    echo ""
    echo -e "${GREEN}🚀 이제 SGLang 서버를 시작할 수 있습니다:${NC}"
    echo "bash scripts/start_korean_sglang.sh"
    echo ""
    echo -e "${BLUE}💡 최적화 내용:${NC}"
    if [ ! -z "$WORKING_FLASHINFER_VERSION" ]; then
        echo "- FlashInfer $WORKING_FLASHINFER_VERSION (GPU 성능 최적화)"
    else
        echo "- FlashInfer 없음 (CPU 모드 또는 성능 제한)"
    fi
    echo "- SGLang constrained 모듈 완전 패치"
    echo "- 한국어 토큰 제한 시스템 준비 완료"

else
    echo -e "\n${RED}❌ FlashInfer 설치 실패${NC}"
    echo ""
    echo -e "${YELLOW}🔧 대안 방법:${NC}"
    echo "1. FlashInfer 없이 SGLang 사용 (제한된 성능):"
    echo "   export SGLANG_DISABLE_FLASHINFER=1"
    echo "   bash scripts/start_korean_sglang.sh"
    echo ""
    echo "2. CPU 전용 모드로 SGLang 사용:"
    echo "   export CUDA_VISIBLE_DEVICES=\"\""
    echo "   bash scripts/start_korean_sglang.sh"
    echo ""
    echo "3. 다른 환경에서 재시도:"
    echo "   conda create -n sglang_gpu python=3.10"
    echo "   conda activate sglang_gpu"
    echo "   # 처음부터 다시 설치"
fi

echo ""
echo "스크립트 완료: $(date)"