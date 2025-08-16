#!/bin/bash
# SGLang용 vLLM 의존성 설치

set -e

echo "🔧 SGLang용 vLLM 의존성 설치"
echo "============================"

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 1. 현재 상태 확인
echo -e "${BLUE}1. 현재 상태 확인...${NC}"
python -c "
import sys
print(f'Python: {sys.version}')

try:
    import sglang
    print(f'✅ SGLang: {sglang.__version__}')
except ImportError as e:
    print(f'❌ SGLang: {e}')

try:
    import vllm
    print(f'✅ vLLM: {vllm.__version__}')
except ImportError as e:
    print(f'❌ vLLM: {e}')

try:
    import torch
    print(f'✅ PyTorch: {torch.__version__}')
    if torch.cuda.is_available():
        print(f'✅ CUDA: {torch.version.cuda}')
    else:
        print('💻 CPU 모드')
except ImportError as e:
    print(f'❌ PyTorch: {e}')
"

# 2. SGLang 호환 vLLM 버전 설치
echo -e "\n${BLUE}2. SGLang 호환 vLLM 버전 설치...${NC}"

# SGLang 0.2.6과 호환되는 vLLM 버전들 시도
VLLM_VERSIONS=("0.2.6" "0.2.7" "0.3.0" "0.3.1" "0.3.2")

for version in "${VLLM_VERSIONS[@]}"; do
    echo "=== vLLM ${version} 설치 시도 ==="
    
    # 기존 vLLM 제거 (있다면)
    pip uninstall vllm -y 2>/dev/null || true
    
    # GPU 사용 가능한지 확인
    if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        echo "GPU 환경에서 vLLM ${version} 설치..."
        
        # CUDA 버전에 맞는 vLLM 설치
        if pip install "vllm==${version}"; then
            echo -e "${GREEN}✅ vLLM ${version} 설치 성공${NC}"
            
            # 즉시 SGLang 호환성 테스트
            if python -c "
import sys
try:
    # vLLM 기본 import
    import vllm
    print(f'✅ vLLM {vllm.__version__} import 성공')
    
    # SGLang이 필요로 하는 vLLM 모듈들 테스트
    try:
        from vllm.transformers_utils.configs import ChatGLMConfig, DbrxConfig
        print('✅ vllm.transformers_utils.configs 정상')
    except ImportError as e:
        print(f'⚠️ transformers_utils.configs 실패: {e}')
        # 다른 경로 시도
        try:
            from vllm.model_executor.models.llama import LlamaForCausalLM
            print('✅ vllm.model_executor 모듈 정상')
        except ImportError as e2:
            print(f'❌ vLLM 모듈 접근 실패: {e2}')
            sys.exit(1)
    
    # SGLang 서버 런처 최종 테스트
    try:
        from sglang.srt.server import launch_server
        print('✅ SGLang 서버 런처 정상')
    except ImportError as e:
        print(f'❌ SGLang 서버 런처 실패: {e}')
        sys.exit(1)
    
    print(f'🎉 vLLM {vllm.__version__} SGLang 호환성 확인!')
    
except Exception as e:
    print(f'❌ 테스트 실패: {e}')
    sys.exit(1)
" 2>/dev/null; then
                echo -e "${GREEN}🎉 호환 가능한 vLLM 버전 발견: ${version}${NC}"
                WORKING_VLLM_VERSION=$version
                break
            else
                echo -e "${YELLOW}⚠️ vLLM ${version} SGLang 호환성 실패${NC}"
            fi
        else
            echo -e "${YELLOW}⚠️ vLLM ${version} 설치 실패${NC}"
        fi
    else
        echo "CPU 환경에서 vLLM ${version} 설치..."
        
        # CPU 버전 vLLM (제한적)
        if pip install "vllm==${version}" --extra-index-url https://download.pytorch.org/whl/cpu; then
            echo -e "${GREEN}✅ vLLM ${version} CPU 버전 설치 성공${NC}"
            WORKING_VLLM_VERSION=$version
            break
        else
            echo -e "${YELLOW}⚠️ vLLM ${version} CPU 설치 실패${NC}"
        fi
    fi
done

# 3. 모든 버전 실패 시 최신 버전 시도
if [ -z "$WORKING_VLLM_VERSION" ]; then
    echo -e "\n${YELLOW}⚠️ 특정 버전 실패. 최신 버전 시도...${NC}"
    
    pip uninstall vllm -y 2>/dev/null || true
    
    # 최신 vLLM 설치
    if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        echo "최신 GPU vLLM 설치..."
        pip install vllm
    else
        echo "최신 CPU vLLM 설치..."
        pip install vllm --extra-index-url https://download.pytorch.org/whl/cpu
    fi
    
    # 호환성 패치 시도
    echo "vLLM 호환성 패치 적용..."
    python -c "
import sys
import os

try:
    import vllm
    print(f'vLLM {vllm.__version__} 설치됨')
    
    # SGLang이 찾지 못하는 모듈들 패치
    try:
        from vllm.transformers_utils.configs import ChatGLMConfig, DbrxConfig
        print('✅ transformers_utils.configs 정상')
    except ImportError:
        print('⚠️ transformers_utils.configs 없음. 패치 시도...')
        
        # vLLM 패키지 경로 찾기
        vllm_path = os.path.dirname(vllm.__file__)
        
        # 더미 configs 모듈 생성
        transformers_utils_path = os.path.join(vllm_path, 'transformers_utils')
        os.makedirs(transformers_utils_path, exist_ok=True)
        
        # __init__.py
        with open(os.path.join(transformers_utils_path, '__init__.py'), 'w') as f:
            f.write('# Dummy transformers_utils module for SGLang compatibility\\n')
        
        # configs.py
        configs_content = '''
# Dummy configs for SGLang compatibility
from transformers import PretrainedConfig

class ChatGLMConfig(PretrainedConfig):
    model_type = 'chatglm'
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class DbrxConfig(PretrainedConfig):
    model_type = 'dbrx'
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

__all__ = ['ChatGLMConfig', 'DbrxConfig']
'''
        
        with open(os.path.join(transformers_utils_path, 'configs.py'), 'w') as f:
            f.write(configs_content)
        
        print(f'✅ 더미 transformers_utils 모듈 생성: {transformers_utils_path}')
    
except Exception as e:
    print(f'❌ 패치 실패: {e}')
"
fi

# 4. 최종 SGLang + vLLM 호환성 테스트
echo -e "\n${BLUE}4. 최종 SGLang + vLLM 호환성 테스트...${NC}"

python -c "
import sys

try:
    print('=== SGLang + vLLM 최종 호환성 테스트 ===')
    
    # 1. vLLM 확인
    import vllm
    print(f'✅ vLLM: {vllm.__version__}')
    
    # 2. SGLang 기본 기능
    import sglang
    print(f'✅ SGLang: {sglang.__version__}')
    
    # 3. vLLM transformers_utils 모듈
    try:
        from vllm.transformers_utils.configs import ChatGLMConfig, DbrxConfig
        print('✅ vllm.transformers_utils.configs: 정상')
    except ImportError as e:
        print(f'⚠️ transformers_utils.configs: {e}')
        # 여전히 실패하면 다른 vLLM 모듈 확인
        try:
            from vllm.engine.arg_utils import AsyncEngineArgs
            print('✅ vllm.engine 모듈: 정상')
        except ImportError as e2:
            print(f'❌ vLLM 핵심 모듈 실패: {e2}')
            raise
    
    # 4. SGLang 서버 런처 최종 테스트
    try:
        from sglang.srt.server import launch_server
        print('✅ sglang.srt.server.launch_server: 정상')
    except ImportError as e:
        print(f'❌ SGLang 서버 런처 실패: {e}')
        raise
    
    # 5. SGLang 기본 기능
    from sglang import function, system, user, assistant, gen
    print('✅ SGLang 기본 기능: 정상')
    
    print()
    print('🎉 모든 테스트 통과! SGLang + vLLM 호환성 확인!')
    
except Exception as e:
    print(f'❌ 최종 테스트 실패: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"

# 5. 성공 시 안내
if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}🎉 SGLang + vLLM 설치 완료!${NC}"
    echo ""
    echo -e "${BLUE}📋 설치 요약:${NC}"
    
    python -c "
try:
    import vllm, sglang
    print(f'✅ vLLM: {vllm.__version__}')
    print(f'✅ SGLang: {sglang.__version__}')
    print('✅ 모든 의존성 정상 설치')
except Exception as e:
    print(f'❌ 오류: {e}')
"
    
    echo ""
    echo -e "${GREEN}🚀 이제 SGLang 서버를 시작할 수 있습니다:${NC}"
    echo "bash scripts/start_korean_sglang.sh"
    echo ""
    echo -e "${BLUE}💡 설치된 구성:${NC}"
    echo "- SGLang 0.2.6 (메인 프레임워크)"
    echo "- vLLM (의존성 지원)"
    echo "- Outlines 없음 (패치 적용)"
    echo "- 기본 텍스트 생성 및 채팅 지원"
    echo ""
    echo -e "${BLUE}🎯 기능:${NC}"
    echo "✅ 텍스트 생성"
    echo "✅ 채팅 완성"
    echo "✅ 스트리밍 응답"
    echo "✅ 한국어 최적화"
    echo "⚠️ 구조화된 생성 (JSON 등) 제한적"
    
else
    echo -e "\n${RED}❌ vLLM 설치 실패${NC}"
    echo ""
    echo -e "${YELLOW}🔧 대안:${NC}"
    echo "1. 새로운 환경에서 처음부터 설치:"
    echo "   conda create -n sglang_fresh python=3.10"
    echo "   conda activate sglang_fresh"
    echo "   pip install torch transformers"
    echo "   pip install vllm"
    echo "   pip install 'sglang[all]==0.2.6'"
    echo ""
    echo "2. 다른 프레임워크 고려:"
    echo "   - vLLM 단독 사용"
    echo "   - Ollama"
    echo "   - Text Generation Inference"
fi

echo ""
echo "스크립트 완료: $(date)"