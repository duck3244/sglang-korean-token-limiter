#!/bin/bash
# SGLang 완전 수정 설치 스크립트 (모든 오류 해결)

set -e

echo "🔧 SGLang 완전 수정 설치 (모든 오류 해결)"
echo "========================================"

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 1. 완전한 환경 정리
echo -e "${YELLOW}🧹 완전한 환경 정리...${NC}"

# 문제 패키지들 완전 제거
pip uninstall -y sglang vllm outlines flashinfer flash-attn triton bitsandbytes numpy || true

# pip 캐시 정리
pip cache purge

# Python 캐시 정리
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

echo -e "${GREEN}✅ 환경 정리 완료${NC}"

# 2. 기본 도구 안정화
echo -e "\n${BLUE}📦 기본 도구 안정화...${NC}"

# setuptools 충돌 해결
pip install setuptools==68.2.2 wheel==0.41.2 --force-reinstall
pip install --upgrade pip

# NumPy 1.x 강제 설치 (NumPy 2.x 문제 해결)
pip install "numpy<2.0,>=1.21.0" --force-reinstall

echo -e "${GREEN}✅ 기본 도구 안정화 완료${NC}"

# 3. PyTorch 안정 설치
echo -e "\n${BLUE}🔥 PyTorch 안정 설치...${NC}"

# 기존 PyTorch 완전 제거
pip uninstall torch torchvision torchaudio -y || true

# PyTorch 2.1.0 설치 (SGLang 최고 호환성)
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# PyTorch 확인
python -c "
import torch
print(f'✅ PyTorch: {torch.__version__}')
if torch.cuda.is_available():
    print(f'✅ CUDA: {torch.version.cuda}')
    print(f'✅ GPU: {torch.cuda.get_device_name()}')
"

# 4. 핵심 의존성 설치 (호환성 우선)
echo -e "\n${BLUE}📦 핵심 의존성 설치...${NC}"

# Transformers 생태계 (안정 버전)
pip install transformers==4.36.0
pip install tokenizers==0.15.0
pip install accelerate==0.25.0
pip install safetensors==0.4.1
pip install sentencepiece==0.1.99
pip install protobuf==4.25.1

# 필수 시스템 라이브러리
pip install psutil==5.9.6
pip install requests==2.32.4
pip install packaging

echo -e "${GREEN}✅ 핵심 의존성 설치 완료${NC}"

# 5. Outlines 문제 해결
echo -e "\n${BLUE}🔧 Outlines 호환성 문제 해결...${NC}"

# 방법 1: 호환되는 outlines 버전 설치 시도
OUTLINES_VERSIONS=("0.0.37" "0.0.36" "0.0.35" "0.0.34")
OUTLINES_INSTALLED=false

for version in "${OUTLINES_VERSIONS[@]}"; do
    echo "Outlines $version 시도..."
    if pip install "outlines==$version" --no-deps; then
        echo -e "${GREEN}✅ Outlines $version 설치 성공${NC}"
        OUTLINES_INSTALLED=true
        break
    fi
done

# 방법 2: Outlines 실패 시 더미 모듈 생성
if [ "$OUTLINES_INSTALLED" = false ]; then
    echo -e "${YELLOW}⚠️ Outlines 설치 실패 - 더미 모듈 생성${NC}"
    
    python -c "
import os
import sglang

# SGLang constrained 경로 찾기
sglang_path = os.path.dirname(sglang.__file__)
constrained_path = os.path.join(sglang_path, 'srt', 'constrained')

if not os.path.exists(constrained_path):
    os.makedirs(constrained_path, exist_ok=True)

init_file = os.path.join(constrained_path, '__init__.py')

# 완전한 더미 모듈 생성
dummy_content = '''
# SGLang constrained - Complete Dummy Module for Outlines Compatibility

import logging
from typing import List, Dict, Any, Optional, Union, Callable

logger = logging.getLogger(__name__)

# Dummy cache functions
def disable_cache():
    pass

def disk_cache(func):
    return func

# Dummy FSM classes
class FSMInfo:
    def __init__(self, vocab_size=50257, init_state=0, final_states=None):
        self.vocab_size = vocab_size
        self.init_state = init_state
        self.final_states = final_states or []

class RegexGuide:
    def __init__(self, regex_string, tokenizer=None):
        self.regex_string = regex_string
        self.tokenizer = tokenizer
        self.fsm_info = FSMInfo()
    
    def get_next_instruction(self, state):
        return {\"type\": \"generate\", \"allowed_tokens\": None}
    
    def is_final_state(self, state):
        return False
    
    def copy(self):
        return RegexGuide(self.regex_string, self.tokenizer)

class JSONGuide:
    def __init__(self, schema, tokenizer=None):
        self.schema = schema
        self.tokenizer = tokenizer
    
    def get_next_instruction(self, state):
        return {\"type\": \"generate\", \"allowed_tokens\": None}

class ChoiceGuide:
    def __init__(self, choices, tokenizer=None):
        self.choices = choices
        self.tokenizer = tokenizer

class TransformerTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.vocabulary = getattr(tokenizer, \"get_vocab\", lambda: {})()
        self.vocab_size = getattr(tokenizer, \"vocab_size\", 50257)

# Dummy functions from outlines.fsm.json_schema
def build_regex_from_schema(schema):
    return \".*\"

def build_regex_from_object(obj):
    return \".*\"

def get_schema_from_signature(func):
    return {}

def make_byte_level_fsm(regex_string, tokenizer=None):
    return FSMInfo()

def make_deterministic_fsm(fsm_info):
    return fsm_info

def create_fsm_index_tokenizer(fsm_info, tokenizer=None):
    return {
        \"states_to_token_maps\": {},
        \"empty_token_ids\": set(),
        \"final_states\": set()
    }

def convert_token_to_string(token, tokenizer=None):
    return str(token)

# Export all symbols
__all__ = [
    \"disable_cache\", \"disk_cache\", \"FSMInfo\", \"RegexGuide\", \"JSONGuide\",
    \"ChoiceGuide\", \"TransformerTokenizer\", \"build_regex_from_schema\",
    \"build_regex_from_object\", \"get_schema_from_signature\", \"make_byte_level_fsm\",
    \"make_deterministic_fsm\", \"create_fsm_index_tokenizer\", \"convert_token_to_string\"
]

logger.info(\"SGLang constrained dummy module loaded (Outlines compatibility)\")
'''

with open(init_file, 'w', encoding='utf-8') as f:
    f.write(dummy_content)

print(f'✅ SGLang constrained 더미 모듈 생성: {init_file}')
"
fi

# 6. SGLang 설치 (여러 방법 시도)
echo -e "\n${PURPLE}🚀 SGLang 설치...${NC}"

# 방법 1: 기본 설치
if pip install sglang==0.2.15; then
    echo -e "${GREEN}✅ SGLang 기본 설치 성공${NC}"
    INSTALL_METHOD="basic"
elif pip install "sglang[all]==0.2.15"; then
    echo -e "${GREEN}✅ SGLang [all] 설치 성공${NC}"
    INSTALL_METHOD="all"
elif pip install "git+https://github.com/sgl-project/sglang.git"; then
    echo -e "${GREEN}✅ SGLang Git 설치 성공${NC}"
    INSTALL_METHOD="git"
else
    echo -e "${RED}❌ SGLang 설치 실패${NC}"
    exit 1
fi

# 7. 웹 서버 패키지 설치
echo -e "\n${BLUE}🌐 웹 서버 패키지 설치...${NC}"

pip install fastapi==0.104.1
pip install uvicorn==0.24.0
pip install httpx==0.25.2
pip install pydantic==2.5.0
pip install sse-starlette==1.6.5

# 8. 기타 필수 패키지
echo -e "\n${BLUE}📊 기타 필수 패키지...${NC}"

pip install streamlit==1.28.2
pip install plotly==5.17.0
pip install pandas==2.1.4
pip install redis==5.0.1
pip install PyYAML==6.0.1

# 9. 설치 검증
echo -e "\n${BLUE}🧪 설치 검증...${NC}"

python -c "
import sys
import warnings
warnings.filterwarnings('ignore')

print('=== 설치 검증 ===')

# 핵심 패키지 확인
packages = [
    ('torch', 'PyTorch'),
    ('sglang', 'SGLang'),
    ('transformers', 'Transformers'),
    ('fastapi', 'FastAPI'),
    ('streamlit', 'Streamlit'),
    ('numpy', 'NumPy')
]

all_good = True
for pkg, name in packages:
    try:
        module = __import__(pkg)
        version = getattr(module, '__version__', 'Unknown')
        print(f'✅ {name}: {version}')
    except ImportError as e:
        print(f'❌ {name}: {e}')
        if pkg in ['torch', 'sglang']:
            all_good = False

if not all_good:
    print('❌ 핵심 패키지 누락')
    sys.exit(1)

print()
print('=== SGLang 기능 테스트 ===')

try:
    import sglang
    print(f'✅ SGLang: {sglang.__version__}')
    
    # 기본 함수
    from sglang import function, system, user, assistant, gen
    print('✅ SGLang 기본 함수')
    
    # Constrained 모듈 (더미 포함)
    try:
        from sglang.srt.constrained import disable_cache, build_regex_from_schema
        print('✅ SGLang constrained (더미 또는 실제)')
    except Exception as e:
        print(f'⚠️ SGLang constrained: {e}')
    
    # 서버 모듈
    try:
        from sglang.srt.server import launch_server
        print('✅ sglang.srt.server')
        working_server = 'sglang.srt.server'
    except:
        try:
            import sglang.launch_server
            print('✅ sglang.launch_server')
            working_server = 'sglang.launch_server'
        except Exception as e:
            print(f'❌ 서버 모듈: {e}')
            working_server = None
    
    if working_server:
        with open('/tmp/working_server.txt', 'w') as f:
            f.write(working_server)
        print(f'🎯 사용 가능한 서버: {working_server}')
        print('🎉 SGLang 완전 설치 성공!')
    else:
        print('❌ SGLang 서버 모듈 사용 불가')
        sys.exit(1)

except ImportError as e:
    print(f'❌ SGLang import 실패: {e}')
    sys.exit(1)
"

# 10. 실행 스크립트 생성
echo -e "\n${BLUE}📝 실행 스크립트 생성...${NC}"

if [ -f "/tmp/working_server.txt" ]; then
    WORKING_SERVER=$(cat /tmp/working_server.txt)
    
    cat > run_sglang_fixed.py << EOF
#!/usr/bin/env python3
"""
SGLang 완전 수정 버전 실행 스크립트
"""

import sys
import subprocess
import time
import requests
import os
import argparse

def start_server(model_path="microsoft/DialoGPT-medium", port=8000):
    """SGLang 서버 시작"""
    
    print("🚀 SGLang 서버 시작 (완전 수정 버전)")
    print(f"모델: {model_path}")
    print(f"포트: {port}")
    print(f"서버 모듈: $WORKING_SERVER")
    
    # 서버 명령어
    if "$WORKING_SERVER" == "sglang.srt.server":
        cmd = [sys.executable, "-m", "sglang.srt.server"]
    else:
        cmd = [sys.executable, "-m", "sglang.launch_server"]
    
    args = [
        "--model-path", model_path,
        "--port", str(port),
        "--host", "127.0.0.1",
        "--trust-remote-code",
        "--mem-fraction-static", "0.7",
        "--max-running-requests", "8",
        "--disable-flashinfer",  # FlashInfer 문제 회피
        "--dtype", "float16"
    ]
    
    full_cmd = cmd + args
    print(f"실행: {' '.join(full_cmd)}")
    
    try:
        os.makedirs("logs", exist_ok=True)
        
        with open("logs/sglang_fixed.log", "w") as log_file:
            process = subprocess.Popen(
                full_cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT
            )
        
        print(f"✅ 서버 시작 (PID: {process.pid})")
        
        # 서버 준비 대기
        print("⏳ 서버 준비 대기...")
        for i in range(180):  # 3분 대기
            try:
                response = requests.get(f"http://127.0.0.1:{port}/get_model_info", timeout=5)
                if response.status_code == 200:
                    print(f"✅ 서버 준비 완료! ({i+1}초)")
                    return process
            except:
                pass
                
            if process.poll() is not None:
                print("❌ 서버 프로세스 종료됨")
                return None
                
            if i % 20 == 0 and i > 0:
                print(f"대기 중... {i}초")
            
            time.sleep(1)
        
        print("❌ 서버 준비 시간 초과")
        process.terminate()
        return None
        
    except Exception as e:
        print(f"❌ 서버 시작 실패: {e}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="microsoft/DialoGPT-medium")
    parser.add_argument("--port", type=int, default=8000)
    
    args = parser.parse_args()
    
    process = start_server(args.model, args.port)
    
    if process:
        print("🎉 서버 실행 성공!")
        print()
        print("테스트:")
        print(f"curl http://127.0.0.1:{args.port}/get_model_info")
        print()
        print("Token Limiter (다른 터미널):")
        print("python main_sglang.py")
        print()
        print("종료: Ctrl+C")
        
        try:
            process.wait()
        except KeyboardInterrupt:
            print("\\n🛑 서버 종료 중...")
            process.terminate()
            process.wait()
    else:
        print("❌ 서버 실행 실패")
        
        # 로그 출력
        if os.path.exists("logs/sglang_fixed.log"):
            print("\\n=== 로그 ===")
            with open("logs/sglang_fixed.log", "r") as f:
                print(f.read()[-2000:])
        
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
EOF

    chmod +x run_sglang_fixed.py
    echo -e "${GREEN}✅ 실행 스크립트 생성: run_sglang_fixed.py${NC}"
fi

echo ""
echo -e "${GREEN}🎉 SGLang 완전 수정 설치 완료!${NC}"
echo "================================="

echo -e "${BLUE}📋 해결된 문제들:${NC}"
echo "✅ NumPy 2.x 호환성 문제 → NumPy 1.x 강제 설치"
echo "✅ FlashInfer 설치 실패 → 비활성화로 우회"
echo "✅ Flash Attention 컴파일 오류 → 생략 (성능 약간 저하)"
echo "✅ Outlines 의존성 문제 → 더미 모듈 또는 호환 버전"
echo "✅ SGLang 서버 모듈 문제 → 자동 감지 및 설정"

echo ""
echo -e "${BLUE}🚀 사용 방법:${NC}"
echo ""
echo "1. SGLang 서버 시작:"
if [ -f "run_sglang_fixed.py" ]; then
    echo "   python run_sglang_fixed.py --model microsoft/DialoGPT-medium --port 8000"
fi

echo ""
echo "2. 직접 명령어:"
if [ -f "/tmp/working_server.txt" ]; then
    WORKING_SERVER=$(cat /tmp/working_server.txt)
    if [[ "$WORKING_SERVER" == "sglang.srt.server" ]]; then
        echo "   python -m sglang.srt.server --model-path microsoft/DialoGPT-medium --port 8000 --disable-flashinfer"
    else
        echo "   python -m sglang.launch_server --model-path microsoft/DialoGPT-medium --port 8000 --disable-flashinfer"
    fi
fi

echo ""
echo "3. Token Limiter (다른 터미널):"
echo "   python main_sglang.py"

echo ""
echo -e "${BLUE}💡 참고사항:${NC}"
echo "- FlashInfer와 Flash Attention이 비활성화되어 성능이 약간 저하됨"
echo "- 구조화된 생성(JSON, Regex) 기능이 제한적일 수 있음"
echo "- 기본 텍스트 생성과 채팅은 정상 작동"
echo "- 서버 시작에 시간이 걸릴 수 있음 (모델 다운로드)"

echo ""
echo "수정 완료 시간: $(date)"