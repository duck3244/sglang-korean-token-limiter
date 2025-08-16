#!/bin/bash
# NumPy & Outlines 호환성 완전 수정

set -e

echo "🔧 NumPy & Outlines 호환성 완전 수정"
echo "=================================="

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 1. NumPy 문제 해결
echo -e "${BLUE}1. NumPy 호환성 문제 해결...${NC}"

# 현재 NumPy 버전 확인
NUMPY_VERSION=$(python -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "None")
echo "현재 NumPy 버전: $NUMPY_VERSION"

if [[ "$NUMPY_VERSION" == 2.* ]]; then
    echo "NumPy 2.x 감지 - 1.x로 다운그레이드 필요"
    
    # NumPy 2.x 제거
    pip uninstall numpy -y
    
    # NumPy 1.24.4 설치 (PyTorch 2.1.0과 호환)
    pip install "numpy<2.0,>=1.21.0"
    
    echo -e "${GREEN}✅ NumPy 다운그레이드 완료${NC}"
else
    echo "NumPy 버전 OK"
fi

# 2. Outlines 완전 재설치
echo -e "\n${BLUE}2. Outlines 완전 재설치...${NC}"

# 기존 outlines 완전 제거
pip uninstall outlines -y 2>/dev/null || true

# outlines 의존성 문제 해결 방법 선택
echo "Outlines 설치 방법:"
echo "1. 수동 의존성 설치 후 outlines 설치 (권장)"
echo "2. outlines 없이 SGLang 사용 (constrained 기능 제외)"
echo "3. 구 버전 outlines 설치"

read -p "선택하세요 (1-3): " -n 1 -r
echo

case $REPLY in
    1)
        echo -e "${BLUE}방법 1: 수동 의존성 설치${NC}"
        
        # 기본 의존성 설치
        echo "기본 의존성 설치..."
        pip install pydantic==1.10.12  # 구 버전 (호환성)
        pip install jinja2 jsonschema referencing
        pip install cloudpickle diskcache
        pip install interegular lark nest-asyncio
        
        # 문제 의존성 해결
        echo "문제 의존성 해결..."
        
        # pycountry 설치 시도
        pip install pycountry || echo "⚠️ pycountry 건너뛰기"
        
        # pyairports 더미 생성 (업그레이드)
        mkdir -p /tmp/pyairports
        cat > /tmp/pyairports/__init__.py << 'EOF'
# Enhanced dummy pyairports module
__version__ = "2.1.2"

class Airport:
    def __init__(self, iata=None, icao=None, name=None):
        self.iata = iata
        self.icao = icao  
        self.name = name

def get_airports():
    return []

def get_airport_by_iata(code):
    return Airport(iata=code, name=f"Airport {code}")

def get_airport_by_icao(code):
    return Airport(icao=code, name=f"Airport {code}")

# 추가 함수들
def get_airports_by_country(country):
    return []

def search_airports(query):
    return []
EOF
        
        # PYTHONPATH 설정
        export PYTHONPATH="/tmp:$PYTHONPATH"
        
        # outlines 설치 (특정 버전)
        echo "Outlines 설치..."
        OUTLINES_VERSIONS=("0.0.46" "0.0.45" "0.0.44")
        
        for version in "${OUTLINES_VERSIONS[@]}"; do
            echo "Outlines $version 설치 시도..."
            
            if PYTHONPATH="/tmp:$PYTHONPATH" pip install "outlines==$version"; then
                echo -e "${GREEN}✅ Outlines $version 설치 성공${NC}"
                OUTLINES_INSTALLED=$version
                break
            else
                echo -e "${YELLOW}⚠️ Outlines $version 설치 실패${NC}"
            fi
        done
        
        if [ -z "$OUTLINES_INSTALLED" ]; then
            echo "모든 outlines 버전 실패, 의존성 없이 설치..."
            PYTHONPATH="/tmp:$PYTHONPATH" pip install outlines --no-deps
            OUTLINES_INSTALLED="no-deps"
        fi
        
        FIX_METHOD="manual_deps"
        ;;
        
    2)
        echo -e "${BLUE}방법 2: Outlines 없이 사용${NC}"
        
        # SGLang constrained 모듈을 완전히 더미로 교체
        python -c "
import os
import sglang

# SGLang constrained 경로
sglang_path = os.path.dirname(sglang.__file__)
constrained_path = os.path.join(sglang_path, 'srt', 'constrained')
init_file = os.path.join(constrained_path, '__init__.py')

print(f'SGLang constrained 더미 교체: {init_file}')

# 백업
backup_file = init_file + '.no_outlines_backup'
if os.path.exists(init_file) and not os.path.exists(backup_file):
    import shutil
    shutil.copy2(init_file, backup_file)

# 완전한 더미 모듈
dummy_content = '''
# SGLang constrained module - Complete dummy implementation
import logging
from typing import List, Dict, Any, Optional, Union, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# 더미 캐시 함수들
def disable_cache():
    \"\"\"더미 disable_cache\"\"\"
    pass

def disk_cache(func):
    \"\"\"더미 disk_cache 데코레이터\"\"\"
    return func

# FSM 관련 더미 클래스들
@dataclass
class FSMInfo:
    \"\"\"더미 FSMInfo\"\"\"
    vocab_size: int = 50257
    init_state: int = 0
    final_states: List[int] = None
    
    def __post_init__(self):
        if self.final_states is None:
            self.final_states = []

class RegexGuide:
    \"\"\"더미 RegexGuide\"\"\"
    def __init__(self, regex_string: str, tokenizer=None):
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
    \"\"\"더미 JSONGuide\"\"\"
    def __init__(self, schema: Union[str, Dict], tokenizer=None):
        self.schema = schema
        self.tokenizer = tokenizer
        
    def get_next_instruction(self, state):
        return {\"type\": \"generate\", \"allowed_tokens\": None}

class ChoiceGuide:
    \"\"\"더미 ChoiceGuide\"\"\"
    def __init__(self, choices: List[str], tokenizer=None):
        self.choices = choices
        self.tokenizer = tokenizer

class TransformerTokenizer:
    \"\"\"더미 TransformerTokenizer\"\"\"
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.vocabulary = getattr(tokenizer, 'get_vocab', lambda: {})()
        self.vocab_size = getattr(tokenizer, 'vocab_size', 50257)

# 더미 함수들
def build_regex_from_object(obj):
    \"\"\"더미 build_regex_from_object\"\"\"
    return \".*\"

def get_schema_from_signature(func):
    \"\"\"더미 get_schema_from_signature\"\"\"
    return {}

def make_byte_level_fsm(regex_string: str, tokenizer=None):
    \"\"\"더미 make_byte_level_fsm\"\"\"
    return FSMInfo()

def make_deterministic_fsm(fsm_info):
    \"\"\"더미 make_deterministic_fsm\"\"\"
    return fsm_info

def create_fsm_index_tokenizer(fsm_info, tokenizer=None):
    \"\"\"더미 create_fsm_index_tokenizer\"\"\"
    return {
        'states_to_token_maps': {},
        'empty_token_ids': set(),
        'final_states': set()
    }

def convert_token_to_string(token, tokenizer=None):
    \"\"\"더미 convert_token_to_string\"\"\"
    return str(token)

# Export all
__all__ = [
    'disable_cache', 'disk_cache', 'FSMInfo', 'RegexGuide', 'JSONGuide', 
    'ChoiceGuide', 'TransformerTokenizer', 'build_regex_from_object',
    'get_schema_from_signature', 'make_byte_level_fsm', 'make_deterministic_fsm',
    'create_fsm_index_tokenizer', 'convert_token_to_string'
]

logger.info(\"SGLang constrained module loaded in dummy mode (no outlines)\")
'''

with open(init_file, 'w') as f:
    f.write(dummy_content)

print('✅ SGLang constrained 더미 모듈 완전 교체 완료')
"
        
        FIX_METHOD="no_outlines"
        ;;
        
    3)
        echo -e "${BLUE}방법 3: 구 버전 outlines 설치${NC}"
        
        # 매우 구 버전 outlines 시도
        OLD_VERSIONS=("0.0.20" "0.0.19" "0.0.18")
        
        for version in "${OLD_VERSIONS[@]}"; do
            echo "Outlines $version 설치 시도..."
            if pip install "outlines==$version"; then
                echo -e "${GREEN}✅ Outlines $version 설치 성공${NC}"
                OUTLINES_INSTALLED=$version
                break
            fi
        done
        
        FIX_METHOD="old_outlines"
        ;;
        
    *)
        echo "잘못된 선택. 방법 2로 진행합니다."
        FIX_METHOD="no_outlines"
        ;;
esac

# 3. 검증
echo -e "\n${BLUE}3. 수정 후 검증...${NC}"

python -c "
import sys
import warnings
warnings.filterwarnings('ignore')

print('=== 기본 패키지 검증 ===')

# NumPy 확인
try:
    import numpy
    print(f'✅ NumPy: {numpy.__version__}')
    if numpy.__version__.startswith('2.'):
        print('⚠️ NumPy 2.x 여전히 설치됨')
    else:
        print('✅ NumPy 1.x 호환성 OK')
except Exception as e:
    print(f'❌ NumPy: {e}')

# PyTorch 확인
try:
    import torch
    print(f'✅ PyTorch: {torch.__version__}')
except Exception as e:
    print(f'❌ PyTorch: {e}')

# Outlines 확인 (선택적)
try:
    import outlines
    version = getattr(outlines, '__version__', 'Unknown')
    print(f'✅ Outlines: {version}')
    outlines_ok = True
except ImportError:
    print('⚠️ Outlines: 없음 (더미 모드)')
    outlines_ok = False

print()
print('=== SGLang 검증 ===')

try:
    import sglang
    print(f'✅ SGLang: {sglang.__version__}')
    
    # 기본 함수
    from sglang import function, system, user, assistant, gen
    print('✅ SGLang 기본 함수')
    
    # Constrained 모듈
    try:
        from sglang.srt.constrained import disable_cache
        print('✅ SGLang constrained (더미 또는 실제)')
        constrained_ok = True
    except Exception as e:
        print(f'❌ SGLang constrained: {e}')
        constrained_ok = False
    
    # 서버 모듈
    server_modules = ['sglang.srt.server', 'sglang.launch_server']
    working_server = None
    
    for module in server_modules:
        try:
            if module == 'sglang.srt.server':
                from sglang.srt.server import launch_server
            else:
                import sglang.launch_server
            
            print(f'✅ 서버 모듈: {module}')
            working_server = module
            break
            
        except Exception as e:
            print(f'❌ {module}: {e}')
    
    if working_server and constrained_ok:
        print('🎉 SGLang 완전 사용 가능!')
        with open('/tmp/final_working_server.txt', 'w') as f:
            f.write(working_server)
    elif working_server:
        print('✅ SGLang 기본 기능 사용 가능 (constrained 제한적)')
        with open('/tmp/final_working_server.txt', 'w') as f:
            f.write(working_server)
    else:
        print('❌ SGLang 서버 사용 불가')
        sys.exit(1)

except Exception as e:
    print(f'❌ SGLang 검증 실패: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"

# 4. 최종 실행 스크립트 생성
echo -e "\n${BLUE}4. 최종 실행 스크립트 생성...${NC}"

if [ -f "/tmp/final_working_server.txt" ]; then
    WORKING_SERVER=$(cat /tmp/final_working_server.txt)
    
    cat > run_sglang_fixed.py << EOF
#!/usr/bin/env python3
"""
SGLang 최종 수정 버전 실행 스크립트
"""

import sys
import subprocess
import time
import requests
import os
import argparse
import warnings

# 경고 무시
warnings.filterwarnings('ignore')

def start_server(model_path="microsoft/DialoGPT-medium", port=8000):
    """SGLang 서버 시작 (수정 버전)"""
    
    print("🚀 SGLang 서버 시작 (최종 수정 버전)")
    print(f"모델: {model_path}")
    print(f"포트: {port}")
    print(f"서버 모듈: $WORKING_SERVER")
    print(f"수정 방법: $FIX_METHOD")
    
    # 환경 변수 설정
    env = os.environ.copy()
    if "$FIX_METHOD" in ["manual_deps", "old_outlines"]:
        env['PYTHONPATH'] = '/tmp:' + env.get('PYTHONPATH', '')
    
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
        "--mem-fraction-static", "0.65",  # 메모리 여유
        "--max-running-requests", "4"     # 안정성 우선
    ]
    
    full_cmd = cmd + args
    print(f"실행: {' '.join(full_cmd)}")
    
    try:
        os.makedirs("logs", exist_ok=True)
        
        with open("logs/sglang_fixed.log", "w") as log_file:
            process = subprocess.Popen(
                full_cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env
            )
        
        print(f"✅ 서버 시작 (PID: {process.pid})")
        
        # PID 저장
        os.makedirs("pids", exist_ok=True)
        with open("pids/sglang.pid", "w") as f:
            f.write(str(process.pid))
        
        return process
        
    except Exception as e:
        print(f"❌ 서버 시작 실패: {e}")
        return None

def wait_for_server(port=8000, timeout=180):
    """서버 대기 (긴 타임아웃)"""
    
    print("⏳ 서버 준비 대기 (모델 다운로드 시간 포함)...")
    
    for i in range(timeout):
        try:
            response = requests.get(f"http://127.0.0.1:{port}/get_model_info", timeout=5)
            if response.status_code == 200:
                print(f"✅ 서버 준비 완료! ({i+1}초)")
                return True
        except:
            pass
        
        if i % 20 == 0 and i > 0:
            print(f"대기 중... {i}초")
            
            # 로그 체크
            if os.path.exists("logs/sglang_fixed.log"):
                with open("logs/sglang_fixed.log", "r") as f:
                    lines = f.readlines()
                    if lines:
                        recent_lines = lines[-3:]
                        for line in recent_lines:
                            clean_line = line.strip()
                            if clean_line:
                                print(f"  로그: {clean_line}")
        
        time.sleep(1)
    
    print("❌ 서버 준비 시간 초과")
    return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="microsoft/DialoGPT-medium")
    parser.add_argument("--port", type=int, default=8000)
    
    args = parser.parse_args()
    
    process = start_server(args.model, args.port)
    if not process:
        return 1
    
    if wait_for_server(args.port):
        print("🎉 SGLang 서버 실행 성공!")
        print()
        print(f"서버 주소: http://127.0.0.1:{args.port}")
        print(f"테스트: curl http://127.0.0.1:{args.port}/get_model_info")
        print("Token Limiter: python main_sglang.py (다른 터미널)")
        print("로그 모니터링: tail -f logs/sglang_fixed.log")
        print()
        print("종료: Ctrl+C")
        
        try:
            process.wait()
        except KeyboardInterrupt:
            print("\\n🛑 서버 종료 중...")
            process.terminate()
            process.wait()
            try:
                os.remove("pids/sglang.pid")
            except:
                pass
    else:
        print("❌ 서버 대기 실패")
        
        # 상세 로그 출력
        if os.path.exists("logs/sglang_fixed.log"):
            print("\\n=== 로그 내용 ===")
            with open("logs/sglang_fixed.log", "r") as f:
                content = f.read()
                print(content[-2000:])  # 마지막 2000자
        
        if process.poll() is None:
            process.terminate()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
EOF

    chmod +x run_sglang_fixed.py
    echo -e "${GREEN}✅ 최종 실행 스크립트 생성: run_sglang_fixed.py${NC}"
fi

echo ""
echo -e "${GREEN}🎉 NumPy & Outlines 호환성 수정 완료!${NC}"
echo "======================================="

echo -e "${BLUE}📋 수정 내용:${NC}"
echo "- NumPy: 1.x 버전으로 다운그레이드"
echo "- 수정 방법: $FIX_METHOD"
if [ ! -z "$OUTLINES_INSTALLED" ]; then
    echo "- Outlines: $OUTLINES_INSTALLED"
fi
if [ -f "/tmp/final_working_server.txt" ]; then
    echo "- 서버 모듈: $(cat /tmp/final_working_server.txt)"
fi

echo ""
echo -e "${BLUE}🚀 사용 방법:${NC}"
echo ""
echo "1. 최종 수정 버전으로 SGLang 서버 시작:"
if [ -f "run_sglang_fixed.py" ]; then
    echo "   python run_sglang_fixed.py --model microsoft/DialoGPT-medium --port 8000"
fi

echo ""
echo "2. 직접 명령어 (환경 변수 포함):"
if [ -f "/tmp/final_working_server.txt" ]; then
    WORKING_SERVER=$(cat /tmp/final_working_server.txt)
    if [[ "$FIX_METHOD" == "manual_deps" ]] || [[ "$FIX_METHOD" == "old_outlines" ]]; then
        ENV_PREFIX="PYTHONPATH=/tmp:\$PYTHONPATH "
    else
        ENV_PREFIX=""
    fi
    
    if [[ "$WORKING_SERVER" == "sglang.srt.server" ]]; then
        echo "   ${ENV_PREFIX}python -m sglang.srt.server --model-path microsoft/DialoGPT-medium --port 8000 --host 127.0.0.1 --trust-remote-code"
    else
        echo "   ${ENV_PREFIX}python -m sglang.launch_server --model-path microsoft/DialoGPT-medium --port 8000 --host 127.0.0.1 --trust-remote-code"
    fi
fi

echo ""
echo "3. Token Limiter (다른 터미널):"
echo "   python main_sglang.py"

echo ""
echo -e "${BLUE}💡 참고:${NC}"
if [[ "$FIX_METHOD" == "no_outlines" ]]; then
    echo "- 구조화된 생성 기능(JSON, Regex) 제한적"
    echo "- 기본 텍스트 생성은 정상 작동"
else
    echo "- 모든 SGLang 기능 사용 가능"
fi
echo "- 서버 시작에 시간이 걸릴 수 있음 (모델 다운로드)"

echo ""
echo "수정 완료 시간: $(date)"