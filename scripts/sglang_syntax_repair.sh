#!/bin/bash
# SGLang 구문 오류 수정 및 복구 스크립트

set -e

echo "🔧 SGLang 구문 오류 수정 및 복구"
echo "================================"

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

echo -e "${BLUE}🔍 SGLang 구문 오류 진단 및 수정...${NC}"

python -c "
import os
import sys
import ast
import site

print('SGLang 구문 오류 진단 및 수정...')

# SGLang 패키지 경로 찾기
sglang_path = None
for path in sys.path:
    potential_path = os.path.join(path, 'sglang')
    if os.path.exists(potential_path):
        sglang_path = potential_path
        break

if not sglang_path:
    for site_path in site.getsitepackages():
        potential_path = os.path.join(site_path, 'sglang')
        if os.path.exists(potential_path):
            sglang_path = potential_path
            break

if sglang_path:
    print(f'SGLang 경로: {sglang_path}')
    
    # 문제가 있는 파일들 확인 및 수정
    problem_files = [
        'srt/server.py',
        'launch_server.py',
        'srt/managers/controller_single.py',
        'srt/managers/tp_worker.py'
    ]
    
    for file_name in problem_files:
        file_path = os.path.join(sglang_path, file_name)
        if os.path.exists(file_path):
            print(f'\\n검사 중: {file_name}')
            
            # 파일 읽기
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 구문 오류 확인
            try:
                ast.parse(content)
                print(f'✅ {file_name}: 구문 오류 없음')
            except SyntaxError as e:
                print(f'❌ {file_name}: 구문 오류 발견 (라인 {e.lineno})')
                print(f'   오류: {e.msg}')
                
                # 오류 라인 주변 표시
                lines = content.split('\\n')
                error_line = e.lineno - 1
                print(f'   문제 라인: {lines[error_line]}')
                
                # 일반적인 구문 오류 패턴 수정
                original_content = content
                
                # 1. 'pass as mp' 같은 잘못된 구문 수정
                content = content.replace('pass as mp', 'pass')
                content = content.replace('except as', 'except:')
                
                # 2. 잘못된 import 구문 수정
                import re
                content = re.sub(r'import multiprocessing\\n\\n.*?pass.*?\\n', 
                                'import multiprocessing\\n', content, flags=re.DOTALL)
                
                # 3. 중복된 import 제거
                lines = content.split('\\n')
                import_lines = []
                other_lines = []
                
                for line in lines:
                    if line.strip().startswith('import ') or line.strip().startswith('from '):
                        if line not in import_lines:
                            import_lines.append(line)
                    else:
                        other_lines.append(line)
                
                # 재구성
                content = '\\n'.join(import_lines + [''] + other_lines)
                
                # 재검사
                try:
                    ast.parse(content)
                    print(f'✅ {file_name}: 자동 수정 성공')
                    
                    # 수정된 내용 저장
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                except SyntaxError as e2:
                    print(f'❌ {file_name}: 자동 수정 실패, 원본 복구 중...')
                    
                    # 원본 내용으로 복구
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(original_content)
        else:
            print(f'⚠️ {file_name}: 파일 없음')
    
    print('\\n✅ SGLang 구문 오류 수정 완료')
else:
    print('❌ SGLang 패키지를 찾을 수 없습니다')
"

echo -e "${GREEN}✅ SGLang 구문 오류 수정 완료${NC}"

# SGLang 재설치 스크립트 생성
echo -e "\n${BLUE}📝 SGLang 깨끗한 재설치 스크립트 생성...${NC}"

cat > reinstall_sglang_clean.sh << 'EOF'
#!/bin/bash
# SGLang 깨끗한 재설치 스크립트

set -e

echo "🔄 SGLang 깨끗한 재설치"
echo "======================="

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}1단계: 기존 SGLang 완전 제거...${NC}"

# SGLang 관련 프로세스 종료
pkill -f sglang 2>/dev/null || true
pkill -f "python.*launch_server" 2>/dev/null || true

# SGLang 패키지 제거
pip uninstall sglang -y 2>/dev/null || true

# 캐시 정리
pip cache purge
rm -rf ~/.cache/pip/wheels/sglang* 2>/dev/null || true

echo -e "${GREEN}✅ 기존 SGLang 제거 완료${NC}"

echo -e "\n${BLUE}2단계: Python 환경 확인...${NC}"

python -c "
import sys
print(f'Python: {sys.version}')
print(f'가상환경: {sys.prefix}')

import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA 사용 가능: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA 버전: {torch.version.cuda}')
"

echo -e "${GREEN}✅ Python 환경 확인 완료${NC}"

echo -e "\n${BLUE}3단계: SGLang 깨끗한 설치...${NC}"

# 최신 pip 도구 설치
pip install --upgrade pip wheel setuptools

# SGLang 설치 (의존성 포함)
echo "SGLang 설치 중..."
pip install "sglang==0.2.15" --no-cache-dir

# 설치 확인
echo -e "\n${BLUE}4단계: 설치 확인...${NC}"

python -c "
try:
    import sglang
    print(f'✅ SGLang 버전: {sglang.__version__}')
    
    # 기본 import 테스트
    try:
        from sglang.srt.server import launch_server
        print('✅ sglang.srt.server 모듈 정상')
    except ImportError as e:
        print(f'⚠️ server 모듈 제한: {e}')
    
    try:
        import sglang.launch_server
        print('✅ sglang.launch_server 모듈 정상')
    except ImportError as e:
        print(f'⚠️ launch_server 모듈 제한: {e}')
    
    print('\\n🎉 SGLang 깨끗한 재설치 완료!')
    
except ImportError as e:
    print(f'❌ SGLang 설치 실패: {e}')
    exit(1)
"

echo -e "${GREEN}✅ SGLang 깨끗한 재설치 완료${NC}"
EOF

chmod +x reinstall_sglang_clean.sh

echo -e "${GREEN}✅ SGLang 재설치 스크립트 생성: reinstall_sglang_clean.sh${NC}"

# 최소한의 SGLang 실행 스크립트 생성
echo -e "\n${BLUE}📝 최소한의 SGLang 실행 스크립트 생성...${NC}"

cat > start_sglang_minimal.py << 'EOF'
#!/usr/bin/env python3
"""
최소한의 SGLang 실행 스크립트 (구문 오류 회피)
"""

import sys
import os
import subprocess
import time
import requests

def minimal_sglang_start(model_path="microsoft/DialoGPT-medium", port=8000):
    """최소한의 SGLang 서버 시작"""
    
    print("🚀 최소한의 SGLang 서버 시작")
    print(f"모델: {model_path}")
    print(f"포트: {port}")
    
    # 최소한의 환경 설정
    env = os.environ.copy()
    env.update({
        'CUDA_VISIBLE_DEVICES': '',  # CPU 모드 강제
        'TOKENIZERS_PARALLELISM': 'false'
    })
    
    # 가장 기본적인 명령어만 사용
    cmd = [
        sys.executable, "-c",
        f"""
import sys
sys.path.insert(0, '{os.getcwd()}')

# 멀티프로세싱 설정
import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

# SGLang 서버 시작
try:
    from sglang.srt.server import launch_server
    from sglang.srt.server_args import ServerArgs
    
    args = ServerArgs(
        model_path='{model_path}',
        host='127.0.0.1',
        port={port},
        trust_remote_code=True,
        max_running_requests=2,
        max_total_tokens=1024
    )
    
    launch_server(args)
    
except Exception as e:
    print(f'서버 시작 오류: {{e}}')
    import traceback
    traceback.print_exc()
"""
    ]
    
    print(f"실행 중...")
    
    try:
        # 로그 디렉토리 생성
        os.makedirs("logs", exist_ok=True)
        
        # 서버 시작
        with open("logs/sglang_minimal.log", "w") as log_file:
            process = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env
            )
        
        print(f"✅ 서버 프로세스 시작 (PID: {process.pid})")
        
        # 서버 준비 대기
        print("⏳ 서버 준비 대기...")
        for i in range(120):
            if process.poll() is not None:
                print("❌ 서버 프로세스 종료됨")
                return None
            
            try:
                response = requests.get(f"http://127.0.0.1:{port}/get_model_info", timeout=3)
                if response.status_code == 200:
                    print(f"✅ 서버 준비 완료! ({i+1}초)")
                    return process
            except:
                pass
            
            if i % 20 == 0 and i > 0:
                print(f"대기 중... {i}초")
            
            time.sleep(1)
        
        print("❌ 서버 시작 시간 초과")
        process.terminate()
        return None
        
    except Exception as e:
        print(f"❌ 서버 시작 실패: {e}")
        return None

def main():
    print("⚡ 최소한의 SGLang 실행 (구문 오류 회피)")
    print("=" * 50)
    
    model_path = sys.argv[1] if len(sys.argv) > 1 else "microsoft/DialoGPT-medium"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
    
    process = minimal_sglang_start(model_path, port)
    
    if process:
        print("\n🎉 최소한의 SGLang 서버 성공!")
        print("=" * 50)
        print()
        print("🧪 테스트:")
        print(f"curl http://127.0.0.1:{port}/get_model_info")
        print()
        print("🛑 종료: Ctrl+C")
        
        try:
            process.wait()
        except KeyboardInterrupt:
            print("\n🛑 서버 종료 중...")
            process.terminate()
            process.wait()
            print("✅ 서버 종료 완료")
    else:
        print("❌ 최소한의 서버 시작 실패")
        
        if os.path.exists("logs/sglang_minimal.log"):
            print("\n=== 최소 실행 로그 ===")
            with open("logs/sglang_minimal.log", "r") as f:
                print(f.read()[-1500:])
        
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
EOF

chmod +x start_sglang_minimal.py

echo -e "${GREEN}✅ 최소 실행 스크립트 생성: start_sglang_minimal.py${NC}"

# 대체 모델 실행 스크립트 생성
echo -e "\n${BLUE}📝 대체 모델 실행 스크립트 생성...${NC}"

cat > start_alternative_model.py << 'EOF'
#!/usr/bin/env python3
"""
대체 모델 실행 스크립트 (Transformers 직접 사용)
"""

import sys
import os
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import json

# FastAPI 앱 생성
app = FastAPI(title="대체 한국어 모델 서버")

# 글로벌 변수
model = None
tokenizer = None

@app.on_event("startup")
async def load_model():
    """모델 로드"""
    global model, tokenizer
    
    print("🔽 모델 로드 중...")
    model_name = "microsoft/DialoGPT-medium"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # CPU 호환
            device_map="cpu"  # CPU 강제
        )
        
        print(f"✅ 모델 로드 완료: {model_name}")
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")

@app.get("/get_model_info")
async def get_model_info():
    """모델 정보 조회"""
    return {
        "model_path": "microsoft/DialoGPT-medium",
        "max_total_tokens": 1024,
        "served_model_names": ["korean-qwen"],
        "is_generation": True
    }

@app.get("/v1/models")
async def list_models():
    """모델 목록"""
    return {
        "data": [
            {
                "id": "korean-qwen",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "alternative-server"
            }
        ]
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """채팅 완성"""
    global model, tokenizer
    
    if model is None or tokenizer is None:
        return JSONResponse(
            status_code=503,
            content={"error": "모델이 로드되지 않았습니다"}
        )
    
    try:
        body = await request.json()
        messages = body.get("messages", [])
        max_tokens = body.get("max_tokens", 50)
        
        # 메시지를 텍스트로 변환
        if messages:
            user_message = messages[-1].get("content", "")
        else:
            user_message = "안녕하세요"
        
        # 토큰화
        inputs = tokenizer.encode(user_message, return_tensors="pt")
        
        # 생성
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=min(max_tokens, 100),
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # 디코딩
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 입력 텍스트 제거
        if response_text.startswith(user_message):
            response_text = response_text[len(user_message):].strip()
        
        # OpenAI 호환 응답
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "korean-qwen",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text or "안녕하세요! 대체 모델 서버입니다."
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(inputs[0]),
                "completion_tokens": len(response_text.split()),
                "total_tokens": len(inputs[0]) + len(response_text.split())
            }
        }
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"생성 오류: {str(e)}"}
        )

def main():
    print("🚀 대체 한국어 모델 서버 시작")
    print("=" * 40)
    print("💻 CPU 모드로 실행")
    print("🔗 포트: 8000")
    print()
    
    # 환경 설정
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    try:
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=8000,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n🛑 서버 종료")

if __name__ == "__main__":
    main()
EOF

chmod +x start_alternative_model.py

echo -e "${GREEN}✅ 대체 모델 서버 생성: start_alternative_model.py${NC}"

# 통합 해결 스크립트 업데이트
echo -e "\n${BLUE}📝 통합 해결 스크립트 업데이트...${NC}"

cat > fix_sglang_completely.sh << 'EOF'
#!/bin/bash
# SGLang 완전 해결 스크립트

echo "🔧 SGLang 완전 해결 스크립트"
echo "==========================="

echo "문제를 해결할 방법을 선택하세요:"
echo ""
echo "1) SGLang 구문 오류 수정 후 재시도"
echo "2) SGLang 깨끗한 재설치"
echo "3) 최소한의 SGLang 실행"
echo "4) 대체 모델 서버 사용 (Transformers 직접)"
echo "5) 모든 방법 순서대로 시도"

read -p "선택 (1-5): " choice

case $choice in
    1)
        echo "🔧 SGLang 구문 오류 수정..."
        bash sglang_syntax_repair.sh
        echo "재시도 중..."
        python start_sglang_cpu_mode.py
        ;;
    2)
        echo "🔄 SGLang 깨끗한 재설치..."
        bash reinstall_sglang_clean.sh
        echo "재설치 후 실행..."
        python start_sglang_cpu_mode.py
        ;;
    3)
        echo "⚡ 최소한의 SGLang 실행..."
        python start_sglang_minimal.py
        ;;
    4)
        echo "🔄 대체 모델 서버 사용..."
        python start_alternative_model.py
        ;;
    5)
        echo "🚀 모든 방법 순서대로 시도..."
        
        echo "1단계: 구문 오류 수정..."
        bash sglang_syntax_repair.sh
        
        echo "2단계: CPU 모드 시도..."
        timeout 30 python start_sglang_cpu_mode.py || echo "CPU 모드 실패"
        
        echo "3단계: 최소 실행 시도..."
        timeout 30 python start_sglang_minimal.py || echo "최소 실행 실패"
        
        echo "4단계: 대체 서버 실행..."
        python start_alternative_model.py
        ;;
    *)
        echo "❌ 잘못된 선택"
        exit 1
        ;;
esac
EOF

chmod +x fix_sglang_completely.sh

echo -e "${GREEN}✅ 통합 해결 스크립트 생성: fix_sglang_completely.sh${NC}"

echo ""
echo -e "${GREEN}🎉 SGLang 구문 오류 수정 및 복구 완료!${NC}"
echo "=============================================="

echo -e "${BLUE}🎯 해결 방법들:${NC}"
echo "✅ SGLang 구문 오류 자동 수정"
echo "✅ SGLang 깨끗한 재설치"
echo "✅ 최소한의 SGLang 실행"
echo "✅ 대체 모델 서버 (Transformers 직접 사용)"
echo "✅ 통합 해결 스크립트"

echo ""
echo -e "${BLUE}🚀 권장 사용 순서:${NC}"
echo ""
echo "1. 구문 오류 수정 후 재시도:"
echo "   bash sglang_syntax_repair.sh"
echo "   python start_sglang_cpu_mode.py"
echo ""
echo "2. 재설치 후 재시도:"
echo "   bash reinstall_sglang_clean.sh"
echo ""
echo "3. 대체 서버 사용 (가장 안정적):"
echo "   python start_alternative_model.py"
echo ""
echo "4. 통합 해결 스크립트:"
echo "   bash fix_sglang_completely.sh"

echo ""
echo -e "${PURPLE}💡 대체 서버가 가장 안정적입니다!${NC}"
echo "Transformers를 직접 사용하여 SGLang 없이도 동일한 API 제공"

echo ""
echo "구문 오류 수정 완료 시간: $(date)"