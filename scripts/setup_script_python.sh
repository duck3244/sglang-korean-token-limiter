#!/bin/bash
# SGLang 완전 새로 설치 스크립트

set -e

echo "🚀 SGLang 완전 새로 설치"
echo "======================="

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

# 1. 완전한 정리
echo -e "${YELLOW}🧹 완전한 환경 정리...${NC}"

# 모든 관련 패키지 강제 제거
echo "기존 패키지 제거 중..."
pip uninstall -y sglang vllm outlines flashinfer xformers flash-attn triton bitsandbytes 2>/dev/null || true

# pip 캐시 완전 정리
pip cache purge

# Python 캐시 정리
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true

echo -e "${GREEN}✅ 환경 정리 완료${NC}"

# 2. Python 환경 확인
echo -e "\n${BLUE}🐍 Python 환경 확인...${NC}"

python -c "
import sys
import os

print(f'Python: {sys.version}')
print(f'실행 경로: {sys.executable}')
print(f'Python 경로: {sys.path[0]}')

# 가상환경 확인
if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
    print('✅ 가상환경 활성화됨')
else:
    print('⚠️ 시스템 Python 사용 중')

# conda 환경 확인
if 'CONDA_DEFAULT_ENV' in os.environ:
    print(f'✅ Conda 환경: {os.environ[\"CONDA_DEFAULT_ENV\"]}')
"

# 3. 기본 도구 업그레이드
echo -e "\n${BLUE}📦 기본 도구 업그레이드...${NC}"

# pip, wheel, setuptools 최신화
pip install --upgrade pip wheel setuptools

# 4. PyTorch 안정 설치
echo -e "\n${BLUE}🔥 PyTorch 안정 설치...${NC}"

# 기존 PyTorch 제거
pip uninstall torch torchvision torchaudio -y 2>/dev/null || true

# PyTorch 2.1.0 설치 (SGLang 최고 호환성)
echo "PyTorch 2.1.0 설치 중..."
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# PyTorch 설치 확인
python -c "
import torch
print(f'✅ PyTorch: {torch.__version__}')
if torch.cuda.is_available():
    print(f'✅ CUDA: {torch.version.cuda}')
    print(f'✅ GPU: {torch.cuda.get_device_name()}')
else:
    print('💻 CPU 모드')
"

# 5. 필수 의존성 설치
echo -e "\n${BLUE}📦 필수 의존성 설치...${NC}"

# Transformers 생태계
pip install transformers==4.36.0
pip install tokenizers==0.15.0
pip install accelerate==0.25.0
pip install safetensors==0.4.1
pip install sentencepiece==0.1.99

# 웹 서버
pip install fastapi==0.104.1
pip install uvicorn==0.24.0
pip install httpx==0.25.2
pip install pydantic==2.5.0

# 기타 필수
pip install numpy==1.24.4 --force-reinstall
pip install requests==2.32.4
pip install psutil==5.9.6

# 6. SGLang 설치 (여러 방법 시도)
echo -e "\n${PURPLE}🚀 SGLang 설치 (여러 방법 시도)...${NC}"

# 방법 1: PyPI 최신 버전
echo "=== 방법 1: PyPI 최신 버전 ==="
if pip install sglang; then
    echo -e "${GREEN}✅ PyPI 설치 성공${NC}"
    INSTALL_METHOD="pypi"
else
    echo -e "${YELLOW}⚠️ PyPI 설치 실패${NC}"

    # 방법 2: PyPI all 버전
    echo "=== 방법 2: PyPI [all] 버전 ==="
    if pip install "sglang[all]"; then
        echo -e "${GREEN}✅ PyPI [all] 설치 성공${NC}"
        INSTALL_METHOD="pypi_all"
    else
        echo -e "${YELLOW}⚠️ PyPI [all] 설치 실패${NC}"

        # 방법 3: Git 개발 버전
        echo "=== 방법 3: Git 개발 버전 ==="
        if pip install "git+https://github.com/sgl-project/sglang.git"; then
            echo -e "${GREEN}✅ Git 설치 성공${NC}"
            INSTALL_METHOD="git"
        else
            echo -e "${RED}❌ 모든 설치 방법 실패${NC}"
            INSTALL_METHOD="failed"
        fi
    fi
fi

# 7. 설치 확인
echo -e "\n${BLUE}🔍 SGLang 설치 확인...${NC}"

python -c "
import sys
import os

print('=== SGLang 설치 확인 ===')

try:
    import sglang

    # 기본 정보
    print(f'✅ SGLang import 성공')
    print(f'파일 위치: {sglang.__file__}')
    print(f'버전: {getattr(sglang, \"__version__\", \"버전 정보 없음\")}')

    # 디렉토리 구조 확인
    if sglang.__file__:
        sglang_dir = os.path.dirname(sglang.__file__)
        print(f'SGLang 디렉토리: {sglang_dir}')

        # 주요 파일들 확인
        important_files = [
            'srt/__init__.py',
            'srt/server.py',
            'launch_server.py',
            'server.py'
        ]

        print('\n📂 주요 파일 확인:')
        for file_path in important_files:
            full_path = os.path.join(sglang_dir, file_path)
            if os.path.exists(full_path):
                print(f'✅ {file_path}')
            else:
                print(f'❌ {file_path}')

        # 실제 파일 구조 출력 (상위 2레벨만)
        print('\n📁 실제 구조:')
        for root, dirs, files in os.walk(sglang_dir):
            level = root.replace(sglang_dir, '').count(os.sep)
            if level > 1:
                dirs.clear()
                continue
            indent = '  ' * level
            print(f'{indent}{os.path.basename(root)}/')
            for file in files:
                if file.endswith('.py'):
                    print(f'{indent}  {file}')

    # 핵심 기능 테스트
    print('\n🧪 핵심 기능 테스트:')

    try:
        from sglang import function, system, user, assistant, gen
        print('✅ sglang 기본 함수들')
    except ImportError as e:
        print(f'❌ 기본 함수 import 실패: {e}')

    try:
        from sglang.srt.server import launch_server
        print('✅ sglang.srt.server.launch_server')
        SERVER_AVAILABLE = True
        SERVER_PATH = 'sglang.srt.server'
    except ImportError:
        try:
            import sglang.launch_server
            print('✅ sglang.launch_server')
            SERVER_AVAILABLE = True
            SERVER_PATH = 'sglang.launch_server'
        except ImportError:
            print('❌ 서버 모듈 없음')
            SERVER_AVAILABLE = False
            SERVER_PATH = None

    if SERVER_AVAILABLE:
        print(f'🎯 사용 가능한 서버 경로: {SERVER_PATH}')
        with open('/tmp/sglang_server_path.txt', 'w') as f:
            f.write(SERVER_PATH)

    print('\n🎉 SGLang 설치 및 확인 완료!')

except ImportError as e:
    print(f'❌ SGLang import 실패: {e}')
    print('설치가 제대로 되지 않았습니다.')
    sys.exit(1)
"

# 8. 실행 가능한 스크립트 생성
echo -e "\n${BLUE}📝 실행 스크립트 생성...${NC}"

if [ -f "/tmp/sglang_server_path.txt" ]; then
    SERVER_PATH=$(cat /tmp/sglang_server_path.txt)
    echo -e "${GREEN}✅ 서버 경로 확인: $SERVER_PATH${NC}"

    # 실행 스크립트 생성
    cat > run_sglang_server.py << EOF
#!/usr/bin/env python3
"""
SGLang 서버 실행 스크립트 (자동 생성)
"""

import sys
import subprocess
import time
import requests
import argparse

def start_server(model_path="microsoft/DialoGPT-medium", port=8000):
    """SGLang 서버 시작"""

    print(f"🚀 SGLang 서버 시작")
    print(f"모델: {model_path}")
    print(f"포트: {port}")

    # 서버 경로: $SERVER_PATH
    if "$SERVER_PATH" == "sglang.srt.server":
        # sglang.srt.server 사용
        cmd = [
            sys.executable, "-m", "sglang.srt.server",
            "--model-path", model_path,
            "--port", str(port),
            "--host", "127.0.0.1",
            "--trust-remote-code"
        ]
    elif "$SERVER_PATH" == "sglang.launch_server":
        # sglang.launch_server 사용
        cmd = [
            sys.executable, "-m", "sglang.launch_server",
            "--model-path", model_path,
            "--port", str(port),
            "--host", "127.0.0.1",
            "--trust-remote-code"
        ]
    else:
        print("❌ 알 수 없는 서버 경로")
        return None

    print(f"실행 명령어: {' '.join(cmd)}")

    try:
        # 백그라운드에서 서버 시작
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        print(f"✅ 서버 프로세스 시작됨 (PID: {process.pid})")

        # 서버 준비 대기
        print("⏳ 서버 준비 대기...")
        for i in range(60):
            try:
                response = requests.get(f"http://127.0.0.1:{port}/get_model_info", timeout=2)
                if response.status_code == 200:
                    print(f"✅ 서버 준비 완료! ({i+1}초)")

                    # 모델 정보 출력
                    try:
                        info = response.json()
                        print(f"모델 경로: {info.get('model_path', 'Unknown')}")
                    except:
                        pass

                    return process
            except:
                pass

            # 프로세스 종료 확인
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                print("❌ 서버 프로세스 종료됨")
                print("STDERR:", stderr[:500])
                return None

            time.sleep(1)

        print("❌ 서버 시작 시간 초과")
        process.terminate()
        return None

    except Exception as e:
        print(f"❌ 서버 시작 실패: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="SGLang 서버 실행")
    parser.add_argument("--model", default="microsoft/DialoGPT-medium", help="모델 경로")
    parser.add_argument("--port", type=int, default=8000, help="포트 번호")

    args = parser.parse_args()

    process = start_server(args.model, args.port)

    if process:
        print("🎉 서버 실행 성공!")
        print()
        print("테스트 명령어:")
        print(f"curl http://127.0.0.1:{args.port}/get_model_info")
        print()
        print("종료하려면 Ctrl+C를 누르세요...")

        try:
            process.wait()
        except KeyboardInterrupt:
            print("\\n🛑 서버 종료 중...")
            process.terminate()
            process.wait()
            print("✅ 서버 종료 완료")
    else:
        print("❌ 서버 실행 실패")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
EOF

    chmod +x run_sglang_server.py
    echo -e "${GREEN}✅ 실행 스크립트 생성 완료: run_sglang_server.py${NC}"
else
    echo -e "${RED}❌ 서버 경로를 찾을 수 없어 실행 스크립트 생성 실패${NC}"
fi

# 9. 최종 테스트
echo -e "\n${BLUE}🧪 최종 테스트...${NC}"

echo "1. SGLang import 테스트:"
if python -c "import sglang; print(f'✅ 성공: {sglang.__file__}')"; then
    echo -e "${GREEN}✅ SGLang import 성공${NC}"
else
    echo -e "${RED}❌ SGLang import 실패${NC}"
    exit 1
fi

echo ""
echo "2. 기본 기능 테스트:"
python -c "
try:
    from sglang import function, system, user, assistant
    print('✅ 기본 함수 import 성공')
except Exception as e:
    print(f'⚠️ 기본 함수 import 실패: {e}')

try:
    from sglang.srt.server import launch_server
    print('✅ 서버 함수 import 성공')
except Exception as e:
    print(f'⚠️ 서버 함수 import 실패: {e}')
"

# 10. 완료 안내
echo ""
echo -e "${GREEN}🎉 SGLang 완전 새로 설치 완료!${NC}"
echo "==============================="

echo -e "${BLUE}📋 설치 정보:${NC}"
echo "- 설치 방법: $INSTALL_METHOD"
if [ -f "/tmp/sglang_server_path.txt" ]; then
    echo "- 서버 경로: $(cat /tmp/sglang_server_path.txt)"
fi
echo "- PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "- CUDA: $(python -c 'import torch; print(torch.version.cuda if torch.cuda.is_available() else "없음")')"

echo ""
echo -e "${BLUE}🚀 사용 방법:${NC}"
echo ""
echo "1. SGLang 서버 시작:"
if [ -f "run_sglang_server.py" ]; then
    echo "   python run_sglang_server.py --model microsoft/DialoGPT-medium --port 8000"
else
    echo "   python -m sglang.srt.server --model-path microsoft/DialoGPT-medium --port 8000 --host 127.0.0.1 --trust-remote-code"
fi

echo ""
echo "2. 백그라운드 실행:"
if [ -f "run_sglang_server.py" ]; then
    echo "   nohup python run_sglang_server.py --model microsoft/DialoGPT-medium --port 8000 > sglang.log 2>&1 &"
fi

echo ""
echo "3. Token Limiter 시작 (다른 터미널):"
echo "   python main_sglang.py"

echo ""
echo "4. 테스트:"
echo "   curl http://127.0.0.1:8000/get_model_info"

echo ""
echo -e "${BLUE}💡 다음 단계:${NC}"
echo "1. 먼저 SGLang 서버가 성공적으로 시작되는지 확인"
echo "2. 서버가 정상 작동하면 Token Limiter 연결"
echo "3. 전체 시스템 테스트"

echo ""
echo "설치 완료 시간: $(date)"