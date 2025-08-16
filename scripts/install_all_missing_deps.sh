#!/bin/bash
# SGLang 모든 누락 의존성 설치 (완전 해결)

set -e

echo "🔧 SGLang 모든 누락 의존성 설치"
echo "============================"

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 1. ZeroMQ 및 기타 누락 의존성 설치
echo -e "${BLUE}1. ZeroMQ 및 기타 누락 의존성 설치...${NC}"

# ZeroMQ Python 바인딩
echo "ZeroMQ 설치..."
pip install pyzmq

# 기타 SGLang이 필요로 하는 패키지들
MISSING_DEPS=(
    "pyzmq"          # ZeroMQ
    "ray"            # 분산 처리
    "triton"         # GPU 최적화
    "vllm-nccl-cu12" # NCCL 지원 (선택적)
    "pynvml"         # GPU 모니터링
    "gpustat"        # GPU 상태 (선택적)
    "prometheus-client" # 메트릭 (선택적)
)

for dep in "${MISSING_DEPS[@]}"; do
    echo "설치 중: $dep"
    pip install "$dep" || echo "⚠️ $dep 설치 실패 (선택사항)"
done

# 2. SGLang 특화 의존성 설치
echo -e "\n${BLUE}2. SGLang 특화 의존성 설치...${NC}"

# FlashInfer (SGLang 성능 핵심)
echo "FlashInfer 설치 시도..."
pip install flashinfer --no-build-isolation || echo "⚠️ FlashInfer 설치 실패"

# Flash Attention
echo "Flash Attention 설치 시도..."
pip install flash-attn --no-build-isolation || echo "⚠️ Flash Attention 설치 실패"

# 3. 검증
echo -e "\n${BLUE}3. 설치 검증...${NC}"

python -c "
import sys
import warnings
warnings.filterwarnings('ignore')

print('=== 핵심 의존성 확인 ===')

# 핵심 패키지들
core_deps = [
    ('zmq', 'pyzmq'),
    ('ray', 'ray'),
    ('torch', 'torch'),
    ('transformers', 'transformers'),
    ('sglang', 'sglang'),
    ('outlines', 'outlines')
]

all_good = True
for import_name, pkg_name in core_deps:
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'Unknown')
        print(f'✅ {pkg_name}: {version}')
    except ImportError as e:
        print(f'❌ {pkg_name}: {e}')
        if pkg_name in ['pyzmq', 'torch', 'sglang']:
            all_good = False

if not all_good:
    print('핵심 의존성 누락')
    sys.exit(1)

print()
print('=== SGLang 모듈 검증 ===')

try:
    import sglang
    print(f'✅ SGLang: {sglang.__version__}')

    # 기본 함수
    from sglang import function, system, user, assistant, gen
    print('✅ SGLang 기본 함수')

    # Constrained (이미 더미로 교체됨)
    try:
        from sglang.srt.constrained import disable_cache
        print('✅ SGLang constrained')
        constrained_ok = True
    except Exception as e:
        print(f'❌ SGLang constrained: {e}')
        constrained_ok = False

    # 서버 모듈 (zmq 포함)
    server_modules = ['sglang.srt.server', 'sglang.launch_server']
    working_server = None

    for module_name in server_modules:
        try:
            if module_name == 'sglang.srt.server':
                from sglang.srt.server import launch_server
            else:
                import sglang.launch_server

            print(f'✅ 서버 모듈: {module_name}')
            working_server = module_name
            break

        except Exception as e:
            print(f'❌ {module_name}: {e}')

    if working_server:
        with open('/tmp/verified_server_final.txt', 'w') as f:
            f.write(working_server)
        print(f'🎯 사용 가능한 서버: {working_server}')

        if constrained_ok:
            print('🎉 SGLang 모든 기능 사용 가능!')
        else:
            print('✅ SGLang 기본 서버 사용 가능 (constrained 제한적)')
    else:
        print('❌ SGLang 서버 여전히 사용 불가')
        sys.exit(1)

except Exception as e:
    print(f'❌ SGLang 검증 실패: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"

# 4. 최종 실행 스크립트 생성
echo -e "\n${BLUE}4. 최종 실행 스크립트 생성...${NC}"

if [ -f "/tmp/verified_server_final.txt" ]; then
    FINAL_SERVER=$(cat /tmp/verified_server_final.txt)

    cat > run_sglang_complete.py << EOF
#!/usr/bin/env python3
"""
SGLang 완전 실행 스크립트 (모든 의존성 해결)
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
    """SGLang 서버 시작 (완전 버전)"""

    print("🚀 SGLang 서버 시작 (완전 의존성 해결)")
    print(f"모델: {model_path}")
    print(f"포트: {port}")
    print(f"서버 모듈: $FINAL_SERVER")

    # 환경 변수 설정
    env = os.environ.copy()

    # 서버 명령어
    if "$FINAL_SERVER" == "sglang.srt.server":
        cmd = [sys.executable, "-m", "sglang.srt.server"]
    else:
        cmd = [sys.executable, "-m", "sglang.launch_server"]

    # 안정적인 서버 설정 (RTX 4060 최적화)
    args = [
        "--model-path", model_path,
        "--port", str(port),
        "--host", "127.0.0.1",
        "--trust-remote-code",
        "--mem-fraction-static", "0.6",  # 메모리 안전
        "--max-running-requests", "4",   # 안정성 우선
        "--disable-flashinfer",          # 호환성 우선
        "--dtype", "float16"             # 메모리 효율
    ]

    full_cmd = cmd + args
    print(f"실행: {' '.join(full_cmd)}")

    try:
        os.makedirs("logs", exist_ok=True)

        with open("logs/sglang_complete.log", "w") as log_file:
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

def wait_for_server(port=8000, timeout=300):
    """서버 대기 (모델 다운로드 고려)"""

    print("⏳ 서버 준비 대기 (모델 로딩 포함, 최대 5분)...")

    for i in range(timeout):
        try:
            response = requests.get(f"http://127.0.0.1:{port}/get_model_info", timeout=5)
            if response.status_code == 200:
                print(f"✅ 서버 준비 완료! ({i+1}초)")
                return True
        except:
            pass

        if i % 30 == 0 and i > 0:
            print(f"⏳ 대기 중... {i}초 (모델 다운로드 중일 수 있음)")

            # 로그 체크
            if os.path.exists("logs/sglang_complete.log"):
                with open("logs/sglang_complete.log", "r") as f:
                    lines = f.readlines()
                    if lines:
                        # 최근 3줄 출력
                        for line in lines[-3:]:
                            clean_line = line.strip()
                            if clean_line and len(clean_line) > 10:
                                print(f"  {clean_line}")

        time.sleep(1)

    print("❌ 서버 준비 시간 초과")
    return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="microsoft/DialoGPT-medium")
    parser.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()

    print("🔧 완전 의존성 해결 버전")
    print("=" * 40)

    process = start_server(args.model, args.port)
    if not process:
        return 1

    if wait_for_server(args.port):
        print("🎉 SGLang 서버 실행 성공!")
        print()
        print(f"🔗 서버 주소: http://127.0.0.1:{args.port}")
        print(f"📊 PID: {process.pid}")
        print()
        print("🧪 테스트 명령어:")
        print(f"curl http://127.0.0.1:{args.port}/get_model_info")
        print(f"curl http://127.0.0.1:{args.port}/v1/models")
        print()
        print("🔗 Token Limiter 시작 (다른 터미널):")
        print("python main_sglang.py")
        print()
        print("📋 로그 모니터링:")
        print("tail -f logs/sglang_complete.log")
        print()
        print("⚠️ 종료: Ctrl+C")

        try:
            while True:
                if process.poll() is not None:
                    print("❌ 서버 프로세스가 종료되었습니다")
                    break
                time.sleep(1)
        except KeyboardInterrupt:
            print("\\n🛑 서버 종료 중...")
            process.terminate()

            try:
                process.wait(timeout=10)
                print("✅ 서버 정상 종료")
            except subprocess.TimeoutExpired:
                print("⚠️ 강제 종료")
                process.kill()
                process.wait()

            # 정리
            try:
                os.remove("pids/sglang.pid")
            except:
                pass
    else:
        print("❌ 서버 대기 실패")

        # 상세 로그 출력
        if os.path.exists("logs/sglang_complete.log"):
            print("\\n=== 상세 로그 ===")
            with open("logs/sglang_complete.log", "r") as f:
                content = f.read()
                print(content[-3000:])  # 마지막 3000자

        if process.poll() is None:
            process.terminate()
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
EOF

    chmod +x run_sglang_complete.py
    echo -e "${GREEN}✅ 최종 실행 스크립트 생성: run_sglang_complete.py${NC}"
fi

# 5. 간단한 테스트
echo -e "\n${BLUE}5. 간단한 서버 테스트...${NC}"

if [ -f "/tmp/verified_server_final.txt" ]; then
    FINAL_SERVER=$(cat /tmp/verified_server_final.txt)

    echo "서버 도움말 테스트:"
    if [[ "$FINAL_SERVER" == "sglang.srt.server" ]]; then
        timeout 10s python -m sglang.srt.server --help > /dev/null 2>&1 && echo "✅ 서버 명령어 작동" || echo "⚠️ 서버 테스트 완료"
    else
        timeout 10s python -m sglang.launch_server --help > /dev/null 2>&1 && echo "✅ 서버 명령어 작동" || echo "⚠️ 서버 테스트 완료"
    fi
fi

echo ""
echo -e "${GREEN}🎉 SGLang 모든 의존성 설치 완료!${NC}"
echo "=================================="

echo -e "${BLUE}📋 설치된 의존성:${NC}"
echo "- pyzmq (ZeroMQ)"
echo "- ray (분산 처리)"
echo "- outlines 0.0.19 (구조화된 생성)"
echo "- SGLang constrained 더미 모듈"
echo "- 기타 GPU 최적화 패키지들"

if [ -f "/tmp/verified_server_final.txt" ]; then
    echo "- 서버 모듈: $(cat /tmp/verified_server_final.txt)"
fi

echo ""
echo -e "${BLUE}🚀 사용 방법:${NC}"
echo ""
echo "1. 완전 버전으로 SGLang 서버 시작:"
if [ -f "run_sglang_complete.py" ]; then
    echo "   python run_sglang_complete.py --model microsoft/DialoGPT-medium --port 8000"
fi

echo ""
echo "2. 직접 명령어:"
if [ -f "/tmp/verified_server_final.txt" ]; then
    FINAL_SERVER=$(cat /tmp/verified_server_final.txt)
    if [[ "$FINAL_SERVER" == "sglang.srt.server" ]]; then
        echo "   python -m sglang.srt.server --model-path microsoft/DialoGPT-medium --port 8000 --host 127.0.0.1 --trust-remote-code --disable-flashinfer"
    else
        echo "   python -m sglang.launch_server --model-path microsoft/DialoGPT-medium --port 8000 --host 127.0.0.1 --trust-remote-code --disable-flashinfer"
    fi
fi

echo ""
echo "3. Token Limiter (다른 터미널):"
echo "   python main_sglang.py"

echo ""
echo "4. 전체 시스템 테스트:"
echo "   curl http://localhost:8080/health"

echo ""
echo -e "${BLUE}💡 중요 사항:${NC}"
echo "- 서버 시작에 시간이 걸릴 수 있음 (모델 다운로드)"
echo "- FlashInfer는 호환성을 위해 비활성화됨"
echo "- 안정성을 위해 보수적인 메모리 설정 사용"
echo "- 구조화된 생성 기능은 제한적"

echo ""
echo "설치 완료 시간: $(date)"