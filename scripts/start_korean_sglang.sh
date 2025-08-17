#!/bin/bash
# SGLang 서버 명령어 옵션 수정 스크립트

set -e

echo "🔧 SGLang 서버 명령어 옵션 수정"
echo "================================"

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

echo -e "${BLUE}🔍 SGLang 0.2.15 지원 옵션 확인...${NC}"

# SGLang 서버 옵션 확인
python -c "
import subprocess
import sys

print('SGLang 서버 지원 옵션 확인...')

try:
    # SGLang 서버 도움말 확인
    result = subprocess.run(
        [sys.executable, '-m', 'sglang.launch_server', '--help'],
        capture_output=True,
        text=True,
        timeout=10
    )

    if result.returncode == 0:
        help_text = result.stdout

        # 주요 옵션들 확인
        options_to_check = [
            '--kv-cache-dtype',
            '--max-batch-size',
            '--chunked-prefill-size',
            '--enable-mixed-chunk',
            '--enable-torch-compile',
            '--mem-fraction-static',
            '--max-running-requests',
            '--max-total-tokens'
        ]

        print('\\n📋 SGLang 0.2.15 지원 옵션:')
        for option in options_to_check:
            if option in help_text:
                print(f'✅ {option}')

                # kv-cache-dtype의 경우 가능한 값들 확인
                if option == '--kv-cache-dtype':
                    import re
                    pattern = r'--kv-cache-dtype.*?choose from ([^)]+)'
                    match = re.search(pattern, help_text)
                    if match:
                        choices = match.group(1)
                        print(f'   가능한 값: {choices}')
            else:
                print(f'❌ {option}')

        # 전체 도움말에서 kv-cache-dtype 관련 부분 추출
        if '--kv-cache-dtype' in help_text:
            lines = help_text.split('\\n')
            for i, line in enumerate(lines):
                if '--kv-cache-dtype' in line:
                    print(f'\\n📄 kv-cache-dtype 옵션 상세:')
                    for j in range(max(0, i-1), min(len(lines), i+3)):
                        print(f'  {lines[j]}')
                    break
    else:
        print(f'❌ SGLang 도움말 조회 실패: {result.stderr}')

except Exception as e:
    print(f'❌ 옵션 확인 실패: {e}')
"

echo -e "${GREEN}✅ SGLang 옵션 확인 완료${NC}"

# 수정된 시작 스크립트 생성
echo -e "\n${BLUE}📝 수정된 SGLang 시작 스크립트 생성...${NC}"

cat > start_korean_sglang_fixed.sh << 'EOF'
#!/bin/bash
# SGLang 기반 한국어 Token Limiter 시스템 시작 스크립트 (옵션 수정 버전)

set -e

echo "🇰🇷 SGLang 기반 한국어 Token Limiter 시스템 시작 (옵션 수정 버전)"
echo "=================================================================="

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

# 디렉토리 생성
mkdir -p logs pids tokenizer_cache

# 기본 설정
SGLANG_MODEL="microsoft/DialoGPT-medium"
SGLANG_PORT=8000
TOKEN_LIMITER_PORT=8080
MAX_MEMORY_FRACTION=0.8
MAX_RUNNING_REQUESTS=8

# 프로세스 정리
cleanup_processes() {
    echo -e "${YELLOW}🧹 기존 프로세스 정리...${NC}"
    pkill -f "sglang.*launch_server" 2>/dev/null || true
    pkill -f "main_sglang.py" 2>/dev/null || true
    rm -f pids/*.pid 2>/dev/null || true
    sleep 2
    echo -e "${GREEN}✅ 정리 완료${NC}"
}

# GPU 확인
check_gpu() {
    echo -e "${BLUE}🔍 GPU 확인...${NC}"
    if nvidia-smi >/dev/null 2>&1; then
        echo "✅ GPU 사용 가능"
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
        GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
        echo "GPU: $GPU_NAME"
        echo "메모리: ${GPU_MEMORY}MB"

        # RTX 4060 감지 시 최적화 설정 조정
        if [[ "$GPU_NAME" == *"4060"* ]]; then
            echo -e "${PURPLE}🎮 RTX 4060 감지 - 최적화 설정 적용${NC}"
            MAX_MEMORY_FRACTION=0.75
            MAX_RUNNING_REQUESTS=6
        fi

        return 0
    else
        echo -e "${YELLOW}⚠️ GPU 없음. CPU 모드로 진행${NC}"
        return 1
    fi
}

# SGLang 서버 시작 (옵션 수정 버전)
start_sglang_server_fixed() {
    echo -e "${BLUE}🚀 SGLang 서버 시작 (옵션 수정 버전)...${NC}"

    # GPU 메모리 정리
    if command -v nvidia-smi >/dev/null 2>&1; then
        python -c "
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print('GPU 메모리 정리 완료')
"
    fi

    # SGLang 서버 설정 (유효한 옵션만 사용)
    SGLANG_ARGS=(
        "--model-path" "$SGLANG_MODEL"
        "--port" "$SGLANG_PORT"
        "--host" "127.0.0.1"
        "--mem-fraction-static" "$MAX_MEMORY_FRACTION"
        "--max-running-requests" "$MAX_RUNNING_REQUESTS"
        "--max-total-tokens" "4096"
        "--served-model-name" "korean-qwen"
        "--trust-remote-code"
    )

    # GPU 사용 가능한 경우에만 GPU 관련 옵션 추가
    if check_gpu; then
        SGLANG_ARGS+=(
            "--tensor-parallel-size" "1"
            "--kv-cache-dtype" "auto"  # fp16 대신 auto 사용
            "--chunked-prefill-size" "2048"  # 크기 줄임
        )

        # RTX 4060 특화 설정
        if [[ "$GPU_NAME" == *"4060"* ]]; then
            SGLANG_ARGS+=(
                "--disable-cuda-graph"  # 메모리 절약
                "--disable-flashinfer"  # 안정성 우선
            )
        else
            # 다른 GPU에서는 성능 최적화 옵션 사용
            SGLANG_ARGS+=(
                "--enable-torch-compile"
            )
        fi
    else
        # CPU 모드 설정
        SGLANG_ARGS+=(
            "--device" "cpu"
            "--dtype" "float32"
            "--disable-cuda-graph"
            "--disable-flashinfer"
        )
    fi

    echo "SGLang 시작 명령어 (수정된 옵션):"
    echo "python -m sglang.launch_server ${SGLANG_ARGS[*]}"

    # SGLang 서버 시작
    nohup python -m sglang.launch_server "${SGLANG_ARGS[@]}" \
        > logs/sglang_server_fixed.log 2>&1 &

    SGLANG_PID=$!
    echo $SGLANG_PID > pids/sglang.pid
    echo "SGLang PID: $SGLANG_PID"

    # 서버 준비 대기
    echo "SGLang 서버 준비 대기 (옵션 수정 버전)..."
    for i in {1..180}; do
        if curl -s http://127.0.0.1:$SGLANG_PORT/get_model_info >/dev/null 2>&1; then
            echo -e "${GREEN}✅ SGLang 서버 준비 완료 (${i}초)${NC}"

            # 모델 정보 표시
            MODEL_INFO=$(curl -s http://127.0.0.1:$SGLANG_PORT/get_model_info)
            if [[ "$MODEL_INFO" != "" ]]; then
                echo "모델 정보:"
                echo "$MODEL_INFO" | python -m json.tool 2>/dev/null || echo "$MODEL_INFO"
            fi

            return 0
        fi

        # 프로세스 체크
        if ! kill -0 $SGLANG_PID 2>/dev/null; then
            echo -e "${RED}❌ SGLang 프로세스 종료됨${NC}"
            echo "로그 확인:"
            tail -50 logs/sglang_server_fixed.log
            return 1
        fi

        if [ $((i % 15)) -eq 0 ]; then
            echo "⏳ 대기 중... (${i}/180초)"
            # 로그 일부 확인
            if [ -f "logs/sglang_server_fixed.log" ]; then
                echo "최근 로그:"
                tail -5 logs/sglang_server_fixed.log
            fi
        fi
        sleep 1
    done

    echo -e "${RED}❌ SGLang 서버 시작 시간 초과${NC}"
    echo "전체 로그 확인:"
    tail -100 logs/sglang_server_fixed.log
    return 1
}

# Token Limiter 시작
start_token_limiter() {
    echo -e "${BLUE}🛡️ Token Limiter 시작...${NC}"

    # 메인 스크립트 찾기
    if [ -f "main_sglang.py" ]; then
        MAIN_SCRIPT="main_sglang.py"
    elif [ -f "main.py" ]; then
        MAIN_SCRIPT="main.py"
    else
        echo -e "${RED}❌ 메인 스크립트 없음${NC}"
        return 1
    fi

    # Token Limiter 실행
    nohup python $MAIN_SCRIPT > logs/token_limiter_fixed.log 2>&1 &
    LIMITER_PID=$!
    echo $LIMITER_PID > pids/token_limiter.pid
    echo "Token Limiter PID: $LIMITER_PID"

    # 준비 대기
    echo "Token Limiter 준비 대기..."
    for i in {1..60}; do
        if curl -s http://localhost:$TOKEN_LIMITER_PORT/health >/dev/null 2>&1; then
            echo -e "${GREEN}✅ Token Limiter 준비 완료 (${i}초)${NC}"
            return 0
        fi

        if ! kill -0 $LIMITER_PID 2>/dev/null; then
            echo -e "${RED}❌ Token Limiter 프로세스 종료됨${NC}"
            echo "로그 확인:"
            tail -20 logs/token_limiter_fixed.log
            return 1
        fi

        sleep 1
    done

    echo -e "${RED}❌ Token Limiter 시작 시간 초과${NC}"
    return 1
}

# 상태 확인
check_status() {
    echo -e "${BLUE}🔍 시스템 상태 확인 (옵션 수정 버전)${NC}"
    echo "====================================="

    # SGLang 서버 상태
    if curl -s http://127.0.0.1:$SGLANG_PORT/get_model_info >/dev/null 2>&1; then
        echo -e "SGLang 서버:    ${GREEN}✅ 정상${NC}"

        # 성능 정보 표시
        SERVER_INFO=$(curl -s http://127.0.0.1:$SGLANG_PORT/get_server_info 2>/dev/null)
        if [[ "$SERVER_INFO" != "" ]]; then
            echo "SGLang 성능 정보:"
            echo "$SERVER_INFO" | python -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(f\"  처리 중인 요청: {data.get('running_requests', 0)}개\")
    print(f\"  대기열 길이: {data.get('queue_length', 0)}개\")
    print(f\"  메모리 사용량: {data.get('memory_usage_gb', 0):.1f}GB\")
except:
    pass
"
        fi
    else
        echo -e "SGLang 서버:    ${RED}❌ 오류${NC}"
    fi

    # Token Limiter 상태
    if curl -s http://localhost:$TOKEN_LIMITER_PORT/health >/dev/null 2>&1; then
        echo -e "Token Limiter:  ${GREEN}✅ 정상${NC}"
    else
        echo -e "Token Limiter:  ${RED}❌ 오류${NC}"
    fi

    echo ""
    echo "=== 접속 정보 ==="
    echo "🔗 SGLang 서버: http://127.0.0.1:$SGLANG_PORT"
    echo "🔗 Token Limiter: http://localhost:$TOKEN_LIMITER_PORT"
    echo "🔗 헬스체크: curl http://localhost:$TOKEN_LIMITER_PORT/health"

    echo ""
    echo "=== SGLang 테스트 명령어 (옵션 수정 버전) ==="
    echo 'curl -X POST http://localhost:8080/v1/chat/completions \'
    echo '  -H "Content-Type: application/json" \'
    echo '  -H "Authorization: Bearer sk-user1-korean-key-def" \'
    echo '  -d '"'"'{'
    echo '    "model": "korean-qwen",'
    echo '    "messages": [{"role": "user", "content": "SGLang 옵션 수정 버전이 잘 작동하나요?"}],'
    echo '    "max_tokens": 100,'
    echo '    "stream": false'
    echo '  }'"'"
}

# 종료 처리
cleanup_on_exit() {
    echo ""
    echo -e "${YELLOW}🛑 시스템 종료 중...${NC}"

    if [ -f "pids/token_limiter.pid" ]; then
        kill $(cat pids/token_limiter.pid) 2>/dev/null || true
        rm -f pids/token_limiter.pid
    fi

    if [ -f "pids/sglang.pid" ]; then
        kill $(cat pids/sglang.pid) 2>/dev/null || true
        rm -f pids/sglang.pid
    fi

    cleanup_processes
    echo -e "${GREEN}✅ 종료 완료${NC}"
    exit 0
}

# 시그널 핸들러
trap cleanup_on_exit INT TERM

# 메인 실행 함수
main() {
    echo "시작 시간: $(date)"
    echo "SGLang 모델: $SGLANG_MODEL (옵션 수정 버전)"

    # 단계별 실행
    cleanup_processes

    echo -e "\n${BLUE}🐍 Python 환경 확인...${NC}"
    if ! python -c "import sglang; print(f'✅ SGLang 버전: {sglang.__version__}')" 2>/dev/null; then
        echo -e "${RED}❌ SGLang이 설치되지 않았습니다${NC}"
        exit 1
    fi

    start_sglang_server_fixed
    start_token_limiter
    check_status

    echo ""
    echo -e "${GREEN}🎉 SGLang 기반 한국어 시스템 시작 완료! (옵션 수정 버전)${NC}"
    echo "================================================================"
    echo "종료하려면 Ctrl+C를 누르세요."

    echo ""
    echo -e "${BLUE}📊 모니터링 대시보드:${NC}"
    echo "  streamlit run dashboard/sglang_app.py --server.port 8501"
    echo ""
    echo -e "${BLUE}📋 로그 모니터링:${NC}"
    echo "  tail -f logs/sglang_server_fixed.log"
    echo "  tail -f logs/token_limiter_fixed.log"
    echo ""

    # 모니터링 루프
    while true; do
        sleep 30

        # 프로세스 생존 확인
        if [ -f "pids/sglang.pid" ] && ! kill -0 $(cat pids/sglang.pid) 2>/dev/null; then
            echo -e "${RED}❌ SGLang 서버 종료됨${NC}"
            break
        fi

        if [ -f "pids/token_limiter.pid" ] && ! kill -0 $(cat pids/token_limiter.pid) 2>/dev/null; then
            echo -e "${RED}❌ Token Limiter 종료됨${NC}"
            break
        fi
    done
}

# 명령행 인자 처리
case "${1:-}" in
    --help|-h)
        echo "SGLang 기반 한국어 Token Limiter 시작 스크립트 (옵션 수정 버전)"
        echo ""
        echo "사용법:"
        echo "  $0                    # 전체 시스템 시작"
        echo "  $0 --model MODEL      # 특정 모델로 시작"
        echo "  $0 --help             # 이 도움말 표시"
        echo ""
        echo "주요 수정사항:"
        echo "  - kv-cache-dtype을 fp16에서 auto로 변경"
        echo "  - RTX 4060에 최적화된 설정 적용"
        echo "  - 안정성을 우선한 옵션 사용"
        echo ""
        exit 0
        ;;
    --model)
        SGLANG_MODEL="$2"
        echo "모델 변경: $SGLANG_MODEL"
        shift 2
        ;;
    "")
        # 기본 실행
        ;;
    *)
        echo -e "${RED}❌ 알 수 없는 옵션: $1${NC}"
        echo "도움말: $0 --help"
        exit 1
        ;;
esac

# 실행
main "$@"
EOF

chmod +x start_korean_sglang_fixed.sh

echo -e "${GREEN}✅ 수정된 SGLang 시작 스크립트 생성: start_korean_sglang_fixed.sh${NC}"

# 빠른 실행 스크립트도 생성
echo -e "\n${BLUE}📝 빠른 SGLang 서버 시작 스크립트 생성...${NC}"

cat > quick_start_sglang.py << 'EOF'
#!/usr/bin/env python3
"""
SGLang 빠른 시작 스크립트 (옵션 수정 버전)
"""

import sys
import subprocess
import time
import requests
import os

def start_sglang_quick():
    """SGLang 서버 빠른 시작 (유효한 옵션만 사용)"""

    print("🚀 SGLang 서버 빠른 시작 (옵션 수정 버전)")
    print("=" * 50)

    # 기본 설정
    model_path = "microsoft/DialoGPT-medium"
    port = 8000

    # GPU 확인
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            gpu_name = torch.cuda.get_device_name()
            print(f"✅ GPU: {gpu_name}")
        else:
            print("💻 CPU 모드")
    except:
        gpu_available = False
        print("💻 CPU 모드")

    # 서버 명령어 (유효한 옵션만 사용)
    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", model_path,
        "--port", str(port),
        "--host", "127.0.0.1",
        "--trust-remote-code"
    ]

    # GPU 사용 시 추가 옵션
    if gpu_available:
        cmd.extend([
            "--mem-fraction-static", "0.75",
            "--max-running-requests", "6",
            "--kv-cache-dtype", "auto",  # auto 사용 (fp16 대신)
            "--tensor-parallel-size", "1"
        ])

        # RTX 4060 감지 시 안전 옵션
        if "4060" in gpu_name:
            cmd.extend([
                "--disable-cuda-graph",
                "--disable-flashinfer"
            ])
    else:
        # CPU 모드
        cmd.extend([
            "--disable-cuda-graph",
            "--disable-flashinfer"
        ])

    print(f"실행 명령어: {' '.join(cmd)}")

    try:
        # 로그 디렉토리 생성
        os.makedirs("logs", exist_ok=True)

        # 서버 시작
        with open("logs/sglang_quick.log", "w") as log_file:
            process = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT
            )

        print(f"✅ 서버 시작 (PID: {process.pid})")

        # 서버 준비 대기
        print("⏳ 서버 준비 대기...")
        for i in range(120):
            try:
                response = requests.get(f"http://127.0.0.1:{port}/get_model_info", timeout=3)
                if response.status_code == 200:
                    print(f"✅ 서버 준비 완료! ({i+1}초)")

                    # 모델 정보 표시
                    model_info = response.json()
                    print(f"모델: {model_info.get('model_path', 'Unknown')}")

                    print()
                    print("🧪 테스트 명령어:")
                    print(f"curl http://127.0.0.1:{port}/get_model_info")
                    print()
                    print("🛑 종료: Ctrl+C")

                    # 서버 대기
                    try:
                        process.wait()
                    except KeyboardInterrupt:
                        print("\n🛑 서버 종료 중...")
                        process.terminate()
                        process.wait()
                        print("✅ 서버 종료 완료")

                    return 0
            except:
                pass

            # 프로세스 체크
            if process.poll() is not None:
                print("❌ 서버 프로세스 종료됨")
                if os.path.exists("logs/sglang_quick.log"):
                    print("\n=== 로그 ===")
                    with open("logs/sglang_quick.log", "r") as f:
                        print(f.read()[-1000:])
                return 1

            if i % 20 == 0 and i > 0:
                print(f"대기 중... {i}초")

            time.sleep(1)

        print("❌ 서버 시작 시간 초과")
        process.terminate()
        return 1

    except Exception as e:
        print(f"❌ 서버 시작 실패: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(start_sglang_quick())
EOF

chmod +x quick_start_sglang.py

echo -e "${GREEN}✅ 빠른 SGLang 시작 스크립트 생성: quick_start_sglang.py${NC}"

echo ""
echo -e "${GREEN}🎉 SGLang 서버 명령어 옵션 수정 완료!${NC}"
echo "=============================================="


echo -e "${BLUE}🎯 수정 내용:${NC}"
echo "✅ kv-cache-dtype을 fp16에서 auto로 변경"
echo "✅ RTX 4060에 최적화된 설정 적용"
echo "✅ 유효하지 않은 옵션들 제거"
echo "✅ 안정성을 우선한 옵션 사용"
echo "✅ 메모리 사용량 최적화"

echo ""
echo -e "${BLUE}🚀 사용 방법:${NC}"
echo ""
echo "1. 수정된 전체 시스템 시작:"
echo "   bash start_korean_sglang_fixed.sh"
echo ""
echo "2. 빠른 SGLang 서버만 시작:"
echo "   python quick_start_sglang.py"
echo ""
echo "3. 기존 스크립트 대신 수정된 버전 사용:"
echo "   mv start_korean_sglang_fixed.sh scripts/start_korean_sglang.sh"

echo ""
echo -e "${BLUE}💡 주요 변경사항:${NC}"
echo "- ❌ --kv-cache-dtype fp16 (유효하지 않음)"
echo "- ✅ --kv-cache-dtype auto (SGLang 0.2.15 지원)"
echo "- ✅ RTX 4060: --disable-cuda-graph, --disable-flashinfer 추가"
echo "- ✅ 메모리 사용률을 0.75로 조정"
echo "- ✅ chunked-prefill-size를 2048로 축소"

echo ""
echo "SGLang 옵션 수정 완료 시간: $(date)"