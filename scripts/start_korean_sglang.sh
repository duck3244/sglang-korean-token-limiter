#!/bin/bash
# SGLang 기반 한국어 Token Limiter 시스템 시작 스크립트

set -e

echo "🇰🇷 SGLang 기반 한국어 Token Limiter 시스템 시작"
echo "=============================================="

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
SGLANG_MODEL="Qwen/Qwen2.5-3B-Instruct"
SGLANG_PORT=8000
TOKEN_LIMITER_PORT=8080
MAX_MEMORY_FRACTION=0.75
MAX_RUNNING_REQUESTS=16

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
            MAX_MEMORY_FRACTION=0.7
            MAX_RUNNING_REQUESTS=12
        fi
        
        return 0
    else
        echo -e "${YELLOW}⚠️ GPU 없음. CPU 모드로 진행${NC}"
        return 1
    fi
}

# Python 환경 확인
check_python_env() {
    echo -e "${BLUE}🐍 Python 환경 확인...${NC}"
    
    if [[ "$VIRTUAL_ENV" == "" ]] && [[ "$CONDA_DEFAULT_ENV" == "" ]]; then
        echo -e "${YELLOW}⚠️ 가상환경이 활성화되지 않았습니다${NC}"
        echo "다음 명령어로 환경을 활성화하세요:"
        echo "  conda activate korean_sglang"
        echo "  또는"
        echo "  source venv/bin/activate"
        read -p "계속 진행하시겠습니까? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    # SGLang 설치 확인
    if ! python -c "import sglang" 2>/dev/null; then
        echo -e "${RED}❌ SGLang이 설치되지 않았습니다${NC}"
        echo "설치 명령어: pip install 'sglang[all]==0.2.6'"
        exit 1
    fi
    
    SGLANG_VERSION=$(python -c "import sglang; print(sglang.__version__)" 2>/dev/null || echo "unknown")
    echo "✅ SGLang 버전: $SGLANG_VERSION"
}

# Redis 시작
start_redis() {
    echo -e "${BLUE}🔴 Redis 시작...${NC}"

    if redis-cli ping >/dev/null 2>&1; then
        echo -e "${GREEN}✅ Redis 실행 중${NC}"
        return 0
    fi

    if command -v docker >/dev/null 2>&1; then
        docker rm korean-redis 2>/dev/null || true
        docker run -d --name korean-redis -p 6379:6379 redis:alpine

        # 연결 대기
        for i in {1..20}; do
            if redis-cli ping >/dev/null 2>&1; then
                echo -e "${GREEN}✅ Redis 연결 완료${NC}"
                return 0
            fi
            sleep 1
        done
    fi

    echo -e "${RED}❌ Redis 시작 실패. SQLite 모드로 전환${NC}"
    return 1
}

# 한국어 모델 다운로드 확인
check_korean_model() {
    echo -e "${BLUE}🇰🇷 한국어 모델 확인...${NC}"
    
    # HuggingFace 캐시 확인
    CACHE_DIR="$HOME/.cache/huggingface/hub"
    MODEL_CACHE_DIR=$(echo "$SGLANG_MODEL" | sed 's/\//_/g')
    
    if [[ -d "$CACHE_DIR" ]] && find "$CACHE_DIR" -name "*$MODEL_CACHE_DIR*" -type d | grep -q .; then
        echo "✅ 한국어 모델 캐시 발견"
        return 0
    fi
    
    echo -e "${YELLOW}⚠️ 한국어 모델이 로컬에 없습니다${NC}"
    echo "모델: $SGLANG_MODEL"
    
    read -p "지금 다운로드하시겠습니까? (Y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Nn]$ ]]; then
        echo "모델 다운로드를 건너뜁니다"
        return 0
    fi
    
    # 모델 다운로드
    echo "모델 다운로드 중..."
    python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print('🔽 토크나이저 다운로드 중...')
tokenizer = AutoTokenizer.from_pretrained('$SGLANG_MODEL', trust_remote_code=True)
print(f'✅ 토크나이저 다운로드 완료 (어휘 크기: {len(tokenizer):,})')

print('🔽 모델 다운로드 중... (시간이 오래 걸릴 수 있습니다)')
model = AutoModelForCausalLM.from_pretrained(
    '$SGLANG_MODEL',
    torch_dtype=torch.float16,
    trust_remote_code=True,
    device_map='auto' if torch.cuda.is_available() else 'cpu'
)
print('✅ 모델 다운로드 완료')
"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ 한국어 모델 다운로드 완료${NC}"
    else
        echo -e "${RED}❌ 모델 다운로드 실패${NC}"
        exit 1
    fi
}

# SGLang 서버 시작
start_sglang_server() {
    echo -e "${BLUE}🚀 SGLang 서버 시작...${NC}"

    # GPU 메모리 정리
    if command -v nvidia-smi >/dev/null 2>&1; then
        python -c "
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print('GPU 메모리 정리 완료')
"
    fi

    # SGLang 서버 설정
    SGLANG_ARGS=(
        "--model-path" "$SGLANG_MODEL"
        "--port" "$SGLANG_PORT"
        "--host" "127.0.0.1"
        "--tp-size" "1"
        "--mem-fraction-static" "$MAX_MEMORY_FRACTION"
        "--max-running-requests" "$MAX_RUNNING_REQUESTS"
        "--max-total-tokens" "8192"
        "--served-model-name" "korean-qwen"
        "--trust-remote-code"
    )

    # GPU 사용 가능한 경우 최적화 옵션 추가
    if check_gpu; then
        SGLANG_ARGS+=(
            "--enable-torch-compile"
            "--chunked-prefill-size" "4096"
            "--enable-mixed-chunk"
        )
        
        # RTX 4060 특화 설정
        if [[ "$GPU_NAME" == *"4060"* ]]; then
            SGLANG_ARGS+=(
                "--kv-cache-dtype" "fp16"
                "--max-batch-size" "16"
            )
        fi
    else
        # CPU 모드 설정
        SGLANG_ARGS+=(
            "--device" "cpu"
            "--dtype" "float32"
        )
    fi

    echo "SGLang 시작 명령어:"
    echo "python -m sglang.launch_server ${SGLANG_ARGS[*]}"
    
    # SGLang 서버 시작
    nohup python -m sglang.launch_server "${SGLANG_ARGS[@]}" \
        > logs/sglang_server.log 2>&1 &

    SGLANG_PID=$!
    echo $SGLANG_PID > pids/sglang.pid
    echo "SGLang PID: $SGLANG_PID"

    # 서버 준비 대기
    echo "SGLang 서버 준비 대기..."
    for i in {1..120}; do
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
            tail -30 logs/sglang_server.log
            return 1
        fi

        if [ $((i % 10)) -eq 0 ]; then
            echo "⏳ 대기 중... (${i}/120초)"
        fi
        sleep 1
    done

    echo -e "${RED}❌ SGLang 서버 시작 시간 초과${NC}"
    echo "로그 확인:"
    tail -50 logs/sglang_server.log
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
    nohup python $MAIN_SCRIPT > logs/token_limiter.log 2>&1 &
    LIMITER_PID=$!
    echo $LIMITER_PID > pids/token_limiter.pid
    echo "Token Limiter PID: $LIMITER_PID"

    # 준비 대기
    echo "Token Limiter 준비 대기..."
    for i in {1..30}; do
        if curl -s http://localhost:$TOKEN_LIMITER_PORT/health >/dev/null 2>&1; then
            echo -e "${GREEN}✅ Token Limiter 준비 완료 (${i}초)${NC}"
            return 0
        fi

        if ! kill -0 $LIMITER_PID 2>/dev/null; then
            echo -e "${RED}❌ Token Limiter 프로세스 종료됨${NC}"
            echo "로그 확인:"
            tail -20 logs/token_limiter.log
            return 1
        fi

        sleep 1
    done

    echo -e "${RED}❌ Token Limiter 시작 시간 초과${NC}"
    return 1
}

# 상태 확인
check_status() {
    echo -e "${BLUE}🔍 시스템 상태 확인${NC}"
    echo "========================="

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

    # Redis 상태
    if redis-cli ping >/dev/null 2>&1; then
        echo -e "Redis:          ${GREEN}✅ 정상${NC}"
    else
        echo -e "Redis:          ${YELLOW}⚠️ SQLite 모드${NC}"
    fi

    echo ""
    echo "=== 접속 정보 ==="
    echo "🔗 SGLang 서버: http://127.0.0.1:$SGLANG_PORT"
    echo "🔗 Token Limiter: http://localhost:$TOKEN_LIMITER_PORT"
    echo "🔗 헬스체크: curl http://localhost:$TOKEN_LIMITER_PORT/health"

    echo ""
    echo "=== SGLang 테스트 명령어 ==="
    echo 'curl -X POST http://localhost:8080/v1/chat/completions \'
    echo '  -H "Content-Type: application/json" \'
    echo '  -H "Authorization: Bearer sk-user1-korean-key-def" \'
    echo '  -d '"'"'{'
    echo '    "model": "korean-qwen",'
    echo '    "messages": [{"role": "user", "content": "안녕하세요! SGLang으로 한국어 대화가 가능한가요?"}],'
    echo '    "max_tokens": 100,'
    echo '    "stream": false'
    echo '  }'"'"
    
    echo ""
    echo "=== 스트리밍 테스트 ==="
    echo 'curl -X POST http://localhost:8080/v1/chat/completions \'
    echo '  -H "Content-Type: application/json" \'
    echo '  -H "Authorization: Bearer sk-user1-korean-key-def" \'
    echo '  -d '"'"'{'
    echo '    "model": "korean-qwen",'
    echo '    "messages": [{"role": "user", "content": "한국어로 자기소개를 해주세요."}],'
    echo '    "max_tokens": 150,'
    echo '    "stream": true'
    echo '  }'"'"
}

# 성능 벤치마크
run_performance_test() {
    echo -e "${BLUE}⚡ 성능 테스트 실행...${NC}"
    
    # 간단한 지연시간 테스트
    echo "지연시간 테스트 (5회):"
    for i in {1..5}; do
        start_time=$(date +%s.%N)
        curl -s -X POST http://localhost:$TOKEN_LIMITER_PORT/v1/chat/completions \
            -H "Content-Type: application/json" \
            -H "Authorization: Bearer sk-user1-korean-key-def" \
            -d '{
                "model": "korean-qwen",
                "messages": [{"role": "user", "content": "안녕"}],
                "max_tokens": 10
            }' >/dev/null
        end_time=$(date +%s.%N)
        
        duration=$(echo "$end_time - $start_time" | bc 2>/dev/null || echo "N/A")
        echo "  테스트 $i: ${duration}초"
        sleep 1
    done
    
    echo ""
    echo "동시 요청 테스트 (3개):"
    for i in {1..3}; do
        curl -s -X POST http://localhost:$TOKEN_LIMITER_PORT/v1/chat/completions \
            -H "Content-Type: application/json" \
            -H "Authorization: Bearer sk-user1-korean-key-def" \
            -d "{
                \"model\": \"korean-qwen\",
                \"messages\": [{\"role\": \"user\", \"content\": \"동시 테스트 $i\"}],
                \"max_tokens\": 20
            }" &
    done
    wait
    echo "✅ 동시 요청 테스트 완료"
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
    echo "SGLang 모델: $SGLANG_MODEL"

    # 단계별 실행
    cleanup_processes
    check_python_env
    check_gpu
    check_korean_model
    start_redis
    start_sglang_server
    start_token_limiter
    check_status

    echo ""
    echo -e "${GREEN}🎉 SGLang 기반 한국어 시스템 시작 완료!${NC}"
    echo "========================================"
    echo "종료하려면 Ctrl+C를 누르세요."
    
    # 성능 테스트 옵션
    echo ""
    read -p "성능 테스트를 실행하시겠습니까? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        run_performance_test
    fi

    echo ""
    echo -e "${BLUE}📊 모니터링 대시보드:${NC}"
    echo "  streamlit run dashboard/sglang_app.py --server.port 8501"
    echo ""
    echo -e "${BLUE}📋 로그 모니터링:${NC}"
    echo "  tail -f logs/sglang_server.log"
    echo "  tail -f logs/token_limiter.log"
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

# 도움말
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "SGLang 기반 한국어 Token Limiter 시작 스크립트"
    echo ""
    echo "사용법:"
    echo "  $0                    # 전체 시스템 시작"
    echo "  $0 --model MODEL      # 특정 모델로 시작"
    echo "  $0 --quick            # 빠른 시작 (확인 생략)"
    echo "  $0 --help             # 이 도움말 표시"
    echo ""
    echo "옵션:"
    echo "  --model MODEL         SGLang 모델 지정 (기본: Qwen/Qwen2.5-3B-Instruct)"
    echo "  --quick               모델 다운로드 확인 건너뛰기"
    echo "  --memory-fraction F   GPU 메모리 사용률 (기본: 0.75)"
    echo "  --max-requests N      최대 동시 요청 수 (기본: 16)"
    echo "  --port PORT           SGLang 포트 (기본: 8000)"
    echo ""
    echo "환경 변수:"
    echo "  SGLANG_MODEL          사용할 모델 (기본: Qwen/Qwen2.5-3B-Instruct)"
    echo "  SGLANG_PORT           SGLang 서버 포트 (기본: 8000)"
    echo "  LIMITER_PORT          Token Limiter 포트 (기본: 8080)"
    echo ""
    echo "예시:"
    echo "  $0 --model beomi/Llama-3-Open-Ko-8B"
    echo "  $0 --memory-fraction 0.6 --max-requests 12"
    echo "  SGLANG_MODEL=upstage/SOLAR-10.7B-Instruct-v1.0 $0"
    echo ""
    exit 0
fi

# 명령행 인자 처리
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            SGLANG_MODEL="$2"
            shift 2
            ;;
        --quick)
            SKIP_MODEL_CHECK=true
            shift
            ;;
        --memory-fraction)
            MAX_MEMORY_FRACTION="$2"
            shift 2
            ;;
        --max-requests)
            MAX_RUNNING_REQUESTS="$2"
            shift 2
            ;;
        --port)
            SGLANG_PORT="$2"
            shift 2
            ;;
        *)
            echo "알 수 없는 옵션: $1"
            echo "도움말: $0 --help"
            exit 1
            ;;
    esac
done

# 환경 변수 적용
SGLANG_MODEL=${SGLANG_MODEL:-"Qwen/Qwen2.5-3B-Instruct"}
SGLANG_PORT=${SGLANG_PORT:-8000}
TOKEN_LIMITER_PORT=${LIMITER_PORT:-8080}

# 모델 확인 함수 수정
check_korean_model() {
    if [ "$SKIP_MODEL_CHECK" = true ]; then
        echo -e "${YELLOW}⚠️ 모델 확인 건너뛰기${NC}"
        return 0
    fi
    
    echo -e "${BLUE}🇰🇷 한국어 모델 확인...${NC}"
    echo "모델: $SGLANG_MODEL"
    
    # HuggingFace 캐시 확인
    CACHE_DIR="$HOME/.cache/huggingface/hub"
    MODEL_CACHE_DIR=$(echo "$SGLANG_MODEL" | sed 's/\//_/g')
    
    if [[ -d "$CACHE_DIR" ]] && find "$CACHE_DIR" -name "*$MODEL_CACHE_DIR*" -type d | grep -q .; then
        echo "✅ 한국어 모델 캐시 발견"
        return 0
    fi
    
    echo -e "${YELLOW}⚠️ 한국어 모델이 로컬에 없습니다${NC}"
    
    read -p "지금 다운로드하시겠습니까? (Y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Nn]$ ]]; then
        echo "모델 다운로드를 건너뜁니다"
        return 0
    fi
    
    # 모델 다운로드
    echo "모델 다운로드 중..."
    python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print('🔽 토크나이저 다운로드 중...')
tokenizer = AutoTokenizer.from_pretrained('$SGLANG_MODEL', trust_remote_code=True)
print(f'✅ 토크나이저 다운로드 완료 (어휘 크기: {len(tokenizer):,})')

print('🔽 모델 다운로드 중... (시간이 오래 걸릴 수 있습니다)')
try:
    model = AutoModelForCausalLM.from_pretrained(
        '$SGLANG_MODEL',
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
        device_map='auto' if torch.cuda.is_available() else 'cpu'
    )
    print('✅ 모델 다운로드 완료')
except Exception as e:
    print(f'⚠️ 모델 다운로드 중 오류: {e}')
    print('SGLang이 시작할 때 자동으로 다운로드됩니다.')
"
    
    echo -e "${GREEN}✅ 한국어 모델 준비 완료${NC}"
}

# 실행
main "$@"