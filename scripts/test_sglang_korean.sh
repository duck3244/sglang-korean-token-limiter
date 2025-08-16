#!/bin/bash
# SGLang 기반 한국어 Token Limiter 테스트 스크립트

set -e

echo "🧪 SGLang 기반 한국어 Token Limiter 테스트 시작"
echo "================================================"

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

TOKEN_LIMITER_URL="http://localhost:8080"
SGLANG_URL="http://127.0.0.1:8000"

# 테스트 결과 저장
PASSED=0
FAILED=0
TOTAL=0

# 테스트 함수
run_test() {
    local test_name="$1"
    local test_command="$2"
    local expected_status="$3"

    echo -e "\n${BLUE}🧪 테스트: $test_name${NC}"
    TOTAL=$((TOTAL + 1))

    # 명령어 실행
    response=$(eval "$test_command" 2>/dev/null)
    exit_code=$?

    if [ $exit_code -eq 0 ] && [ "$expected_status" = "success" ]; then
        echo -e "${GREEN}✅ 통과${NC}"
        PASSED=$((PASSED + 1))
        return 0
    elif [ $exit_code -ne 0 ] && [ "$expected_status" = "fail" ]; then
        echo -e "${GREEN}✅ 통과 (예상된 실패)${NC}"
        PASSED=$((PASSED + 1))
        return 0
    else
        echo -e "${RED}❌ 실패${NC}"
        echo "응답: $response"
        FAILED=$((FAILED + 1))
        return 1
    fi
}

# HTTP 요청 헬퍼 함수
make_request() {
    local method="$1"
    local url="$2"
    local headers="$3"
    local data="$4"

    if [ -n "$data" ]; then
        curl -s -w "%{http_code}" -X "$method" "$url" $headers -d "$data"
    else
        curl -s -w "%{http_code}" -X "$method" "$url" $headers
    fi
}

# 시스템 상태 확인
echo -e "${BLUE}🔍 시스템 상태 확인...${NC}"

# Token Limiter 헬스체크
if curl -s "$TOKEN_LIMITER_URL/health" | grep -q "healthy"; then
    echo -e "${GREEN}✅ Token Limiter 정상${NC}"
else
    echo -e "${RED}❌ Token Limiter 오류 - 테스트를 중단합니다${NC}"
    exit 1
fi

# SGLang 서버 헬스체크
if curl -s "$SGLANG_URL/get_model_info" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ SGLang 서버 정상${NC}"
    
    # SGLang 모델 정보 표시
    echo "SGLang 모델 정보:"
    curl -s "$SGLANG_URL/get_model_info" | python -m json.tool 2>/dev/null | head -10
else
    echo -e "${YELLOW}⚠️ SGLang 서버 연결 불가 - 일부 테스트가 실패할 수 있습니다${NC}"
fi

echo ""
echo "=== 기본 기능 테스트 ==="

# 1. 헬스체크 테스트
run_test "시스템 헬스체크" "curl -s $TOKEN_LIMITER_URL/health | grep -q healthy" "success"

# 2. SGLang 런타임 정보 테스트
run_test "SGLang 런타임 정보" "curl -s $TOKEN_LIMITER_URL/sglang/runtime-info | grep -q model_info" "success"

# 3. 토큰 정보 테스트
run_test "한국어 토큰 정보 조회" "curl -s '$TOKEN_LIMITER_URL/token-info?text=안녕하세요' | grep -q token_count" "success"

# 4. 사용자 통계 조회 테스트
run_test "사용자 통계 조회" "curl -s $TOKEN_LIMITER_URL/stats/사용자1 | grep -q user_id" "success"

# 5. 모델 목록 테스트
run_test "모델 목록 조회" "curl -s $TOKEN_LIMITER_URL/models | grep -q korean-qwen" "success"

echo ""
echo "=== SGLang 성능 테스트 ==="

# SGLang 성능 메트릭 조회
run_test "SGLang 성능 메트릭" "curl -s $TOKEN_LIMITER_URL/admin/sglang/performance | grep -q timestamp" "success"

echo ""
echo "=== 한국어 사용자별 테스트 ==="

# 한국어 사용자 및 API 키 배열
korean_users=("사용자1" "사용자2" "개발자1" "테스트" "게스트")
api_keys=("sk-user1-korean-key-def" "sk-user2-korean-key-ghi" "sk-dev1-korean-key-789" "sk-test-korean-key-stu" "sk-guest-korean-key-vwx")

# SGLang 최적화를 반영한 한국어 테스트 메시지들
korean_messages=(
    "SGLang의 동적 배치 처리 기능에 대해 설명해주세요."
    "한국어 토큰화에서 SGLang의 장점은 무엇인가요?"
    "KV 캐시 최적화가 성능에 미치는 영향을 알려주세요."
    "프리픽스 캐싱이 한국어 처리에 유용한 이유는?"
    "SGLang과 vLLM의 차이점을 간단히 설명해주세요."
)

# 사용자별 채팅 완성 요청 테스트
for i in ${!korean_users[@]}; do
    user=${korean_users[$i]}
    api_key=${api_keys[$i]}
    message=${korean_messages[$i]}

    echo -e "\n${PURPLE}🚀 [$user] SGLang 채팅 완성 테스트${NC}"
    echo "메시지: $message"

    # 채팅 완성 요청
    response=$(curl -s -w "%{http_code}" -X POST "$TOKEN_LIMITER_URL/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $api_key" \
        -d "{
            \"model\": \"korean-qwen\",
            \"messages\": [
                {\"role\": \"system\", \"content\": \"당신은 SGLang 기반의 한국어 AI 어시스턴트입니다. 기술적인 질문에 정확하고 간결하게 답변해주세요.\"},
                {\"role\": \"user\", \"content\": \"$message\"}
            ],
            \"max_tokens\": 150,
            \"temperature\": 0.7
        }")

    http_code="${response: -3}"
    response_body="${response%???}"

    if [ "$http_code" = "200" ]; then
        echo -e "${GREEN}✅ [$user] SGLang 요청 성공 (HTTP $http_code)${NC}"
        PASSED=$((PASSED + 1))

        # 응답 내용 일부 표시
        echo "$response_body" | jq -r '.choices[0].message.content // "응답 파싱 실패"' 2>/dev/null | head -3 || echo "응답 파싱 실패"

        # SGLang 성능 정보 표시
        usage=$(echo "$response_body" | jq -r '.usage // {}' 2>/dev/null)
        if [ "$usage" != "{}" ] && [ "$usage" != "null" ]; then
            echo "토큰 사용량: $(echo "$usage" | jq -r '.total_tokens // "N/A"')"
        fi

    elif [ "$http_code" = "429" ]; then
        echo -e "${YELLOW}⚠️ [$user] 속도 제한 감지 (HTTP $http_code)${NC}"
        echo "$response_body" | jq -r '.error.message // "제한 메시지 없음"' 2>/dev/null
        PASSED=$((PASSED + 1))  # 예상된 동작

    else
        echo -e "${RED}❌ [$user] 요청 실패 (HTTP $http_code)${NC}"
        echo "$response_body" | head -2
        FAILED=$((FAILED + 1))
    fi

    TOTAL=$((TOTAL + 1))

    # 사용량 통계 확인
    echo "📊 [$user] 사용량 확인..."
    stats=$(curl -s "$TOKEN_LIMITER_URL/stats/$user")
    if echo "$stats" | grep -q "user_id"; then
        echo "$stats" | jq -r '"토큰(분): \(.tokens_this_minute // 0)/\(.limits.tpm // 0), 요청(분): \(.requests_this_minute // 0)/\(.limits.rpm // 0)"' 2>/dev/null || echo "통계 파싱 실패"
    else
        echo "❌ 통계 조회 실패"
    fi

    sleep 1
done

echo ""
echo "=== SGLang 스트리밍 테스트 ==="

# 스트리밍 응답 테스트
echo -e "${BLUE}📡 SGLang 스트리밍 응답 테스트${NC}"

echo "스트리밍 테스트 명령어:"
echo 'curl -X POST http://localhost:8080/v1/chat/completions \'
echo '  -H "Content-Type: application/json" \'
echo '  -H "Authorization: Bearer sk-user1-korean-key-def" \'
echo '  -d '"'"'{'
echo '    "model": "korean-qwen",'
echo '    "messages": [{"role": "user", "content": "SGLang 스트리밍 테스트입니다. 한국어로 간단한 응답을 해주세요."}],'
echo '    "max_tokens": 100,'
echo '    "stream": true'
echo '  }'"'"

# 스트리밍 응답 확인 (간단한 연결 테스트)
streaming_response=$(curl -s -w "%{http_code}" -X POST "$TOKEN_LIMITER_URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer sk-user1-korean-key-def" \
    -d '{
        "model": "korean-qwen",
        "messages": [{"role": "user", "content": "짧은 인사를 해주세요."}],
        "max_tokens": 30,
        "stream": true
    }' | head -5)

streaming_code="${streaming_response: -3}"
if [ "$streaming_code" = "200" ]; then
    echo -e "${GREEN}✅ 스트리밍 연결 성공${NC}"
    PASSED=$((PASSED + 1))
else
    echo -e "${RED}❌ 스트리밍 연결 실패 (HTTP $streaming_code)${NC}"
    FAILED=$((FAILED + 1))
fi
TOTAL=$((TOTAL + 1))

echo ""
echo "=== 속도 제한 테스트 ==="

# 부하 테스트 (테스트 계정으로 연속 요청)
echo -e "${BLUE}🚀 SGLang 부하 테스트 (테스트 계정으로 연속 요청)...${NC}"

for i in {1..6}; do
    echo "요청 #$i..."
    response=$(curl -s -w "%{http_code}" -X POST "$TOKEN_LIMITER_URL/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer sk-test-korean-key-stu" \
        -d "{
            \"model\": \"korean-qwen\",
            \"messages\": [
                {\"role\": \"user\", \"content\": \"SGLang 테스트 ${i}번째 요청입니다. 짧게 응답해주세요.\"}
            ],
            \"max_tokens\": 20
        }")

    http_code="${response: -3}"

    if [ "$http_code" = "200" ]; then
        echo -e "${GREEN}✅ 응답: HTTP $http_code${NC}"
    elif [ "$http_code" = "429" ]; then
        echo -e "${YELLOW}🎯 속도 제한 성공적으로 작동! (HTTP $http_code)${NC}"
        echo "${response%???}" | jq -r '.error.message // "제한 메시지 없음"' 2>/dev/null
        break
    else
        echo -e "${RED}❌ 예상치 못한 응답: HTTP $http_code${NC}"
    fi

    sleep 0.3  # SGLang 빠른 처리 반영
done

echo ""
echo "=== SGLang 성능 벤치마크 ==="

# 동시 요청 성능 테스트
echo -e "${BLUE}⚡ SGLang 동시 요청 성능 테스트${NC}"

# 백그라운드에서 동시에 여러 요청 실행
pids=()
start_time=$(date +%s.%N)

for i in {1..4}; do  # SGLang 성능을 고려해 4개 동시 요청
    (
        response=$(curl -s -w "%{http_code}" -X POST "$TOKEN_LIMITER_URL/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -H "Authorization: Bearer sk-user1-korean-key-def" \
            -d "{
                \"model\": \"korean-qwen\",
                \"messages\": [{\"role\": \"user\", \"content\": \"SGLang 동시 요청 테스트 $i\"}],
                \"max_tokens\": 30
            }")
        echo "요청 $i: ${response: -3}"
    ) &
    pids+=($!)
done

# 모든 백그라운드 작업 완료 대기
for pid in "${pids[@]}"; do
    wait $pid
done

end_time=$(date +%s.%N)
duration=$(echo "$end_time - $start_time" | bc 2>/dev/null || echo "시간 측정 실패")

echo -e "${GREEN}✅ SGLang 동시 요청 4개 완료 (소요 시간: ${duration}초)${NC}"

echo ""
echo "=== 한국어 토큰 계산 테스트 ==="

# 다양한 한국어 텍스트의 토큰 계산 테스트
korean_texts=(
    "안녕하세요"
    "SGLang 기반 한국어 토큰 계산 테스트입니다"
    "복잡한 한국어 문장: 안녕하세요! SGLang은 정말 빠른 LLM 서빙 프레임워크네요. 어떻게 생각하시나요?"
    "English mixed 한국어 텍스트 with SGLang 123 테스트"
    "이모지 포함 😊 SGLang 한국어 텍스트 🚀"
)

for text in "${korean_texts[@]}"; do
    echo -e "\n${BLUE}🔤 토큰 계산: \"$text\"${NC}"

    response=$(curl -s "$TOKEN_LIMITER_URL/token-info" \
        --data-urlencode "text=$text")

    if echo "$response" | grep -q "token_count"; then
        token_count=$(echo "$response" | jq -r '.token_count // "N/A"' 2>/dev/null)
        method=$(echo "$response" | jq -r '.method // "unknown"' 2>/dev/null)
        echo -e "${GREEN}✅ 토큰 수: $token_count (방법: $method)${NC}"
        PASSED=$((PASSED + 1))
    else
        echo -e "${RED}❌ 토큰 계산 실패${NC}"
        FAILED=$((FAILED + 1))
    fi
    TOTAL=$((TOTAL + 1))
done

echo ""
echo "=== 관리자 API 테스트 ==="

# 관리자 기능 테스트
run_test "사용자 목록 조회" "curl -s $TOKEN_LIMITER_URL/admin/users | grep -q users" "success"

run_test "SGLang 정보 새로고침"