#!/usr/bin/env python3
"""
Korean SGLang Token Limiter - SGLang 기반 한국어 토큰 제한 시스템
"""

import asyncio
import json
import time
import logging
import sys
import os
import urllib.parse
from typing import Optional, Dict, Any, AsyncGenerator
import traceback

try:
    import uvicorn
    from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
    from fastapi.responses import JSONResponse, StreamingResponse
    from fastapi.middleware.cors import CORSMiddleware
    import httpx
    from sse_starlette.sse import EventSourceResponse
except ImportError as e:
    print(f"❌ 필수 패키지 누락: {e}")
    print("pip install fastapi uvicorn httpx sse-starlette 를 실행하세요.")
    sys.exit(1)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI(
    title="🇰🇷 Korean SGLang Token Limiter",
    description="SGLang 기반 한국어 LLM 토큰 사용량 제한 시스템",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 변수
SGLANG_SERVER_URL = "http://127.0.0.1:8000"
ACTUAL_MODEL_NAME = None
SGLANG_RUNTIME_INFO = {}


class KoreanTokenCounter:
    """한국어 특화 토큰 카운터 (SGLang용)"""

    @staticmethod
    def count_tokens(text: str) -> int:
        """텍스트의 대략적인 토큰 수 계산"""
        if not text:
            return 0

        # 한국어 특화 계산 (SGLang 최적화)
        korean_chars = len([c for c in text if '\uac00' <= c <= '\ud7af'])
        english_chars = len([c for c in text if c.isalpha() and ord(c) < 128])
        other_chars = len(text) - korean_chars - english_chars

        # SGLang의 효율적인 토큰화 특성 반영
        tokens = int(korean_chars * 1.15 + english_chars * 0.25 + other_chars * 0.5)
        return max(1, tokens)

    @staticmethod
    def count_messages_tokens(messages) -> int:
        """메시지의 토큰 수 계산"""
        total = 0
        for msg in messages:
            if isinstance(msg, dict) and 'content' in msg:
                total += KoreanTokenCounter.count_tokens(str(msg['content']))
                total += 3  # 역할 오버헤드
        return total + 4  # 대화 오버헤드


class SGLangRateLimiter:
    """SGLang 특화 속도 제한기"""

    def __init__(self):
        self.users = {}
        # SGLang의 높은 성능을 반영한 기본 제한
        self.default_limits = {
            'rpm': 40,      # SGLang 성능 향상으로 증가
            'tpm': 8000,    # 분당 토큰 수 증가
            'daily': 1000000  # 일일 토큰 수 증가
        }

        # 사용자별 API 키 매핑
        self.api_keys = {
            'sk-user1-korean-key-def': 'user1',
            'sk-user2-korean-key-ghi': 'user2',
            'sk-dev1-korean-key-789': 'developer1',
            'sk-test-korean-key-stu': 'test',
            'sk-guest-korean-key-vwx': 'guest'
        }

        # 영어 -> 한국어 매핑 (표시용)
        self.user_display_names = {
            'user1': '사용자1',
            'user2': '사용자2',
            'developer1': '개발자1',
            'test': '테스트',
            'guest': '게스트'
        }

    def get_user_from_api_key(self, api_key: str) -> str:
        """API 키에서 사용자 ID 추출"""
        return self.api_keys.get(api_key, 'guest')

    def get_display_name(self, user_id: str) -> str:
        """사용자 표시명 조회"""
        return self.user_display_names.get(user_id, user_id)

    def check_limits(self, user_id: str, tokens: int) -> tuple:
        """사용량 제한 확인"""
        now = time.time()

        if user_id not in self.users:
            self.users[user_id] = {
                'requests_minute': [],
                'tokens_minute': [],
                'tokens_daily': [],
                'total_requests': 0,
                'total_tokens': 0
            }

        user_data = self.users[user_id]

        # 1분 이내 데이터만 유지
        minute_ago = now - 60
        user_data['requests_minute'] = [t for t in user_data['requests_minute'] if t > minute_ago]
        user_data['tokens_minute'] = [t for t in user_data['tokens_minute'] if t[0] > minute_ago]

        # 하루 이내 데이터만 유지
        day_ago = now - 86400
        user_data['tokens_daily'] = [t for t in user_data['tokens_daily'] if t[0] > day_ago]

        # 현재 사용량 계산
        current_rpm = len(user_data['requests_minute'])
        current_tpm = sum(t[1] for t in user_data['tokens_minute'])
        current_daily = sum(t[1] for t in user_data['tokens_daily'])

        # 제한 확인
        if current_rpm >= self.default_limits['rpm']:
            return False, f"분당 요청 제한 초과 ({self.default_limits['rpm']}개)"

        if current_tpm + tokens > self.default_limits['tpm']:
            return False, f"분당 토큰 제한 초과 ({self.default_limits['tpm']}개)"

        if current_daily + tokens > self.default_limits['daily']:
            return False, f"일일 토큰 제한 초과 ({self.default_limits['daily']}개)"

        return True, None

    def record_usage(self, user_id: str, tokens: int):
        """사용량 기록"""
        now = time.time()

        if user_id not in self.users:
            self.users[user_id] = {
                'requests_minute': [],
                'tokens_minute': [],
                'tokens_daily': [],
                'total_requests': 0,
                'total_tokens': 0
            }

        user_data = self.users[user_id]
        user_data['requests_minute'].append(now)
        user_data['tokens_minute'].append((now, tokens))
        user_data['tokens_daily'].append((now, tokens))
        user_data['total_requests'] += 1
        user_data['total_tokens'] += tokens

    def get_user_stats(self, user_id: str) -> dict:
        """사용자 통계 조회"""
        if user_id not in self.users:
            return {
                'user_id': user_id,
                'display_name': self.get_display_name(user_id),
                'requests_this_minute': 0,
                'tokens_this_minute': 0,
                'tokens_today': 0,
                'total_requests': 0,
                'total_tokens': 0,
                'limits': self.default_limits
            }

        now = time.time()
        minute_ago = now - 60
        day_ago = now - 86400

        user_data = self.users[user_id]

        return {
            'user_id': user_id,
            'display_name': self.get_display_name(user_id),
            'requests_this_minute': len([t for t in user_data['requests_minute'] if t > minute_ago]),
            'tokens_this_minute': sum(t[1] for t in user_data['tokens_minute'] if t[0] > minute_ago),
            'tokens_today': sum(t[1] for t in user_data['tokens_daily'] if t[0] > day_ago),
            'total_requests': user_data['total_requests'],
            'total_tokens': user_data['total_tokens'],
            'limits': self.default_limits
        }


# 전역 인스턴스
token_counter = KoreanTokenCounter()
rate_limiter = SGLangRateLimiter()


async def get_sglang_models():
    """SGLang 서버에서 모델 정보 조회"""
    global ACTUAL_MODEL_NAME, SGLANG_RUNTIME_INFO

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            # SGLang 서버 상태 확인
            response = await client.get(f"{SGLANG_SERVER_URL}/get_model_info")
            
            if response.status_code == 200:
                model_info = response.json()
                ACTUAL_MODEL_NAME = model_info.get('model_path', 'korean-sglang')
                SGLANG_RUNTIME_INFO = model_info
                logger.info(f"✅ SGLang 모델 정보: {ACTUAL_MODEL_NAME}")
                return ACTUAL_MODEL_NAME
            
    except Exception as e:
        logger.warning(f"⚠️ SGLang 모델 정보 조회 실패: {e}")

    # 기본값 설정
    ACTUAL_MODEL_NAME = "korean-llama"
    return ACTUAL_MODEL_NAME


def extract_user_id(request: Request) -> str:
    """요청에서 사용자 ID 추출"""
    auth_header = request.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        api_key = auth_header[7:]
        return rate_limiter.get_user_from_api_key(api_key)

    user_id = request.headers.get("x-user-id")
    if user_id:
        return user_id

    return "guest"


def convert_to_sglang_format(messages: list, model: str = "korean-llama") -> dict:
    """채팅 메시지를 SGLang 형태로 변환"""
    # SGLang은 OpenAI 호환 형식을 직접 지원
    return {
        "model": model,
        "messages": messages,
        "stream": False,
        "temperature": 0.7,
        "max_tokens": 512
    }


@app.middleware("http")
async def token_limit_middleware(request: Request, call_next):
    """토큰 제한 미들웨어 (SGLang 최적화)"""
    
    # API 경로가 아니면 통과
    if not any(path in request.url.path for path in ["/v1/chat/completions", "/v1/completions"]):
        return await call_next(request)

    user_id = extract_user_id(request)

    # 요청 본문 읽기
    body = await request.body()

    try:
        request_data = json.loads(body) if body else {}
    except json.JSONDecodeError:
        return JSONResponse(
            status_code=400,
            content={"error": "잘못된 JSON 형식입니다"}
        )

    # 토큰 계산
    estimated_tokens = 0
    if 'messages' in request_data:
        estimated_tokens = token_counter.count_messages_tokens(request_data['messages'])
    elif 'prompt' in request_data:
        estimated_tokens = token_counter.count_tokens(str(request_data['prompt']))

    estimated_tokens += request_data.get('max_tokens', 100)

    # 제한 확인
    allowed, reason = rate_limiter.check_limits(user_id, estimated_tokens)

    if not allowed:
        logger.warning(f"Rate limit exceeded for user '{user_id}': {reason}")
        return JSONResponse(
            status_code=429,
            content={
                "error": {
                    "message": reason,
                    "type": "rate_limit_exceeded",
                    "user_id": user_id,
                    "estimated_tokens": estimated_tokens
                }
            }
        )

    # 사용량 기록
    rate_limiter.record_usage(user_id, estimated_tokens)

    # 요청 본문 복원
    async def receive():
        return {"type": "http.request", "body": body}

    request._receive = receive

    # 요청 처리
    response = await call_next(request)

    # 사용자 ID 헤더 추가
    safe_user_id = urllib.parse.quote(user_id.encode('utf-8'))
    response.headers["X-User-ID"] = safe_user_id

    return response


async def stream_sglang_response(messages: list, model: str, max_tokens: int = 512, temperature: float = 0.7) -> AsyncGenerator[str, None]:
    """SGLang 스트리밍 응답 처리"""
    try:
        request_data = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            async with client.stream(
                "POST",
                f"{SGLANG_SERVER_URL}/v1/chat/completions",
                json=request_data,
                headers={"Content-Type": "application/json"}
            ) as response:
                
                if response.status_code != 200:
                    error_text = await response.aread()
                    yield f"data: {json.dumps({'error': f'SGLang error: {error_text.decode()}'})}\n\n"
                    return

                async for chunk in response.aiter_lines():
                    if chunk.startswith("data: "):
                        data = chunk[6:]  # "data: " 제거
                        if data.strip() == "[DONE]":
                            yield "data: [DONE]\n\n"
                            break
                        
                        try:
                            # SGLang 응답을 OpenAI 형식으로 변환
                            chunk_data = json.loads(data)
                            openai_chunk = {
                                "id": f"chatcmpl-{int(time.time())}",
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": "korean-llama",
                                "choices": [{
                                    "index": 0,
                                    "delta": chunk_data.get("choices", [{}])[0].get("delta", {}),
                                    "finish_reason": chunk_data.get("choices", [{}])[0].get("finish_reason")
                                }]
                            }
                            yield f"data: {json.dumps(openai_chunk)}\n\n"
                            
                        except json.JSONDecodeError:
                            continue

    except Exception as e:
        logger.error(f"❌ SGLang streaming error: {e}")
        yield f"data: {json.dumps({'error': f'Streaming error: {str(e)}'})}\n\n"


@app.post("/v1/chat/completions")
async def chat_completions_proxy(request: Request):
    """채팅 완성 프록시 (SGLang 기반)"""
    
    body = await request.body()
    user_id = extract_user_id(request)

    try:
        request_data = json.loads(body)
        messages = request_data.get('messages', [])
        max_tokens = request_data.get('max_tokens', 512)
        temperature = request_data.get('temperature', 0.7)
        stream = request_data.get('stream', False)

        # 실제 모델명 조회
        actual_model = await get_sglang_models()

        logger.info(f"🔄 SGLang 요청: 모델={actual_model}, 사용자={user_id}, 스트림={stream}")

        # 스트리밍 응답
        if stream:
            return EventSourceResponse(
                stream_sglang_response(messages, actual_model, max_tokens, temperature),
                media_type="text/event-stream"
            )

        # 일반 응답
        sglang_request = {
            "model": actual_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            sglang_response = await client.post(
                f"{SGLANG_SERVER_URL}/v1/chat/completions",
                json=sglang_request,
                headers={"Content-Type": "application/json"}
            )

        if sglang_response.status_code != 200:
            error_detail = sglang_response.text
            logger.error(f"❌ SGLang 오류 (모델: {actual_model}): {error_detail}")
            return JSONResponse(
                status_code=sglang_response.status_code,
                content={
                    "error": "SGLang 서버 오류",
                    "detail": error_detail,
                    "model_used": actual_model
                }
            )

        result = sglang_response.json()
        
        # OpenAI 호환 형식으로 응답 (SGLang이 이미 호환 형식 제공)
        if 'model' not in result:
            result['model'] = "korean-llama"
        
        logger.info(f"✅ SGLang 응답 생성 완료: 사용자={user_id}")
        return JSONResponse(content=result)

    except httpx.ConnectError:
        logger.error(f"SGLang server connection error for user '{user_id}'")
        return JSONResponse(
            status_code=503,
            content={"error": "SGLang 서버에 연결할 수 없습니다"}
        )
    except Exception as e:
        logger.error(f"Chat completion error for user '{user_id}': {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"채팅 완성 오류: {str(e)}"}
        )


@app.post("/v1/completions")
async def completions_proxy(request: Request):
    """텍스트 완성 프록시 (SGLang 기반)"""
    
    body = await request.body()
    user_id = extract_user_id(request)

    try:
        request_data = json.loads(body)
        prompt = request_data.get('prompt', '')
        max_tokens = request_data.get('max_tokens', 100)
        temperature = request_data.get('temperature', 0.7)

        # 실제 모델명으로 변경
        actual_model = await get_sglang_models()
        request_data["model"] = actual_model

        # SGLang 서버로 요청 전달
        async with httpx.AsyncClient(timeout=60.0) as client:
            sglang_response = await client.post(
                f"{SGLANG_SERVER_URL}/v1/completions",
                json=request_data,
                headers={"Content-Type": "application/json"}
            )

        # 응답 반환
        if sglang_response.status_code == 200:
            result = sglang_response.json()
            if 'model' not in result:
                result['model'] = "korean-llama"
            return JSONResponse(content=result)
        else:
            return JSONResponse(
                status_code=sglang_response.status_code,
                content={"error": f"SGLang 서버 오류: {sglang_response.text}"}
            )

    except httpx.ConnectError:
        logger.error(f"SGLang server connection error for user '{user_id}'")
        return JSONResponse(
            status_code=503,
            content={"error": "SGLang 서버에 연결할 수 없습니다"}
        )
    except Exception as e:
        logger.error(f"Completion proxy error for user '{user_id}': {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"텍스트 완성 오류: {str(e)}"}
        )


@app.get("/health")
async def health_check():
    """헬스체크 (SGLang 상태 포함)"""
    try:
        # SGLang 서버 확인
        async with httpx.AsyncClient(timeout=5.0) as client:
            sglang_response = await client.get(f"{SGLANG_SERVER_URL}/get_model_info")
            sglang_status = sglang_response.status_code == 200
            
            if sglang_status:
                model_info = sglang_response.json()
                runtime_info = {
                    "model_path": model_info.get("model_path", "unknown"),
                    "max_total_tokens": model_info.get("max_total_tokens", 0),
                    "served_model_names": model_info.get("served_model_names", []),
                    "is_generation": model_info.get("is_generation", True)
                }
            else:
                runtime_info = {}

        actual_model = await get_sglang_models()

    except Exception as e:
        sglang_status = False
        runtime_info = {"error": str(e)}
        actual_model = "unknown"

    return {
        "status": "healthy",
        "sglang_server": "connected" if sglang_status else "disconnected",
        "model": "korean-llama",
        "actual_sglang_model": actual_model,
        "runtime_info": runtime_info,
        "supports_korean": True,
        "supports_streaming": True,
        "framework": "sglang",
        "encoding": "utf-8_safe",
        "timestamp": time.time()
    }


@app.get("/models")
async def list_models():
    """사용 가능한 모델 목록"""
    try:
        actual_model = await get_sglang_models()
        return {
            "data": [
                {
                    "id": "korean-llama",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "korean-sglang-limiter",
                    "actual_model": actual_model,
                    "framework": "sglang",
                    "supports_streaming": True
                }
            ]
        }
    except Exception as e:
        return {"error": f"모델 목록 조회 실패: {str(e)}"}


@app.get("/sglang/runtime-info")
async def get_sglang_runtime_info():
    """SGLang 런타임 정보 조회"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            # SGLang 상태 정보
            model_response = await client.get(f"{SGLANG_SERVER_URL}/get_model_info")
            server_response = await client.get(f"{SGLANG_SERVER_URL}/get_server_info")
            
            if model_response.status_code == 200 and server_response.status_code == 200:
                model_info = model_response.json()
                server_info = server_response.json()
                
                return {
                    "model_info": model_info,
                    "server_info": server_info,
                    "performance": {
                        "supports_dynamic_batching": True,
                        "supports_prefix_caching": True,
                        "supports_chunked_prefill": True,
                        "kv_cache_optimized": True
                    },
                    "timestamp": time.time()
                }
            else:
                return {"error": "SGLang 런타임 정보 조회 실패"}
                
    except Exception as e:
        return {"error": f"런타임 정보 오류: {str(e)}"}


@app.get("/stats/{user_id}")
async def get_user_stats(user_id: str):
    """사용자 통계 조회"""
    try:
        user_id = urllib.parse.unquote(user_id)
        stats = rate_limiter.get_user_stats(user_id)
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"통계 조회 실패: {str(e)}")


@app.get("/token-info")
async def get_token_info(text: str = "안녕하세요! SGLang 기반 한국어 토큰 계산 테스트입니다."):
    """토큰 계산 정보"""
    try:
        token_count = token_counter.count_tokens(text)
        return {
            "text": text,
            "token_count": token_count,
            "method": "korean_sglang_optimized",
            "framework": "sglang"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"토큰 정보 조회 실패: {str(e)}")


@app.get("/admin/users")
async def list_users():
    """사용자 목록 조회"""
    try:
        users_with_display = []
        for user_id in rate_limiter.users.keys():
            users_with_display.append({
                "user_id": user_id,
                "display_name": rate_limiter.get_display_name(user_id)
            })

        return {
            "users": users_with_display,
            "total_count": len(users_with_display)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"사용자 목록 조회 실패: {str(e)}")


@app.post("/admin/reload-sglang")
async def reload_sglang_info():
    """SGLang 모델 정보 다시 로드"""
    try:
        global ACTUAL_MODEL_NAME, SGLANG_RUNTIME_INFO
        ACTUAL_MODEL_NAME = None
        SGLANG_RUNTIME_INFO = {}
        
        model_name = await get_sglang_models()
        
        return {
            "message": "SGLang 정보가 다시 로드되었습니다",
            "model": model_name,
            "runtime_info": SGLANG_RUNTIME_INFO
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SGLang 정보 로드 실패: {str(e)}")


@app.get("/admin/sglang/performance")
async def get_sglang_performance():
    """SGLang 성능 메트릭 조회"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{SGLANG_SERVER_URL}/get_server_info")
            
            if response.status_code == 200:
                server_info = response.json()
                
                # SGLang 성능 정보 추출
                performance_metrics = {
                    "requests_per_second": server_info.get("requests_per_second", 0),
                    "tokens_per_second": server_info.get("tokens_per_second", 0),
                    "queue_length": server_info.get("queue_length", 0),
                    "running_requests": server_info.get("running_requests", 0),
                    "memory_usage": server_info.get("memory_usage_gb", 0),
                    "cache_hit_rate": server_info.get("cache_hit_rate", 0),
                    "framework": "sglang",
                    "timestamp": time.time()
                }
                
                return performance_metrics
            else:
                return {"error": "SGLang 성능 정보 조회 실패"}
                
    except Exception as e:
        return {"error": f"성능 메트릭 오류: {str(e)}"}


if __name__ == "__main__":
    print("🇰🇷 Korean SGLang Token Limiter 시작 중...")

    # 로그 디렉토리 생성
    os.makedirs("logs", exist_ok=True)

    # 서버 실행
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        log_level="info",
        access_log=True
    )