"""
SGLang API 프록시 - 한국어 최적화 및 토큰 제한 기능이 포함된 API 엔드포인트
"""

import json
import time
import asyncio
import logging
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from sse_starlette.sse import EventSourceResponse

from src.core.sglang_client import get_sglang_client, SGLangClient
from src.core.korean_token_counter import KoreanTokenCounter
from src.core.rate_limiter import SGLangRateLimiter
from src.utils.performance import get_performance_monitor, collect_sglang_metrics
from src.core.config import config

logger = logging.getLogger(__name__)

# 라우터 생성
router = APIRouter(prefix="/v1", tags=["SGLang API"])

# 글로벌 인스턴스들
token_counter = KoreanTokenCounter()
rate_limiter = SGLangRateLimiter()
performance_monitor = get_performance_monitor()


def extract_user_id_from_request(request: Request) -> str:
    """요청에서 사용자 ID 추출"""
    # Authorization 헤더에서 추출
    auth_header = request.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        api_key = auth_header[7:]
        return rate_limiter.get_user_from_api_key(api_key)

    # X-User-ID 헤더에서 추출
    user_id = request.headers.get("x-user-id")
    if user_id:
        return user_id

    # 기본값
    return "guest"


@router.post("/chat/completions")
async def chat_completions_proxy(request: Request, background_tasks: BackgroundTasks):
    """SGLang 채팅 완성 프록시 (한국어 최적화)"""
    start_time = time.time()
    user_id = extract_user_id_from_request(request)

    try:
        # 요청 데이터 파싱
        request_data = await request.json()

        # 필수 필드 검증
        if "messages" not in request_data:
            raise HTTPException(status_code=400, detail="'messages' 필드가 필요합니다.")

        messages = request_data["messages"]
        model = request_data.get("model", "korean-qwen")
        max_tokens = request_data.get("max_tokens", 512)
        temperature = request_data.get("temperature", 0.7)
        top_p = request_data.get("top_p", 1.0)
        frequency_penalty = request_data.get("frequency_penalty", 0.0)
        presence_penalty = request_data.get("presence_penalty", 0.0)
        stop = request_data.get("stop")
        stream = request_data.get("stream", False)

        # 한국어 최적화 옵션
        korean_optimized = request_data.get("korean_optimized", True)

        logger.info(f"🔄 SGLang 채팅 요청: 사용자={user_id}, 모델={model}, 스트림={stream}")

        # 토큰 수 계산
        input_tokens = token_counter.count_messages_tokens(messages)
        estimated_total_tokens = input_tokens + max_tokens

        # 한국어 텍스트 분석
        text_content = " ".join([msg.get("content", "") for msg in messages if isinstance(msg.get("content"), str)])
        korean_analysis = _analyze_korean_content(text_content)

        # 속도 제한 확인
        allowed, limit_reason = await rate_limiter.check_limit(
            user_id,
            estimated_total_tokens,
            request_type="chat_completion",
            is_stream=stream
        )

        if not allowed:
            logger.warning(f"⚠️ 사용자 '{user_id}' 속도 제한: {limit_reason}")
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "message": limit_reason,
                        "type": "rate_limit_exceeded",
                        "code": "rate_limit_exceeded",
                        "details": {
                            "user_id": user_id,
                            "estimated_tokens": estimated_total_tokens,
                            "korean_analysis": korean_analysis
                        }
                    }
                }
            )

        # SGLang 클라이언트 가져오기
        sglang_client = get_sglang_client(config.sglang_server_url)

        # 스트리밍 응답 처리
        if stream:
            return EventSourceResponse(
                _stream_chat_completion(
                    sglang_client, messages, model, max_tokens, temperature,
                    top_p, frequency_penalty, presence_penalty, stop,
                    korean_optimized, user_id, input_tokens, start_time
                )
            )

        # 일반 응답 처리
        result = await sglang_client.chat_completion(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
            stream=False,
            korean_optimized=korean_optimized
        )

        response_time = time.time() - start_time

        # 에러 처리
        if "error" in result:
            logger.error(f"❌ SGLang 채팅 완성 실패: {result['error']}")

            # 사용량 기록 (에러도 기록)
            background_tasks.add_task(
                record_usage_background,
                user_id, input_tokens, 0, 1, response_time, False, False, korean_analysis
            )

            return JSONResponse(
                status_code=500,
                content=result
            )

        # 실제 출력 토큰 수 계산
        output_tokens = 0
        if "choices" in result and result["choices"]:
            choice = result["choices"][0]
            if "message" in choice and "content" in choice["message"]:
                output_content = choice["message"]["content"]
                output_tokens = token_counter.count_tokens(output_content)

        # 사용량 통계에 실제 토큰 수 반영
        if "usage" not in result:
            result["usage"] = {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens
            }

        # 사용량 기록
        background_tasks.add_task(
            record_usage_background,
            user_id, input_tokens, output_tokens, 1, response_time, False, False, korean_analysis
        )

        # 성능 메트릭 기록
        background_tasks.add_task(
            record_performance_metrics,
            sglang_client, response_time, True
        )

        logger.info(f"✅ SGLang 채팅 완성 성공: 사용자={user_id}, 응답시간={response_time:.2f}초")

        return JSONResponse(content=result)

    except json.JSONDecodeError:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "잘못된 JSON 형식입니다.", "type": "invalid_request"}}
        )

    except Exception as e:
        response_time = time.time() - start_time
        logger.error(f"❌ 채팅 완성 프록시 오류: {e}")

        # 에러 사용량 기록
        background_tasks.add_task(
            record_usage_background,
            user_id, 0, 0, 1, response_time, False, False, {}
        )

        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": f"SGLang 프록시 오류: {str(e)}",
                    "type": "proxy_error",
                    "code": "internal_server_error"
                }
            }
        )


async def _stream_chat_completion(sglang_client: SGLangClient, messages: List[Dict],
                                  model: str, max_tokens: int, temperature: float,
                                  top_p: float, frequency_penalty: float, presence_penalty: float,
                                  stop: Optional[List[str]], korean_optimized: bool,
                                  user_id: str, input_tokens: int, start_time: float):
    """스트리밍 채팅 완성 처리"""
    try:
        output_tokens = 0
        chunk_count = 0

        async for chunk in sglang_client.chat_completion_stream(
                messages=messages,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                korean_optimized=korean_optimized
        ):
            chunk_count += 1

            # 에러 처리
            if "error" in chunk:
                logger.error(f"❌ SGLang 스트리밍 오류: {chunk['error']}")
                yield f"data: {json.dumps(chunk)}\n\n"
                break

            # 정상 청크 처리
            if "choices" in chunk and chunk["choices"]:
                choice = chunk["choices"][0]
                if "delta" in choice and "content" in choice["delta"]:
                    content = choice["delta"]["content"]
                    if content:
                        output_tokens += token_counter.count_tokens(content)

            yield f"data: {json.dumps(chunk)}\n\n"

        # 스트리밍 완료
        yield "data: [DONE]\n\n"

        response_time = time.time() - start_time

        # 스트리밍 사용량 기록
        asyncio.create_task(
            record_usage_background(
                user_id, input_tokens, output_tokens, 1, response_time, False, True, {}
            )
        )

        logger.info(f"✅ SGLang 스트리밍 완료: 사용자={user_id}, 청크={chunk_count}개, 응답시간={response_time:.2f}초")

    except Exception as e:
        logger.error(f"❌ 스트리밍 처리 오류: {e}")
        yield f"data: {json.dumps({'error': {'message': str(e), 'type': 'stream_error'}})}\n\n"


@router.post("/completions")
async def text_completions_proxy(request: Request, background_tasks: BackgroundTasks):
    """SGLang 텍스트 완성 프록시"""
    start_time = time.time()
    user_id = extract_user_id_from_request(request)

    try:
        request_data = await request.json()

        # 필수 필드 검증
        if "prompt" not in request_data:
            raise HTTPException(status_code=400, detail="'prompt' 필드가 필요합니다.")

        prompt = request_data["prompt"]
        model = request_data.get("model", "korean-qwen")
        max_tokens = request_data.get("max_tokens", 256)
        temperature = request_data.get("temperature", 0.7)
        korean_optimized = request_data.get("korean_optimized", True)

        logger.info(f"🔄 SGLang 텍스트 완성 요청: 사용자={user_id}, 모델={model}")

        # 토큰 수 계산
        input_tokens = token_counter.count_tokens(prompt)
        estimated_total_tokens = input_tokens + max_tokens

        # 한국어 텍스트 분석
        korean_analysis = _analyze_korean_content(prompt)

        # 속도 제한 확인
        allowed, limit_reason = await rate_limiter.check_limit(
            user_id,
            estimated_total_tokens,
            request_type="text_completion"
        )

        if not allowed:
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "message": limit_reason,
                        "type": "rate_limit_exceeded",
                        "code": "rate_limit_exceeded"
                    }
                }
            )

        # SGLang 클라이언트 가져오기
        sglang_client = get_sglang_client(config.sglang_server_url)

        # 텍스트 완성 요청
        result = await sglang_client.text_completion(
            prompt=prompt,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            korean_optimized=korean_optimized
        )

        response_time = time.time() - start_time

        # 에러 처리
        if "error" in result:
            logger.error(f"❌ SGLang 텍스트 완성 실패: {result['error']}")

            background_tasks.add_task(
                record_usage_background,
                user_id, input_tokens, 0, 1, response_time, False, False, korean_analysis
            )

            return JSONResponse(status_code=500, content=result)

        # 출력 토큰 수 계산
        output_tokens = 0
        if "choices" in result and result["choices"]:
            choice = result["choices"][0]
            if "text" in choice:
                output_tokens = token_counter.count_tokens(choice["text"])

        # 사용량 통계 추가
        if "usage" not in result:
            result["usage"] = {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens
            }

        # 사용량 기록
        background_tasks.add_task(
            record_usage_background,
            user_id, input_tokens, output_tokens, 1, response_time, False, False, korean_analysis
        )

        logger.info(f"✅ SGLang 텍스트 완성 성공: 사용자={user_id}, 응답시간={response_time:.2f}초")

        return JSONResponse(content=result)

    except Exception as e:
        response_time = time.time() - start_time
        logger.error(f"❌ 텍스트 완성 프록시 오류: {e}")

        background_tasks.add_task(
            record_usage_background,
            user_id, 0, 0, 1, response_time, False, False, {}
        )

        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": f"SGLang 프록시 오류: {str(e)}",
                    "type": "proxy_error"
                }
            }
        )


@router.get("/models")
async def list_models():
    """사용 가능한 모델 목록 조회 (SGLang 서버에서)"""
    try:
        sglang_client = get_sglang_client(config.sglang_server_url)
        model_info = await sglang_client.get_model_info()

        if model_info:
            models_data = [
                {
                    "id": name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "sglang-korean-limiter",
                    "framework": "SGLang",
                    "korean_optimized": True,
                    "context_length": model_info.max_total_tokens,
                    "architecture": model_info.architecture if model_info.architecture else "unknown"
                }
                for name in model_info.served_model_names
            ]

            # 기본 모델도 추가
            if not models_data:
                models_data.append({
                    "id": "korean-qwen",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "sglang-korean-limiter",
                    "framework": "SGLang",
                    "korean_optimized": True,
                    "context_length": model_info.max_total_tokens,
                    "architecture": "transformer"
                })
        else:
            # SGLang 서버에 연결할 수 없는 경우 기본 모델 반환
            models_data = [
                {
                    "id": "korean-qwen",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "sglang-korean-limiter",
                    "framework": "SGLang",
                    "korean_optimized": True,
                    "context_length": 8192,
                    "architecture": "transformer"
                }
            ]

        return {"object": "list", "data": models_data}

    except Exception as e:
        logger.error(f"❌ 모델 목록 조회 실패: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": f"모델 목록 조회 실패: {str(e)}",
                    "type": "model_list_error"
                }
            }
        )


# 관리자 API 엔드포인트
@router.get("/admin/sglang/status")
async def get_sglang_status():
    """SGLang 서버 상태 조회 (관리자용)"""
    try:
        sglang_client = get_sglang_client(config.sglang_server_url)

        # 헬스 체크
        is_healthy = await sglang_client.health_check()

        # 모델 정보
        model_info = await sglang_client.get_model_info()

        # 서버 정보
        server_info = await sglang_client.get_server_info()

        # 클라이언트 통계
        client_stats = await sglang_client.get_client_statistics()

        return {
            "status": "healthy" if is_healthy else "unhealthy",
            "framework": "SGLang",
            "timestamp": time.time(),
            "model_info": model_info.__dict__ if model_info else None,
            "server_info": server_info.__dict__ if server_info else None,
            "client_statistics": client_stats,
            "korean_optimizations": sglang_client.korean_optimizations
        }

    except Exception as e:
        logger.error(f"❌ SGLang 상태 조회 실패: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(e),
                "framework": "SGLang"
            }
        )


@router.get("/admin/sglang/performance")
async def get_sglang_performance():
    """SGLang 성능 메트릭 조회 (관리자용)"""
    try:
        sglang_client = get_sglang_client(config.sglang_server_url)

        # SGLang 성능 메트릭 수집
        metrics = await collect_sglang_metrics(sglang_client)

        if metrics:
            # 성능 모니터에 기록
            performance_monitor.record_sglang_metrics(metrics)

            # 최근 성능 요약 추가
            performance_summary = performance_monitor.get_performance_summary(minutes=10)

            return {
                "framework": "SGLang",
                "timestamp": time.time(),
                "current_metrics": metrics,
                "performance_summary": performance_summary,
                "optimization_recommendations": performance_monitor.get_optimization_recommendations()
            }
        else:
            return JSONResponse(
                status_code=503,
                content={
                    "error": "SGLang 성능 메트릭을 수집할 수 없습니다",
                    "framework": "SGLang"
                }
            )

    except Exception as e:
        logger.error(f"❌ SGLang 성능 메트릭 조회 실패: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "framework": "SGLang"
            }
        )


@router.post("/admin/sglang/optimize")
async def optimize_sglang_performance():
    """SGLang 성능 자동 최적화 (관리자용)"""
    try:
        sglang_client = get_sglang_client(config.sglang_server_url)

        # 현재 부하 상태 분석
        server_info = await sglang_client.get_server_info()
        current_load = 0.0

        if server_info:
            # 대기열 길이와 실행 중인 요청 수를 기반으로 부하 계산
            max_requests = config.sglang_server.max_running_requests
            current_load = (server_info.running_requests + server_info.queue_length) / max_requests

        # 적응적 배치 최적화 수행
        optimization_result = await sglang_client.adaptive_batch_optimization(current_load)

        # 성능 모니터에서 추가 분석
        performance_summary = performance_monitor.get_performance_summary(minutes=5)
        recommendations = performance_monitor.get_optimization_recommendations()

        return {
            "status": "optimization_completed",
            "framework": "SGLang",
            "timestamp": time.time(),
            "current_load": current_load,
            "optimization_result": optimization_result,
            "performance_analysis": performance_summary,
            "recommendations": recommendations
        }

    except Exception as e:
        logger.error(f"❌ SGLang 성능 최적화 실패: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "optimization_failed",
                "error": str(e),
                "framework": "SGLang"
            }
        )


@router.post("/admin/sglang/warmup")
async def warmup_sglang():
    """SGLang 서버 워밍업 (관리자용)"""
    try:
        sglang_client = get_sglang_client(config.sglang_server_url)

        # 한국어 테스트 메시지 준비
        korean_test_messages = [
            {"role": "user", "content": "안녕하세요! SGLang 워밍업 테스트입니다."},
            {"role": "user", "content": "한국어 토큰화 테스트를 진행합니다."},
            {"role": "user", "content": "SGLang의 성능은 어떤가요?"}
        ]

        # 워밍업 실행
        warmup_success = await sglang_client.warmup(korean_test_messages)

        if warmup_success:
            # 워밍업 후 상태 확인
            model_info = await sglang_client.get_model_info()
            server_info = await sglang_client.get_server_info()

            return {
                "status": "warmup_completed",
                "success": True,
                "framework": "SGLang",
                "timestamp": time.time(),
                "model_info": model_info.__dict__ if model_info else None,
                "server_info": server_info.__dict__ if server_info else None,
                "korean_test_completed": True
            }
        else:
            return {
                "status": "warmup_failed",
                "success": False,
                "framework": "SGLang",
                "timestamp": time.time(),
                "error": "SGLang 워밍업 실패"
            }

    except Exception as e:
        logger.error(f"❌ SGLang 워밍업 실패: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "warmup_error",
                "success": False,
                "error": str(e),
                "framework": "SGLang"
            }
        )


# 헬퍼 함수들
def _analyze_korean_content(text: str) -> Dict[str, Any]:
    """한국어 콘텐츠 분석"""
    import re

    if not text:
        return {
            "korean_chars": 0,
            "total_chars": 0,
            "korean_ratio": 0.0,
            "complexity_score": 0.0
        }

    korean_chars = len(re.findall(r'[가-힣]', text))
    total_chars = len(text)
    korean_ratio = korean_chars / total_chars if total_chars > 0 else 0

    # 복잡도 점수 (복합어, 한영 혼용 등)
    compound_words = len(re.findall(r'[가-힣]{3,}', text))
    english_chars = len(re.findall(r'[a-zA-Z]', text))
    complexity_score = (compound_words * 0.3) + (english_chars > 0 and korean_chars > 0) * 0.5

    return {
        "korean_chars": korean_chars,
        "total_chars": total_chars,
        "korean_ratio": korean_ratio,
        "complexity_score": complexity_score,
        "compound_words": compound_words,
        "has_mixed_language": english_chars > 0 and korean_chars > 0
    }


async def record_usage_background(user_id: str, input_tokens: int, output_tokens: int,
                                  requests: int, response_time: float, cache_hit: bool,
                                  is_stream: bool, korean_analysis: Dict[str, Any]):
    """백그라운드에서 사용량 기록"""
    try:
        # Rate Limiter에 사용량 기록
        await rate_limiter.record_usage(
            user_id=user_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            requests=requests,
            response_time=response_time,
            cache_hit=cache_hit,
            is_stream=is_stream
        )

        # 한국어 토큰 메트릭 기록
        if korean_analysis:
            performance_monitor.record_korean_token_metrics(
                korean_chars=korean_analysis.get("korean_chars", 0),
                total_chars=korean_analysis.get("total_chars", 0),
                estimated_tokens=input_tokens + output_tokens,
                actual_tokens=input_tokens + output_tokens
            )

        logger.debug(f"📊 사용량 기록 완료: {user_id} -> {input_tokens + output_tokens} 토큰")

    except Exception as e:
        logger.error(f"❌ 사용량 기록 실패: {e}")


async def record_performance_metrics(sglang_client: SGLangClient, response_time: float, success: bool):
    """백그라운드에서 성능 메트릭 기록"""
    try:
        # SGLang 성능 메트릭 수집
        metrics = await collect_sglang_metrics(sglang_client)

        if metrics:
            # 현재 요청의 성능 정보 추가
            metrics["last_response_time"] = response_time
            metrics["last_request_success"] = success

            # 성능 모니터에 기록
            performance_monitor.record_sglang_metrics(metrics)

    except Exception as e:
        logger.debug(f"성능 메트릭 기록 실패: {e}")


# 초기화 함수
async def initialize_sglang_proxy():
    """SGLang 프록시 초기화"""
    try:
        # SGLang 클라이언트 초기화
        sglang_client = get_sglang_client(config.sglang_server_url)

        # 워밍업 수행
        await sglang_client.warmup()

        # 성능 모니터링 시작
        performance_monitor.start_monitoring()

        logger.info("✅ SGLang 프록시 초기화 완료")

    except Exception as e:
        logger.error(f"❌ SGLang 프록시 초기화 실패: {e}")
        raise


async def cleanup_sglang_proxy():
    """SGLang 프록시 정리"""
    try:
        # 성능 모니터링 중지
        performance_monitor.stop_monitoring()

        # SGLang 클라이언트 정리
        sglang_client = get_sglang_client(config.sglang_server_url)
        await sglang_client.close()

        logger.info("✅ SGLang 프록시 정리 완료")

    except Exception as e:
        logger.error(f"❌ SGLang 프록시 정리 실패: {e}")


# 에러 핸들러
@router.exception_handler(Exception)
async def sglang_proxy_exception_handler(request: Request, exc: Exception):
    """SGLang 프록시 전용 예외 처리기"""
    logger.error(f"❌ SGLang 프록시 예상치 못한 오류: {exc}")

    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "SGLang 프록시에서 내부 오류가 발생했습니다",
                "type": "sglang_proxy_error",
                "details": str(exc) if config.debug else None,
                "framework": "SGLang"
            }
        }
    )


# 프록시 상태 체크 엔드포인트
@router.get("/health")
async def proxy_health_check():
    """프록시 헬스 체크"""
    try:
        sglang_client = get_sglang_client(config.sglang_server_url)

        # SGLang 서버 상태 확인
        sglang_healthy = await sglang_client.health_check()

        # 성능 모니터 상태 확인
        monitor_active = performance_monitor.monitoring

        # Rate Limiter 상태 확인 (간단한 테스트)
        test_allowed, _ = await rate_limiter.check_limit("health_check", 1)

        return {
            "status": "healthy" if all([sglang_healthy, monitor_active]) else "degraded",
            "framework": "SGLang",
            "components": {
                "sglang_server": "healthy" if sglang_healthy else "unhealthy",
                "performance_monitor": "active" if monitor_active else "inactive",
                "rate_limiter": "healthy" if test_allowed else "limited",
                "korean_optimizer": "active"
            },
            "timestamp": time.time(),
            "version": "1.0.0"
        }

    except Exception as e:
        logger.error(f"❌ 프록시 헬스 체크 실패: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "framework": "SGLang",
                "error": str(e),
                "timestamp": time.time()
            }
        )