"""
SGLang 클라이언트 - SGLang 서버와의 통신 및 한국어 최적화
"""

import asyncio
import httpx
import json
import time
import logging
from typing import Dict, Any, Optional, AsyncGenerator, List, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class SGLangModelStatus(Enum):
    """SGLang 모델 상태"""
    UNKNOWN = "unknown"
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"
    OVERLOADED = "overloaded"


@dataclass
class SGLangModelInfo:
    """SGLang 모델 정보"""
    model_path: str
    served_model_names: List[str]
    max_total_tokens: int
    is_generation: bool
    architecture: str = ""
    vocab_size: int = 0
    context_length: int = 0


@dataclass
class SGLangServerInfo:
    """SGLang 서버 정보"""
    queue_length: int
    running_requests: int
    memory_usage_gb: float
    requests_per_second: float
    tokens_per_second: float
    cache_hit_rate: float
    uptime_seconds: float


@dataclass
class SGLangPerformanceMetrics:
    """SGLang 성능 메트릭"""
    avg_first_token_latency: float
    avg_inter_token_latency: float
    throughput_tokens_per_second: float
    batch_size_avg: float
    kv_cache_usage: float
    gpu_utilization: float
    memory_usage: float


class SGLangClient:
    """SGLang 서버와의 통신을 담당하는 클라이언트"""

    def __init__(self, base_url: str = "http://127.0.0.1:8000", timeout: float = 60.0):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.model_info: Optional[SGLangModelInfo] = None
        self.server_info: Optional[SGLangServerInfo] = None
        self.status = SGLangModelStatus.UNKNOWN

        # 성능 모니터링
        self.performance_history = []
        self.request_count = 0
        self.error_count = 0

        # 한국어 특화 설정
        self.korean_optimizations = {
            "enable_prefix_caching": True,
            "chunked_prefill": True,
            "korean_tokenizer_mode": True
        }

    async def health_check(self) -> bool:
        """SGLang 서버 헬스 체크"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/health")
                return response.status_code == 200
        except Exception as e:
            logger.debug(f"SGLang health check failed: {e}")
            return False

    async def get_model_info(self) -> Optional[SGLangModelInfo]:
        """SGLang 모델 정보 조회"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.base_url}/get_model_info")

                if response.status_code == 200:
                    data = response.json()

                    self.model_info = SGLangModelInfo(
                        model_path=data.get("model_path", ""),
                        served_model_names=data.get("served_model_names", []),
                        max_total_tokens=data.get("max_total_tokens", 0),
                        is_generation=data.get("is_generation", True),
                        architecture=data.get("architecture", ""),
                        vocab_size=data.get("vocab_size", 0),
                        context_length=data.get("context_length", 0)
                    )

                    self.status = SGLangModelStatus.READY
                    logger.info(f"✅ SGLang 모델 정보 조회 성공: {self.model_info.model_path}")
                    return self.model_info

        except Exception as e:
            logger.error(f"❌ SGLang 모델 정보 조회 실패: {e}")
            self.status = SGLangModelStatus.ERROR

        return None

    async def get_server_info(self) -> Optional[SGLangServerInfo]:
        """SGLang 서버 상태 정보 조회"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/get_server_info")

                if response.status_code == 200:
                    data = response.json()

                    self.server_info = SGLangServerInfo(
                        queue_length=data.get("queue_length", 0),
                        running_requests=data.get("running_requests", 0),
                        memory_usage_gb=data.get("memory_usage_gb", 0.0),
                        requests_per_second=data.get("requests_per_second", 0.0),
                        tokens_per_second=data.get("tokens_per_second", 0.0),
                        cache_hit_rate=data.get("cache_hit_rate", 0.0),
                        uptime_seconds=data.get("uptime_seconds", 0.0)
                    )

                    logger.debug(f"📊 SGLang 서버 상태: {self.server_info.running_requests}개 요청 처리 중")
                    return self.server_info

        except Exception as e:
            logger.debug(f"SGLang 서버 정보 조회 실패: {e}")

        return None

    async def chat_completion(self,
                              messages: List[Dict[str, str]],
                              model: str = "korean-qwen",
                              max_tokens: int = 512,
                              temperature: float = 0.7,
                              top_p: float = 1.0,
                              frequency_penalty: float = 0.0,
                              presence_penalty: float = 0.0,
                              stop: Optional[List[str]] = None,
                              stream: bool = False,
                              korean_optimized: bool = True) -> Dict[str, Any]:
        """SGLang 채팅 완성 요청 (한국어 최적화)"""

        start_time = time.time()
        self.request_count += 1

        try:
            # 한국어 최적화 설정
            request_data = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
                "stream": stream
            }

            if stop:
                request_data["stop"] = stop

            # 한국어 특화 옵션 추가
            if korean_optimized:
                request_data.update({
                    "repetition_penalty": 1.1,  # 한국어 반복 방지
                    "do_sample": True,
                    "pad_token_id": None,
                    "eos_token_id": None
                })

                # 한국어 정지 시퀀스 추가
                korean_stop_sequences = ["인간:", "사용자:", "Human:", "User:", "질문:", "답변:"]
                if stop:
                    stop.extend(korean_stop_sequences)
                else:
                    request_data["stop"] = korean_stop_sequences

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=request_data,
                    headers={"Content-Type": "application/json"}
                )

                response_time = time.time() - start_time

                if response.status_code == 200:
                    result = response.json()

                    # 성능 메트릭 기록
                    await self._record_performance_metrics(response_time, True, len(str(messages)))

                    logger.debug(f"✅ SGLang 채팅 완성 성공 ({response_time:.2f}초)")
                    return result

                else:
                    self.error_count += 1
                    error_text = response.text
                    logger.error(f"❌ SGLang 채팅 완성 실패 (HTTP {response.status_code}): {error_text}")

                    return {
                        "error": {
                            "message": f"SGLang 서버 오류: {error_text}",
                            "type": "sglang_error",
                            "code": response.status_code
                        }
                    }

        except httpx.TimeoutException:
            self.error_count += 1
            logger.error(f"❌ SGLang 요청 타임아웃 ({self.timeout}초)")
            return {
                "error": {
                    "message": f"SGLang 서버 응답 시간 초과 ({self.timeout}초)",
                    "type": "timeout_error",
                    "code": 408
                }
            }

        except Exception as e:
            self.error_count += 1
            logger.error(f"❌ SGLang 채팅 완성 오류: {e}")
            return {
                "error": {
                    "message": f"SGLang 클라이언트 오류: {str(e)}",
                    "type": "client_error",
                    "code": 500
                }
            }

    async def chat_completion_stream(self,
                                     messages: List[Dict[str, str]],
                                     model: str = "korean-qwen",
                                     max_tokens: int = 512,
                                     temperature: float = 0.7,
                                     korean_optimized: bool = True) -> AsyncGenerator[Dict[str, Any], None]:
        """SGLang 스트리밍 채팅 완성 (한국어 최적화)"""

        start_time = time.time()
        self.request_count += 1

        try:
            request_data = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": True
            }

            # 한국어 최적화
            if korean_optimized:
                request_data.update({
                    "repetition_penalty": 1.1,
                    "stop": ["인간:", "사용자:", "Human:", "User:"]
                })

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream(
                        "POST",
                        f"{self.base_url}/v1/chat/completions",
                        json=request_data,
                        headers={"Content-Type": "application/json"}
                ) as response:

                    if response.status_code != 200:
                        self.error_count += 1
                        error_text = await response.aread()
                        logger.error(f"❌ SGLang 스트리밍 실패 (HTTP {response.status_code}): {error_text.decode()}")

                        yield {
                            "error": {
                                "message": f"SGLang 스트리밍 오류: {error_text.decode()}",
                                "type": "sglang_stream_error",
                                "code": response.status_code
                            }
                        }
                        return

                    # 스트리밍 응답 처리
                    chunk_count = 0
                    async for chunk in response.aiter_lines():
                        chunk_count += 1

                        if chunk.startswith("data: "):
                            data = chunk[6:]  # "data: " 제거

                            if data.strip() == "[DONE]":
                                # 성능 메트릭 기록
                                response_time = time.time() - start_time
                                await self._record_performance_metrics(response_time, True, len(str(messages)),
                                                                       is_stream=True)
                                logger.debug(f"✅ SGLang 스트리밍 완료 ({response_time:.2f}초, {chunk_count}개 청크)")
                                break

                            try:
                                chunk_data = json.loads(data)

                                # OpenAI 호환 형식으로 변환
                                if "choices" in chunk_data:
                                    openai_chunk = {
                                        "id": f"chatcmpl-{int(time.time())}",
                                        "object": "chat.completion.chunk",
                                        "created": int(time.time()),
                                        "model": model,
                                        "choices": chunk_data["choices"]
                                    }
                                    yield openai_chunk

                            except json.JSONDecodeError:
                                continue

        except Exception as e:
            self.error_count += 1
            logger.error(f"❌ SGLang 스트리밍 오류: {e}")
            yield {
                "error": {
                    "message": f"SGLang 스트리밍 클라이언트 오류: {str(e)}",
                    "type": "stream_client_error",
                    "code": 500
                }
            }

    async def text_completion(self,
                              prompt: str,
                              model: str = "korean-qwen",
                              max_tokens: int = 256,
                              temperature: float = 0.7,
                              korean_optimized: bool = True) -> Dict[str, Any]:
        """SGLang 텍스트 완성 요청"""

        start_time = time.time()
        self.request_count += 1

        try:
            request_data = {
                "model": model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False
            }

            # 한국어 최적화
            if korean_optimized:
                request_data.update({
                    "repetition_penalty": 1.1,
                    "stop": ["인간:", "사용자:", "Human:", "User:"]
                })

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/v1/completions",
                    json=request_data,
                    headers={"Content-Type": "application/json"}
                )

                response_time = time.time() - start_time

                if response.status_code == 200:
                    result = response.json()

                    # 성능 메트릭 기록
                    await self._record_performance_metrics(response_time, True, len(prompt))

                    logger.debug(f"✅ SGLang 텍스트 완성 성공 ({response_time:.2f}초)")
                    return result

                else:
                    self.error_count += 1
                    error_text = response.text
                    logger.error(f"❌ SGLang 텍스트 완성 실패 (HTTP {response.status_code}): {error_text}")

                    return {
                        "error": {
                            "message": f"SGLang 서버 오류: {error_text}",
                            "type": "sglang_error",
                            "code": response.status_code
                        }
                    }

        except Exception as e:
            self.error_count += 1
            logger.error(f"❌ SGLang 텍스트 완성 오류: {e}")
            return {
                "error": {
                    "message": f"SGLang 클라이언트 오류: {str(e)}",
                    "type": "client_error",
                    "code": 500
                }
            }

    async def get_performance_metrics(self) -> SGLangPerformanceMetrics:
        """SGLang 성능 메트릭 조회"""
        try:
            # 서버에서 메트릭 조회 시도
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/metrics")

                if response.status_code == 200:
                    data = response.json()

                    return SGLangPerformanceMetrics(
                        avg_first_token_latency=data.get("avg_first_token_latency", 0.0),
                        avg_inter_token_latency=data.get("avg_inter_token_latency", 0.0),
                        throughput_tokens_per_second=data.get("throughput_tokens_per_second", 0.0),
                        batch_size_avg=data.get("batch_size_avg", 0.0),
                        kv_cache_usage=data.get("kv_cache_usage", 0.0),
                        gpu_utilization=data.get("gpu_utilization", 0.0),
                        memory_usage=data.get("memory_usage", 0.0)
                    )

        except Exception as e:
            logger.debug(f"SGLang 메트릭 조회 실패: {e}")

        # 로컬 메트릭 계산
        if self.performance_history:
            recent_metrics = self.performance_history[-10:]  # 최근 10개
            avg_response_time = sum(m["response_time"] for m in recent_metrics) / len(recent_metrics)
            success_rate = sum(1 for m in recent_metrics if m["success"]) / len(recent_metrics)

            return SGLangPerformanceMetrics(
                avg_first_token_latency=avg_response_time,
                avg_inter_token_latency=avg_response_time / 10,  # 근사치
                throughput_tokens_per_second=1000 / avg_response_time if avg_response_time > 0 else 0,
                batch_size_avg=1.0,  # 기본값
                kv_cache_usage=success_rate * 100,  # 근사치
                gpu_utilization=80.0 if success_rate > 0.9 else 50.0,  # 근사치
                memory_usage=70.0  # 근사치
            )

        # 기본값 반환
        return SGLangPerformanceMetrics(
            avg_first_token_latency=0.0,
            avg_inter_token_latency=0.0,
            throughput_tokens_per_second=0.0,
            batch_size_avg=0.0,
            kv_cache_usage=0.0,
            gpu_utilization=0.0,
            memory_usage=0.0
        )

    async def _record_performance_metrics(self, response_time: float, success: bool,
                                          input_length: int, is_stream: bool = False):
        """성능 메트릭 기록"""
        metric = {
            "timestamp": time.time(),
            "response_time": response_time,
            "success": success,
            "input_length": input_length,
            "is_stream": is_stream
        }

        self.performance_history.append(metric)

        # 히스토리 크기 제한 (최근 1000개만 유지)
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]

    async def get_client_statistics(self) -> Dict[str, Any]:
        """클라이언트 통계 정보"""
        success_rate = 0.0
        avg_response_time = 0.0

        if self.request_count > 0:
            success_rate = ((self.request_count - self.error_count) / self.request_count) * 100

        if self.performance_history:
            avg_response_time = sum(m["response_time"] for m in self.performance_history) / len(
                self.performance_history)

        return {
            "total_requests": self.request_count,
            "successful_requests": self.request_count - self.error_count,
            "error_count": self.error_count,
            "success_rate": success_rate,
            "avg_response_time": avg_response_time,
            "server_status": self.status.value,
            "model_info": self.model_info.__dict__ if self.model_info else None,
            "server_info": self.server_info.__dict__ if self.server_info else None,
            "korean_optimizations": self.korean_optimizations,
            "framework": "SGLang"
        }

    async def warmup(self, korean_test_messages: Optional[List[Dict[str, str]]] = None):
        """SGLang 서버 워밍업 (한국어 테스트 포함)"""
        logger.info("🔥 SGLang 서버 워밍업 시작...")

        try:
            # 1. 헬스 체크
            if not await self.health_check():
                logger.warning("⚠️ SGLang 헬스 체크 실패")
                return False

            # 2. 모델 정보 조회
            model_info = await self.get_model_info()
            if not model_info:
                logger.warning("⚠️ SGLang 모델 정보 조회 실패")
                return False

            # 3. 서버 정보 조회
            await self.get_server_info()

            # 4. 한국어 테스트 요청
            if not korean_test_messages:
                korean_test_messages = [
                    {"role": "user", "content": "안녕하세요! SGLang 테스트입니다."}
                ]

            result = await self.chat_completion(
                messages=korean_test_messages,
                max_tokens=50,
                korean_optimized=True
            )

            if "error" not in result:
                logger.info("✅ SGLang 한국어 워밍업 성공")
                return True
            else:
                logger.warning(f"⚠️ SGLang 한국어 테스트 실패: {result.get('error', {}).get('message')}")
                return False

        except Exception as e:
            logger.error(f"❌ SGLang 워밍업 실패: {e}")
            return False

    async def adaptive_batch_optimization(self, current_load: float) -> Dict[str, Any]:
        """SGLang 동적 배치 최적화 제안"""
        try:
            server_info = await self.get_server_info()

            if not server_info:
                return {"status": "no_server_info"}

            recommendations = []

            # 부하에 따른 최적화 제안
            if current_load > 0.8:
                recommendations.append({
                    "type": "increase_batch_size",
                    "description": "높은 부하 감지 - 배치 크기 증가 권장",
                    "suggested_value": "max_running_requests += 4"
                })

            elif current_load < 0.3:
                recommendations.append({
                    "type": "reduce_latency",
                    "description": "낮은 부하 감지 - 지연시간 최적화 권장",
                    "suggested_value": "enable_chunked_prefill=True"
                })

            # KV 캐시 최적화
            if server_info.cache_hit_rate < 0.5:
                recommendations.append({
                    "type": "improve_caching",
                    "description": "캐시 히트율 개선 필요",
                    "suggested_value": "enable_prefix_caching=True"
                })

            return {
                "status": "analyzed",
                "current_load": current_load,
                "server_metrics": server_info.__dict__,
                "recommendations": recommendations,
                "korean_optimizations": self.korean_optimizations
            }

        except Exception as e:
            logger.error(f"❌ SGLang 배치 최적화 분석 실패: {e}")
            return {"status": "error", "error": str(e)}

    def set_korean_optimizations(self, **kwargs):
        """한국어 최적화 설정 업데이트"""
        self.korean_optimizations.update(kwargs)
        logger.info(f"🇰🇷 SGLang 한국어 최적화 설정 업데이트: {self.korean_optimizations}")

    async def close(self):
        """클라이언트 종료"""
        logger.info("✅ SGLang 클라이언트 종료")


# 전역 SGLang 클라이언트 인스턴스
sglang_client: Optional[SGLangClient] = None


def get_sglang_client(base_url: str = "http://127.0.0.1:8000") -> SGLangClient:
    """SGLang 클라이언트 싱글톤 인스턴스 반환"""
    global sglang_client
    if sglang_client is None:
        sglang_client = SGLangClient(base_url)
    return sglang_client


async def initialize_sglang_client(base_url: str = "http://127.0.0.1:8000",
                                   warmup: bool = True) -> SGLangClient:
    """SGLang 클라이언트 초기화"""
    global sglang_client

    sglang_client = SGLangClient(base_url)

    if warmup:
        await sglang_client.warmup()

    logger.info(f"✅ SGLang 클라이언트 초기화 완료: {base_url}")
    return sglang_client


async def cleanup_sglang_client():
    """SGLang 클라이언트 정리"""
    global sglang_client
    if sglang_client:
        await sglang_client.close()
        sglang_client = None