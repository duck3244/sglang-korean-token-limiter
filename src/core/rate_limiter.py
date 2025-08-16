"""
SGLang 특화 한국어 토큰 사용량 기반 속도 제한기
"""

import time
import asyncio
from typing import Dict, Optional, Tuple, List, Any
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import json
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class LimitType(Enum):
    """제한 타입"""
    REQUESTS_PER_MINUTE = "rpm"
    TOKENS_PER_MINUTE = "tpm"
    TOKENS_PER_HOUR = "tph"
    DAILY_TOKEN_LIMIT = "daily"


@dataclass
class UserLimits:
    """사용자별 제한 설정 (SGLang 성능 향상 반영)"""
    rpm: int = 40           # SGLang 성능 향상으로 기본값 증가 (vLLM: 30)
    tpm: int = 8000         # 분당 토큰 수 증가 (vLLM: 5000)
    tph: int = 500000       # 시간당 토큰 수 증가 (vLLM: 300000)
    daily: int = 1000000    # 일일 토큰 수 증가 (vLLM: 500000)
    cooldown_minutes: int = 2  # 쿨다운 시간 단축 (vLLM: 3, SGLang 빠른 처리)
    description: str = ""   # 사용자 설명

    def to_dict(self):
        return asdict(self)


@dataclass
class UsageInfo:
    """사용량 정보"""
    requests_this_minute: int = 0
    tokens_this_minute: int = 0
    tokens_this_hour: int = 0
    tokens_today: int = 0
    total_requests: int = 0
    total_tokens: int = 0
    last_request_time: float = 0
    cooldown_until: float = 0

    # SGLang 특화 메트릭
    avg_response_time: float = 0.0
    cache_hit_count: int = 0
    stream_requests: int = 0


class SGLangRateLimiter:
    """SGLang 특화 속도 제한기"""

    def __init__(self, storage=None):
        self.storage = storage
        self.user_limits: Dict[str, UserLimits] = {}

        # SGLang 고성능을 반영한 기본 제한 (더 관대함)
        self.default_limits = UserLimits(
            rpm=40,         # vLLM 30 → SGLang 40 (33% 증가)
            tpm=8000,       # vLLM 5000 → SGLang 8000 (60% 증가)
            tph=500000,     # vLLM 300000 → SGLang 500000 (67% 증가)
            daily=1000000,  # vLLM 500000 → SGLang 1000000 (100% 증가)
            cooldown_minutes=2  # vLLM 3분 → SGLang 2분 (빠른 복구)
        )

        self.api_key_mapping: Dict[str, str] = {}  # API 키 → 사용자 ID 매핑
        self.user_display_names: Dict[str, str] = {}  # 영어 → 한국어 매핑

        # SGLang 성능 메트릭
        self.performance_metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "rate_limited_requests": 0,
            "avg_processing_time": 0.0,
            "cache_hits": 0,
            "stream_requests": 0
        }

    def set_user_limits(self, user_id: str, limits: UserLimits):
        """사용자별 제한 설정"""
        self.user_limits[user_id] = limits
        logger.info(f"✅ SGLang 사용자 제한 설정 '{user_id}': RPM={limits.rpm}, TPM={limits.tpm}")

    def set_api_key_mapping(self, api_key: str, user_id: str, display_name: str = None):
        """API 키와 사용자 ID 매핑 설정"""
        self.api_key_mapping[api_key] = user_id
        if display_name:
            self.user_display_names[user_id] = display_name
        logger.debug(f"✅ API 키 매핑: {api_key[:8]}... → {user_id}")

    def get_user_from_api_key(self, api_key: str) -> str:
        """API 키로부터 사용자 ID 조회"""
        return self.api_key_mapping.get(api_key, "guest")

    def get_display_name(self, user_id: str) -> str:
        """사용자 표시명 조회"""
        return self.user_display_names.get(user_id, user_id)

    def get_user_limits(self, user_id: str) -> UserLimits:
        """사용자 제한 설정 조회"""
        return self.user_limits.get(user_id, self.default_limits)

    async def check_limit(self, user_id: str, estimated_tokens: int,
                         request_type: str = "chat", is_stream: bool = False) -> Tuple[bool, Optional[str]]:
        """사용량 제한 확인 (SGLang 특화 기능 포함)"""
        try:
            limits = self.get_user_limits(user_id)
            current_time = time.time()

            # 성능 메트릭 업데이트
            self.performance_metrics["total_requests"] += 1
            if is_stream:
                self.performance_metrics["stream_requests"] += 1

            # 현재 사용량 조회
            if self.storage:
                usage = await self.storage.get_user_usage(user_id)
            else:
                usage = self._get_memory_usage(user_id)

            # 쿨다운 상태 확인
            cooldown_until = usage.get('cooldown_until', 0)
            if cooldown_until > current_time:
                remaining_cooldown = int(cooldown_until - current_time)
                self.performance_metrics["rate_limited_requests"] += 1
                return False, f"🚫 쿨다운 중입니다. {remaining_cooldown}초 후 다시 시도하세요. (SGLang 빠른 복구)"

            # 분당 요청 수 확인
            current_requests = usage.get('requests_this_minute', 0)
            if current_requests >= limits.rpm:
                await self._apply_cooldown(user_id, limits.cooldown_minutes)
                self.performance_metrics["rate_limited_requests"] += 1
                return False, f"⏰ 분당 요청 제한 초과 ({limits.rpm}개). SGLang 고성능으로 {limits.cooldown_minutes}분 후 재시도 가능."

            # 분당 토큰 수 확인
            current_minute_tokens = usage.get('tokens_this_minute', 0)
            if current_minute_tokens + estimated_tokens > limits.tpm:
                await self._apply_cooldown(user_id, limits.cooldown_minutes)
                self.performance_metrics["rate_limited_requests"] += 1
                return False, f"🔢 분당 토큰 제한 초과 ({limits.tpm:,}개). SGLang 최적화로 현재: {current_minute_tokens:,}, 요청: {estimated_tokens:,}"

            # 시간당 토큰 수 확인
            current_hour_tokens = usage.get('tokens_this_hour', 0)
            if current_hour_tokens + estimated_tokens > limits.tph:
                await self._apply_cooldown(user_id, limits.cooldown_minutes)
                self.performance_metrics["rate_limited_requests"] += 1
                return False, f"⏳ 시간당 토큰 제한 초과 ({limits.tph:,}개). SGLang 처리량: 현재 {current_hour_tokens:,}, 요청: {estimated_tokens:,}"

            # 일일 토큰 수 확인
            current_daily_tokens = usage.get('tokens_today', 0)
            if current_daily_tokens + estimated_tokens > limits.daily:
                await self._apply_cooldown(user_id, limits.cooldown_minutes * 2)  # 일일 제한은 더 긴 쿨다운
                self.performance_metrics["rate_limited_requests"] += 1
                return False, f"📅 일일 토큰 제한 초과 ({limits.daily:,}개). SGLang 효율성: 현재 {current_daily_tokens:,}, 요청: {estimated_tokens:,}"

            # SGLang 동적 배치 고려한 추가 검증
            if hasattr(self, '_check_sglang_capacity'):
                capacity_ok, capacity_msg = await self._check_sglang_capacity(user_id, estimated_tokens)
                if not capacity_ok:
                    return False, capacity_msg

            self.performance_metrics["successful_requests"] += 1
            return True, None

        except Exception as e:
            logger.error(f"❌ SGLang rate limit check failed for user {user_id}: {e}")
            # 에러 시 허용 (fail-open 정책)
            return True, None

    async def _apply_cooldown(self, user_id: str, cooldown_minutes: int):
        """쿨다운 적용 (SGLang 빠른 복구 특성 반영)"""
        cooldown_until = time.time() + (cooldown_minutes * 60)

        if self.storage:
            await self.storage.set_user_cooldown(user_id, cooldown_until)
        else:
            self._set_memory_cooldown(user_id, cooldown_until)

        logger.warning(f"⚠️ SGLang 쿨다운 적용: '{user_id}' - {cooldown_minutes}분 (빠른 복구 모드)")

    async def record_usage(self, user_id: str, input_tokens: int, output_tokens: int,
                          requests: int = 1, response_time: float = 0.0,
                          cache_hit: bool = False, is_stream: bool = False):
        """사용량 기록 (SGLang 성능 메트릭 포함)"""
        try:
            total_tokens = input_tokens + output_tokens

            if self.storage:
                await self.storage.record_usage(user_id, total_tokens, requests)

                # SGLang 특화 메트릭 기록
                await self._record_sglang_metrics(user_id, response_time, cache_hit, is_stream)
            else:
                self._record_memory_usage(user_id, total_tokens, requests, response_time, cache_hit, is_stream)

            # 전역 성능 메트릭 업데이트
            if cache_hit:
                self.performance_metrics["cache_hits"] += 1

            # 평균 처리 시간 업데이트
            if response_time > 0:
                current_avg = self.performance_metrics["avg_processing_time"]
                total_requests = self.performance_metrics["successful_requests"]
                self.performance_metrics["avg_processing_time"] = (
                    (current_avg * (total_requests - 1) + response_time) / total_requests
                )

            logger.debug(f"📊 SGLang 사용량 기록: '{user_id}' -> {total_tokens} tokens, 응답시간: {response_time:.2f}s")

        except Exception as e:
            logger.error(f"❌ SGLang usage recording failed for user {user_id}: {e}")

    async def _record_sglang_metrics(self, user_id: str, response_time: float,
                                   cache_hit: bool, is_stream: bool):
        """SGLang 특화 메트릭 기록"""
        try:
            metrics = {
                "response_time": response_time,
                "cache_hit": cache_hit,
                "is_stream": is_stream,
                "timestamp": time.time()
            }

            if self.storage and hasattr(self.storage, 'record_sglang_metrics'):
                await self.storage.record_sglang_metrics(user_id, metrics)

        except Exception as e:
            logger.debug(f"SGLang 메트릭 기록 실패: {e}")

    async def get_user_status(self, user_id: str) -> Dict:
        """사용자 상태 조회 (SGLang 성능 정보 포함)"""
        try:
            if self.storage:
                usage = await self.storage.get_user_usage(user_id)
            else:
                usage = self._get_memory_usage(user_id)

            limits = self.get_user_limits(user_id)
            current_time = time.time()

            # 남은 할당량 계산
            remaining_rpm = max(0, limits.rpm - usage.get('requests_this_minute', 0))
            remaining_tpm = max(0, limits.tpm - usage.get('tokens_this_minute', 0))
            remaining_tph = max(0, limits.tph - usage.get('tokens_this_hour', 0))
            remaining_daily = max(0, limits.daily - usage.get('tokens_today', 0))

            # 쿨다운 상태
            cooldown_until = usage.get('cooldown_until', 0)
            is_cooldown = cooldown_until > current_time
            cooldown_remaining = max(0, int(cooldown_until - current_time)) if is_cooldown else 0

            # 사용률 계산
            rpm_percent = (usage.get('requests_this_minute', 0) / limits.rpm) * 100 if limits.rpm > 0 else 0
            tpm_percent = (usage.get('tokens_this_minute', 0) / limits.tpm) * 100 if limits.tpm > 0 else 0
            tph_percent = (usage.get('tokens_this_hour', 0) / limits.tph) * 100 if limits.tph > 0 else 0
            daily_percent = (usage.get('tokens_today', 0) / limits.daily) * 100 if limits.daily > 0 else 0

            # SGLang 특화 메트릭
            sglang_metrics = await self._get_sglang_user_metrics(user_id)

            return {
                'user_id': user_id,
                'display_name': self.get_display_name(user_id),
                'user_type': 'sglang_korean_user',
                'framework': 'SGLang',
                'limits': limits.to_dict(),
                'usage': usage,
                'remaining': {
                    'requests_this_minute': remaining_rpm,
                    'tokens_this_minute': remaining_tpm,
                    'tokens_this_hour': remaining_tph,
                    'tokens_today': remaining_daily
                },
                'cooldown': {
                    'is_active': is_cooldown,
                    'remaining_seconds': cooldown_remaining,
                    'status_message': f"SGLang 쿨다운 {cooldown_remaining}초 남음" if is_cooldown else "정상"
                },
                'utilization': {
                    'rpm_percent': round(rpm_percent, 1),
                    'tpm_percent': round(tpm_percent, 1),
                    'tph_percent': round(tph_percent, 1),
                    'daily_percent': round(daily_percent, 1)
                },
                'sglang_metrics': sglang_metrics,
                'status_summary': self._get_status_summary(rpm_percent, tpm_percent, tph_percent, daily_percent, is_cooldown),
                'performance_tier': self._get_performance_tier(limits)
            }

        except Exception as e:
            logger.error(f"❌ SGLang user status retrieval failed for {user_id}: {e}")
            return {
                'user_id': user_id,
                'error': f"SGLang 통계 조회 실패: {str(e)}",
                'framework': 'SGLang'
            }

    async def _get_sglang_user_metrics(self, user_id: str) -> Dict:
        """SGLang 사용자별 성능 메트릭 조회"""
        try:
            if self.storage and hasattr(self.storage, 'get_sglang_user_metrics'):
                return await self.storage.get_sglang_user_metrics(user_id)
            else:
                # 기본 메트릭
                return {
                    "avg_response_time": 0.0,
                    "cache_hit_rate": 0.0,
                    "stream_usage_rate": 0.0,
                    "total_cache_hits": 0,
                    "total_stream_requests": 0
                }
        except Exception:
            return {}

    def _get_status_summary(self, rpm_percent: float, tpm_percent: float,
                          tph_percent: float, daily_percent: float, is_cooldown: bool) -> str:
        """상태 요약 메시지 생성 (SGLang 특화)"""
        if is_cooldown:
            return "🚫 SGLang 쿨다운 중 (빠른 복구)"

        max_usage = max(rpm_percent, tpm_percent, tph_percent, daily_percent)

        if max_usage >= 90:
            return "🔴 위험 (90% 이상) - SGLang 성능 한계"
        elif max_usage >= 70:
            return "🟡 주의 (70% 이상) - SGLang 최적화 권장"
        elif max_usage >= 50:
            return "🟢 보통 (50% 이상) - SGLang 안정적"
        else:
            return "🔵 여유 (50% 미만) - SGLang 고성능 활용"

    def _get_performance_tier(self, limits: UserLimits) -> str:
        """성능 등급 결정 (SGLang 기준)"""
        if limits.tpm >= 15000:
            return "🚀 SGLang Premium"
        elif limits.tpm >= 8000:
            return "⚡ SGLang Pro"
        elif limits.tpm >= 5000:
            return "💼 SGLang Standard"
        else:
            return "🥉 SGLang Basic"

    async def reset_user_usage(self, user_id: str):
        """사용자 사용량 초기화"""
        try:
            if self.storage:
                await self.storage.reset_user_usage(user_id)
            else:
                self._reset_memory_usage(user_id)
            logger.info(f"🔄 SGLang 사용량 초기화: '{user_id}'")
        except Exception as e:
            logger.error(f"❌ SGLang usage reset failed for user {user_id}: {e}")
            raise

    def set_default_limits(self, limits: UserLimits):
        """기본 제한 설정 (SGLang 최적화)"""
        self.default_limits = limits
        logger.info(f"✅ SGLang 기본 제한 설정: {limits}")

    async def cleanup_expired_data(self):
        """만료된 데이터 정리"""
        try:
            if self.storage:
                await self.storage.cleanup_expired_data()
            logger.debug("🧹 SGLang 만료 데이터 정리 완료")
        except Exception as e:
            logger.error(f"❌ SGLang data cleanup failed: {e}")

    async def get_top_users(self, limit: int = 10, period: str = "today") -> List[Dict]:
        """상위 사용자 조회 (SGLang 성능 기준)"""
        try:
            if self.storage:
                return await self.storage.get_top_users(limit, period)
            else:
                return []
        except Exception as e:
            logger.error(f"❌ SGLang top users retrieval failed: {e}")
            return []

    async def get_usage_statistics(self) -> Dict:
        """전체 사용량 통계 (SGLang 성능 정보 포함)"""
        try:
            if self.storage:
                stats = await self.storage.get_usage_statistics()
            else:
                stats = self._get_memory_statistics()

            # SGLang 성능 메트릭 추가
            stats.update({
                'framework': 'SGLang',
                'performance_metrics': self.performance_metrics.copy(),
                'labels': {
                    'total_users': '총 사용자 수',
                    'active_users_today': '오늘 활성 사용자',
                    'total_tokens_today': '오늘 총 토큰 사용량 (SGLang 처리)',
                    'total_requests_today': '오늘 총 요청 수',
                    'avg_response_time': 'SGLang 평균 응답 시간',
                    'cache_hit_rate': 'SGLang 캐시 히트율'
                },
                'sglang_efficiency': {
                    'cache_hit_rate': self._calculate_cache_hit_rate(),
                    'avg_response_time': self.performance_metrics['avg_processing_time'],
                    'stream_usage_rate': self._calculate_stream_usage_rate(),
                    'success_rate': self._calculate_success_rate()
                }
            })

            return stats
        except Exception as e:
            logger.error(f"❌ SGLang usage statistics retrieval failed: {e}")
            return {'framework': 'SGLang', 'error': str(e)}

    def _calculate_cache_hit_rate(self) -> float:
        """캐시 히트율 계산"""
        total = self.performance_metrics['total_requests']
        if total == 0:
            return 0.0
        return (self.performance_metrics['cache_hits'] / total) * 100

    def _calculate_stream_usage_rate(self) -> float:
        """스트리밍 사용률 계산"""
        total = self.performance_metrics['total_requests']
        if total == 0:
            return 0.0
        return (self.performance_metrics['stream_requests'] / total) * 100

    def _calculate_success_rate(self) -> float:
        """성공률 계산"""
        total = self.performance_metrics['total_requests']
        if total == 0:
            return 100.0
        return (self.performance_metrics['successful_requests'] / total) * 100

    # 메모리 기반 저장소 (Storage가 없는 경우)
    def __init_memory_storage(self):
        """메모리 기반 저장소 초기화"""
        if not hasattr(self, '_memory_storage'):
            self._memory_storage = {}

    def _get_memory_usage(self, user_id: str) -> Dict:
        """메모리에서 사용량 조회"""
        self.__init_memory_storage()
        return self._memory_storage.get(user_id, {
            'requests_this_minute': 0,
            'tokens_this_minute': 0,
            'tokens_this_hour': 0,
            'tokens_today': 0,
            'total_requests': 0,
            'total_tokens': 0,
            'last_request_time': 0,
            'cooldown_until': 0
        })

    def _record_memory_usage(self, user_id: str, tokens: int, requests: int,
                           response_time: float, cache_hit: bool, is_stream: bool):
        """메모리에 사용량 기록"""
        self.__init_memory_storage()
        now = time.time()

        if user_id not in self._memory_storage:
            self._memory_storage[user_id] = self._get_memory_usage(user_id)

        user_data = self._memory_storage[user_id]

        # 시간대별 데이터 정리 (간단한 구현)
        minute_ago = now - 60
        hour_ago = now - 3600
        day_ago = now - 86400

        if user_data.get('last_request_time', 0) < minute_ago:
            user_data['requests_this_minute'] = 0
            user_data['tokens_this_minute'] = 0

        if user_data.get('last_request_time', 0) < hour_ago:
            user_data['tokens_this_hour'] = 0

        if user_data.get('last_request_time', 0) < day_ago:
            user_data['tokens_today'] = 0

        # 사용량 업데이트
        user_data['requests_this_minute'] += requests
        user_data['tokens_this_minute'] += tokens
        user_data['tokens_this_hour'] += tokens
        user_data['tokens_today'] += tokens
        user_data['total_requests'] += requests
        user_data['total_tokens'] += tokens
        user_data['last_request_time'] = now

    def _set_memory_cooldown(self, user_id: str, cooldown_until: float):
        """메모리에 쿨다운 설정"""
        self.__init_memory_storage()
        if user_id not in self._memory_storage:
            self._memory_storage[user_id] = self._get_memory_usage(user_id)
        self._memory_storage[user_id]['cooldown_until'] = cooldown_until

    def _reset_memory_usage(self, user_id: str):
        """메모리 사용량 초기화"""
        self.__init_memory_storage()
        if user_id in self._memory_storage:
            del self._memory_storage[user_id]

    def _get_memory_statistics(self) -> Dict:
        """메모리 기반 통계"""
        self.__init_memory_storage()

        total_users = len(self._memory_storage)
        active_users = sum(1 for data in self._memory_storage.values()
                          if data.get('tokens_today', 0) > 0)
        total_tokens = sum(data.get('total_tokens', 0)
                          for data in self._memory_storage.values())
        total_requests = sum(data.get('total_requests', 0)
                           for data in self._memory_storage.values())

        return {
            'total_users': total_users,
            'active_users_today': active_users,
            'total_tokens_today': total_tokens,
            'total_requests_today': total_requests,
            'average_tokens_per_user': total_tokens / max(active_users, 1),
            'timestamp': time.time(),
            'system_type': 'sglang_korean_limiter'
        }

    async def bulk_update_limits(self, user_limits_dict: Dict[str, UserLimits]):
        """여러 사용자 제한 일괄 업데이트"""
        updated_count = 0
        for user_id, limits in user_limits_dict.items():
            try:
                self.set_user_limits(user_id, limits)
                updated_count += 1
            except Exception as e:
                logger.error(f"❌ Failed to update SGLang limits for {user_id}: {e}")

        logger.info(f"✅ SGLang 일괄 제한 업데이트: {updated_count}명")
        return updated_count


# 호환성을 위한 별칭
KoreanRateLimiter = SGLangRateLimiter
RateLimiter = SGLangRateLimiter