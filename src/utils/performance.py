"""
SGLang 기반 한국어 Token Limiter 성능 최적화 유틸리티
"""

import time
import asyncio
import psutil
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import threading
from collections import deque

try:
    import pynvml

    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """시스템 메트릭"""
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    disk_usage_percent: float
    timestamp: float


@dataclass
class GPUMetrics:
    """GPU 메트릭"""
    gpu_id: int
    name: str
    memory_used_gb: float
    memory_total_gb: float
    memory_percent: float
    utilization_percent: float
    temperature: int
    power_draw_watts: float
    timestamp: float


@dataclass
class SGLangPerformanceMetrics:
    """SGLang 성능 메트릭"""
    requests_per_second: float
    tokens_per_second: float
    avg_response_time: float
    queue_length: int
    running_requests: int
    cache_hit_rate: float
    batch_size_avg: float
    memory_usage_gb: float
    error_rate: float
    timestamp: float


@dataclass
class KoreanTokenMetrics:
    """한국어 토큰 처리 메트릭"""
    korean_char_count: int
    total_char_count: int
    korean_ratio: float
    estimated_tokens: int
    actual_tokens: int
    tokenization_efficiency: float
    timestamp: float


class PerformanceMonitor:
    """SGLang 성능 모니터링 시스템"""

    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.system_metrics_history = deque(maxlen=history_size)
        self.gpu_metrics_history = deque(maxlen=history_size)
        self.sglang_metrics_history = deque(maxlen=history_size)
        self.korean_token_metrics_history = deque(maxlen=history_size)

        self.monitoring = False
        self.monitor_thread = None
        self.monitor_interval = 10  # 10초 간격

        # GPU 초기화
        self.gpu_available = self._init_gpu()

        # 성능 임계값 (SGLang 최적화 기준)
        self.thresholds = {
            "cpu_warning": 80.0,
            "cpu_critical": 95.0,
            "memory_warning": 85.0,
            "memory_critical": 95.0,
            "gpu_memory_warning": 90.0,
            "gpu_memory_critical": 98.0,
            "gpu_temp_warning": 80,
            "gpu_temp_critical": 90,
            "response_time_warning": 3.0,  # SGLang 빠른 응답
            "response_time_critical": 5.0,
            "cache_hit_rate_warning": 0.6,  # SGLang 캐시 효율
            "error_rate_warning": 0.05,  # 5% 에러율
            "error_rate_critical": 0.10  # 10% 에러율
        }

    def _init_gpu(self) -> bool:
        """GPU 모니터링 초기화"""
        if not PYNVML_AVAILABLE:
            logger.info("pynvml 없음 - GPU 모니터링 비활성화")
            return False

        try:
            pynvml.nvmlInit()
            gpu_count = pynvml.nvmlDeviceGetCount()
            logger.info(f"✅ GPU 모니터링 초기화: {gpu_count}개 GPU 감지")
            return True
        except Exception as e:
            logger.warning(f"⚠️ GPU 모니터링 초기화 실패: {e}")
            return False

    def start_monitoring(self):
        """성능 모니터링 시작"""
        if self.monitoring:
            logger.warning("성능 모니터링이 이미 실행 중입니다")
            return

        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info(f"🚀 SGLang 성능 모니터링 시작 (간격: {self.monitor_interval}초)")

    def stop_monitoring(self):
        """성능 모니터링 중지"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("⏹️ SGLang 성능 모니터링 중지")

    def _monitor_loop(self):
        """모니터링 루프"""
        while self.monitoring:
            try:
                # 시스템 메트릭 수집
                system_metrics = self.get_system_metrics()
                self.system_metrics_history.append(system_metrics)

                # GPU 메트릭 수집
                if self.gpu_available:
                    gpu_metrics = self.get_gpu_metrics()
                    if gpu_metrics:
                        self.gpu_metrics_history.extend(gpu_metrics)

                # 경고 상태 체크
                self._check_alerts(system_metrics)

                time.sleep(self.monitor_interval)

            except Exception as e:
                logger.error(f"❌ 모니터링 루프 오류: {e}")
                time.sleep(5)

    def get_system_metrics(self) -> SystemMetrics:
        """시스템 메트릭 수집"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            return SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_gb=memory.used / (1024 ** 3),
                memory_total_gb=memory.total / (1024 ** 3),
                disk_usage_percent=disk.percent,
                timestamp=time.time()
            )
        except Exception as e:
            logger.error(f"❌ 시스템 메트릭 수집 실패: {e}")
            return SystemMetrics(0, 0, 0, 0, 0, time.time())

    def get_gpu_metrics(self) -> List[GPUMetrics]:
        """GPU 메트릭 수집"""
        if not self.gpu_available:
            return []

        gpu_metrics = []
        try:
            gpu_count = pynvml.nvmlDeviceGetCount()

            for i in range(gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)

                # GPU 기본 정보
                name = pynvml.nvmlDeviceGetName(handle).decode()

                # 메모리 정보
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                memory_total_gb = memory_info.total / (1024 ** 3)
                memory_used_gb = memory_info.used / (1024 ** 3)
                memory_percent = (memory_info.used / memory_info.total) * 100

                # 사용률
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                utilization_percent = utilization.gpu

                # 온도
                temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)

                # 전력 사용량
                try:
                    power_draw = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W
                except:
                    power_draw = 0.0

                gpu_metrics.append(GPUMetrics(
                    gpu_id=i,
                    name=name,
                    memory_used_gb=memory_used_gb,
                    memory_total_gb=memory_total_gb,
                    memory_percent=memory_percent,
                    utilization_percent=utilization_percent,
                    temperature=temperature,
                    power_draw_watts=power_draw,
                    timestamp=time.time()
                ))

        except Exception as e:
            logger.error(f"❌ GPU 메트릭 수집 실패: {e}")

        return gpu_metrics

    def record_sglang_metrics(self, metrics: Dict[str, Any]):
        """SGLang 성능 메트릭 기록"""
        try:
            sglang_metrics = SGLangPerformanceMetrics(
                requests_per_second=metrics.get("requests_per_second", 0.0),
                tokens_per_second=metrics.get("tokens_per_second", 0.0),
                avg_response_time=metrics.get("avg_response_time", 0.0),
                queue_length=metrics.get("queue_length", 0),
                running_requests=metrics.get("running_requests", 0),
                cache_hit_rate=metrics.get("cache_hit_rate", 0.0),
                batch_size_avg=metrics.get("batch_size_avg", 0.0),
                memory_usage_gb=metrics.get("memory_usage_gb", 0.0),
                error_rate=metrics.get("error_rate", 0.0),
                timestamp=time.time()
            )

            self.sglang_metrics_history.append(sglang_metrics)
            logger.debug(
                f"📊 SGLang 메트릭 기록: RPS={sglang_metrics.requests_per_second:.1f}, TPS={sglang_metrics.tokens_per_second:.1f}")

        except Exception as e:
            logger.error(f"❌ SGLang 메트릭 기록 실패: {e}")

    def record_korean_token_metrics(self, korean_chars: int, total_chars: int,
                                    estimated_tokens: int, actual_tokens: Optional[int] = None):
        """한국어 토큰 메트릭 기록"""
        try:
            korean_ratio = korean_chars / total_chars if total_chars > 0 else 0

            if actual_tokens is not None:
                efficiency = actual_tokens / estimated_tokens if estimated_tokens > 0 else 1.0
            else:
                efficiency = 1.0
                actual_tokens = estimated_tokens

            korean_metrics = KoreanTokenMetrics(
                korean_char_count=korean_chars,
                total_char_count=total_chars,
                korean_ratio=korean_ratio,
                estimated_tokens=estimated_tokens,
                actual_tokens=actual_tokens,
                tokenization_efficiency=efficiency,
                timestamp=time.time()
            )

            self.korean_token_metrics_history.append(korean_metrics)
            logger.debug(f"🇰🇷 한국어 토큰 메트릭: {korean_chars}/{total_chars} 한글, 효율성: {efficiency:.2f}")

        except Exception as e:
            logger.error(f"❌ 한국어 토큰 메트릭 기록 실패: {e}")

    def _check_alerts(self, system_metrics: SystemMetrics):
        """경고 상태 체크"""
        alerts = []

        # CPU 경고
        if system_metrics.cpu_percent > self.thresholds["cpu_critical"]:
            alerts.append(f"🔴 CPU 위험: {system_metrics.cpu_percent:.1f}%")
        elif system_metrics.cpu_percent > self.thresholds["cpu_warning"]:
            alerts.append(f"🟡 CPU 주의: {system_metrics.cpu_percent:.1f}%")

        # 메모리 경고
        if system_metrics.memory_percent > self.thresholds["memory_critical"]:
            alerts.append(f"🔴 메모리 위험: {system_metrics.memory_percent:.1f}%")
        elif system_metrics.memory_percent > self.thresholds["memory_warning"]:
            alerts.append(f"🟡 메모리 주의: {system_metrics.memory_percent:.1f}%")

        # GPU 경고 (최신 GPU 메트릭 사용)
        if self.gpu_metrics_history:
            latest_gpu = self.gpu_metrics_history[-1]

            if latest_gpu.memory_percent > self.thresholds["gpu_memory_critical"]:
                alerts.append(f"🔴 GPU 메모리 위험: {latest_gpu.memory_percent:.1f}%")
            elif latest_gpu.memory_percent > self.thresholds["gpu_memory_warning"]:
                alerts.append(f"🟡 GPU 메모리 주의: {latest_gpu.memory_percent:.1f}%")

            if latest_gpu.temperature > self.thresholds["gpu_temp_critical"]:
                alerts.append(f"🔴 GPU 온도 위험: {latest_gpu.temperature}°C")
            elif latest_gpu.temperature > self.thresholds["gpu_temp_warning"]:
                alerts.append(f"🟡 GPU 온도 주의: {latest_gpu.temperature}°C")

        # SGLang 성능 경고
        if self.sglang_metrics_history:
            latest_sglang = self.sglang_metrics_history[-1]

            if latest_sglang.avg_response_time > self.thresholds["response_time_critical"]:
                alerts.append(f"🔴 SGLang 응답 시간 위험: {latest_sglang.avg_response_time:.2f}초")
            elif latest_sglang.avg_response_time > self.thresholds["response_time_warning"]:
                alerts.append(f"🟡 SGLang 응답 시간 주의: {latest_sglang.avg_response_time:.2f}초")

            if latest_sglang.cache_hit_rate < self.thresholds["cache_hit_rate_warning"]:
                alerts.append(f"🟡 SGLang 캐시 히트율 낮음: {latest_sglang.cache_hit_rate:.1%}")

            if latest_sglang.error_rate > self.thresholds["error_rate_critical"]:
                alerts.append(f"🔴 SGLang 에러율 위험: {latest_sglang.error_rate:.1%}")
            elif latest_sglang.error_rate > self.thresholds["error_rate_warning"]:
                alerts.append(f"🟡 SGLang 에러율 주의: {latest_sglang.error_rate:.1%}")

        # 경고 로깅
        for alert in alerts:
            logger.warning(alert)

    def get_performance_summary(self, minutes: int = 10) -> Dict[str, Any]:
        """성능 요약 정보 (최근 N분)"""
        cutoff_time = time.time() - (minutes * 60)

        # 최근 시스템 메트릭
        recent_system = [m for m in self.system_metrics_history if m.timestamp > cutoff_time]

        # 최근 GPU 메트릭
        recent_gpu = [m for m in self.gpu_metrics_history if m.timestamp > cutoff_time]

        # 최근 SGLang 메트릭
        recent_sglang = [m for m in self.sglang_metrics_history if m.timestamp > cutoff_time]

        # 최근 한국어 토큰 메트릭
        recent_korean = [m for m in self.korean_token_metrics_history if m.timestamp > cutoff_time]

        summary = {
            "period_minutes": minutes,
            "timestamp": time.time(),
            "system": self._summarize_system_metrics(recent_system),
            "gpu": self._summarize_gpu_metrics(recent_gpu),
            "sglang": self._summarize_sglang_metrics(recent_sglang),
            "korean_tokens": self._summarize_korean_metrics(recent_korean),
            "framework": "SGLang",
            "korean_optimized": True
        }

        return summary

    def _summarize_system_metrics(self, metrics: List[SystemMetrics]) -> Dict[str, Any]:
        """시스템 메트릭 요약"""
        if not metrics:
            return {"status": "no_data"}

        cpu_values = [m.cpu_percent for m in metrics]
        memory_values = [m.memory_percent for m in metrics]

        return {
            "cpu": {
                "avg": sum(cpu_values) / len(cpu_values),
                "max": max(cpu_values),
                "min": min(cpu_values),
                "current": cpu_values[-1] if cpu_values else 0
            },
            "memory": {
                "avg": sum(memory_values) / len(memory_values),
                "max": max(memory_values),
                "min": min(memory_values),
                "current": memory_values[-1] if memory_values else 0,
                "total_gb": metrics[-1].memory_total_gb if metrics else 0
            },
            "disk_usage": metrics[-1].disk_usage_percent if metrics else 0,
            "sample_count": len(metrics)
        }

    def _summarize_gpu_metrics(self, metrics: List[GPUMetrics]) -> Dict[str, Any]:
        """GPU 메트릭 요약"""
        if not metrics:
            return {"status": "no_data", "available": False}

        # GPU별로 그룹화
        gpu_summary = {}
        for metric in metrics:
            gpu_id = metric.gpu_id
            if gpu_id not in gpu_summary:
                gpu_summary[gpu_id] = {
                    "name": metric.name,
                    "memory_percent": [],
                    "utilization": [],
                    "temperature": [],
                    "power_draw": [],
                    "memory_total_gb": metric.memory_total_gb
                }

            gpu_summary[gpu_id]["memory_percent"].append(metric.memory_percent)
            gpu_summary[gpu_id]["utilization"].append(metric.utilization_percent)
            gpu_summary[gpu_id]["temperature"].append(metric.temperature)
            gpu_summary[gpu_id]["power_draw"].append(metric.power_draw_watts)

        # 요약 통계 계산
        for gpu_id in gpu_summary:
            gpu_data = gpu_summary[gpu_id]
            gpu_summary[gpu_id] = {
                "name": gpu_data["name"],
                "memory_total_gb": gpu_data["memory_total_gb"],
                "memory_percent": {
                    "avg": sum(gpu_data["memory_percent"]) / len(gpu_data["memory_percent"]),
                    "max": max(gpu_data["memory_percent"]),
                    "current": gpu_data["memory_percent"][-1]
                },
                "utilization": {
                    "avg": sum(gpu_data["utilization"]) / len(gpu_data["utilization"]),
                    "max": max(gpu_data["utilization"]),
                    "current": gpu_data["utilization"][-1]
                },
                "temperature": {
                    "avg": sum(gpu_data["temperature"]) / len(gpu_data["temperature"]),
                    "max": max(gpu_data["temperature"]),
                    "current": gpu_data["temperature"][-1]
                },
                "power_draw": {
                    "avg": sum(gpu_data["power_draw"]) / len(gpu_data["power_draw"]),
                    "max": max(gpu_data["power_draw"]),
                    "current": gpu_data["power_draw"][-1]
                }
            }

        return {
            "available": True,
            "gpus": gpu_summary,
            "sample_count": len(metrics)
        }

    def _summarize_sglang_metrics(self, metrics: List[SGLangPerformanceMetrics]) -> Dict[str, Any]:
        """SGLang 메트릭 요약"""
        if not metrics:
            return {"status": "no_data", "framework": "SGLang"}

        rps_values = [m.requests_per_second for m in metrics]
        tps_values = [m.tokens_per_second for m in metrics]
        response_times = [m.avg_response_time for m in metrics]
        cache_rates = [m.cache_hit_rate for m in metrics]
        error_rates = [m.error_rate for m in metrics]

        return {
            "framework": "SGLang",
            "requests_per_second": {
                "avg": sum(rps_values) / len(rps_values),
                "max": max(rps_values),
                "current": rps_values[-1]
            },
            "tokens_per_second": {
                "avg": sum(tps_values) / len(tps_values),
                "max": max(tps_values),
                "current": tps_values[-1]
            },
            "response_time": {
                "avg": sum(response_times) / len(response_times),
                "max": max(response_times),
                "min": min(response_times),
                "current": response_times[-1]
            },
            "cache_hit_rate": {
                "avg": sum(cache_rates) / len(cache_rates),
                "current": cache_rates[-1]
            },
            "error_rate": {
                "avg": sum(error_rates) / len(error_rates),
                "current": error_rates[-1]
            },
            "queue_length": metrics[-1].queue_length,
            "running_requests": metrics[-1].running_requests,
            "memory_usage_gb": metrics[-1].memory_usage_gb,
            "sample_count": len(metrics)
        }

    def _summarize_korean_metrics(self, metrics: List[KoreanTokenMetrics]) -> Dict[str, Any]:
        """한국어 토큰 메트릭 요약"""
        if not metrics:
            return {"status": "no_data", "language": "korean"}

        korean_ratios = [m.korean_ratio for m in metrics]
        efficiencies = [m.tokenization_efficiency for m in metrics]
        total_korean_chars = sum(m.korean_char_count for m in metrics)
        total_chars = sum(m.total_char_count for m in metrics)
        total_estimated = sum(m.estimated_tokens for m in metrics)
        total_actual = sum(m.actual_tokens for m in metrics)

        return {
            "language": "korean",
            "korean_ratio": {
                "avg": sum(korean_ratios) / len(korean_ratios),
                "max": max(korean_ratios),
                "current": korean_ratios[-1]
            },
            "tokenization_efficiency": {
                "avg": sum(efficiencies) / len(efficiencies),
                "current": efficiencies[-1]
            },
            "totals": {
                "korean_chars": total_korean_chars,
                "total_chars": total_chars,
                "estimated_tokens": total_estimated,
                "actual_tokens": total_actual,
                "overall_korean_ratio": total_korean_chars / total_chars if total_chars > 0 else 0
            },
            "sample_count": len(metrics)
        }

    def export_metrics(self, filepath: str, minutes: int = 60):
        """메트릭을 JSON 파일로 내보내기"""
        try:
            cutoff_time = time.time() - (minutes * 60)

            export_data = {
                "export_info": {
                    "timestamp": time.time(),
                    "period_minutes": minutes,
                    "framework": "SGLang",
                    "korean_optimized": True
                },
                "system_metrics": [
                    asdict(m) for m in self.system_metrics_history
                    if m.timestamp > cutoff_time
                ],
                "gpu_metrics": [
                    asdict(m) for m in self.gpu_metrics_history
                    if m.timestamp > cutoff_time
                ],
                "sglang_metrics": [
                    asdict(m) for m in self.sglang_metrics_history
                    if m.timestamp > cutoff_time
                ],
                "korean_token_metrics": [
                    asdict(m) for m in self.korean_token_metrics_history
                    if m.timestamp > cutoff_time
                ],
                "performance_summary": self.get_performance_summary(minutes)
            }

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

            logger.info(f"📊 성능 메트릭 내보내기 완료: {filepath}")

        except Exception as e:
            logger.error(f"❌ 메트릭 내보내기 실패: {e}")

    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """성능 최적화 권장사항"""
        recommendations = []

        # 최근 메트릭 분석
        if self.system_metrics_history:
            latest_system = self.system_metrics_history[-1]

            # CPU 최적화
            if latest_system.cpu_percent > 80:
                recommendations.append({
                    "category": "CPU",
                    "severity": "high" if latest_system.cpu_percent > 90 else "medium",
                    "title": "CPU 사용률 높음",
                    "description": f"현재 CPU 사용률: {latest_system.cpu_percent:.1f}%",
                    "suggestions": [
                        "SGLang max_running_requests 값 감소",
                        "동시 요청 수 제한",
                        "배치 크기 조정"
                    ]
                })

            # 메모리 최적화
            if latest_system.memory_percent > 85:
                recommendations.append({
                    "category": "Memory",
                    "severity": "high" if latest_system.memory_percent > 95 else "medium",
                    "title": "메모리 사용률 높음",
                    "description": f"현재 메모리 사용률: {latest_system.memory_percent:.1f}%",
                    "suggestions": [
                        "SGLang mem_fraction_static 값 감소",
                        "KV 캐시 크기 제한",
                        "모델 양자화 고려"
                    ]
                })

        # GPU 최적화
        if self.gpu_metrics_history:
            latest_gpu = self.gpu_metrics_history[-1]

            if latest_gpu.memory_percent > 90:
                recommendations.append({
                    "category": "GPU",
                    "severity": "high" if latest_gpu.memory_percent > 95 else "medium",
                    "title": "GPU 메모리 사용률 높음",
                    "description": f"현재 GPU 메모리: {latest_gpu.memory_percent:.1f}%",
                    "suggestions": [
                        "SGLang chunked_prefill_size 감소",
                        "max_total_tokens 조정",
                        "FlashAttention 활성화"
                    ]
                })

            if latest_gpu.temperature > 80:
                recommendations.append({
                    "category": "GPU",
                    "severity": "high" if latest_gpu.temperature > 85 else "medium",
                    "title": "GPU 온도 높음",
                    "description": f"현재 GPU 온도: {latest_gpu.temperature}°C",
                    "suggestions": [
                        "팬 설정 확인",
                        "처리량 감소",
                        "냉각 시스템 점검"
                    ]
                })

        # SGLang 성능 최적화
        if self.sglang_metrics_history:
            latest_sglang = self.sglang_metrics_history[-1]

            if latest_sglang.avg_response_time > 3.0:
                recommendations.append({
                    "category": "SGLang",
                    "severity": "medium",
                    "title": "응답 시간 느림",
                    "description": f"평균 응답 시간: {latest_sglang.avg_response_time:.2f}초",
                    "suggestions": [
                        "enable_torch_compile=True 설정",
                        "prefix_caching 활성화",
                        "dynamic_batching 최적화"
                    ]
                })

            if latest_sglang.cache_hit_rate < 0.6:
                recommendations.append({
                    "category": "SGLang",
                    "severity": "medium",
                    "title": "캐시 효율성 낮음",
                    "description": f"캐시 히트율: {latest_sglang.cache_hit_rate:.1%}",
                    "suggestions": [
                        "프리픽스 캐싱 활성화",
                        "반복적인 프롬프트 패턴 활용",
                        "KV 캐시 크기 증가"
                    ]
                })

        # 한국어 토큰 최적화
        if self.korean_token_metrics_history:
            recent_korean = self.korean_token_metrics_history[-10:]  # 최근 10개
            avg_efficiency = sum(m.tokenization_efficiency for m in recent_korean) / len(recent_korean)

            if avg_efficiency < 0.8:
                recommendations.append({
                    "category": "Korean",
                    "severity": "low",
                    "title": "한국어 토큰화 효율성 개선 필요",
                    "description": f"토큰화 효율성: {avg_efficiency:.1%}",
                    "suggestions": [
                        "한국어 전용 토크나이저 사용",
                        "토큰 계산 알고리즘 조정",
                        "한국어 특화 모델 고려"
                    ]
                })

        return recommendations

    def clear_history(self):
        """메트릭 히스토리 초기화"""
        self.system_metrics_history.clear()
        self.gpu_metrics_history.clear()
        self.sglang_metrics_history.clear()
        self.korean_token_metrics_history.clear()
        logger.info("🧹 성능 메트릭 히스토리 초기화 완료")


# 전역 성능 모니터 인스턴스
performance_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> PerformanceMonitor:
    """성능 모니터 싱글톤 인스턴스 반환"""
    global performance_monitor
    if performance_monitor is None:
        performance_monitor = PerformanceMonitor()
    return performance_monitor


def start_performance_monitoring(interval: int = 10):
    """성능 모니터링 시작"""
    monitor = get_performance_monitor()
    monitor.monitor_interval = interval
    monitor.start_monitoring()


def stop_performance_monitoring():
    """성능 모니터링 중지"""
    global performance_monitor
    if performance_monitor:
        performance_monitor.stop_monitoring()


async def collect_sglang_metrics(sglang_client) -> Optional[Dict[str, Any]]:
    """SGLang 클라이언트에서 메트릭 수집"""
    try:
        # SGLang 서버 정보 조회
        server_info = await sglang_client.get_server_info()
        if not server_info:
            return None

        # 클라이언트 통계 조회
        client_stats = await sglang_client.get_client_statistics()

        # 메트릭 조합
        metrics = {
            "requests_per_second": server_info.requests_per_second,
            "tokens_per_second": server_info.tokens_per_second,
            "avg_response_time": client_stats.get("avg_response_time", 0.0),
            "queue_length": server_info.queue_length,
            "running_requests": server_info.running_requests,
            "cache_hit_rate": server_info.cache_hit_rate,
            "batch_size_avg": 1.0,  # SGLang에서 제공되면 업데이트
            "memory_usage_gb": server_info.memory_usage_gb,
            "error_rate": client_stats.get("error_count", 0) / max(client_stats.get("total_requests", 1), 1),
            "success_rate": client_stats.get("success_rate", 100.0) / 100.0
        }

        return metrics

    except Exception as e:
        logger.error(f"❌ SGLang 메트릭 수집 실패: {e}")
        return None


class SGLangOptimizer:
    """SGLang 성능 최적화 도구"""

    def __init__(self, performance_monitor: PerformanceMonitor):
        self.monitor = performance_monitor
        self.optimization_history = []

    def analyze_performance_bottlenecks(self) -> Dict[str, Any]:
        """성능 병목 지점 분석"""
        analysis = {
            "timestamp": time.time(),
            "bottlenecks": [],
            "recommendations": [],
            "severity": "none"
        }

        # 최근 메트릭 분석
        recent_summary = self.monitor.get_performance_summary(minutes=5)

        # CPU 병목
        if recent_summary["system"]["cpu"]["avg"] > 80:
            analysis["bottlenecks"].append({
                "type": "cpu",
                "severity": "high" if recent_summary["system"]["cpu"]["avg"] > 90 else "medium",
                "value": recent_summary["system"]["cpu"]["avg"],
                "description": "CPU 사용률이 높아 처리 속도 저하"
            })

        # GPU 메모리 병목
        if recent_summary["gpu"]["available"]:
            for gpu_id, gpu_data in recent_summary["gpu"]["gpus"].items():
                if gpu_data["memory_percent"]["avg"] > 90:
                    analysis["bottlenecks"].append({
                        "type": "gpu_memory",
                        "severity": "high",
                        "value": gpu_data["memory_percent"]["avg"],
                        "description": f"GPU {gpu_id} 메모리 부족"
                    })

        # SGLang 성능 병목
        if recent_summary["sglang"].get("response_time", {}).get("avg", 0) > 3.0:
            analysis["bottlenecks"].append({
                "type": "response_time",
                "severity": "medium",
                "value": recent_summary["sglang"]["response_time"]["avg"],
                "description": "SGLang 응답 시간 지연"
            })

        # 캐시 효율성 문제
        if recent_summary["sglang"].get("cache_hit_rate", {}).get("avg", 0) < 0.6:
            analysis["bottlenecks"].append({
                "type": "cache_efficiency",
                "severity": "medium",
                "value": recent_summary["sglang"]["cache_hit_rate"]["avg"],
                "description": "SGLang 캐시 효율성 낮음"
            })

        # 권장사항 생성
        analysis["recommendations"] = self._generate_optimization_recommendations(analysis["bottlenecks"])

        # 전체 심각도 결정
        if any(b["severity"] == "high" for b in analysis["bottlenecks"]):
            analysis["severity"] = "high"
        elif any(b["severity"] == "medium" for b in analysis["bottlenecks"]):
            analysis["severity"] = "medium"
        else:
            analysis["severity"] = "low"

        return analysis

    def _generate_optimization_recommendations(self, bottlenecks: List[Dict]) -> List[Dict]:
        """최적화 권장사항 생성"""
        recommendations = []

        for bottleneck in bottlenecks:
            if bottleneck["type"] == "cpu":
                recommendations.append({
                    "category": "SGLang Configuration",
                    "action": "Reduce max_running_requests",
                    "description": "동시 처리 요청 수를 줄여 CPU 부하 감소",
                    "config_change": "--max-running-requests 8",
                    "expected_impact": "CPU 사용률 20-30% 감소"
                })

            elif bottleneck["type"] == "gpu_memory":
                recommendations.append({
                    "category": "SGLang Memory",
                    "action": "Reduce memory allocation",
                    "description": "GPU 메모리 사용률 조정",
                    "config_change": "--mem-fraction-static 0.6",
                    "expected_impact": "GPU 메모리 사용률 15-20% 감소"
                })

            elif bottleneck["type"] == "response_time":
                recommendations.append({
                    "category": "SGLang Performance",
                    "action": "Enable optimizations",
                    "description": "SGLang 성능 최적화 기능 활성화",
                    "config_change": "--enable-torch-compile --chunked-prefill-size 4096",
                    "expected_impact": "응답 시간 25-40% 개선"
                })

            elif bottleneck["type"] == "cache_efficiency":
                recommendations.append({
                    "category": "SGLang Caching",
                    "action": "Improve caching",
                    "description": "프리픽스 캐싱 및 KV 캐시 최적화",
                    "config_change": "--enable-prefix-caching",
                    "expected_impact": "캐시 히트율 30-50% 향상"
                })

        return recommendations

    def suggest_sglang_config_optimization(self, current_config: Dict) -> Dict[str, Any]:
        """SGLang 설정 최적화 제안"""
        optimized_config = current_config.copy()
        changes = []

        # 최근 성능 데이터 분석
        recent_summary = self.monitor.get_performance_summary(minutes=10)

        # CPU 기반 최적화
        cpu_avg = recent_summary["system"]["cpu"]["avg"]
        if cpu_avg > 85:
            # CPU 부하가 높으면 동시 요청 수 감소
            new_max_requests = max(4, current_config.get("max_running_requests", 16) - 4)
            optimized_config["max_running_requests"] = new_max_requests
            changes.append({
                "parameter": "max_running_requests",
                "old_value": current_config.get("max_running_requests", 16),
                "new_value": new_max_requests,
                "reason": f"CPU 사용률 높음 ({cpu_avg:.1f}%)"
            })
        elif cpu_avg < 50:
            # CPU 여유 있으면 동시 요청 수 증가
            new_max_requests = min(32, current_config.get("max_running_requests", 16) + 4)
            optimized_config["max_running_requests"] = new_max_requests
            changes.append({
                "parameter": "max_running_requests",
                "old_value": current_config.get("max_running_requests", 16),
                "new_value": new_max_requests,
                "reason": f"CPU 여유 있음 ({cpu_avg:.1f}%)"
            })

        # GPU 메모리 기반 최적화
        if recent_summary["gpu"]["available"]:
            for gpu_id, gpu_data in recent_summary["gpu"]["gpus"].items():
                gpu_mem_avg = gpu_data["memory_percent"]["avg"]

                if gpu_mem_avg > 90:
                    # GPU 메모리 부족 시 메모리 사용률 감소
                    new_mem_fraction = max(0.5, current_config.get("mem_fraction_static", 0.75) - 0.1)
                    optimized_config["mem_fraction_static"] = new_mem_fraction
                    changes.append({
                        "parameter": "mem_fraction_static",
                        "old_value": current_config.get("mem_fraction_static", 0.75),
                        "new_value": new_mem_fraction,
                        "reason": f"GPU 메모리 부족 ({gpu_mem_avg:.1f}%)"
                    })
                elif gpu_mem_avg < 60:
                    # GPU 메모리 여유 시 증가
                    new_mem_fraction = min(0.85, current_config.get("mem_fraction_static", 0.75) + 0.05)
                    optimized_config["mem_fraction_static"] = new_mem_fraction
                    changes.append({
                        "parameter": "mem_fraction_static",
                        "old_value": current_config.get("mem_fraction_static", 0.75),
                        "new_value": new_mem_fraction,
                        "reason": f"GPU 메모리 여유 ({gpu_mem_avg:.1f}%)"
                    })

        # 응답 시간 기반 최적화
        if recent_summary["sglang"].get("response_time", {}).get("avg", 0) > 3.0:
            # 응답 시간이 느리면 청크 크기 조정
            current_chunk_size = current_config.get("chunked_prefill_size", 4096)
            new_chunk_size = max(2048, current_chunk_size - 1024)
            optimized_config["chunked_prefill_size"] = new_chunk_size
            changes.append({
                "parameter": "chunked_prefill_size",
                "old_value": current_chunk_size,
                "new_value": new_chunk_size,
                "reason": "응답 시간 개선"
            })

            # Torch compile 활성화 제안
            if not current_config.get("enable_torch_compile", False):
                optimized_config["enable_torch_compile"] = True
                changes.append({
                    "parameter": "enable_torch_compile",
                    "old_value": False,
                    "new_value": True,
                    "reason": "성능 최적화"
                })

        # 캐시 효율성 기반 최적화
        cache_hit_rate = recent_summary["sglang"].get("cache_hit_rate", {}).get("avg", 0)
        if cache_hit_rate < 0.6:
            if not current_config.get("enable_prefix_caching", False):
                optimized_config["enable_prefix_caching"] = True
                changes.append({
                    "parameter": "enable_prefix_caching",
                    "old_value": False,
                    "new_value": True,
                    "reason": f"캐시 효율성 개선 ({cache_hit_rate:.1%})"
                })

        return {
            "optimized_config": optimized_config,
            "changes": changes,
            "performance_analysis": recent_summary,
            "estimated_improvements": self._estimate_performance_improvements(changes)
        }

    def _estimate_performance_improvements(self, changes: List[Dict]) -> Dict[str, str]:
        """성능 개선 예상치 계산"""
        improvements = {}

        for change in changes:
            param = change["parameter"]

            if param == "max_running_requests":
                if change["new_value"] < change["old_value"]:
                    improvements["cpu_usage"] = "15-25% 감소"
                    improvements["response_stability"] = "향상"
                else:
                    improvements["throughput"] = "20-30% 증가"

            elif param == "mem_fraction_static":
                if change["new_value"] < change["old_value"]:
                    improvements["gpu_memory"] = "안정성 향상"
                else:
                    improvements["model_capacity"] = "증가"

            elif param == "enable_torch_compile":
                improvements["inference_speed"] = "15-25% 향상"
                improvements["first_token_latency"] = "감소"

            elif param == "enable_prefix_caching":
                improvements["cache_efficiency"] = "30-50% 향상"
                improvements["repeat_request_speed"] = "크게 향상"

            elif param == "chunked_prefill_size":
                improvements["memory_efficiency"] = "향상"
                improvements["large_context_handling"] = "개선"

        return improvements


class KoreanTokenOptimizer:
    """한국어 토큰 처리 최적화"""

    def __init__(self):
        self.optimization_cache = {}

    def analyze_korean_text_characteristics(self, text: str) -> Dict[str, Any]:
        """한국어 텍스트 특성 분석"""
        import re

        # 문자 유형별 분류
        korean_chars = len(re.findall(r'[가-힣]', text))
        hanja_chars = len(re.findall(r'[一-龯]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        number_chars = len(re.findall(r'[0-9]', text))
        punctuation_chars = len(re.findall(r'[^\w\s가-힣一-龯]', text))
        space_chars = len(re.findall(r'\s', text))

        total_chars = len(text)

        # 한국어 복합어 분석
        compound_words = len(re.findall(r'[가-힣]{3,}', text))  # 3글자 이상 한글 단어

        # 문장 구조 분석
        sentences = len(re.findall(r'[.!?]', text))
        avg_sentence_length = total_chars / max(sentences, 1)

        return {
            "total_chars": total_chars,
            "korean_chars": korean_chars,
            "hanja_chars": hanja_chars,
            "english_chars": english_chars,
            "number_chars": number_chars,
            "punctuation_chars": punctuation_chars,
            "space_chars": space_chars,
            "korean_ratio": korean_chars / total_chars if total_chars > 0 else 0,
            "compound_words": compound_words,
            "sentences": sentences,
            "avg_sentence_length": avg_sentence_length,
            "complexity_score": self._calculate_complexity_score(korean_chars, compound_words, english_chars)
        }

    def _calculate_complexity_score(self, korean_chars: int, compound_words: int, english_chars: int) -> float:
        """텍스트 복잡도 점수 계산"""
        # 한국어 복합어가 많고, 영어가 섞이면 복잡도 증가
        base_score = korean_chars * 0.8  # 한글 기본 점수
        compound_penalty = compound_words * 0.3  # 복합어 가중치
        mixed_penalty = (english_chars > 0 and korean_chars > 0) * 0.5  # 혼용 가중치

        return base_score + compound_penalty + mixed_penalty

    def optimize_korean_tokenization_factor(self, historical_data: List[Dict]) -> float:
        """한국어 토큰화 팩터 최적화"""
        if not historical_data:
            return 1.15  # 기본값

        # 실제 토큰 수와 예상 토큰 수 비교
        efficiency_scores = []
        for data in historical_data:
            if data.get("actual_tokens") and data.get("estimated_tokens"):
                efficiency = data["actual_tokens"] / data["estimated_tokens"]
                efficiency_scores.append(efficiency)

        if efficiency_scores:
            avg_efficiency = sum(efficiency_scores) / len(efficiency_scores)

            # 효율성 기반 팩터 조정
            if avg_efficiency > 1.1:
                return 1.25  # 실제가 예상보다 높으면 팩터 증가
            elif avg_efficiency < 0.9:
                return 1.05  # 실제가 예상보다 낮으면 팩터 감소
            else:
                return 1.15  # 적정 수준

        return 1.15

    def suggest_korean_prompt_optimization(self, prompt: str) -> Dict[str, Any]:
        """한국어 프롬프트 최적화 제안"""
        analysis = self.analyze_korean_text_characteristics(prompt)
        suggestions = []

        # 너무 긴 문장 체크
        if analysis["avg_sentence_length"] > 100:
            suggestions.append({
                "type": "sentence_length",
                "description": "문장이 너무 길어 토큰 효율성이 떨어질 수 있습니다",
                "suggestion": "문장을 짧게 나누어 작성하세요"
            })

        # 영어/한국어 혼용 체크
        if 0.2 < analysis["korean_ratio"] < 0.8:
            suggestions.append({
                "type": "language_mixing",
                "description": "한국어와 영어가 혼재되어 토큰화 효율이 떨어집니다",
                "suggestion": "가능하면 한 언어로 통일하거나 명확히 구분하세요"
            })

        # 복합어 과다 사용 체크
        complexity_ratio = analysis["compound_words"] / analysis["korean_chars"] if analysis["korean_chars"] > 0 else 0
        if complexity_ratio > 0.3:
            suggestions.append({
                "type": "complexity",
                "description": "복잡한 복합어가 많아 토큰 수가 증가할 수 있습니다",
                "suggestion": "간단한 표현으로 바꾸거나 설명을 추가하세요"
            })

        return {
            "analysis": analysis,
            "suggestions": suggestions,
            "optimized_prompt": self._optimize_prompt_structure(prompt, analysis),
            "expected_token_reduction": len(suggestions) * 5  # 제안사항당 약 5% 토큰 절약
        }

    def _optimize_prompt_structure(self, prompt: str, analysis: Dict) -> str:
        """프롬프트 구조 최적화"""
        import re

        optimized = prompt

        # 긴 문장을 짧게 분할
        if analysis["avg_sentence_length"] > 100:
            # 접속사 기준으로 문장 분할
            optimized = re.sub(r'(\s+(그리고|또한|하지만|그러나|따라서)\s+)', r'.\n\1', optimized)

        # 반복되는 표현 압축
        optimized = re.sub(r'(\w+)\s+\1', r'\1', optimized)  # 중복 단어 제거

        # 불필요한 공백 제거
        optimized = re.sub(r'\s+', ' ', optimized).strip()

        return optimized


# 유틸리티 함수들
def benchmark_sglang_performance(sglang_client, test_prompts: List[str],
                                 iterations: int = 5) -> Dict[str, Any]:
    """SGLang 성능 벤치마크"""
    results = {
        "total_iterations": iterations,
        "test_prompts_count": len(test_prompts),
        "results": [],
        "summary": {}
    }

    for i in range(iterations):
        iteration_start = time.time()
        iteration_results = []

        for j, prompt in enumerate(test_prompts):
            start_time = time.time()

            # 비동기 함수를 동기적으로 실행
            try:
                result = asyncio.get_event_loop().run_until_complete(
                    sglang_client.chat_completion([{"role": "user", "content": prompt}])
                )

                response_time = time.time() - start_time
                success = "error" not in result

                iteration_results.append({
                    "prompt_index": j,
                    "response_time": response_time,
                    "success": success,
                    "prompt_length": len(prompt)
                })

            except Exception as e:
                iteration_results.append({
                    "prompt_index": j,
                    "response_time": time.time() - start_time,
                    "success": False,
                    "error": str(e),
                    "prompt_length": len(prompt)
                })

        iteration_time = time.time() - iteration_start
        results["results"].append({
            "iteration": i + 1,
            "total_time": iteration_time,
            "requests": iteration_results
        })

    # 결과 요약
    all_response_times = []
    success_count = 0
    total_requests = 0

    for iteration in results["results"]:
        for request in iteration["requests"]:
            all_response_times.append(request["response_time"])
            if request["success"]:
                success_count += 1
            total_requests += 1

    results["summary"] = {
        "avg_response_time": sum(all_response_times) / len(all_response_times),
        "min_response_time": min(all_response_times),
        "max_response_time": max(all_response_times),
        "success_rate": success_count / total_requests,
        "total_requests": total_requests,
        "requests_per_second": total_requests / sum(r["total_time"] for r in results["results"])
    }

    return results


def compare_sglang_vs_baseline(sglang_metrics: Dict, baseline_metrics: Dict) -> Dict[str, Any]:
    """SGLang vs 기준선 성능 비교"""
    comparison = {
        "framework_comparison": "SGLang vs Baseline",
        "metrics": {},
        "improvements": {},
        "timestamp": time.time()
    }

    metric_comparisons = [
        ("requests_per_second", "처리량 (RPS)"),
        ("tokens_per_second", "토큰 처리율 (TPS)"),
        ("avg_response_time", "평균 응답 시간"),
        ("cache_hit_rate", "캐시 히트율"),
        ("memory_usage_gb", "메모리 사용량"),
        ("error_rate", "에러율")
    ]

    for metric_key, metric_name in metric_comparisons:
        sglang_value = sglang_metrics.get(metric_key, 0)
        baseline_value = baseline_metrics.get(metric_key, 0)

        if baseline_value > 0:
            if metric_key in ["avg_response_time", "memory_usage_gb", "error_rate"]:
                # 낮을수록 좋은 메트릭
                improvement = ((baseline_value - sglang_value) / baseline_value) * 100
            else:
                # 높을수록 좋은 메트릭
                improvement = ((sglang_value - baseline_value) / baseline_value) * 100
        else:
            improvement = 0

        comparison["metrics"][metric_key] = {
            "name": metric_name,
            "sglang": sglang_value,
            "baseline": baseline_value,
            "improvement_percent": improvement
        }

        if abs(improvement) > 5:  # 5% 이상 차이나는 경우만
            comparison["improvements"][metric_key] = improvement

    return comparison