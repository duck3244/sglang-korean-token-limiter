"""
SGLang 기반 한국어 Token Limiter 설정 관리
"""

import os
import yaml
from typing import Optional, Dict, Any, List
from pydantic import BaseSettings, Field, validator
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class SGLangServerConfig(BaseSettings):
    """SGLang 서버 설정"""

    # 기본 서버 설정
    host: str = Field(default="127.0.0.1", env="SGLANG_HOST")
    port: int = Field(default=8000, env="SGLANG_PORT")
    model_path: str = Field(default="Qwen/Qwen2.5-3B-Instruct", env="SGLANG_MODEL_PATH")

    # SGLang 특화 설정
    tp_size: int = Field(default=1, env="SGLANG_TP_SIZE")
    mem_fraction_static: float = Field(default=0.75, env="SGLANG_MEM_FRACTION")
    max_running_requests: int = Field(default=16, env="SGLANG_MAX_RUNNING_REQUESTS")
    max_total_tokens: int = Field(default=8192, env="SGLANG_MAX_TOTAL_TOKENS")

    # 스케줄링 및 최적화
    schedule_policy: str = Field(default="lpm", env="SGLANG_SCHEDULE_POLICY")  # lpm, fcfs
    enable_torch_compile: bool = Field(default=True, env="SGLANG_TORCH_COMPILE")
    disable_flashinfer: bool = Field(default=False, env="SGLANG_DISABLE_FLASHINFER")
    chunked_prefill_size: int = Field(default=4096, env="SGLANG_CHUNKED_PREFILL_SIZE")
    enable_mixed_chunk: bool = Field(default=True, env="SGLANG_MIXED_CHUNK")

    # KV 캐시 설정
    kv_cache_dtype: str = Field(default="fp16", env="SGLANG_KV_CACHE_DTYPE")
    enable_prefix_caching: bool = Field(default=True, env="SGLANG_PREFIX_CACHING")

    # 보안 및 신뢰성
    trust_remote_code: bool = Field(default=True, env="SGLANG_TRUST_REMOTE_CODE")
    served_model_name: str = Field(default="korean-qwen", env="SGLANG_SERVED_MODEL_NAME")

    @validator('schedule_policy')
    def validate_schedule_policy(cls, v):
        valid_policies = ['lpm', 'fcfs', 'priority']
        if v not in valid_policies:
            raise ValueError(f'schedule_policy must be one of {valid_policies}')
        return v

    @validator('kv_cache_dtype')
    def validate_kv_cache_dtype(cls, v):
        valid_dtypes = ['fp16', 'fp8', 'int8']
        if v not in valid_dtypes:
            raise ValueError(f'kv_cache_dtype must be one of {valid_dtypes}')
        return v

    def get_sglang_args(self) -> List[str]:
        """SGLang 서버 시작 인자 생성"""
        args = [
            "--model-path", self.model_path,
            "--port", str(self.port),
            "--host", self.host,
            "--tp-size", str(self.tp_size),
            "--mem-fraction-static", str(self.mem_fraction_static),
            "--max-running-requests", str(self.max_running_requests),
            "--max-total-tokens", str(self.max_total_tokens),
            "--schedule-policy", self.schedule_policy,
            "--chunked-prefill-size", str(self.chunked_prefill_size),
            "--kv-cache-dtype", self.kv_cache_dtype,
            "--served-model-name", self.served_model_name,
        ]

        # Boolean 플래그들
        if self.enable_torch_compile:
            args.append("--enable-torch-compile")
        if self.disable_flashinfer:
            args.append("--disable-flashinfer")
        if self.enable_mixed_chunk:
            args.append("--enable-mixed-chunk")
        if self.enable_prefix_caching:
            args.append("--enable-prefix-caching")
        if self.trust_remote_code:
            args.append("--trust-remote-code")

        return args


class TokenizerConfig(BaseSettings):
    """토크나이저 설정"""

    model_name: str = Field(default="Qwen/Qwen2.5-3B-Instruct", env="TOKENIZER_MODEL")
    max_length: int = Field(default=8192, env="TOKENIZER_MAX_LENGTH")
    korean_factor: float = Field(default=1.15, env="KOREAN_FACTOR")  # SGLang 최적화 반영
    cache_dir: str = Field(default="./tokenizer_cache", env="TOKENIZER_CACHE_DIR")

    # SGLang 특화 토큰 설정
    enable_fast_tokenizer: bool = Field(default=True, env="ENABLE_FAST_TOKENIZER")
    add_special_tokens: bool = Field(default=True, env="ADD_SPECIAL_TOKENS")


class PerformanceConfig(BaseSettings):
    """SGLang 성능 설정"""

    # 동시성 설정
    max_concurrent_requests: int = Field(default=20, env="MAX_CONCURRENT_REQUESTS")
    request_timeout: int = Field(default=120, env="REQUEST_TIMEOUT")
    batch_size: int = Field(default=8, env="BATCH_SIZE")

    # SGLang 동적 배치 설정
    enable_dynamic_batching: bool = Field(default=True, env="ENABLE_DYNAMIC_BATCHING")
    batch_expansion_factor: float = Field(default=2.0, env="BATCH_EXPANSION_FACTOR")

    # KV 캐시 관리
    kv_cache_capacity: float = Field(default=0.8, env="KV_CACHE_CAPACITY")
    prefix_cache_max_size: int = Field(default=1000, env="PREFIX_CACHE_MAX_SIZE")

    # 스트리밍 설정
    enable_streaming: bool = Field(default=True, env="ENABLE_STREAMING")
    stream_chunk_size: int = Field(default=1, env="STREAM_CHUNK_SIZE")


class DefaultLimitsConfig(BaseSettings):
    """기본 제한 설정 (SGLang 성능 향상 반영)"""

    rpm: int = Field(default=40, env="DEFAULT_RPM")      # vLLM 30 → SGLang 40
    tpm: int = Field(default=8000, env="DEFAULT_TPM")    # vLLM 5000 → SGLang 8000
    tph: int = Field(default=500000, env="DEFAULT_TPH")  # vLLM 300000 → SGLang 500000
    daily: int = Field(default=1000000, env="DEFAULT_DAILY")  # vLLM 500000 → SGLang 1000000
    cooldown_minutes: int = Field(default=2, env="DEFAULT_COOLDOWN")  # vLLM 3 → SGLang 2


class MonitoringConfig(BaseSettings):
    """모니터링 설정"""

    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    metrics_port: int = Field(default=9090, env="METRICS_PORT")
    health_check_interval: int = Field(default=15, env="HEALTH_CHECK_INTERVAL")  # SGLang 빠른 응답
    collect_sglang_stats: bool = Field(default=True, env="COLLECT_SGLANG_STATS")

    # 성능 모니터링
    enable_performance_logging: bool = Field(default=True, env="ENABLE_PERF_LOGGING")
    performance_log_interval: int = Field(default=30, env="PERF_LOG_INTERVAL")


class Config(BaseSettings):
    """메인 애플리케이션 설정"""

    # 서버 설정
    server_host: str = Field(default="0.0.0.0", env="SERVER_HOST")
    server_port: int = Field(default=8080, env="SERVER_PORT")
    debug: bool = Field(default=False, env="DEBUG")

    # 저장소 설정
    storage_type: str = Field(default="redis", env="STORAGE_TYPE")  # redis or sqlite
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    sqlite_path: str = Field(default="korean_sglang_usage.db", env="SQLITE_PATH")

    # 로깅 설정
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: Optional[str] = Field(default="logs/korean_sglang_limiter.log", env="LOG_FILE")

    # 하위 설정들
    sglang_server: SGLangServerConfig = SGLangServerConfig()
    tokenizer: TokenizerConfig = TokenizerConfig()
    performance: PerformanceConfig = PerformanceConfig()
    default_limits: DefaultLimitsConfig = DefaultLimitsConfig()
    monitoring: MonitoringConfig = MonitoringConfig()

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        env_nested_delimiter = "__"  # SGLANG_SERVER__HOST 형식 지원

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._load_yaml_config()

    def _load_yaml_config(self):
        """YAML 설정 파일 로드 (SGLang 특화)"""
        config_path = Path("config/sglang_korean.yaml")
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    yaml_config = yaml.safe_load(f)

                self._apply_yaml_config(yaml_config)
                logger.info(f"✅ SGLang 설정 파일 로드 완료: {config_path}")

            except Exception as e:
                logger.warning(f"⚠️ SGLang 설정 파일 로드 실패: {e}")

    def _apply_yaml_config(self, yaml_config: Dict[str, Any]):
        """YAML 설정 적용 (SGLang 특화)"""

        # 서버 설정
        if 'server' in yaml_config:
            server_config = yaml_config['server']
            if not os.getenv('SERVER_HOST'):
                self.server_host = server_config.get('host', self.server_host)
            if not os.getenv('SERVER_PORT'):
                self.server_port = server_config.get('port', self.server_port)

        # SGLang 서버 설정
        if 'sglang_server' in yaml_config:
            sglang_config = yaml_config['sglang_server']

            # 기본 설정
            if not os.getenv('SGLANG_HOST'):
                self.sglang_server.host = sglang_config.get('host', self.sglang_server.host)
            if not os.getenv('SGLANG_PORT'):
                self.sglang_server.port = sglang_config.get('port', self.sglang_server.port)
            if not os.getenv('SGLANG_MODEL_PATH'):
                self.sglang_server.model_path = sglang_config.get('model_path', self.sglang_server.model_path)

            # SGLang args 설정
            if 'sglang_args' in sglang_config:
                sglang_args = sglang_config['sglang_args']

                # 환경변수가 없는 경우에만 YAML 값 적용
                if not os.getenv('SGLANG_TP_SIZE'):
                    self.sglang_server.tp_size = sglang_args.get('tp_size', self.sglang_server.tp_size)
                if not os.getenv('SGLANG_MEM_FRACTION'):
                    self.sglang_server.mem_fraction_static = sglang_args.get('mem_fraction_static', self.sglang_server.mem_fraction_static)
                if not os.getenv('SGLANG_MAX_RUNNING_REQUESTS'):
                    self.sglang_server.max_running_requests = sglang_args.get('max_running_requests', self.sglang_server.max_running_requests)
                if not os.getenv('SGLANG_MAX_TOTAL_TOKENS'):
                    self.sglang_server.max_total_tokens = sglang_args.get('max_total_tokens', self.sglang_server.max_total_tokens)

        # 저장소 설정
        if 'storage' in yaml_config:
            storage_config = yaml_config['storage']
            if not os.getenv('STORAGE_TYPE'):
                self.storage_type = storage_config.get('type', self.storage_type)
            if not os.getenv('REDIS_URL'):
                self.redis_url = storage_config.get('redis_url', self.redis_url)
            if not os.getenv('SQLITE_PATH'):
                self.sqlite_path = storage_config.get('sqlite_path', self.sqlite_path)

        # 기본 제한 설정 (SGLang 성능 반영)
        if 'default_limits' in yaml_config:
            limits_config = yaml_config['default_limits']
            if not os.getenv('DEFAULT_RPM'):
                self.default_limits.rpm = limits_config.get('rpm', self.default_limits.rpm)
            if not os.getenv('DEFAULT_TPM'):
                self.default_limits.tpm = limits_config.get('tpm', self.default_limits.tpm)
            if not os.getenv('DEFAULT_TPH'):
                self.default_limits.tph = limits_config.get('tph', self.default_limits.tph)
            if not os.getenv('DEFAULT_DAILY'):
                self.default_limits.daily = limits_config.get('daily', self.default_limits.daily)
            if not os.getenv('DEFAULT_COOLDOWN'):
                self.default_limits.cooldown_minutes = limits_config.get('cooldown_minutes', self.default_limits.cooldown_minutes)

        # 토크나이저 설정
        if 'tokenizer' in yaml_config:
            tokenizer_config = yaml_config['tokenizer']
            if not os.getenv('TOKENIZER_MODEL'):
                self.tokenizer.model_name = tokenizer_config.get('model_name', self.tokenizer.model_name)
            if not os.getenv('TOKENIZER_MAX_LENGTH'):
                self.tokenizer.max_length = tokenizer_config.get('max_length', self.tokenizer.max_length)
            if not os.getenv('KOREAN_FACTOR'):
                self.tokenizer.korean_factor = tokenizer_config.get('korean_factor', self.tokenizer.korean_factor)

        # 성능 설정
        if 'performance' in yaml_config:
            perf_config = yaml_config['performance']
            if not os.getenv('MAX_CONCURRENT_REQUESTS'):
                self.performance.max_concurrent_requests = perf_config.get('max_concurrent_requests', self.performance.max_concurrent_requests)
            if not os.getenv('ENABLE_DYNAMIC_BATCHING'):
                self.performance.enable_dynamic_batching = perf_config.get('dynamic_batching', self.performance.enable_dynamic_batching)
            if not os.getenv('ENABLE_STREAMING'):
                self.performance.enable_streaming = perf_config.get('enable_streaming', self.performance.enable_streaming)

    @property
    def use_redis(self) -> bool:
        """Redis 사용 여부"""
        return self.storage_type.lower() == "redis"

    @property
    def sglang_server_url(self) -> str:
        """SGLang 서버 URL"""
        return f"http://{self.sglang_server.host}:{self.sglang_server.port}"

    def get_default_limits(self) -> Dict[str, int]:
        """기본 제한 설정 반환"""
        return {
            "rpm": self.default_limits.rpm,
            "tpm": self.default_limits.tpm,
            "tph": self.default_limits.tph,
            "daily": self.default_limits.daily,
            "cooldown_minutes": self.default_limits.cooldown_minutes
        }

    def get_sglang_launch_command(self) -> List[str]:
        """SGLang 서버 시작 명령어 생성"""
        base_cmd = ["python", "-m", "sglang.launch_server"]
        args = self.sglang_server.get_sglang_args()
        return base_cmd + args

    def validate_config(self) -> bool:
        """설정 유효성 검증"""
        try:
            # 기본 검증
            assert self.server_port > 0, "서버 포트는 0보다 커야 합니다"
            assert self.sglang_server.port > 0, "SGLang 포트는 0보다 커야 합니다"
            assert self.sglang_server.tp_size > 0, "TP size는 0보다 커야 합니다"
            assert 0 < self.sglang_server.mem_fraction_static <= 1, "메모리 사용률은 0과 1 사이여야 합니다"

            # SGLang 특화 검증
            assert self.sglang_server.max_running_requests > 0, "최대 동시 요청 수는 0보다 커야 합니다"
            assert self.sglang_server.max_total_tokens > 0, "최대 토큰 수는 0보다 커야 합니다"
            assert self.tokenizer.korean_factor > 0, "한국어 팩터는 0보다 커야 합니다"

            logger.info("✅ SGLang 설정 검증 완료")
            return True

        except AssertionError as e:
            logger.error(f"❌ 설정 검증 실패: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ 설정 검증 중 오류: {e}")
            return False

    def get_config_summary(self) -> Dict[str, Any]:
        """설정 요약 정보"""
        return {
            "framework": "SGLang",
            "server": {
                "host": self.server_host,
                "port": self.server_port,
                "debug": self.debug
            },
            "sglang_server": {
                "url": self.sglang_server_url,
                "model": self.sglang_server.model_path,
                "max_requests": self.sglang_server.max_running_requests,
                "max_tokens": self.sglang_server.max_total_tokens,
                "memory_fraction": self.sglang_server.mem_fraction_static
            },
            "performance": {
                "concurrent_requests": self.performance.max_concurrent_requests,
                "dynamic_batching": self.performance.enable_dynamic_batching,
                "streaming": self.performance.enable_streaming
            },
            "limits": self.get_default_limits(),
            "storage": {
                "type": self.storage_type,
                "url": self.redis_url if self.use_redis else self.sqlite_path
            }
        }


# 전역 설정 인스턴스
config = Config()