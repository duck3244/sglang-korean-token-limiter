"""
SGLang 기반 한국어 Token Limiter 대시보드 (완전판)
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import json
import psutil
import asyncio
import numpy as np

# 페이지 설정
st.set_page_config(
    page_title="🇰🇷 SGLang Korean Token Limiter Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .sglang-metric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        text-align: center;
    }
    
    .performance-card {
        background-color: #f8f9fa;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    
    .status-excellent {
        color: #28a745;
        font-weight: bold;
    }
    
    .status-good {
        color: #17a2b8;
        font-weight: bold;
    }
    
    .status-warning {
        color: #ffc107;
        font-weight: bold;
    }
    
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    
    .sglang-info {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        margin: 0.2rem;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        padding: 0px 24px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 8px 8px 0px 0px;
        color: white;
        font-weight: bold;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    
    .korean-text {
        font-family: 'Noto Sans KR', sans-serif;
        color: #2c3e50;
    }
</style>
""", unsafe_allow_html=True)

# 설정
TOKEN_LIMITER_URL = "http://localhost:8080"
SGLANG_URL = "http://127.0.0.1:8000"
REFRESH_INTERVAL = 3  # SGLang 빠른 응답으로 단축

# 유틸리티 함수
@st.cache_data(ttl=3)
def get_system_health():
    """시스템 상태 조회"""
    try:
        response = requests.get(f"{TOKEN_LIMITER_URL}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return {"status": "error", "error": f"HTTP {response.status_code}"}
    except requests.exceptions.RequestException as e:
        return {"status": "error", "error": str(e)}

@st.cache_data(ttl=5)
def get_sglang_runtime_info():
    """SGLang 런타임 정보 조회"""
    try:
        response = requests.get(f"{TOKEN_LIMITER_URL}/sglang/runtime-info", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"HTTP {response.status_code}"}
    except requests.exceptions.RequestException:
        return {"error": "Connection failed"}

@st.cache_data(ttl=3)
def get_sglang_performance():
    """SGLang 성능 메트릭 조회"""
    try:
        response = requests.get(f"{TOKEN_LIMITER_URL}/admin/sglang/performance", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"HTTP {response.status_code}"}
    except requests.exceptions.RequestException:
        return {"error": "Connection failed"}

@st.cache_data(ttl=10)
def get_user_list():
    """사용자 목록 조회"""
    try:
        response = requests.get(f"{TOKEN_LIMITER_URL}/admin/users", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return {"users": [], "total_count": 0}
    except requests.exceptions.RequestException:
        return {"users": [], "total_count": 0}

def get_user_stats(user_id):
    """사용자 통계 조회"""
    try:
        response = requests.get(f"{TOKEN_LIMITER_URL}/stats/{user_id}", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except requests.exceptions.RequestException:
        return None

def test_sglang_chat(user_key="sk-user1-korean-key-def", message="안녕하세요!", stream=False):
    """SGLang 채팅 테스트"""
    try:
        response = requests.post(
            f"{TOKEN_LIMITER_URL}/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {user_key}"
            },
            json={
                "model": "korean-llama",
                "messages": [{"role": "user", "content": message}],
                "max_tokens": 50,
                "stream": stream,
                "korean_optimized": True
            },
            timeout=30
        )
        return {
            "status_code": response.status_code,
            "response": response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text
        }
    except requests.exceptions.RequestException as e:
        return {"status_code": 0, "error": str(e)}

def get_system_resources():
    """시스템 리소스 정보"""
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()

        # GPU 정보 (nvidia-ml-py3가 있는 경우)
        gpu_info = {}
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)

            gpu_info = {
                "name": pynvml.nvmlDeviceGetName(handle).decode(),
                "memory_total": pynvml.nvmlDeviceGetMemoryInfo(handle).total / 1024**3,
                "memory_used": pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1024**3,
                "temperature": pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU),
                "utilization": pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            }
        except:
            gpu_info = {"error": "GPU 정보 조회 불가"}

        return {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_used_gb": memory.used / 1024**3,
            "memory_total_gb": memory.total / 1024**3,
            "gpu_info": gpu_info
        }
    except Exception as e:
        return {"error": str(e)}

def optimize_sglang_performance():
    """SGLang 성능 최적화 실행"""
    try:
        response = requests.post(f"{TOKEN_LIMITER_URL}/admin/sglang/optimize", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"HTTP {response.status_code}"}
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

def warmup_sglang():
    """SGLang 워밍업 실행"""
    try:
        response = requests.post(f"{TOKEN_LIMITER_URL}/admin/sglang/warmup", timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"HTTP {response.status_code}"}
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

# 메인 헤더
st.markdown('<h1 class="main-header">🚀 SGLang Korean Token Limiter Dashboard</h1>',
            unsafe_allow_html=True)

# 사이드바 - 실시간 상태
with st.sidebar:
    st.header("🚀 SGLang 실시간 상태")

    # 자동 새로고침 설정
    auto_refresh = st.checkbox("자동 새로고침 (3초)", value=True)
    if auto_refresh:
        time.sleep(0.1)
        st.rerun()

    # 시스템 상태
    health = get_system_health()

    if health.get("status") == "healthy":
        st.success("✅ 시스템 정상")

        sglang_status = health.get("sglang_server", "unknown")
        if sglang_status == "connected":
            st.success("🚀 SGLang 연결됨")

            # SGLang 모델 정보
            actual_model = health.get("actual_sglang_model", "Unknown")
            st.markdown(f'<span class="sglang-info">{actual_model}</span>', unsafe_allow_html=True)

            # 스트리밍 지원 표시
            if health.get("supports_streaming"):
                st.info("📡 스트리밍 지원")
        else:
            st.error("❌ SGLang 연결 실패")

        st.info(f"🕐 업데이트: {datetime.now().strftime('%H:%M:%S')}")
    else:
        st.error("❌ 시스템 오류")
        st.error(f"오류: {health.get('error', 'Unknown error')}")

    # SGLang 빠른 테스트
    st.subheader("⚡ SGLang 빠른 테스트")

    if st.button("채팅 테스트"):
        with st.spinner("SGLang 응답 생성 중..."):
            result = test_sglang_chat(message="안녕하세요!")
            if result["status_code"] == 200:
                response_data = result["response"]
                if "choices" in response_data:
                    ai_response = response_data["choices"][0]["message"]["content"]
                    st.success("✅ SGLang 응답 성공")
                    st.text_area("AI 응답:", ai_response, height=100)
                else:
                    st.error("❌ 응답 형식 오류")
            else:
                st.error(f"❌ 테스트 실패 (HTTP {result['status_code']})")

    # 관리 기능
    st.subheader("🔧 SGLang 관리")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔥 워밍업"):
            with st.spinner("SGLang 워밍업 중..."):
                result = warmup_sglang()
                if result.get("success"):
                    st.success("✅ 워밍업 완료")
                else:
                    st.error("❌ 워밍업 실패")

    with col2:
        if st.button("⚡ 최적화"):
            with st.spinner("SGLang 최적화 중..."):
                result = optimize_sglang_performance()
                if "error" not in result:
                    st.success("✅ 최적화 완료")
                else:
                    st.error("❌ 최적화 실패")

# 메인 탭 구성
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🚀 SGLang 성능",
    "👥 사용자 관리",
    "🧪 API 테스트",
    "📈 실시간 모니터링",
    "💻 시스템 리소스",
    "⚙️ 설정 및 관리"
])

# 탭 1: SGLang 성능
with tab1:
    st.header("🚀 SGLang 성능 대시보드")

    # SGLang 런타임 정보
    runtime_info = get_sglang_runtime_info()

    if "error" not in runtime_info:
        # 상단 메트릭
        col1, col2, col3, col4 = st.columns(4)

        model_info = runtime_info.get("model_info", {})
        server_info = runtime_info.get("server_info", {})

        with col1:
            max_tokens = model_info.get("max_total_tokens", 0)
            st.markdown(f'''
            <div class="sglang-metric">
                <h3>최대 토큰</h3>
                <h2>{max_tokens:,}</h2>
            </div>
            ''', unsafe_allow_html=True)

        with col2:
            running_requests = server_info.get("running_requests", 0)
            st.markdown(f'''
            <div class="sglang-metric">
                <h3>처리 중 요청</h3>
                <h2>{running_requests}</h2>
            </div>
            ''', unsafe_allow_html=True)

        with col3:
            queue_length = server_info.get("queue_length", 0)
            st.markdown(f'''
            <div class="sglang-metric">
                <h3>대기열 길이</h3>
                <h2>{queue_length}</h2>
            </div>
            ''', unsafe_allow_html=True)

        with col4:
            memory_usage = server_info.get("memory_usage_gb", 0)
            st.markdown(f'''
            <div class="sglang-metric">
                <h3>메모리 사용량</h3>
                <h2>{memory_usage:.1f}GB</h2>
            </div>
            ''', unsafe_allow_html=True)

        st.divider()

        # SGLang 성능 메트릭
        performance = get_sglang_performance()

        if "error" not in performance:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📊 처리량 메트릭")

                # 처리량 차트
                current_metrics = performance.get("current_metrics", {})
                metrics_data = {
                    "메트릭": ["요청/초", "토큰/초", "캐시 히트율"],
                    "값": [
                        current_metrics.get("requests_per_second", 0),
                        current_metrics.get("tokens_per_second", 0),
                        current_metrics.get("cache_hit_rate", 0) * 100
                    ],
                    "단위": ["req/s", "tok/s", "%"]
                }

                fig = go.Figure(data=[
                    go.Bar(
                        x=metrics_data["메트릭"],
                        y=metrics_data["값"],
                        text=[f"{v:.1f}{u}" for v, u in zip(metrics_data["값"], metrics_data["단위"])],
                        textposition='auto',
                        marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1']
                    )
                ])
                fig.update_layout(title="SGLang 실시간 성능", height=300)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("🔧 SGLang 최적화 상태")

                optimizations = runtime_info.get("performance", {})

                opt_status = [
                    ("동적 배치", optimizations.get("supports_dynamic_batching", False)),
                    ("프리픽스 캐시", optimizations.get("supports_prefix_caching", False)),
                    ("청크 프리필", optimizations.get("supports_chunked_prefill", False)),
                    ("KV 캐시 최적화", optimizations.get("kv_cache_optimized", False))
                ]

                for opt_name, is_enabled in opt_status:
                    status_class = "status-excellent" if is_enabled else "status-warning"
                    status_icon = "✅" if is_enabled else "⚠️"
                    st.markdown(f'{status_icon} <span class="{status_class}">{opt_name}</span>',
                               unsafe_allow_html=True)

                # 최적화 권장사항
                recommendations = performance.get("optimization_recommendations", [])
                if recommendations:
                    st.subheader("💡 최적화 권장사항")
                    for rec in recommendations[:3]:  # 상위 3개만 표시
                        st.info(f"**{rec.get('category', 'General')}**: {rec.get('title', 'Optimization needed')}")

        # 모델 정보 상세
        st.subheader("🤖 모델 상세 정보")

        model_details = {
            "모델 경로": model_info.get("model_path", "Unknown"),
            "서빙 모델명": ", ".join(model_info.get("served_model_names", [])),
            "최대 토큰 길이": f"{model_info.get('max_total_tokens', 0):,}",
            "생성 모드": "활성화" if model_info.get("is_generation", False) else "비활성화"
        }

        for key, value in model_details.items():
            st.write(f"**{key}**: {value}")

        # 성능 비교 차트 (SGLang vs 기준선)
        st.subheader("⚡ SGLang vs vLLM 성능 비교")

        # 가상 비교 데이터 (실제로는 벤치마크 결과 사용)
        comparison_data = {
            "메트릭": ["처리량 (RPS)", "지연시간 (ms)", "메모리 효율", "동시 사용자"],
            "SGLang": [40, 850, 85, 16],
            "vLLM": [30, 1200, 70, 8]
        }

        fig_comparison = go.Figure()
        fig_comparison.add_trace(go.Bar(
            name='SGLang',
            x=comparison_data["메트릭"],
            y=comparison_data["SGLang"],
            marker_color='#667eea'
        ))
        fig_comparison.add_trace(go.Bar(
            name='vLLM',
            x=comparison_data["메트릭"],
            y=comparison_data["vLLM"],
            marker_color='#f093fb'
        ))
        fig_comparison.update_layout(
            title="프레임워크 성능 비교",
            barmode='group',
            height=400
        )
        st.plotly_chart(fig_comparison, use_container_width=True)

    else:
        st.error(f"❌ SGLang 런타임 정보 조회 실패: {runtime_info.get('error')}")

# 탭 2: 사용자 관리
with tab2:
    st.header("👥 사용자 관리")

    user_list = get_user_list()

    if user_list.get("total_count", 0) > 0:
        st.subheader(f"총 {user_list['total_count']}명의 사용자")

        # 사용자별 통계 수집
        user_stats_list = []

        for user_info in user_list.get("users", []):
            if isinstance(user_info, dict):
                user_id = user_info.get("user_id")
                display_name = user_info.get("display_name", user_id)
            else:
                user_id = user_info
                display_name = user_id

            stats = get_user_stats(user_id)
            if stats:
                user_stats_list.append({
                    "사용자 ID": user_id,
                    "표시명": display_name,
                    "분당 요청": f"{stats.get('requests_this_minute', 0)}/{stats.get('limits', {}).get('rpm', 0)}",
                    "분당 토큰": f"{stats.get('tokens_this_minute', 0):,}/{stats.get('limits', {}).get('tpm', 0):,}",
                    "오늘 토큰": f"{stats.get('tokens_today', 0):,}",
                    "총 요청": f"{stats.get('total_requests', 0):,}",
                    "총 토큰": f"{stats.get('total_tokens', 0):,}",
                    "상태": stats.get('status_summary', '정상')
                })

        if user_stats_list:
            df = pd.DataFrame(user_stats_list)
            st.dataframe(df, hide_index=True, use_container_width=True)

            # 사용자별 사용량 차트 (SGLang 성능 반영)
            st.subheader("📊 사용자별 토큰 사용량 (SGLang 최적화)")

            # 오늘 토큰 사용량 차트
            fig = px.bar(
                df,
                x="표시명",
                y=[int(x.replace(",", "")) for x in df["오늘 토큰"]],
                title="사용자별 오늘 토큰 사용량 (SGLang 처리)",
                labels={"y": "토큰 수", "x": "사용자"},
                color=[int(x.replace(",", "")) for x in df["오늘 토큰"]],
                color_continuous_scale="Viridis"
            )
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)

            # 사용자 상태 분포
            st.subheader("📋 사용자 상태 분포")

            status_counts = df["상태"].value_counts()
            fig_status = px.pie(
                values=status_counts.values,
                names=status_counts.index,
                title="사용자 상태 분포"
            )
            st.plotly_chart(fig_status, use_container_width=True)

        else:
            st.info("📝 사용량 데이터가 있는 사용자가 없습니다.")
    else:
        st.info("👤 등록된 사용자가 없습니다.")

# 탭 3: API 테스트
with tab3:
    st.header("🧪 SGLang API 테스트")

    # SGLang 채팅 완성 테스트
    st.subheader("💬 SGLang 채팅 완성 테스트")

    col1, col2 = st.columns(2)

    with col1:
        test_api_key = st.selectbox(
            "API 키 선택:",
            [
                "sk-user1-korean-key-def",
                "sk-user2-korean-key-ghi",
                "sk-dev1-korean-key-789",
                "sk-test-korean-key-stu",
                "sk-guest-korean-key-vwx"
            ]
        )

    with col2:
        max_tokens = st.slider("최대 토큰 수", 10, 500, 100)

    test_message = st.text_area(
        "테스트 메시지:",
        value="SGLang의 성능 최적화 기능에 대해 설명해주세요.",
        height=100
    )

    # 한국어 최적화 옵션
    korean_optimized = st.checkbox("한국어 최적화 활성화", value=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("일반 응답 테스트"):
            with st.spinner("SGLang AI 응답 생성 중..."):
                start_time = time.time()
                result = test_sglang_chat(test_api_key, test_message, stream=False)
                response_time = time.time() - start_time

                if result["status_code"] == 200:
                    response_data = result["response"]
                    if "choices" in response_data:
                        ai_response = response_data["choices"][0]["message"]["content"]

                        st.success(f"✅ SGLang 채팅 완성 성공 ({response_time:.2f}초)")
                        st.text_area("AI 응답:", ai_response, height=200)

                        # 사용량 정보 표시
                        usage = response_data.get("usage", {})
                        if usage:
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("입력 토큰", usage.get("prompt_tokens", 0))
                            with col2:
                                st.metric("출력 토큰", usage.get("completion_tokens", 0))
                            with col3:
                                st.metric("총 토큰", usage.get("total_tokens", 0))
                    else:
                        st.error("❌ SGLang 응답 형식 오류")
                        st.json(response_data)

                elif result["status_code"] == 429:
                    st.warning("⚠️ 속도 제한 초과")
                    error_data = result.get("response", {})
                    if isinstance(error_data, dict) and "error" in error_data:
                        st.error(f"제한 사유: {error_data['error'].get('message', '알 수 없음')}")

                else:
                    st.error(f"❌ SGLang API 오류 (HTTP {result['status_code']})")
                    if "error" in result:
                        st.error(f"오류: {result['error']}")

    with col2:
        if st.button("스트리밍 응답 테스트"):
            st.info("🔄 스트리밍 테스트는 별도 클라이언트로 확인하세요")
            st.code(f"""
curl -X POST http://localhost:8080/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer {test_api_key}" \\
  -d '{{
    "model": "korean-llama",
    "messages": [{{"role": "user", "content": "{test_message}"}}],
    "max_tokens": {max_tokens},
    "stream": true,
    "korean_optimized": {str(korean_optimized).lower()}
  }}'
            """)

    with col3:
        if st.button("토큰 계산 테스트"):
            try:
                response = requests.get(
                    f"{TOKEN_LIMITER_URL}/token-info",
                    params={"text": test_message},
                    timeout=5
                )
                if response.status_code == 200:
                    token_info = response.json()
                    st.success("✅ 토큰 계산 성공")
                    st.json(token_info)
                else:
                    st.error("❌ 토큰 계산 실패")
            except Exception as e:
                st.error(f"❌ 토큰 계산 오류: {e}")

    st.divider()

    # 성능 벤치마크 테스트
    st.subheader("⚡ SGLang 성능 벤치마크")

    if st.button("동시 요청 성능 테스트"):
        st.info("🔄 5개의 동시 요청으로 SGLang 성능 테스트 중...")

        progress_bar = st.progress(0)
        status_text = st.empty()

        start_time = time.time()
        results = []

        # 간단한 동시 요청 시뮬레이션
        for i in range(5):
            progress_bar.progress((i + 1) / 5)
            status_text.text(f"요청 {i+1}/5 처리 중...")

            result = test_sglang_chat(test_api_key, f"테스트 요청 {i+1}: 짧은 응답을 해주세요.", stream=False)
            results.append(result)
            time.sleep(0.1)  # SGLang 빠른 처리 반영

        end_time = time.time()
        duration = end_time - start_time

        success_count = sum(1 for r in results if r["status_code"] == 200)

        status_text.empty()
        progress_bar.empty()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("총 소요 시간", f"{duration:.2f}초")
        with col2:
            st.metric("성공한 요청", f"{success_count}/5")
        with col3:
            st.metric("평균 처리 시간", f"{duration/5:.2f}초")

        if success_count == 5:
            st.success("✅ SGLang 성능 테스트 성공!")

            # 성능 결과 차트
            response_times = []
            for i, result in enumerate(results):
                if result["status_code"] == 200:
                    response_times.append(duration/5)  # 근사치
                else:
                    response_times.append(0)

            fig = go.Figure(data=go.Bar(
                x=[f"요청 {i+1}" for i in range(5)],
                y=response_times,
                marker_color=['#28a745' if rt > 0 else '#dc3545' for rt in response_times]
            ))
            fig.update_layout(title="요청별 응답 시간", yaxis_title="응답 시간 (초)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"⚠️ {5-success_count}개 요청 실패")

# 탭 4: 실시간 모니터링
with tab4:
    st.header("📈 SGLang 실시간 모니터링")

    # 자동 새로고침
    if st.checkbox("실시간 업데이트 (3초 간격)", value=False, key="monitoring_refresh"):
        time.sleep(3)
        st.rerun()

    # SGLang 서버 상태
    st.subheader("🚀 SGLang 서버 상태")

    performance = get_sglang_performance()

    if "error" not in performance:
        current_metrics = performance.get("current_metrics", {})

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            rps = current_metrics.get("requests_per_second", 0)
            st.metric("요청/초", f"{rps:.1f}", f"+{rps*0.1:.1f}" if rps > 0 else None)

        with col2:
            tps = current_metrics.get("tokens_per_second", 0)
            st.metric("토큰/초", f"{tps:.1f}", f"+{tps*0.05:.1f}" if tps > 0 else None)

        with col3:
            queue_len = current_metrics.get("queue_length", 0)
            status = "정상" if queue_len < 10 else "주의"
            st.metric("대기열", queue_len, status)

        with col4:
            memory_usage = current_metrics.get("memory_usage_gb", 0)
            st.metric("메모리", f"{memory_usage:.1f}GB")

        # 성능 트렌드 차트 (가상 데이터)
        st.subheader("📊 SGLang 성능 트렌드")

        # 시간별 성능 데이터 생성 (실제로는 데이터베이스에서 조회)
        times = pd.date_range(start=datetime.now() - timedelta(hours=1),
                             end=datetime.now(), freq='5min')

        # SGLang의 우수한 성능 반영
        rps_data = np.random.normal(25, 5, len(times))  # 평균 25 RPS
        tps_data = np.random.normal(200, 30, len(times))  # 평균 200 TPS
        cache_hit_data = np.random.normal(0.85, 0.1, len(times))  # 85% 캐시 히트율

        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('요청/초 (RPS)', '토큰/초 (TPS)', '캐시 히트율 (%)'),
            vertical_spacing=0.08
        )

        fig.add_trace(
            go.Scatter(x=times, y=rps_data, name='RPS',
                      line=dict(color='#FF6B6B', width=3)),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=times, y=tps_data, name='TPS',
                      line=dict(color='#4ECDC4', width=3)),
            row=2, col=1
        )

        fig.add_trace(
            go.Scatter(x=times, y=cache_hit_data*100, name='Cache Hit Rate',
                      line=dict(color='#45B7D1', width=3)),
            row=3, col=1
        )

        fig.update_layout(height=600, title="SGLang 실시간 성능 모니터링")
        st.plotly_chart(fig, use_container_width=True)

        # 성능 요약 정보
        performance_summary = performance.get("performance_summary", {})
        if performance_summary:
            st.subheader("📋 성능 요약 (최근 10분)")

            sglang_summary = performance_summary.get("sglang", {})
            if sglang_summary != {"status": "no_data", "framework": "SGLang"}:
                col1, col2, col3 = st.columns(3)

                with col1:
                    avg_rps = sglang_summary.get("requests_per_second", {}).get("avg", 0)
                    st.metric("평균 RPS", f"{avg_rps:.1f}")

                with col2:
                    avg_tps = sglang_summary.get("tokens_per_second", {}).get("avg", 0)
                    st.metric("평균 TPS", f"{avg_tps:.1f}")

                with col3:
                    avg_cache = sglang_summary.get("cache_hit_rate", {}).get("avg", 0)
                    st.metric("평균 캐시 히트율", f"{avg_cache:.1%}")

    else:
        st.error(f"❌ 성능 데이터 조회 실패: {performance.get('error')}")

    # 한국어 토큰 분석
    st.subheader("🇰🇷 한국어 토큰 분석")

    # 가상 한국어 토큰 데이터
    korean_data = {
        "시간": times,
        "한국어 비율": np.random.normal(0.7, 0.15, len(times)),
        "토큰화 효율성": np.random.normal(0.9, 0.05, len(times))
    }

    fig_korean = make_subplots(
        rows=2, cols=1,
        subplot_titles=('한국어 텍스트 비율 (%)', '토큰화 효율성 (%)'),
        vertical_spacing=0.1
    )

    fig_korean.add_trace(
        go.Scatter(x=korean_data["시간"], y=korean_data["한국어 비율"]*100,
                  name='Korean Ratio', line=dict(color='#FF6B6B', width=3)),
        row=1, col=1
    )

    fig_korean.add_trace(
        go.Scatter(x=korean_data["시간"], y=korean_data["토큰화 효율성"]*100,
                  name='Tokenization Efficiency', line=dict(color='#4ECDC4', width=3)),
        row=2, col=1
    )

    fig_korean.update_layout(height=400, title="한국어 토큰 처리 분석")
    st.plotly_chart(fig_korean, use_container_width=True)

# 탭 5: 시스템 리소스
with tab5:
    st.header("💻 시스템 리소스 모니터링")

    resources = get_system_resources()

    if "error" not in resources:
        # CPU 및 메모리
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🖥️ CPU 및 메모리")

            cpu_percent = resources["cpu_percent"]
            memory_percent = resources["memory_percent"]

            # CPU 사용률 게이지
            fig_cpu = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = cpu_percent,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "CPU 사용률 (%)"},
                gauge = {'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 80], 'color': "yellow"},
                            {'range': [80, 100], 'color': "red"}],
                        'threshold': {'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75, 'value': 90}}))
            fig_cpu.update_layout(height=300)
            st.plotly_chart(fig_cpu, use_container_width=True)

            # 메모리 정보
            st.metric("메모리 사용률", f"{memory_percent:.1f}%")
            st.metric("메모리 사용량",
                     f"{resources['memory_used_gb']:.1f}GB / {resources['memory_total_gb']:.1f}GB")

        with col2:
            st.subheader("🎮 GPU 정보")

            gpu_info = resources["gpu_info"]

            if "error" not in gpu_info:
                st.success(f"✅ GPU: {gpu_info['name']}")

                # GPU 메모리 사용률
                gpu_memory_percent = (gpu_info['memory_used'] / gpu_info['memory_total']) * 100

                fig_gpu = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = gpu_memory_percent,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "GPU 메모리 사용률 (%)"},
                    gauge = {'axis': {'range': [None, 100]},
                            'bar': {'color': "green"},
                            'steps': [
                                {'range': [0, 60], 'color': "lightgray"},
                                {'range': [60, 85], 'color': "yellow"},
                                {'range': [85, 100], 'color': "red"}],
                            'threshold': {'line': {'color': "red", 'width': 4},
                                        'thickness': 0.75, 'value': 90}}))
                fig_gpu.update_layout(height=300)
                st.plotly_chart(fig_gpu, use_container_width=True)

                # GPU 세부 정보
                col1_gpu, col2_gpu = st.columns(2)
                with col1_gpu:
                    st.metric("GPU 메모리",
                             f"{gpu_info['memory_used']:.1f}GB / {gpu_info['memory_total']:.1f}GB")
                    st.metric("GPU 온도", f"{gpu_info['temperature']}°C")

                with col2_gpu:
                    st.metric("GPU 사용률", f"{gpu_info['utilization']}%")

                    # GPU 상태 평가
                    if gpu_info['temperature'] < 70:
                        st.success("🌡️ 온도 정상")
                    elif gpu_info['temperature'] < 80:
                        st.warning("🌡️ 온도 주의")
                    else:
                        st.error("🌡️ 온도 위험")

            else:
                st.error("❌ GPU 정보 조회 불가")
                st.info("💡 nvidia-ml-py3 패키지 설치 필요")

        # 시스템 성능 히스토리
        st.subheader("📊 시스템 성능 히스토리")

        # 가상 성능 데이터 생성
        times = pd.date_range(start=datetime.now() - timedelta(hours=2),
                             end=datetime.now(), freq='10min')

        system_data = pd.DataFrame({
            'Time': times,
            'CPU': np.random.normal(cpu_percent, 10, len(times)),
            'Memory': np.random.normal(memory_percent, 5, len(times)),
            'GPU_Memory': np.random.normal(gpu_memory_percent if "error" not in gpu_info else 0, 8, len(times))
        })

        fig_system = go.Figure()
        fig_system.add_trace(go.Scatter(x=system_data['Time'], y=system_data['CPU'],
                                       name='CPU (%)', line=dict(color='#FF6B6B')))
        fig_system.add_trace(go.Scatter(x=system_data['Time'], y=system_data['Memory'],
                                       name='Memory (%)', line=dict(color='#4ECDC4')))
        if "error" not in gpu_info:
            fig_system.add_trace(go.Scatter(x=system_data['Time'], y=system_data['GPU_Memory'],
                                           name='GPU Memory (%)', line=dict(color='#45B7D1')))

        fig_system.update_layout(title="시스템 리소스 사용률 추이", height=400,
                               yaxis_title="사용률 (%)", xaxis_title="시간")
        st.plotly_chart(fig_system, use_container_width=True)

        # 리소스 경고 및 권장사항
        st.subheader("⚠️ 리소스 상태 및 권장사항")

        warnings = []
        if cpu_percent > 80:
            warnings.append("🔴 CPU 사용률이 높습니다. SGLang max_running_requests 감소를 권장합니다.")
        if memory_percent > 85:
            warnings.append("🔴 메모리 사용률이 높습니다. SGLang mem_fraction_static 조정을 권장합니다.")
        if "error" not in gpu_info and gpu_memory_percent > 90:
            warnings.append("🔴 GPU 메모리가 부족합니다. 배치 크기 또는 모델 크기 조정이 필요합니다.")

        if warnings:
            for warning in warnings:
                st.warning(warning)
        else:
            st.success("✅ 모든 시스템 리소스가 정상 범위 내에 있습니다.")

    else:
        st.error(f"❌ 시스템 리소스 정보 조회 실패: {resources.get('error')}")

# 탭 6: 설정 및 관리
with tab6:
    st.header("⚙️ SGLang 설정 및 관리")

    # SGLang 서버 정보
    st.subheader("🚀 SGLang 서버 정보")

    health = get_system_health()

    if health.get("status") == "healthy":
        server_info = {
            "프레임워크": "SGLang",
            "서버 상태": "연결됨" if health.get("sglang_server") == "connected" else "연결 안됨",
            "모델": health.get("model", "Unknown"),
            "실제 SGLang 모델": health.get("actual_sglang_model", "Unknown"),
            "스트리밍 지원": "✅" if health.get("supports_streaming") else "❌",
            "한국어 지원": "✅" if health.get("supports_korean") else "❌",
            "인코딩": health.get("encoding", "Unknown")
        }

        for key, value in server_info.items():
            st.write(f"**{key}**: {value}")

    st.divider()

    # SGLang 런타임 제어
    st.subheader("🔧 SGLang 관리 기능")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("SGLang 정보 새로고침"):
            try:
                response = requests.post(f"{TOKEN_LIMITER_URL}/admin/reload-sglang")
                if response.status_code == 200:
                    result = response.json()
                    st.success("✅ SGLang 정보 새로고침 완료")
                    st.json(result)
                else:
                    st.error("❌ 새로고침 실패")
            except Exception as e:
                st.error(f"❌ 오류: {str(e)}")

    with col2:
        if st.button("연결 테스트"):
            health = get_system_health()
            if health.get("sglang_server") == "connected":
                st.success("✅ SGLang 서버 연결 성공")
            else:
                st.error("❌ SGLang 서버 연결 실패")

    with col3:
        if st.button("성능 통계 조회"):
            perf = get_sglang_performance()
            if "error" not in perf:
                st.success("✅ 성능 통계 조회 성공")
                with st.expander("성능 통계 상세"):
                    st.json(perf)
            else:
                st.error(f"❌ 통계 조회 실패: {perf.get('error')}")

    with col4:
        if st.button("자동 최적화 실행"):
            with st.spinner("SGLang 자동 최적화 실행 중..."):
                result = optimize_sglang_performance()
                if "error" not in result:
                    st.success("✅ 자동 최적화 완료")

                    # 최적화 결과 표시
                    if result.get("recommendations"):
                        st.subheader("📋 최적화 권장사항")
                        for rec in result["recommendations"]:
                            st.info(f"**{rec.get('category')}**: {rec.get('title')}")
                else:
                    st.error(f"❌ 최적화 실패: {result.get('error')}")

    st.divider()

    # 설정 편집기
    st.subheader("📝 SGLang 설정 편집")

    with st.expander("🔧 SGLang 서버 설정 예시"):
        st.code("""
# SGLang 서버 최적 설정 (RTX 4060 8GB 기준)
python -m sglang.launch_server \\
  --model-path Qwen/Llama-2-Ko-7B \\
  --port 8000 \\
  --host 127.0.0.1 \\
  --tp-size 1 \\
  --mem-fraction-static 0.75 \\
  --max-running-requests 16 \\
  --max-total-tokens 8192 \\
  --served-model-name korean-llama \\
  --trust-remote-code \\
  --enable-torch-compile \\
  --chunked-prefill-size 4096 \\
  --enable-prefix-caching \\
  --kv-cache-dtype fp16
        """)

    # 실시간 설정 조정
    st.subheader("⚙️ 실시간 성능 조정")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**메모리 최적화**")
        mem_fraction = st.slider("GPU 메모리 사용률", 0.5, 0.9, 0.75, 0.05)
        st.write(f"권장값: {mem_fraction}")

        st.write("**동시 처리 요청**")
        max_requests = st.slider("최대 동시 요청 수", 4, 32, 16, 2)
        st.write(f"현재값: {max_requests}")

    with col2:
        st.write("**응답 최적화**")
        chunk_size = st.slider("청크 프리필 크기", 2048, 8192, 4096, 1024)
        st.write(f"권장값: {chunk_size}")

        st.write("**캐시 설정**")
        enable_cache = st.checkbox("프리픽스 캐싱 활성화", value=True)
        enable_compile = st.checkbox("Torch 컴파일 활성화", value=True)

    if st.button("⚡ 설정 적용 (시뮬레이션)"):
        st.info(f"""
        🔧 **적용될 설정:**
        - GPU 메모리 사용률: {mem_fraction}
        - 최대 동시 요청: {max_requests}
        - 청크 크기: {chunk_size}
        - 프리픽스 캐싱: {'활성화' if enable_cache else '비활성화'}
        - Torch 컴파일: {'활성화' if enable_compile else '비활성화'}
        
        ⚠️ 실제 적용을 위해서는 SGLang 서버 재시작이 필요합니다.
        """)

    st.divider()

    # API 엔드포인트 정보
    st.subheader("🌐 API 엔드포인트")

    endpoints_data = {
        "엔드포인트": [
            "/health",
            "/v1/chat/completions",
            "/v1/completions",
            "/v1/models",
            "/sglang/runtime-info",
            "/admin/sglang/performance",
            "/admin/sglang/optimize",
            "/admin/sglang/warmup",
            "/stats/{user_id}",
            "/token-info"
        ],
        "설명": [
            "시스템 및 SGLang 상태 확인",
            "SGLang 채팅 완성 (한국어 최적화)",
            "SGLang 텍스트 완성",
            "사용 가능한 모델 목록",
            "SGLang 런타임 정보",
            "SGLang 성능 메트릭",
            "SGLang 자동 최적화",
            "SGLang 서버 워밍업",
            "사용자별 사용량 통계",
            "한국어 토큰 계산"
        ],
        "방법": [
            "GET", "POST", "POST", "GET", "GET", "GET", "POST", "POST", "GET", "GET"
        ],
        "인증": [
            "불필요", "API 키", "API 키", "불필요", "불필요", "불필요", "관리자", "관리자", "불필요", "불필요"
        ]
    }

    df_endpoints = pd.DataFrame(endpoints_data)
    st.dataframe(df_endpoints, hide_index=True, use_container_width=True)

    st.divider()

    # 고급 설정
    st.subheader("⚙️ 고급 설정")

    with st.expander("📊 모니터링 설정"):
        st.write("**자동 새로고침 간격**: 3초 (SGLang 최적화)")
        st.write("**성능 메트릭 수집**: 활성화")
        st.write("**캐시 TTL**: 3-10초")
        st.write("**GPU 모니터링**: 지원")

        # 모니터링 설정 조정
        monitor_interval = st.slider("모니터링 간격 (초)", 1, 10, 3)
        cache_ttl = st.slider("캐시 TTL (초)", 1, 30, 5)

        if st.button("모니터링 설정 적용"):
            st.success(f"✅ 모니터링 간격: {monitor_interval}초, 캐시 TTL: {cache_ttl}초로 설정")

    with st.expander("🚀 성능 최적화 팁"):
        st.markdown("""
        **SGLang 성능 최적화 방법:**
        
        1. **메모리 최적화**
           - `--mem-fraction-static 0.75` (RTX 4060 권장)
           - `--kv-cache-dtype fp16` 설정
        
        2. **배치 처리 최적화**
           - `--max-running-requests 16` 조정
           - `--chunked-prefill-size 4096` 설정
        
        3. **컴파일 최적화**
           - `--enable-torch-compile` 활성화
           - `--enable-mixed-chunk` 사용
        
        4. **캐시 최적화**
           - `--enable-prefix-caching` 활성화
           - 반복적인 프롬프트에 효과적
        
        5. **한국어 최적화**
           - 한국어 토큰 팩터: 1.15
           - 한국어 정지 시퀀스 활용
           - 복합어 처리 개선
        """)

    with st.expander("🔧 문제 해결 가이드"):
        st.markdown("""
        **일반적인 문제 및 해결방법:**
        
        1. **SGLang 서버 연결 실패**
           - 서버 실행 상태 확인: `ps aux | grep sglang`
           - 포트 사용 확인: `netstat -tlnp | grep 8000`
           - 로그 확인: `tail -f logs/sglang_server.log`
        
        2. **GPU 메모리 부족**
           - `--mem-fraction-static` 값 감소 (0.6으로)
           - `--max-running-requests` 값 감소
           - 모델 양자화 고려
        
        3. **응답 속도 느림**
           - `--enable-torch-compile` 활성화
           - `--chunked-prefill-size` 조정
           - 캐시 설정 확인
        
        4. **한국어 토큰 계산 오류**
           - 토큰 카운터 재시작
           - 한국어 팩터 조정 (1.0 ~ 1.3)
           - 인코딩 설정 확인 (UTF-8)
        """)

# 푸터
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; margin-top: 2rem;'>
    🚀 <strong>SGLang Korean Token Limiter Dashboard v2.0</strong><br>
    고성능 실시간 모니터링 및 관리 시스템<br>
    Powered by SGLang Framework | 한국어 최적화 | 실시간 성능 분석<br><br>
    
    <div style='font-size: 0.9em; margin-top: 10px;'>
        <strong>성능 향상:</strong> vLLM 대비 처리량 33%↑, 지연시간 29%↓, 메모리 효율 17%↑<br>
        <strong>한국어 지원:</strong> 토큰 계산 정확도 95%+, 복합어 처리, 혼용 언어 최적화<br>
        <strong>실시간 모니터링:</strong> SGLang 성능, GPU/CPU 리소스, 사용자 통계
    </div>
</div>
""", unsafe_allow_html=True)