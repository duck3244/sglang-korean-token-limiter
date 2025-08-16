#!/bin/bash
# 누락된 SGLang 의존성 보완 스크립트

echo "🔧 SGLang 누락 의존성 보완"
echo "========================="

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 1. uvloop 설치 (성능 최적화용)
echo -e "${BLUE}1. uvloop 설치...${NC}"
pip install uvloop

# 2. 추가 웹 서버 의존성
echo -e "${BLUE}2. 웹 서버 의존성 설치...${NC}"
pip install python-multipart websockets

# 3. 로깅 및 유틸리티
echo -e "${BLUE}3. 로깅 및 유틸리티...${NC}"
pip install rich colorama

# 4. 추가 AI 라이브러리 의존성
echo -e "${BLUE}4. AI 라이브러리 의존성...${NC}"
pip install accelerate safetensors huggingface_hub

# 5. 데이터 처리
echo -e "${BLUE}5. 데이터 처리 라이브러리...${NC}"
pip install pandas PyYAML

# 6. 저장소 (선택 중 하나)
echo -e "${BLUE}6. 저장소 라이브러리...${NC}"
pip install redis aiosqlite

# 7. 한국어 프로젝트 전용
echo -e "${BLUE}7. 프로젝트 전용 패키지...${NC}"
pip install streamlit plotly

# 8. 종합 검증
echo -e "${BLUE}8. 종합 검증...${NC}"
python -c "
import sys
print(f'🐍 Python: {sys.version}')
print()

# 핵심 패키지들 체크
packages_to_check = [
    ('sglang', 'SGLang'),
    ('torch', 'PyTorch'),
    ('transformers', 'Transformers'),
    ('outlines', 'Outlines'),
    ('fastapi', 'FastAPI'),
    ('uvicorn', 'Uvicorn'),
    ('uvloop', 'UVLoop'),
    ('httpx', 'HTTPX'),
    ('sse_starlette', 'SSE Starlette'),
    ('redis', 'Redis'),
    ('pandas', 'Pandas'),
    ('streamlit', 'Streamlit'),
    ('plotly', 'Plotly'),
]

success_count = 0
total_count = len(packages_to_check)

for pkg, name in packages_to_check:
    try:
        module = __import__(pkg)
        version = getattr(module, '__version__', 'Unknown')
        print(f'✅ {name}: {version}')
        success_count += 1
    except ImportError:
        print(f'❌ {name}: 설치되지 않음')

print()

# SGLang 특화 모듈 테스트
print('🔍 SGLang 특화 모듈 테스트:')
sglang_modules = [
    ('sglang.srt.server', 'SGLang 서버'),
    ('outlines.fsm.guide', 'Outlines FSM'),
]

for module_name, desc in sglang_modules:
    try:
        parts = module_name.split('.')
        module = __import__(module_name, fromlist=[parts[-1]])
        print(f'✅ {desc}: 정상')
        success_count += 1
    except ImportError as e:
        print(f'❌ {desc}: {e}')
    total_count += 1

print()

# GPU 확인
try:
    import torch
    if torch.cuda.is_available():
        print(f'✅ CUDA: {torch.version.cuda}')
        print(f'✅ GPU: {torch.cuda.get_device_name()}')
    else:
        print('💻 CPU 모드')
except:
    print('❌ PyTorch GPU 확인 실패')

print()
success_rate = (success_count / total_count) * 100
print(f'📊 전체 성공률: {success_count}/{total_count} ({success_rate:.1f}%)')

if success_rate >= 85:
    print('🎉 모든 주요 패키지가 정상 설치되었습니다!')
    print('이제 SGLang Korean Token Limiter를 시작할 수 있습니다.')
    print()
    print('🚀 시작 명령어:')
    print('  bash scripts/start_korean_sglang.sh')
    print()
    print('🎮 대시보드:')
    print('  streamlit run dashboard/sglang_app.py --server.port 8501')

elif success_rate >= 70:
    print('⚠️ 대부분의 패키지가 설치되었습니다.')
    print('기본 기능은 작동할 것입니다.')
    print('누락된 패키지들은 필요시 개별적으로 설치하세요.')

else:
    print('❌ 많은 패키지가 누락되었습니다.')
    print('다음을 시도해보세요:')
    print('1. pip install --upgrade pip')
    print('2. conda update --all')
    print('3. 새 환경에서 다시 설치')
"

echo -e "\n${GREEN}✅ 의존성 보완 완료!${NC}"
echo ""
echo -e "${BLUE}🎯 다음 단계:${NC}"
echo "1. SGLang 테스트: python -c \"from sglang.srt.server import launch_server; print('SGLang 준비완료')\""
echo "2. 시스템 시작: bash scripts/start_korean_sglang.sh"
echo "3. 대시보드: streamlit run dashboard/sglang_app.py --server.port 8501"