#!/bin/bash
# SGLang 환경 문제 해결 스크립트

set -e

echo "🔧 SGLang 환경 문제 해결 중..."
echo "================================"

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 1. NumPy 다운그레이드
echo -e "${BLUE}1. NumPy 버전 문제 해결...${NC}"
echo "현재 NumPy 버전 확인:"
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')" || echo "NumPy 설치되지 않음"

echo "NumPy를 1.x 버전으로 다운그레이드..."
pip install "numpy<2.0" --force-reinstall

echo "NumPy 설치 확인:"
python -c "import numpy; print(f'✅ NumPy version: {numpy.__version__}')"

# 2. Outlines 라이브러리 특정 버전 설치 (SGLang 호환성 확보)
echo -e "\n${BLUE}2. Outlines 라이브러리 호환 버전 설치...${NC}"
echo "SGLang과 호환되는 Outlines 버전 설치 중..."

# SGLang 0.2.6과 호환되는 Outlines 버전들 시도
pip install "outlines==0.0.46" || \
pip install "outlines==0.0.45" || \
pip install "outlines==0.0.44" || \
pip install "outlines<0.1.0" --force-reinstall

echo "Outlines 설치 확인:"
python -c "
import outlines
print(f'✅ Outlines version: {outlines.__version__}')

# 핵심 모듈 체크
try:
    from outlines.fsm.guide import RegexGuide
    print('✅ outlines.fsm.guide 모듈 정상')
except ImportError as e:
    print(f'⚠️ outlines.fsm.guide 오류: {e}')
    try:
        from outlines.fsm.regex import RegexGuide
        print('✅ outlines.fsm.regex 모듈 사용')
    except ImportError as e2:
        print(f'❌ outlines.fsm 모듈들 모두 실패: {e2}')

try:
    from outlines.fsm.json_schema import build_regex_from_object
    print('✅ outlines.fsm.json_schema 모듈 정상')
except ImportError as e:
    print(f'⚠️ outlines.fsm.json_schema 일부 기능 누락: {e}')
"

# 3. SGLang 재설치 (호환성 확보)
echo -e "\n${BLUE}3. SGLang 재설치...${NC}"
pip uninstall sglang -y || true
pip install "sglang[all]==0.2.6" --no-deps --force-reinstall

# 4. 필수 의존성 수동 설치
echo -e "\n${BLUE}4. 필수 의존성 설치...${NC}"
pip install transformers==4.36.0
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install accelerate==0.25.0
pip install fastapi==0.104.1
pip install uvicorn==0.24.0
pip install httpx==0.25.2

# 5. 추가 SGLang 의존성
echo -e "\n${BLUE}5. SGLang 관련 패키지 설치...${NC}"
pip install flashinfer==0.0.5 --no-build-isolation || echo "⚠️ FlashInfer 설치 실패 (선택사항)"
pip install triton==2.1.0 || echo "⚠️ Triton 설치 실패 (선택사항)"

# 6. 환경 확인
echo -e "\n${BLUE}6. 설치 확인...${NC}"
python -c "
try:
    import numpy
    print(f'✅ NumPy: {numpy.__version__}')
    
    import torch
    print(f'✅ PyTorch: {torch.__version__}')
    
    import outlines
    print(f'✅ Outlines: {outlines.__version__}')
    
    import sglang
    print(f'✅ SGLang: {sglang.__version__}')
    
    from outlines.fsm.guide import RegexGuide
    print('✅ Outlines FSM 모듈 정상')
    
    # SGLang 핵심 모듈 체크
    from sglang.srt.server import launch_server
    print('✅ SGLang 서버 런처 정상')
    
    print('\n🎉 모든 패키지가 정상적으로 설치되었습니다!')
    
except ImportError as e:
    print(f'❌ Import 오류: {e}')
    print('추가 문제 해결이 필요합니다.')
"

echo -e "\n${GREEN}✅ 환경 수정 완료!${NC}"
echo "이제 다시 SGLang을 시작해보세요:"
echo "bash scripts/start_korean_sglang.sh"
