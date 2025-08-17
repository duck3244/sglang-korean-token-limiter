#!/bin/bash
# SGLang 깨끗한 재설치 스크립트

set -e

echo "🔄 SGLang 깨끗한 재설치"
echo "======================="

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}1단계: 기존 SGLang 완전 제거...${NC}"

# SGLang 관련 프로세스 종료
pkill -f sglang 2>/dev/null || true
pkill -f "python.*launch_server" 2>/dev/null || true

# SGLang 패키지 제거
pip uninstall sglang -y 2>/dev/null || true

# 캐시 정리
pip cache purge
rm -rf ~/.cache/pip/wheels/sglang* 2>/dev/null || true

echo -e "${GREEN}✅ 기존 SGLang 제거 완료${NC}"

echo -e "\n${BLUE}2단계: Python 환경 확인...${NC}"

python -c "
import sys
print(f'Python: {sys.version}')
print(f'가상환경: {sys.prefix}')

import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA 사용 가능: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA 버전: {torch.version.cuda}')
"

echo -e "${GREEN}✅ Python 환경 확인 완료${NC}"

echo -e "\n${BLUE}3단계: SGLang 깨끗한 설치...${NC}"

# 최신 pip 도구 설치
pip install --upgrade pip wheel setuptools

# SGLang 설치 (의존성 포함)
echo "SGLang 설치 중..."
pip install "sglang==0.2.15" --no-cache-dir

# 설치 확인
echo -e "\n${BLUE}4단계: 설치 확인...${NC}"

python -c "
try:
    import sglang
    print(f'✅ SGLang 버전: {sglang.__version__}')
    
    # 기본 import 테스트
    try:
        from sglang.srt.server import launch_server
        print('✅ sglang.srt.server 모듈 정상')
    except ImportError as e:
        print(f'⚠️ server 모듈 제한: {e}')
    
    try:
        import sglang.launch_server
        print('✅ sglang.launch_server 모듈 정상')
    except ImportError as e:
        print(f'⚠️ launch_server 모듈 제한: {e}')
    
    print('\\n🎉 SGLang 깨끗한 재설치 완료!')
    
except ImportError as e:
    print(f'❌ SGLang 설치 실패: {e}')
    exit(1)
"

echo -e "${GREEN}✅ SGLang 깨끗한 재설치 완료${NC}"
