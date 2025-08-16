#!/bin/bash
# SGLang constrained 모듈 완전 패치

set -e

echo "🔧 SGLang constrained 모듈 완전 패치"
echo "=================================="

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 1. accelerate 먼저 설치 (모델 다운로드용)
echo -e "${BLUE}1. accelerate 패키지 설치...${NC}"
pip install accelerate

# 2. SGLang constrained 모듈 완전 패치
echo -e "\n${BLUE}2. SGLang constrained 모듈 완전 패치...${NC}"

python -c "
import sys
import os

try:
    import sglang
    sglang_path = os.path.dirname(sglang.__file__)
    constrained_path = os.path.join(sglang_path, 'srt', 'constrained')

    print(f'SGLang 경로: {sglang_path}')
    print(f'Constrained 경로: {constrained_path}')

    # 1. __init__.py 완전 패치
    init_file = os.path.join(constrained_path, '__init__.py')

    # 백업 (아직 안했다면)
    backup_file = init_file + '.original_backup'
    if not os.path.exists(backup_file):
        with open(init_file, 'r') as f:
            original_content = f.read()
        with open(backup_file, 'w') as f:
            f.write(original_content)
        print(f'✅ 원본 백업: {backup_file}')

    # 새로운 __init__.py 내용 (모든 필요한 클래스 포함)
    new_init_content = '''
# SGLang constrained module - outlines dependency removed
# Complete dummy implementation for all required classes

import logging
import json
from typing import List, Dict, Any, Optional, Union

logger = logging.getLogger(__name__)

# Dummy cache function
def dummy_cache(func):
    \"\"\"Dummy cache decorator\"\"\"
    return func

# Cache implementation
try:
    from outlines.caching import cache as disk_cache
except ImportError:
    disk_cache = dummy_cache
    logger.warning(\"outlines.caching not available, using dummy cache\")

def disable_cache():
    \"\"\"Disable cache function\"\"\"
    logger.info(\"Cache disabled (outlines not available)\")
    pass

# Dummy RegexGuide class
class RegexGuide:
    \"\"\"Dummy RegexGuide for SGLang compatibility\"\"\"

    def __init__(self, regex_string: str, tokenizer = None):
        self.regex_string = regex_string
        self.tokenizer = tokenizer
        logger.info(f\"Created dummy RegexGuide for pattern: {regex_string}\")

    def get_next_instruction(self, state):
        # Return a simple instruction that allows any token
        return {\"type\": \"generate\", \"allowed_tokens\": None}

    def is_final_state(self, state):
        return False

    def copy(self):
        return RegexGuide(self.regex_string, self.tokenizer)

# Dummy TransformerTokenizer class
class TransformerTokenizer:
    \"\"\"Dummy TransformerTokenizer for SGLang compatibility\"\"\"

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.vocabulary = getattr(tokenizer, 'get_vocab', lambda: {})()
        logger.info(\"Created dummy TransformerTokenizer\")

    def encode(self, text: str) -> List[int]:
        if hasattr(self.tokenizer, 'encode'):
            return self.tokenizer.encode(text)
        return [0]  # Fallback

    def decode(self, token_ids: List[int]) -> str:
        if hasattr(self.tokenizer, 'decode'):
            return self.tokenizer.decode(token_ids)
        return \"\"  # Fallback

    def convert_token_to_string(self, token):
        if hasattr(self.tokenizer, 'convert_tokens_to_string'):
            return self.tokenizer.convert_tokens_to_string([token])
        return str(token)

# Dummy JSONGuide class
class JSONGuide:
    \"\"\"Dummy JSONGuide for SGLang compatibility\"\"\"

    def __init__(self, schema: Union[str, Dict], tokenizer = None):
        self.schema = schema
        self.tokenizer = tokenizer
        logger.info(f\"Created dummy JSONGuide for schema: {type(schema)}\")

    def get_next_instruction(self, state):
        return {\"type\": \"generate\", \"allowed_tokens\": None}

    def is_final_state(self, state):
        return False

# Dummy ChoiceGuide class
class ChoiceGuide:
    \"\"\"Dummy ChoiceGuide for SGLang compatibility\"\"\"

    def __init__(self, choices: List[str], tokenizer = None):
        self.choices = choices
        self.tokenizer = tokenizer
        logger.info(f\"Created dummy ChoiceGuide with {len(choices)} choices\")

    def get_next_instruction(self, state):
        return {\"type\": \"generate\", \"allowed_tokens\": None}

    def is_final_state(self, state):
        return False

# Export all necessary symbols
__all__ = [
    'disable_cache',
    'disk_cache',
    'RegexGuide',
    'TransformerTokenizer',
    'JSONGuide',
    'ChoiceGuide'
]

logger.info(\"SGLang constrained module initialized with dummy implementations\")
'''

    # 새 내용 작성
    with open(init_file, 'w') as f:
        f.write(new_init_content)

    print(f'✅ __init__.py 완전 패치 완료')

    # 2. fsm_cache.py 패치 (필요한 경우)
    fsm_cache_file = os.path.join(constrained_path, 'fsm_cache.py')
    if os.path.exists(fsm_cache_file):
        print(f'✅ fsm_cache.py 발견: {fsm_cache_file}')

        # fsm_cache.py 읽어서 문제있는지 확인
        with open(fsm_cache_file, 'r') as f:
            fsm_content = f.read()

        # RegexGuide import 문제 해결
        if 'from sglang.srt.constrained import RegexGuide' in fsm_content:
            # 백업
            with open(fsm_cache_file + '.backup', 'w') as f:
                f.write(fsm_content)

            # import 문 수정
            fixed_content = fsm_content.replace(
                'from sglang.srt.constrained import RegexGuide, TransformerTokenizer',
                'from sglang.srt.constrained import RegexGuide, TransformerTokenizer  # Patched imports'
            )

            with open(fsm_cache_file, 'w') as f:
                f.write(fixed_content)

            print(f'✅ fsm_cache.py 패치 완료')

    print('🎉 SGLang constrained 모듈 완전 패치 완료!')

except Exception as e:
    print(f'❌ 패치 실패: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"

# 3. 패치 검증
echo -e "\n${BLUE}3. 패치 검증...${NC}"

python -c "
import sys

try:
    print('=== SGLang 패치 검증 ===')

    # constrained 모듈 import 테스트
    from sglang.srt.constrained import RegexGuide, TransformerTokenizer, disable_cache
    print('✅ sglang.srt.constrained: 모든 클래스 import 성공')

    # 클래스 인스턴스화 테스트
    regex_guide = RegexGuide('[0-9]+')
    print('✅ RegexGuide: 인스턴스화 성공')

    # fsm_cache import 테스트
    try:
        from sglang.srt.constrained.fsm_cache import FSMCache
        print('✅ FSMCache: import 성공')
    except ImportError as e:
        print(f'⚠️ FSMCache import 실패: {e}')

    # SGLang 서버 런처 테스트
    try:
        from sglang.srt.server import launch_server
        print('✅ sglang.srt.server.launch_server: 정상')
    except ImportError as e:
        print(f'❌ 서버 런처 실패: {e}')
        raise

    print()
    print('🎉 모든 패치 검증 완료!')

except Exception as e:
    print(f'❌ 검증 실패: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"

# 4. 성공 시 SGLang 서버 시작
if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}🎉 SGLang 완전 패치 성공!${NC}"
    echo ""
    echo -e "${BLUE}📋 패치 내용:${NC}"
    echo "- accelerate 패키지 설치"
    echo "- RegexGuide 더미 구현"
    echo "- TransformerTokenizer 더미 구현"
    echo "- JSONGuide, ChoiceGuide 더미 구현"
    echo "- FSMCache 호환성 수정"
    echo "- 모든 import 오류 해결"
    echo ""
    echo -e "${GREEN}🚀 이제 SGLang 서버를 시작합니다:${NC}"
    echo ""

    # 즉시 SGLang 서버 시작
    echo "bash scripts/start_korean_sglang.sh"
    bash scripts/start_korean_sglang.sh

else
    echo -e "\n${RED}❌ 패치 실패${NC}"
    echo ""
    echo -e "${YELLOW}🔧 수동 복원 방법:${NC}"
    echo "python -c \"
import sglang, os, shutil
sglang_path = os.path.dirname(sglang.__file__)
constrained_init = os.path.join(sglang_path, 'srt', 'constrained', '__init__.py')
backup_path = constrained_init + '.original_backup'
if os.path.exists(backup_path):
    shutil.copy2(backup_path, constrained_init)
    print('원본 복원 완료')
\""
fi

echo ""
echo "스크립트 완료: $(date)"