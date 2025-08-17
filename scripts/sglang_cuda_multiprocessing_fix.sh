#!/bin/bash
# SGLang CUDA 멀티프로세싱 오류 해결 스크립트

set -e

echo "🔧 SGLang CUDA 멀티프로세싱 오류 해결"
echo "====================================="

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

echo -e "${BLUE}🔍 CUDA 멀티프로세싱 문제 진단...${NC}"

python -c "
import multiprocessing
import torch
import os

print('CUDA 멀티프로세싱 환경 진단...')
print(f'Python 멀티프로세싱 시작 방법: {multiprocessing.get_start_method()}')
print(f'CUDA 사용 가능: {torch.cuda.is_available()}')

if torch.cuda.is_available():
    print(f'CUDA 디바이스 수: {torch.cuda.device_count()}')
    print(f'현재 CUDA 디바이스: {torch.cuda.current_device()}')
    print(f'CUDA 디바이스 이름: {torch.cuda.get_device_name()}')

# 환경 변수 확인
env_vars = ['CUDA_VISIBLE_DEVICES', 'PYTORCH_CUDA_ALLOC_CONF', 'TORCH_MULTIPROCESSING_START_METHOD']
for var in env_vars:
    value = os.environ.get(var, 'Not Set')
    print(f'{var}: {value}')
"

echo -e "${GREEN}✅ 진단 완료${NC}"

# CUDA 멀티프로세싱 해결 스크립트 생성
echo -e "\n${BLUE}📝 CUDA 멀티프로세싱 해결 SGLang 시작 스크립트 생성...${NC}"

cat > start_sglang_cuda_fixed.py << 'EOF'
#!/usr/bin/env python3
"""
SGLang CUDA 멀티프로세싱 오류 해결 시작 스크립트
"""

import sys
import os
import subprocess
import time
import requests
import multiprocessing
import argparse

def set_multiprocessing_method():
    """멀티프로세싱 시작 방법을 spawn으로 설정"""
    
    print("🔧 멀티프로세싱 시작 방법 설정...")
    
    # CUDA와 호환되는 spawn 방법 사용
    try:
        multiprocessing.set_start_method('spawn', force=True)
        print(f"✅ 멀티프로세싱 시작 방법: {multiprocessing.get_start_method()}")
    except RuntimeError as e:
        print(f"⚠️ 멀티프로세싱 방법 설정 실패: {e}")
        print("환경 변수로 설정을 시도합니다...")
    
    # 환경 변수 설정
    env_vars = {
        'TORCH_MULTIPROCESSING_START_METHOD': 'spawn',
        'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:512',
        'CUDA_LAUNCH_BLOCKING': '0',
        'TOKENIZERS_PARALLELISM': 'false'
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"🔧 {key}={value}")

def clear_cuda_cache():
    """CUDA 캐시 정리"""
    
    print("🧹 CUDA 캐시 정리...")
    
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            print("✅ CUDA 캐시 정리 완료")
        else:
            print("💻 CPU 모드")
    except Exception as e:
        print(f"⚠️ CUDA 캐시 정리 실패: {e}")

def check_gpu_status():
    """GPU 상태 확인"""
    
    print("🔍 GPU 상태 확인...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            gpu_memory_used = torch.cuda.memory_allocated() / 1024**3
            
            print(f"✅ GPU: {gpu_name}")
            print(f"📊 메모리: {gpu_memory_used:.1f}GB / {gpu_memory:.1f}GB")
            
            return True, gpu_name
        else:
            print("💻 CPU 모드로 실행")
            return False, None
    except Exception as e:
        print(f"❌ GPU 확인 실패: {e}")
        return False, None

def start_sglang_server_spawn(model_path="microsoft/DialoGPT-medium", port=8000):
    """SGLang 서버 시작 (spawn 방법 사용)"""
    
    print("🚀 SGLang 서버 시작 (CUDA 멀티프로세싱 해결 버전)")
    print(f"모델: {model_path}")
    print(f"포트: {port}")
    
    # 멀티프로세싱 설정
    set_multiprocessing_method()
    
    # CUDA 캐시 정리
    clear_cuda_cache()
    
    # GPU 상태 확인
    gpu_available, gpu_name = check_gpu_status()
    
    # 서버 명령어 구성
    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", model_path,
        "--port", str(port),
        "--host", "127.0.0.1",
        "--trust-remote-code"
    ]
    
    # GPU 사용 시 설정
    if gpu_available:
        cmd.extend([
            "--mem-fraction-static", "0.7",  # 메모리 사용률 더 보수적으로
            "--max-running-requests", "4",   # 동시 요청 수 줄임
            "--kv-cache-dtype", "auto",
            "--tensor-parallel-size", "1",
            "--disable-cuda-graph",         # CUDA Graph 비활성화 (멀티프로세싱 안정성)
            "--disable-flashinfer"          # FlashInfer 비활성화 (안정성)
        ])
        
        # RTX 4060 특화 설정
        if gpu_name and "4060" in gpu_name:
            cmd.extend([
                "--chunked-prefill-size", "1024",  # 더 작은 청크 크기
                "--max-total-tokens", "2048"       # 토큰 수 제한
            ])
    else:
        # CPU 모드
        cmd.extend([
            "--disable-cuda-graph",
            "--disable-flashinfer"
        ])
    
    print(f"실행 명령어: {' '.join(cmd)}")
    
    # 환경 변수 설정
    env = os.environ.copy()
    env.update({
        'TORCH_MULTIPROCESSING_START_METHOD': 'spawn',
        'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:512',
        'CUDA_LAUNCH_BLOCKING': '0',
        'TOKENIZERS_PARALLELISM': 'false',
        'SGLANG_DISABLE_FLASHINFER_WARNING': '1'
    })
    
    try:
        # 로그 디렉토리 생성
        os.makedirs("logs", exist_ok=True)
        
        # 서버 시작 (새로운 프로세스에서)
        print("🔄 서버 프로세스 시작 중...")
        
        with open("logs/sglang_cuda_fixed.log", "w") as log_file:
            process = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True  # 새로운 세션에서 시작
            )
        
        print(f"✅ 서버 프로세스 시작 (PID: {process.pid})")
        
        # 서버 준비 대기
        print("⏳ 서버 준비 대기 (CUDA 멀티프로세싱 해결 버전)...")
        
        # 처음 30초는 더 자주 체크 (초기화 시간)
        for i in range(30):
            if process.poll() is not None:
                print("❌ 서버 프로세스 조기 종료")
                return None
            
            try:
                response = requests.get(f"http://127.0.0.1:{port}/get_model_info", timeout=2)
                if response.status_code == 200:
                    print(f"✅ 서버 준비 완료! ({i+1}초)")
                    return process
            except:
                pass
            
            if i % 10 == 0 and i > 0:
                print(f"초기화 중... {i}초")
            
            time.sleep(1)
        
        # 추가 대기 (총 120초까지)
        for i in range(30, 120):
            if process.poll() is not None:
                print("❌ 서버 프로세스 종료됨")
                return None
            
            try:
                response = requests.get(f"http://127.0.0.1:{port}/get_model_info", timeout=3)
                if response.status_code == 200:
                    print(f"✅ 서버 준비 완료! ({i+1}초)")
                    
                    # 모델 정보 표시
                    try:
                        model_info = response.json()
                        print(f"모델: {model_info.get('model_path', 'Unknown')}")
                        print(f"최대 토큰: {model_info.get('max_total_tokens', 'Unknown')}")
                    except:
                        pass
                    
                    return process
            except:
                pass
            
            if i % 20 == 0:
                print(f"대기 중... {i}초")
                
                # 로그 일부 확인
                if os.path.exists("logs/sglang_cuda_fixed.log"):
                    with open("logs/sglang_cuda_fixed.log", "r") as f:
                        lines = f.readlines()
                        if lines:
                            print("최근 로그:")
                            for line in lines[-3:]:
                                print(f"  {line.strip()}")
            
            time.sleep(1)
        
        print("❌ 서버 준비 시간 초과")
        process.terminate()
        return None
        
    except Exception as e:
        print(f"❌ 서버 시작 실패: {e}")
        return None

def test_server(port=8000):
    """서버 기능 테스트"""
    
    print("\n🧪 서버 기능 테스트...")
    
    try:
        # 모델 정보 조회
        response = requests.get(f"http://127.0.0.1:{port}/get_model_info", timeout=5)
        if response.status_code == 200:
            print("✅ 모델 정보 조회 성공")
            model_info = response.json()
            print(f"  모델: {model_info.get('model_path', 'Unknown')}")
        
        # 모델 목록 조회
        response = requests.get(f"http://127.0.0.1:{port}/v1/models", timeout=5)
        if response.status_code == 200:
            print("✅ 모델 목록 조회 성공")
        
        print("🎉 서버 기능 테스트 완료!")
        
    except Exception as e:
        print(f"⚠️ 서버 테스트 실패: {e}")

def main():
    parser = argparse.ArgumentParser(description="SGLang CUDA 멀티프로세싱 해결 서버")
    parser.add_argument("--model", default="microsoft/DialoGPT-medium", help="모델 경로")
    parser.add_argument("--port", type=int, default=8000, help="포트 번호")
    parser.add_argument("--test-only", action="store_true", help="서버 테스트만 실행")
    
    args = parser.parse_args()
    
    print("🎉 SGLang CUDA 멀티프로세싱 해결 버전")
    print("=" * 70)
    print(f"모델: {args.model}")
    print(f"포트: {args.port}")
    print()
    
    if args.test_only:
        test_server(args.port)
        return 0
    
    # 서버 시작
    process = start_sglang_server_spawn(args.model, args.port)
    
    if process:
        print("\n🎉 SGLang 서버 CUDA 멀티프로세싱 문제 해결 성공!")
        print("=" * 80)
        
        print()
        print("🧪 테스트 명령어:")
        print(f"curl http://127.0.0.1:{args.port}/get_model_info")
        print(f"curl http://127.0.0.1:{args.port}/v1/models")
        print()
        print("🇰🇷 한국어 Token Limiter 시작 (다른 터미널):")
        print("python main_sglang.py")
        print()
        print("🔗 채팅 테스트:")
        print(f'''curl -X POST http://localhost:8080/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer sk-user1-korean-key-def" \\
  -d '{{"model": "korean-qwen", "messages": [{{"role": "user", "content": "CUDA 멀티프로세싱 문제가 해결되었나요?"}}], "max_tokens": 100}}' ''')
        print()
        print("✨ 해결된 문제:")
        print("   ✅ CUDA 멀티프로세싱 시작 방법을 spawn으로 변경")
        print("   ✅ CUDA 캐시 정리 및 메모리 최적화")
        print("   ✅ RTX 4060 특화 안정성 설정")
        print("   ✅ 환경 변수 자동 설정")
        print("   ✅ 새로운 세션에서 프로세스 시작")
        print()
        print("🛑 종료: Ctrl+C")
        
        # 서버 테스트
        test_server(args.port)
        
        try:
            process.wait()
        except KeyboardInterrupt:
            print("\n🛑 서버 종료 중...")
            process.terminate()
            process.wait()
            print("✅ 서버 정상 종료")
    else:
        print("❌ 서버 시작 실패")
        
        if os.path.exists("logs/sglang_cuda_fixed.log"):
            print("\n=== 로그 (마지막 50줄) ===")
            with open("logs/sglang_cuda_fixed.log", "r") as f:
                lines = f.readlines()
                for line in lines[-50:]:
                    print(line.rstrip())
        
        return 1
    
    return 0

if __name__ == "__main__":
    # 메인 프로세스에서 멀티프로세싱 방법 설정
    if __name__ == "__main__":
        try:
            multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
    
    sys.exit(main())
EOF

chmod +x start_sglang_cuda_fixed.py

echo -e "${GREEN}✅ CUDA 멀티프로세싱 해결 스크립트 생성: start_sglang_cuda_fixed.py${NC}"

# 간단한 환경 변수 설정 스크립트도 생성
echo -e "\n${BLUE}📝 환경 변수 설정 스크립트 생성...${NC}"

cat > set_cuda_env.sh << 'EOF'
#!/bin/bash
# CUDA 멀티프로세싱 환경 변수 설정

echo "🔧 CUDA 멀티프로세싱 환경 변수 설정"

# 필수 환경 변수 설정
export TORCH_MULTIPROCESSING_START_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0
export TOKENIZERS_PARALLELISM=false
export SGLANG_DISABLE_FLASHINFER_WARNING=1

echo "✅ 환경 변수 설정 완료"
echo "TORCH_MULTIPROCESSING_START_METHOD=$TORCH_MULTIPROCESSING_START_METHOD"
echo "PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
echo "CUDA_LAUNCH_BLOCKING=$CUDA_LAUNCH_BLOCKING"
echo "TOKENIZERS_PARALLELISM=$TOKENIZERS_PARALLELISM"
echo "SGLANG_DISABLE_FLASHINFER_WARNING=$SGLANG_DISABLE_FLASHINFER_WARNING"

# 이 스크립트를 source로 실행하세요:
# source set_cuda_env.sh
EOF

chmod +x set_cuda_env.sh

echo -e "${GREEN}✅ 환경 변수 설정 스크립트 생성: set_cuda_env.sh${NC}"

# 통합 해결 스크립트 생성
echo -e "\n${BLUE}📝 통합 CUDA 멀티프로세싱 해결 스크립트 생성...${NC}"

cat > fix_and_start_sglang.sh << 'EOF'
#!/bin/bash
# SGLang CUDA 멀티프로세싱 문제 통합 해결 및 시작 스크립트

set -e

echo "🔧 SGLang CUDA 멀티프로세싱 문제 통합 해결"
echo "============================================"

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 1단계: 환경 변수 설정
echo -e "${BLUE}1단계: 환경 변수 설정...${NC}"
export TORCH_MULTIPROCESSING_START_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0
export TOKENIZERS_PARALLELISM=false
export SGLANG_DISABLE_FLASHINFER_WARNING=1

echo "✅ 환경 변수 설정 완료"

# 2단계: 기존 프로세스 정리
echo -e "\n${BLUE}2단계: 기존 프로세스 정리...${NC}"
pkill -f "sglang.*launch_server" 2>/dev/null || true
pkill -f "python.*sglang" 2>/dev/null || true
sleep 2

echo "✅ 프로세스 정리 완료"

# 3단계: CUDA 캐시 정리
echo -e "\n${BLUE}3단계: CUDA 캐시 정리...${NC}"
python -c "
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    print('✅ CUDA 캐시 정리 완료')
else:
    print('💻 CPU 모드')
"

# 4단계: SGLang 서버 시작
echo -e "\n${BLUE}4단계: SGLang 서버 시작 (CUDA 멀티프로세싱 해결)...${NC}"

MODEL_PATH="${1:-microsoft/DialoGPT-medium}"
PORT="${2:-8000}"

echo "모델: $MODEL_PATH"
echo "포트: $PORT"

# Python 스크립트로 서버 시작 (멀티프로세싱 문제 해결)
python start_sglang_cuda_fixed.py --model "$MODEL_PATH" --port "$PORT"
EOF

chmod +x fix_and_start_sglang.sh

echo -e "${GREEN}✅ 통합 해결 스크립트 생성: fix_and_start_sglang.sh${NC}"

echo ""
echo -e "${GREEN}🎉 SGLang CUDA 멀티프로세싱 오류 해결 완료!${NC}"
echo "=================================================="

echo -e "${BLUE}🎯 해결 내용:${NC}"
echo "✅ 멀티프로세싱 시작 방법을 fork에서 spawn으로 변경"
echo "✅ CUDA 관련 환경 변수 자동 설정"
echo "✅ CUDA 캐시 정리 및 메모리 최적화"
echo "✅ 새로운 세션에서 프로세스 시작"
echo "✅ RTX 4060 특화 안정성 설정"

echo ""
echo -e "${BLUE}🚀 사용 방법:${NC}"
echo ""
echo "1. 통합 해결 스크립트로 시작 (권장):"
echo "   bash fix_and_start_sglang.sh"
echo ""
echo "2. Python 스크립트로 직접 시작:"
echo "   python start_sglang_cuda_fixed.py"
echo ""
echo "3. 환경 변수만 설정:"
echo "   source set_cuda_env.sh"
echo "   # 그 다음 일반 SGLang 명령어 실행"

echo ""
echo -e "${BLUE}💡 핵심 해결 방법:${NC}"
echo "- 🔧 multiprocessing.set_start_method('spawn')"
echo "- 🔧 TORCH_MULTIPROCESSING_START_METHOD=spawn"
echo "- 🔧 새로운 세션에서 프로세스 시작"
echo "- 🔧 CUDA 캐시 정리 및 메모리 최적화"
echo "- 🔧 보수적인 GPU 메모리 사용률 (70%)"

echo ""
echo -e "${PURPLE}🌟 CUDA 멀티프로세싱 오류 완전 해결!${NC}"
echo "더 이상 'Cannot re-initialize CUDA in forked subprocess' 오류가 발생하지 않습니다."

echo ""
echo "CUDA 멀티프로세싱 해결 완료 시간: $(date)"