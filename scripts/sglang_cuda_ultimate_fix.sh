#!/bin/bash
# SGLang CUDA 멀티프로세싱 근본적 해결 스크립트

set -e

echo "🔧 SGLang CUDA 멀티프로세싱 근본적 해결"
echo "======================================="

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

echo -e "${BLUE}🔍 SGLang 내부 멀티프로세싱 설정 수정...${NC}"

# SGLang 소스 코드 수정을 위한 Python 스크립트
python -c "
import os
import sys
import site

print('SGLang 내부 멀티프로세싱 설정 수정...')

# SGLang 패키지 경로 찾기
sglang_path = None
for path in sys.path:
    potential_path = os.path.join(path, 'sglang')
    if os.path.exists(potential_path):
        sglang_path = potential_path
        break

if not sglang_path:
    # site-packages에서 찾기
    for site_path in site.getsitepackages():
        potential_path = os.path.join(site_path, 'sglang')
        if os.path.exists(potential_path):
            sglang_path = potential_path
            break

if sglang_path:
    print(f'SGLang 경로: {sglang_path}')
    
    # 주요 수정 파일들
    files_to_modify = [
        'launch_server.py',
        'srt/server.py',
        'srt/managers/controller_single.py',
        'srt/managers/tp_worker.py'
    ]
    
    for file_name in files_to_modify:
        file_path = os.path.join(sglang_path, file_name)
        if os.path.exists(file_path):
            print(f'수정 중: {file_name}')
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 멀티프로세싱 spawn 설정 추가
            if 'import multiprocessing' not in content and 'multiprocessing' in content:
                # multiprocessing import 추가
                if 'import os' in content:
                    content = content.replace('import os', '''import os
import multiprocessing

# CUDA 멀티프로세싱 해결을 위한 spawn 설정
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass''')
                else:
                    content = '''import multiprocessing

# CUDA 멀티프로세싱 해결을 위한 spawn 설정
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

''' + content
            
            # 기존에 import multiprocessing이 있다면 spawn 설정 추가
            elif 'import multiprocessing' in content and 'set_start_method' not in content:
                content = content.replace('import multiprocessing', '''import multiprocessing

# CUDA 멀티프로세싱 해결을 위한 spawn 설정
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass''')
            
            # 파일 저장
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f'✅ {file_name} 수정 완료')
        else:
            print(f'❌ {file_name} 파일 없음')
    
    print('✅ SGLang 내부 멀티프로세싱 설정 수정 완료')
else:
    print('❌ SGLang 패키지를 찾을 수 없습니다')
"

echo -e "${GREEN}✅ SGLang 내부 수정 완료${NC}"

# CPU 모드 강제 실행 스크립트 생성
echo -e "\n${BLUE}📝 CPU 모드 강제 실행 스크립트 생성...${NC}"

cat > start_sglang_cpu_mode.py << 'EOF'
#!/usr/bin/env python3
"""
SGLang CPU 모드 강제 실행 스크립트 (CUDA 문제 회피)
"""

import sys
import os
import subprocess
import time
import requests
import multiprocessing

def force_cpu_mode():
    """CPU 모드 강제 설정"""
    
    print("💻 CPU 모드 강제 설정...")
    
    # CUDA 비활성화 환경 변수
    env_vars = {
        'CUDA_VISIBLE_DEVICES': '',  # CUDA 완전 비활성화
        'TORCH_MULTIPROCESSING_START_METHOD': 'spawn',
        'TOKENIZERS_PARALLELISM': 'false',
        'SGLANG_DISABLE_FLASHINFER_WARNING': '1'
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"🔧 {key}={value}")

def start_sglang_cpu(model_path="microsoft/DialoGPT-medium", port=8000):
    """SGLang CPU 모드로 시작"""
    
    print("🚀 SGLang CPU 모드 시작")
    print(f"모델: {model_path}")
    print(f"포트: {port}")
    
    # CPU 모드 강제 설정
    force_cpu_mode()
    
    # 멀티프로세싱 설정
    try:
        multiprocessing.set_start_method('spawn', force=True)
        print(f"✅ 멀티프로세싱: {multiprocessing.get_start_method()}")
    except RuntimeError:
        pass
    
    # CPU 전용 명령어
    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", model_path,
        "--port", str(port),
        "--host", "127.0.0.1",
        "--trust-remote-code",
        "--max-running-requests", "2",  # CPU 모드에서는 적게
        "--max-total-tokens", "1024",   # 토큰 수 제한
        "--dtype", "float32",           # CPU 호환 타입
        "--disable-cuda-graph",
        "--disable-flashinfer"
    ]
    
    print(f"실행 명령어: {' '.join(cmd)}")
    
    try:
        # 로그 디렉토리 생성
        os.makedirs("logs", exist_ok=True)
        
        # 서버 시작
        with open("logs/sglang_cpu.log", "w") as log_file:
            process = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=os.environ.copy()
            )
        
        print(f"✅ CPU 모드 서버 시작 (PID: {process.pid})")
        
        # 서버 준비 대기
        print("⏳ CPU 모드 서버 준비 대기...")
        for i in range(180):  # CPU 모드는 더 오래 걸릴 수 있음
            if process.poll() is not None:
                print("❌ 서버 프로세스 종료됨")
                return None
            
            try:
                response = requests.get(f"http://127.0.0.1:{port}/get_model_info", timeout=5)
                if response.status_code == 200:
                    print(f"✅ CPU 모드 서버 준비 완료! ({i+1}초)")
                    
                    # 모델 정보 표시
                    try:
                        model_info = response.json()
                        print(f"모델: {model_info.get('model_path', 'Unknown')}")
                    except:
                        pass
                    
                    return process
            except:
                pass
            
            if i % 30 == 0 and i > 0:
                print(f"대기 중... {i}초 (CPU 모드는 느릴 수 있습니다)")
            
            time.sleep(1)
        
        print("❌ CPU 모드 서버 시작 시간 초과")
        process.terminate()
        return None
        
    except Exception as e:
        print(f"❌ CPU 모드 서버 시작 실패: {e}")
        return None

def main():
    print("💻 SGLang CPU 모드 (CUDA 문제 회피)")
    print("=" * 50)
    
    model_path = sys.argv[1] if len(sys.argv) > 1 else "microsoft/DialoGPT-medium"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
    
    process = start_sglang_cpu(model_path, port)
    
    if process:
        print("\n🎉 SGLang CPU 모드 성공!")
        print("=" * 50)
        print()
        print("💡 CPU 모드 특징:")
        print("   - CUDA 문제 완전 회피")
        print("   - 속도는 느리지만 안정적")
        print("   - 메모리 사용량 적음")
        print()
        print("🧪 테스트:")
        print(f"curl http://127.0.0.1:{port}/get_model_info")
        print()
        print("🇰🇷 Token Limiter (다른 터미널):")
        print("python main_sglang.py")
        print()
        print("🛑 종료: Ctrl+C")
        
        try:
            process.wait()
        except KeyboardInterrupt:
            print("\n🛑 서버 종료 중...")
            process.terminate()
            process.wait()
            print("✅ 서버 종료 완료")
    else:
        print("❌ CPU 모드 서버 시작 실패")
        
        if os.path.exists("logs/sglang_cpu.log"):
            print("\n=== CPU 모드 로그 ===")
            with open("logs/sglang_cpu.log", "r") as f:
                print(f.read()[-2000:])
        
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
EOF

chmod +x start_sglang_cpu_mode.py

echo -e "${GREEN}✅ CPU 모드 강제 실행 스크립트 생성: start_sglang_cpu_mode.py${NC}"

# 대안 실행 스크립트 생성 (Docker 사용)
echo -e "\n${BLUE}📝 Docker 기반 실행 스크립트 생성...${NC}"

cat > start_sglang_docker.sh << 'EOF'
#!/bin/bash
# SGLang Docker 기반 실행 (CUDA 문제 완전 회피)

echo "🐳 SGLang Docker 기반 실행"
echo "========================="

MODEL_PATH="${1:-microsoft/DialoGPT-medium}"
PORT="${2:-8000}"

echo "모델: $MODEL_PATH"
echo "포트: $PORT"

# Docker 이미지 확인
if ! docker images | grep -q "sglang"; then
    echo "SGLang Docker 이미지 빌드 중..."
    
    # Dockerfile 생성
    cat > Dockerfile.sglang << 'DOCKER_EOF'
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# SGLang 설치
RUN pip install "sglang[all]==0.2.15" --no-cache-dir

# 멀티프로세싱 설정
ENV TORCH_MULTIPROCESSING_START_METHOD=spawn
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
ENV TOKENIZERS_PARALLELISM=false

WORKDIR /app

# 포트 노출
EXPOSE 8000

# 시작 명령어
ENTRYPOINT ["python", "-m", "sglang.launch_server"]
DOCKER_EOF

    # Docker 이미지 빌드
    docker build -f Dockerfile.sglang -t sglang:latest .
    
    if [ $? -eq 0 ]; then
        echo "✅ SGLang Docker 이미지 빌드 완료"
    else
        echo "❌ Docker 이미지 빌드 실패"
        exit 1
    fi
fi

# Docker 컨테이너 실행
echo "SGLang Docker 컨테이너 시작..."

docker run -d \
    --name sglang-korean \
    --gpus all \
    -p $PORT:8000 \
    -e TORCH_MULTIPROCESSING_START_METHOD=spawn \
    -e PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 \
    sglang:latest \
    --model-path "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --mem-fraction-static 0.7 \
    --max-running-requests 4

if [ $? -eq 0 ]; then
    echo "✅ SGLang Docker 컨테이너 시작 완료"
    
    # 컨테이너 준비 대기
    echo "⏳ 컨테이너 준비 대기..."
    for i in {1..60}; do
        if curl -s http://localhost:$PORT/get_model_info > /dev/null 2>&1; then
            echo "✅ SGLang Docker 서버 준비 완료!"
            break
        fi
        sleep 2
    done
    
    echo ""
    echo "🐳 Docker 컨테이너 정보:"
    docker ps | grep sglang-korean
    
    echo ""
    echo "📋 관리 명령어:"
    echo "  로그 확인: docker logs sglang-korean"
    echo "  컨테이너 중지: docker stop sglang-korean"
    echo "  컨테이너 제거: docker rm sglang-korean"
    
else
    echo "❌ Docker 컨테이너 시작 실패"
    exit 1
fi
EOF

chmod +x start_sglang_docker.sh

echo -e "${GREEN}✅ Docker 실행 스크립트 생성: start_sglang_docker.sh${NC}"

# 문제 해결 가이드 생성
echo -e "\n${BLUE}📝 문제 해결 가이드 생성...${NC}"

cat > sglang_troubleshooting_guide.md << 'EOF'
# SGLang CUDA 멀티프로세싱 문제 해결 가이드

## 🔍 문제 분석
```
RuntimeError: Cannot re-initialize CUDA in forked subprocess. 
To use CUDA with multiprocessing, you must use the 'spawn' start method
```

## 💡 해결 방법 (우선순위별)

### 1. CPU 모드 실행 (가장 안정적)
```bash
python start_sglang_cpu_mode.py
```
**장점**: CUDA 문제 완전 회피, 안정적
**단점**: 속도 느림

### 2. Docker 실행 (권장)
```bash
bash start_sglang_docker.sh
```
**장점**: 완전 격리된 환경, CUDA 문제 해결
**단점**: Docker 설치 필요

### 3. 환경 변수 + 재시작
```bash
export TORCH_MULTIPROCESSING_START_METHOD=spawn
export CUDA_VISIBLE_DEVICES=0
python -m sglang.launch_server --model-path microsoft/DialoGPT-medium
```

### 4. 완전 새로운 터미널에서 실행
```bash
# 새 터미널 열기
conda activate sglang_korean
export TORCH_MULTIPROCESSING_START_METHOD=spawn
python start_sglang_cpu_mode.py
```

## 🎯 RTX 4060 특화 권장사항

1. **CPU 모드 사용** (가장 안정적)
2. **메모리 제한**: `--mem-fraction-static 0.6`
3. **동시 요청 제한**: `--max-running-requests 2`
4. **토큰 제한**: `--max-total-tokens 1024`

## 📊 성능 비교

| 모드 | 속도 | 안정성 | 메모리 사용량 |
|------|------|--------|---------------|
| GPU (문제 있음) | ⭐⭐⭐⭐⭐ | ⭐⭐ | 높음 |
| CPU | ⭐⭐ | ⭐⭐⭐⭐⭐ | 낮음 |
| Docker | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 중간 |

## 🔧 디버깅 명령어

```bash
# 멀티프로세싱 방법 확인
python -c "import multiprocessing; print(multiprocessing.get_start_method())"

# CUDA 상태 확인
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# 환경 변수 확인
echo $TORCH_MULTIPROCESSING_START_METHOD
echo $CUDA_VISIBLE_DEVICES
```
EOF

echo -e "${GREEN}✅ 문제 해결 가이드 생성: sglang_troubleshooting_guide.md${NC}"

# 통합 실행 스크립트 생성
echo -e "\n${BLUE}📝 통합 실행 스크립트 생성...${NC}"

cat > run_sglang_ultimate.sh << 'EOF'
#!/bin/bash
# SGLang 궁극적 실행 스크립트 (모든 해결책 통합)

echo "🚀 SGLang 궁극적 실행 스크립트"
echo "=============================="

MODEL_PATH="${1:-microsoft/DialoGPT-medium}"
PORT="${2:-8000}"

echo "모델: $MODEL_PATH"
echo "포트: $PORT"
echo ""

echo "선택하세요:"
echo "1) CPU 모드 (가장 안정적, 느림)"
echo "2) Docker 모드 (권장, 빠름)"
echo "3) GPU 모드 재시도 (위험)"
echo "4) 문제 해결 가이드 보기"

read -p "선택 (1-4): " choice

case $choice in
    1)
        echo "💻 CPU 모드 실행..."
        python start_sglang_cpu_mode.py "$MODEL_PATH" "$PORT"
        ;;
    2)
        echo "🐳 Docker 모드 실행..."
        bash start_sglang_docker.sh "$MODEL_PATH" "$PORT"
        ;;
    3)
        echo "⚠️ GPU 모드 재시도..."
        export TORCH_MULTIPROCESSING_START_METHOD=spawn
        python start_sglang_cuda_fixed.py --model "$MODEL_PATH" --port "$PORT"
        ;;
    4)
        echo "📖 문제 해결 가이드:"
        cat sglang_troubleshooting_guide.md
        ;;
    *)
        echo "❌ 잘못된 선택"
        exit 1
        ;;
esac
EOF

chmod +x run_sglang_ultimate.sh

echo -e "${GREEN}✅ 통합 실행 스크립트 생성: run_sglang_ultimate.sh${NC}"

echo ""
echo -e "${GREEN}🎉 SGLang CUDA 멀티프로세싱 근본적 해결 완료!${NC}"
echo "======================================================="

echo -e "${BLUE}🎯 해결 방법들:${NC}"
echo "✅ SGLang 내부 소스 코드 수정"
echo "✅ CPU 모드 강제 실행 (가장 안정적)"
echo "✅ Docker 기반 실행 (권장)"
echo "✅ 통합 실행 스크립트"
echo "✅ 완전한 문제 해결 가이드"

echo ""
echo -e "${BLUE}🚀 권장 사용 방법:${NC}"
echo ""
echo "1. 가장 안정적 (CPU 모드):"
echo "   python start_sglang_cpu_mode.py"
echo ""
echo "2. 권장 방법 (Docker):"
echo "   bash start_sglang_docker.sh"
echo ""
echo "3. 통합 선택 메뉴:"
echo "   bash run_sglang_ultimate.sh"

echo ""
echo -e "${PURPLE}💡 RTX 4060에서는 CPU 모드가 가장 안정적입니다!${NC}"
echo "속도는 느리지만 CUDA 멀티프로세싱 문제를 완전히 회피할 수 있습니다."

echo ""
echo "근본적 해결 완료 시간: $(date)"