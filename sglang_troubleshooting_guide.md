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
