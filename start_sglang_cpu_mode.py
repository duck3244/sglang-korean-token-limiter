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
