#!/usr/bin/env python3
"""
최소한의 SGLang 실행 스크립트 (구문 오류 회피)
"""

import sys
import os
import subprocess
import time
import requests

def minimal_sglang_start(model_path="microsoft/DialoGPT-medium", port=8000):
    """최소한의 SGLang 서버 시작"""
    
    print("🚀 최소한의 SGLang 서버 시작")
    print(f"모델: {model_path}")
    print(f"포트: {port}")
    
    # 최소한의 환경 설정
    env = os.environ.copy()
    env.update({
        'CUDA_VISIBLE_DEVICES': '',  # CPU 모드 강제
        'TOKENIZERS_PARALLELISM': 'false'
    })
    
    # 가장 기본적인 명령어만 사용
    cmd = [
        sys.executable, "-c",
        f"""
import sys
sys.path.insert(0, '{os.getcwd()}')

# 멀티프로세싱 설정
import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

# SGLang 서버 시작
try:
    from sglang.srt.server import launch_server
    from sglang.srt.server_args import ServerArgs
    
    args = ServerArgs(
        model_path='{model_path}',
        host='127.0.0.1',
        port={port},
        trust_remote_code=True,
        max_running_requests=2,
        max_total_tokens=1024
    )
    
    launch_server(args)
    
except Exception as e:
    print(f'서버 시작 오류: {{e}}')
    import traceback
    traceback.print_exc()
"""
    ]
    
    print(f"실행 중...")
    
    try:
        # 로그 디렉토리 생성
        os.makedirs("logs", exist_ok=True)
        
        # 서버 시작
        with open("logs/sglang_minimal.log", "w") as log_file:
            process = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env
            )
        
        print(f"✅ 서버 프로세스 시작 (PID: {process.pid})")
        
        # 서버 준비 대기
        print("⏳ 서버 준비 대기...")
        for i in range(120):
            if process.poll() is not None:
                print("❌ 서버 프로세스 종료됨")
                return None
            
            try:
                response = requests.get(f"http://127.0.0.1:{port}/get_model_info", timeout=3)
                if response.status_code == 200:
                    print(f"✅ 서버 준비 완료! ({i+1}초)")
                    return process
            except:
                pass
            
            if i % 20 == 0 and i > 0:
                print(f"대기 중... {i}초")
            
            time.sleep(1)
        
        print("❌ 서버 시작 시간 초과")
        process.terminate()
        return None
        
    except Exception as e:
        print(f"❌ 서버 시작 실패: {e}")
        return None

def main():
    print("⚡ 최소한의 SGLang 실행 (구문 오류 회피)")
    print("=" * 50)
    
    model_path = sys.argv[1] if len(sys.argv) > 1 else "microsoft/DialoGPT-medium"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
    
    process = minimal_sglang_start(model_path, port)
    
    if process:
        print("\n🎉 최소한의 SGLang 서버 성공!")
        print("=" * 50)
        print()
        print("🧪 테스트:")
        print(f"curl http://127.0.0.1:{port}/get_model_info")
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
        print("❌ 최소한의 서버 시작 실패")
        
        if os.path.exists("logs/sglang_minimal.log"):
            print("\n=== 최소 실행 로그 ===")
            with open("logs/sglang_minimal.log", "r") as f:
                print(f.read()[-1500:])
        
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
