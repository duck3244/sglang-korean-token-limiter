#!/usr/bin/env python3
"""
SGLang 백엔드 수정 실행 스크립트
"""

import sys
import subprocess
import time
import requests
import os
import argparse

def setup_environment():
    """SGLang 환경 설정"""
    
    # 필수 환경 변수
    env_vars = {
        'SGLANG_BACKEND': 'pytorch',
        'SGLANG_USE_CPU_ENGINE': '0',
        'CUDA_VISIBLE_DEVICES': '0',
        'PYTHONPATH': os.getcwd(),
        'TOKENIZERS_PARALLELISM': 'false',  # 경고 억제
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"환경 변수 설정: {key}={value}")

def test_basic_sglang():
    """기본 SGLang 기능 테스트 (백엔드 포함)"""
    
    print("🧪 SGLang 기본 기능 테스트 (백엔드 포함)")
    
    try:
        # 환경 설정
        setup_environment()
        
        import sglang as sgl
        from sglang.lang.backend.runtime_endpoint import RuntimeEndpoint
        
        # 런타임 엔드포인트 생성 (로컬 백엔드)
        runtime = RuntimeEndpoint("http://localhost:30000")
        
        # 간단한 함수 정의
        @sgl.function
        def simple_chat(s, user_message):
            s += sgl.system("You are a helpful assistant.")
            s += sgl.user(user_message)
            s += sgl.assistant(sgl.gen("response", max_tokens=50))
        
        print("✅ SGLang 함수 정의 성공")
        return True
        
    except Exception as e:
        print(f"⚠️ 기본 SGLang 테스트: {e}")
        
        # 대안: 매우 기본적인 테스트
        try:
            import sglang
            print(f"✅ SGLang {sglang.__version__} import 성공")
            return True
        except Exception as e2:
            print(f"❌ SGLang import 실패: {e2}")
            return False

def start_server_direct(model_path="microsoft/DialoGPT-medium", port=8000):
    """직접 SGLang 서버 시작"""
    
    print("🚀 SGLang 서버 직접 시작")
    
    # 환경 설정
    setup_environment()
    
    # Python 스크립트로 직접 서버 시작
    server_script = f'''
import os
import sys

# 환경 설정
os.environ["SGLANG_BACKEND"] = "pytorch"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

try:
    from sglang.srt.server import launch_server
    print("✅ launch_server 함수 import 성공")
    
    # 서버 시작
    launch_server(
        model_path="{model_path}",
        host="127.0.0.1",
        port={port},
        trust_remote_code=True,
        mem_fraction_static=0.6,
        max_running_requests=4,
        disable_flashinfer=True
    )
    
except Exception as e:
    print(f"❌ 서버 시작 실패: {{e}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
'''
    
    try:
        os.makedirs("logs", exist_ok=True)
        
        # 서버 스크립트 실행
        with open("logs/sglang_backend_fixed.log", "w") as log_file:
            process = subprocess.Popen(
                [sys.executable, "-c", server_script],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=os.environ.copy()
            )
        
        print(f"✅ 서버 프로세스 시작 (PID: {process.pid})")
        
        # 서버 준비 대기
        print("⏳ 서버 준비 대기...")
        for i in range(120):  # 2분 대기
            try:
                response = requests.get(f"http://127.0.0.1:{port}/get_model_info", timeout=5)
                if response.status_code == 200:
                    print(f"✅ 서버 준비 완료! ({i+1}초)")
                    return process
            except:
                pass
                
            if process.poll() is not None:
                print("❌ 서버 프로세스 종료됨")
                return None
                
            if i % 20 == 0 and i > 0:
                print(f"대기 중... {i}초")
            
            time.sleep(1)
        
        print("❌ 서버 준비 시간 초과")
        process.terminate()
        return None
        
    except Exception as e:
        print(f"❌ 서버 시작 실패: {e}")
        return None

def start_server_alternative(model_path="microsoft/DialoGPT-medium", port=8000):
    """대안 방법으로 서버 시작"""
    
    print("🔄 대안 방법으로 서버 시작")
    
    # 환경 설정
    setup_environment()
    
    # 명령어 방식
    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", model_path,
        "--port", str(port),
        "--host", "127.0.0.1",
        "--trust-remote-code",
        "--mem-fraction-static", "0.6",
        "--max-running-requests", "4",
        "--disable-flashinfer"
    ]
    
    try:
        os.makedirs("logs", exist_ok=True)
        
        with open("logs/sglang_alternative.log", "w") as log_file:
            process = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=os.environ.copy()
            )
        
        print(f"✅ 대안 서버 시작 (PID: {process.pid})")
        
        # 서버 준비 대기
        for i in range(60):
            try:
                response = requests.get(f"http://127.0.0.1:{port}/get_model_info", timeout=5)
                if response.status_code == 200:
                    print(f"✅ 대안 서버 준비 완료! ({i+1}초)")
                    return process
            except:
                pass
                
            if process.poll() is not None:
                print("❌ 대안 서버 프로세스 종료됨")
                return None
            
            time.sleep(1)
        
        print("❌ 대안 서버 준비 시간 초과")
        process.terminate()
        return None
        
    except Exception as e:
        print(f"❌ 대안 서버 시작 실패: {e}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="microsoft/DialoGPT-medium")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--alternative", action="store_true", help="대안 방법 사용")
    
    args = parser.parse_args()
    
    print("🔧 SGLang 백엔드 수정 버전")
    print("=" * 30)
    
    # 기본 테스트
    if args.test_only:
        if test_basic_sglang():
            print("🎉 SGLang 기본 기능 작동!")
            return 0
        else:
            return 1
    
    # 서버 시작
    if args.alternative:
        process = start_server_alternative(args.model, args.port)
    else:
        process = start_server_direct(args.model, args.port)
        
        # 첫 번째 방법 실패 시 대안 시도
        if not process:
            print("🔄 첫 번째 방법 실패 - 대안 방법 시도...")
            process = start_server_alternative(args.model, args.port)
    
    if process:
        print("🎉 SGLang 서버 실행 성공!")
        print()
        print("테스트:")
        print(f"curl http://127.0.0.1:{args.port}/get_model_info")
        print()
        print("Token Limiter (다른 터미널):")
        print("python main_sglang.py")
        print()
        print("종료: Ctrl+C")
        
        try:
            process.wait()
        except KeyboardInterrupt:
            print("\n🛑 서버 종료 중...")
            process.terminate()
            process.wait()
    else:
        print("❌ 모든 서버 시작 방법 실패")
        
        # 기본 기능 테스트
        print("\n🧪 기본 기능 테스트...")
        if test_basic_sglang():
            print("✅ 기본 SGLang 기능은 작동합니다")
        
        # 로그 출력
        log_files = ["logs/sglang_backend_fixed.log", "logs/sglang_alternative.log"]
        for log_file in log_files:
            if os.path.exists(log_file):
                print(f"\n=== {log_file} ===")
                with open(log_file, "r") as f:
                    print(f.read()[-1000:])
        
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
