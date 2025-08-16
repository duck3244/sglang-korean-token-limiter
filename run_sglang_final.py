#!/usr/bin/env python3
"""
SGLang 최종 수정 버전 실행 스크립트
"""

import sys
import subprocess
import time
import requests
import os
import argparse

def start_server(model_path="microsoft/DialoGPT-medium", port=8000):
    """SGLang 서버 시작 (최종 수정 버전)"""

    print("🚀 SGLang 서버 시작 (최종 수정 버전)")
    print(f"모델: {model_path}")
    print(f"포트: {port}")
    print(f"서버 모듈: sglang_basic")
    print(f"설치 방법: vllm_sglang")

    # 서버 명령어 결정
    if "sglang_basic" == "sglang.srt.server":
        cmd = [sys.executable, "-m", "sglang.srt.server"]
    elif "sglang_basic" == "sglang.launch_server":
        cmd = [sys.executable, "-m", "sglang.launch_server"]
    elif "sglang_basic" == "sglang_basic":
        print("⚠️ 기본 SGLang만 사용 - 서버 기능 제한적")
        return None
    else:
        cmd = [sys.executable, "-m", "sglang.launch_server"]  # 기본값

    # 안전한 서버 설정
    args = [
        "--model-path", model_path,
        "--port", str(port),
        "--host", "127.0.0.1",
        "--trust-remote-code",
        "--mem-fraction-static", "0.6",  # 안전한 메모리 사용
        "--max-running-requests", "4",   # 안정성 우선
        "--disable-flashinfer",          # 호환성 문제 방지
        "--dtype", "float16"
    ]

    full_cmd = cmd + args
    print(f"실행: {' '.join(full_cmd)}")

    try:
        os.makedirs("logs", exist_ok=True)

        with open("logs/sglang_final.log", "w") as log_file:
            process = subprocess.Popen(
                full_cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT
            )

        print(f"✅ 서버 시작 (PID: {process.pid})")

        # 서버 준비 대기
        print("⏳ 서버 준비 대기...")
        for i in range(180):  # 3분 대기
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

            if i % 30 == 0 and i > 0:
                print(f"대기 중... {i}초")

            time.sleep(1)

        print("❌ 서버 준비 시간 초과")
        process.terminate()
        return None

    except Exception as e:
        print(f"❌ 서버 시작 실패: {e}")
        return None

def test_sglang_basic():
    """기본 SGLang 기능 테스트"""
    print("🧪 기본 SGLang 기능 테스트")

    try:
        import sglang as sgl

        @sgl.function
        def simple_chat(s, user_message):
            s += "User: " + user_message + "\n"
            s += "Assistant: " + sgl.gen("response", max_tokens=50)

        # 간단한 테스트
        state = simple_chat.run(user_message="Hello, how are you?")
        print("✅ 기본 SGLang 기능 작동")
        print(f"응답: {state['response']}")

        return True

    except Exception as e:
        print(f"❌ 기본 SGLang 테스트 실패: {e}")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="microsoft/DialoGPT-medium")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--test-only", action="store_true", help="기본 기능만 테스트")

    args = parser.parse_args()

    if args.test_only or "sglang_basic" == "sglang_basic":
        print("🧪 기본 SGLang 테스트 모드")
        if test_sglang_basic():
            print("🎉 SGLang 기본 기능 작동!")
            return 0
        else:
            return 1

    process = start_server(args.model, args.port)

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
        print("❌ 서버 실행 실패")
        print("\n🧪 기본 기능 테스트 시도...")
        if test_sglang_basic():
            print("✅ 기본 SGLang 기능은 작동합니다")
            print("서버 없이 기본 기능만 사용 가능")

        # 로그 출력
        if os.path.exists("logs/sglang_final.log"):
            print("\n=== 로그 ===")
            with open("logs/sglang_final.log", "r") as f:
                print(f.read()[-2000:])

        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
