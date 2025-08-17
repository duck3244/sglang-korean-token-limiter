#!/usr/bin/env python3
"""
SGLang 빠른 시작 스크립트 (옵션 수정 버전)
"""

import sys
import subprocess
import time
import requests
import os

def start_sglang_quick():
    """SGLang 서버 빠른 시작 (유효한 옵션만 사용)"""

    print("🚀 SGLang 서버 빠른 시작 (옵션 수정 버전)")
    print("=" * 50)

    # 기본 설정
    model_path = "microsoft/DialoGPT-medium"
    port = 8000

    # GPU 확인
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            gpu_name = torch.cuda.get_device_name()
            print(f"✅ GPU: {gpu_name}")
        else:
            print("💻 CPU 모드")
    except:
        gpu_available = False
        print("💻 CPU 모드")

    # 서버 명령어 (유효한 옵션만 사용)
    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", model_path,
        "--port", str(port),
        "--host", "127.0.0.1",
        "--trust-remote-code"
    ]

    # GPU 사용 시 추가 옵션
    if gpu_available:
        cmd.extend([
            "--mem-fraction-static", "0.75",
            "--max-running-requests", "6",
            "--kv-cache-dtype", "auto",  # auto 사용 (fp16 대신)
            "--tensor-parallel-size", "1"
        ])

        # RTX 4060 감지 시 안전 옵션
        if "4060" in gpu_name:
            cmd.extend([
                "--disable-cuda-graph",
                "--disable-flashinfer"
            ])
    else:
        # CPU 모드
        cmd.extend([
            "--disable-cuda-graph",
            "--disable-flashinfer"
        ])

    print(f"실행 명령어: {' '.join(cmd)}")

    try:
        # 로그 디렉토리 생성
        os.makedirs("logs", exist_ok=True)

        # 서버 시작
        with open("logs/sglang_quick.log", "w") as log_file:
            process = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT
            )

        print(f"✅ 서버 시작 (PID: {process.pid})")

        # 서버 준비 대기
        print("⏳ 서버 준비 대기...")
        for i in range(120):
            try:
                response = requests.get(f"http://127.0.0.1:{port}/get_model_info", timeout=3)
                if response.status_code == 200:
                    print(f"✅ 서버 준비 완료! ({i+1}초)")

                    # 모델 정보 표시
                    model_info = response.json()
                    print(f"모델: {model_info.get('model_path', 'Unknown')}")

                    print()
                    print("🧪 테스트 명령어:")
                    print(f"curl http://127.0.0.1:{port}/get_model_info")
                    print()
                    print("🛑 종료: Ctrl+C")

                    # 서버 대기
                    try:
                        process.wait()
                    except KeyboardInterrupt:
                        print("\n🛑 서버 종료 중...")
                        process.terminate()
                        process.wait()
                        print("✅ 서버 종료 완료")

                    return 0
            except:
                pass

            # 프로세스 체크
            if process.poll() is not None:
                print("❌ 서버 프로세스 종료됨")
                if os.path.exists("logs/sglang_quick.log"):
                    print("\n=== 로그 ===")
                    with open("logs/sglang_quick.log", "r") as f:
                        print(f.read()[-1000:])
                return 1

            if i % 20 == 0 and i > 0:
                print(f"대기 중... {i}초")

            time.sleep(1)

        print("❌ 서버 시작 시간 초과")
        process.terminate()
        return 1

    except Exception as e:
        print(f"❌ 서버 시작 실패: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(start_sglang_quick())
