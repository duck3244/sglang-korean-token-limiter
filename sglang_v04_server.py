#!/usr/bin/env python3
"""
SGLang 0.4+ 버전용 서버 시작 스크립트
"""

import sys
import subprocess
import time
import requests
import argparse
import os

def find_sglang_command():
    """SGLang 0.4+ 버전의 서버 시작 방법 찾기"""
    
    methods = []
    
    # 방법 1: python -m sglang.launch_server
    try:
        result = subprocess.run([sys.executable, '-m', 'sglang.launch_server', '--help'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            methods.append(('module_launch', [sys.executable, '-m', 'sglang.launch_server']))
            print("✅ sglang.launch_server 모듈 사용 가능")
    except:
        print("❌ sglang.launch_server 모듈 실패")
    
    # 방법 2: 직접 launch_server.py 실행
    try:
        import sglang
        sglang_dir = os.path.dirname(sglang.__file__)
        launch_server_path = os.path.join(sglang_dir, 'launch_server.py')
        
        if os.path.exists(launch_server_path):
            result = subprocess.run([sys.executable, launch_server_path, '--help'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                methods.append(('direct_script', [sys.executable, launch_server_path]))
                print("✅ 직접 launch_server.py 실행 가능")
    except:
        print("❌ 직접 launch_server.py 실행 실패")
    
    # 방법 3: sglang 명령어 (설치되어 있다면)
    try:
        result = subprocess.run(['sglang', '--help'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            methods.append(('command', ['sglang']))
            print("✅ sglang 명령어 사용 가능")
    except:
        print("❌ sglang 명령어 없음")
    
    return methods

def start_sglang_server(model_path="microsoft/DialoGPT-medium", port=8000, host="127.0.0.1"):
    """SGLang 서버 시작"""
    
    print("🔍 SGLang 0.4+ 서버 시작 방법 찾는 중...")
    
    methods = find_sglang_command()
    
    if not methods:
        print("❌ SGLang 서버 시작 방법을 찾을 수 없습니다")
        return None
    
    # 첫 번째 사용 가능한 방법 사용
    method_name, base_cmd = methods[0]
    print(f"🚀 {method_name} 방법으로 서버 시작...")
    
    # SGLang 0.4+ 인자 구성
    server_args = [
        "--model-path", model_path,
        "--port", str(port),
        "--host", host,
        "--trust-remote-code"
    ]
    
    # GPU 메모리 설정 (RTX 4060 최적화)
    server_args.extend([
        "--mem-fraction-static", "0.75",
        "--max-running-requests", "8",
        "--tp-size", "1"
    ])
    
    full_cmd = base_cmd + server_args
    
    print(f"실행 명령어: {' '.join(full_cmd)}")
    
    try:
        # 서버 프로세스 시작
        process = subprocess.Popen(
            full_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        print(f"✅ SGLang 서버 프로세스 시작 (PID: {process.pid})")
        
        # PID 저장
        os.makedirs("pids", exist_ok=True)
        with open("pids/sglang.pid", "w") as f:
            f.write(str(process.pid))
        
        return process
        
    except Exception as e:
        print(f"❌ 서버 시작 실패: {e}")
        return None

def wait_for_server(port=8000, timeout=120):
    """SGLang 서버 준비 대기"""
    
    print(f"⏳ SGLang 서버 준비 대기 (포트 {port})...")
    
    # 가능한 엔드포인트들
    endpoints = [
        f"http://127.0.0.1:{port}/get_model_info",
        f"http://127.0.0.1:{port}/health", 
        f"http://127.0.0.1:{port}/v1/models"
    ]
    
    for i in range(timeout):
        for endpoint in endpoints:
            try:
                response = requests.get(endpoint, timeout=2)
                if response.status_code == 200:
                    print(f"✅ SGLang 서버 준비 완료! ({i+1}초)")
                    print(f"✅ 응답 엔드포인트: {endpoint}")
                    
                    # 모델 정보 출력 시도
                    try:
                        if "get_model_info" in endpoint:
                            info = response.json()
                            print(f"모델: {info.get('model_path', 'Unknown')}")
                            print(f"최대 토큰: {info.get('max_total_tokens', 'Unknown')}")
                    except:
                        pass
                    
                    return True
            except:
                continue
        
        if i % 15 == 0 and i > 0:
            print(f"⏳ 대기 중... ({i}/{timeout}초)")
        
        time.sleep(1)
    
    print(f"❌ 서버 준비 시간 초과 ({timeout}초)")
    return False

def monitor_server_output(process, max_lines=50):
    """서버 출력 모니터링"""
    
    print("📋 서버 출력 모니터링 (처음 50줄):")
    print("-" * 50)
    
    lines_shown = 0
    while lines_shown < max_lines and process.poll() is None:
        try:
            line = process.stdout.readline()
            if line:
                print(line.strip())
                lines_shown += 1
            else:
                time.sleep(0.1)
        except:
            break
    
    print("-" * 50)

def main():
    parser = argparse.ArgumentParser(description="SGLang 0.4+ 서버 실행")
    parser.add_argument("--model", default="microsoft/DialoGPT-medium", help="모델 경로")
    parser.add_argument("--port", type=int, default=8000, help="서버 포트")
    parser.add_argument("--host", default="127.0.0.1", help="호스트 주소")
    parser.add_argument("--monitor", action="store_true", help="서버 출력 모니터링")
    
    args = parser.parse_args()
    
    print("🚀 SGLang 0.4+ 서버 시작")
    print("=" * 30)
    print(f"모델: {args.model}")
    print(f"포트: {args.port}")
    print(f"호스트: {args.host}")
    print()
    
    # 로그 디렉토리 생성
    os.makedirs("logs", exist_ok=True)
    
    # 서버 시작
    process = start_sglang_server(args.model, args.port, args.host)
    
    if not process:
        print("❌ 서버 시작 실패")
        return 1
    
    # 서버 출력 모니터링 (옵션)
    if args.monitor:
        monitor_server_output(process)
    
    # 서버 준비 대기
    if wait_for_server(args.port):
        print("🎉 SGLang 서버 실행 성공!")
        print()
        print("📊 서버 정보:")
        print(f"- 주소: http://{args.host}:{args.port}")
        print(f"- PID: {process.pid}")
        print()
        print("🧪 테스트 명령어:")
        print(f"curl http://{args.host}:{args.port}/get_model_info")
        print(f"curl http://{args.host}:{args.port}/v1/models")
        print()
        print("🔗 Token Limiter 연결:")
        print("다른 터미널에서: python main_sglang.py")
        print()
        print("종료하려면 Ctrl+C를 누르세요...")
        
        try:
            # 서버 모니터링 루프
            while True:
                if process.poll() is not None:
                    print("❌ 서버 프로세스가 종료되었습니다")
                    stdout, stderr = process.communicate()
                    if stderr:
                        print("오류 출력:", stderr)
                    break
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 서버 종료 중...")
            process.terminate()
            
            # 종료 대기
            try:
                process.wait(timeout=10)
                print("✅ 서버 정상 종료")
            except subprocess.TimeoutExpired:
                print("⚠️ 강제 종료")
                process.kill()
                process.wait()
            
            # PID 파일 정리
            try:
                os.remove("pids/sglang.pid")
            except:
                pass
    else:
        print("❌ 서버 대기 실패")
        
        # 프로세스 출력 확인
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            print("프로세스 출력:")
            if stdout:
                print("STDOUT:", stdout[-1000:])  # 마지막 1000자만
            if stderr:
                print("STDERR:", stderr[-1000:])
        else:
            print("프로세스는 여전히 실행 중입니다...")
            process.terminate()
        
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
