#!/bin/bash
# SGLang 완전 해결 스크립트

echo "🔧 SGLang 완전 해결 스크립트"
echo "==========================="

echo "문제를 해결할 방법을 선택하세요:"
echo ""
echo "1) SGLang 구문 오류 수정 후 재시도"
echo "2) SGLang 깨끗한 재설치"
echo "3) 최소한의 SGLang 실행"
echo "4) 대체 모델 서버 사용 (Transformers 직접)"
echo "5) 모든 방법 순서대로 시도"

read -p "선택 (1-5): " choice

case $choice in
    1)
        echo "🔧 SGLang 구문 오류 수정..."
        bash sglang_syntax_repair.sh
        echo "재시도 중..."
        python start_sglang_cpu_mode.py
        ;;
    2)
        echo "🔄 SGLang 깨끗한 재설치..."
        bash reinstall_sglang_clean.sh
        echo "재설치 후 실행..."
        python start_sglang_cpu_mode.py
        ;;
    3)
        echo "⚡ 최소한의 SGLang 실행..."
        python start_sglang_minimal.py
        ;;
    4)
        echo "🔄 대체 모델 서버 사용..."
        python start_alternative_model.py
        ;;
    5)
        echo "🚀 모든 방법 순서대로 시도..."
        
        echo "1단계: 구문 오류 수정..."
        bash sglang_syntax_repair.sh
        
        echo "2단계: CPU 모드 시도..."
        timeout 30 python start_sglang_cpu_mode.py || echo "CPU 모드 실패"
        
        echo "3단계: 최소 실행 시도..."
        timeout 30 python start_sglang_minimal.py || echo "최소 실행 실패"
        
        echo "4단계: 대체 서버 실행..."
        python start_alternative_model.py
        ;;
    *)
        echo "❌ 잘못된 선택"
        exit 1
        ;;
esac
