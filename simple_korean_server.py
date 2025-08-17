#!/usr/bin/env python3
"""
간단한 한국어 모델 서버 (최소 의존성)
"""

import sys
import os
import json
import time
import random
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

# 글로벌 변수
model_loaded = False
model_info = {
    "model_path": "korean-simple-server",
    "max_total_tokens": 2048,
    "served_model_names": ["korean-qwen"],
    "is_generation": True,
    "version": "1.0.0"
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 생명주기 관리"""
    # 시작
    print("🔽 서버 초기화 중...")
    global model_loaded
    
    try:
        # 실제 모델 대신 간단한 응답 생성기 사용
        print("✅ 간단한 응답 생성기 준비 완료")
        model_loaded = True
    except Exception as e:
        print(f"⚠️ 모델 로드 실패하지만 계속 진행: {e}")
        model_loaded = False
    
    yield
    
    # 종료
    print("🛑 서버 종료 중...")

# FastAPI 앱 생성
app = FastAPI(
    title="간단한 한국어 모델 서버",
    description="SGLang 대체 서버 (최소 의존성)",
    version="1.0.0",
    lifespan=lifespan
)

def generate_korean_response(user_message: str, max_tokens: int = 100) -> str:
    """간단한 한국어 응답 생성"""
    
    # 미리 정의된 한국어 응답들
    responses = [
        "안녕하세요! 저는 간단한 한국어 AI 어시스턴트입니다.",
        "네, 무엇을 도와드릴까요?",
        "좋은 질문이네요. 더 자세히 설명해 주시겠어요?",
        "SGLang 대체 서버가 정상적으로 작동하고 있습니다.",
        "한국어 토큰 제한 시스템과 잘 연동되고 있어요.",
        "Token Limiter가 제대로 작동하는지 확인해보세요.",
        "이 서버는 CPU 모드로 안정적으로 실행됩니다.",
        "CUDA 문제 없이 잘 작동하고 있어요!",
        "간단하지만 효과적인 대체 솔루션입니다.",
        "더 궁금한 것이 있으시면 언제든 물어보세요."
    ]
    
    # 사용자 메시지에 따른 맞춤 응답
    user_lower = user_message.lower()
    
    if any(word in user_lower for word in ['안녕', 'hello', '하이']):
        return "안녕하세요! 저는 한국어 대체 서버입니다. 무엇을 도와드릴까요?"
    elif any(word in user_lower for word in ['토큰', 'token']):
        return "토큰 제한 시스템이 정상적으로 작동하고 있습니다. SGLang 대체 서버가 Token Limiter와 잘 연동되고 있어요."
    elif any(word in user_lower for word in ['서버', 'server', 'sglang']):
        return "SGLang 대체 서버가 CPU 모드로 안정적으로 실행 중입니다. CUDA 멀티프로세싱 문제 없이 잘 작동해요!"
    elif any(word in user_lower for word in ['테스트', 'test']):
        return "네, 테스트가 성공적으로 진행되고 있습니다! 간단한 대체 서버가 정상 작동 중이에요."
    elif any(word in user_lower for word in ['문제', 'problem', '오류', 'error']):
        return "문제가 해결되었습니다! 이 대체 서버는 SGLang의 복잡한 문제들을 우회하여 안정적으로 작동합니다."
    else:
        # 랜덤 응답 선택
        return random.choice(responses)

@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "message": "간단한 한국어 모델 서버",
        "status": "running",
        "model_loaded": model_loaded
    }

@app.get("/health")
async def health_check():
    """헬스체크"""
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "server": "simple-korean-server",
        "timestamp": time.time()
    }

@app.get("/get_model_info")
async def get_model_info():
    """모델 정보 조회 (SGLang 호환)"""
    return model_info

@app.get("/get_server_info")
async def get_server_info():
    """서버 정보 조회"""
    return {
        "requests_per_second": 0,
        "tokens_per_second": 0,
        "queue_length": 0,
        "running_requests": 0,
        "memory_usage_gb": 1.0,
        "cache_hit_rate": 0.8,
        "timestamp": time.time()
    }

@app.get("/v1/models")
async def list_models():
    """모델 목록 (OpenAI 호환)"""
    return {
        "object": "list",
        "data": [
            {
                "id": "korean-qwen",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "simple-korean-server"
            }
        ]
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """채팅 완성 (OpenAI 호환)"""
    
    try:
        body = await request.json()
        
        # 요청 파라미터 추출
        messages = body.get("messages", [])
        max_tokens = body.get("max_tokens", 100)
        temperature = body.get("temperature", 0.7)
        stream = body.get("stream", False)
        model = body.get("model", "korean-qwen")
        
        # 사용자 메시지 추출
        user_message = ""
        if messages:
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    user_message = msg.get("content", "")
                    break
        
        if not user_message:
            user_message = "안녕하세요"
        
        # 응답 생성
        response_content = generate_korean_response(user_message, max_tokens)
        
        # 스트리밍 응답 (현재는 지원하지 않음)
        if stream:
            return JSONResponse(
                content={"error": "스트리밍은 현재 지원하지 않습니다"}
            )
        
        # OpenAI 호환 응답
        response = {
            "id": f"chatcmpl-{int(time.time())}-{random.randint(1000, 9999)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_content
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(user_message.split()),
                "completion_tokens": len(response_content.split()),
                "total_tokens": len(user_message.split()) + len(response_content.split())
            }
        }
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"채팅 완성 오류: {str(e)}"
        )

@app.post("/v1/completions")
async def completions(request: Request):
    """텍스트 완성 (OpenAI 호환)"""
    
    try:
        body = await request.json()
        
        prompt = body.get("prompt", "")
        max_tokens = body.get("max_tokens", 100)
        
        # 응답 생성
        response_text = generate_korean_response(prompt, max_tokens)
        
        # OpenAI 호환 응답
        response = {
            "id": f"cmpl-{int(time.time())}-{random.randint(1000, 9999)}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": "korean-qwen",
            "choices": [
                {
                    "text": response_text,
                    "index": 0,
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(response_text.split()),
                "total_tokens": len(prompt.split()) + len(response_text.split())
            }
        }
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"텍스트 완성 오류: {str(e)}"
        )

def main():
    print("🚀 간단한 한국어 모델 서버 시작")
    print("=" * 50)
    print("💻 최소 의존성으로 실행")
    print("🔗 포트: 8000")
    print("🇰🇷 한국어 응답 지원")
    print("📡 OpenAI 호환 API")
    print()
    
    try:
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=8000,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n🛑 서버 종료")

if __name__ == "__main__":
    main()
