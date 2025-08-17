#!/usr/bin/env python3
"""
대체 모델 실행 스크립트 (Transformers 직접 사용)
"""

import sys
import os
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import json

# FastAPI 앱 생성
app = FastAPI(title="대체 한국어 모델 서버")

# 글로벌 변수
model = None
tokenizer = None

@app.on_event("startup")
async def load_model():
    """모델 로드"""
    global model, tokenizer
    
    print("🔽 모델 로드 중...")
    model_name = "microsoft/DialoGPT-medium"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # CPU 호환
            device_map="cpu"  # CPU 강제
        )
        
        print(f"✅ 모델 로드 완료: {model_name}")
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")

@app.get("/get_model_info")
async def get_model_info():
    """모델 정보 조회"""
    return {
        "model_path": "microsoft/DialoGPT-medium",
        "max_total_tokens": 1024,
        "served_model_names": ["korean-qwen"],
        "is_generation": True
    }

@app.get("/v1/models")
async def list_models():
    """모델 목록"""
    return {
        "data": [
            {
                "id": "korean-qwen",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "alternative-server"
            }
        ]
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """채팅 완성"""
    global model, tokenizer
    
    if model is None or tokenizer is None:
        return JSONResponse(
            status_code=503,
            content={"error": "모델이 로드되지 않았습니다"}
        )
    
    try:
        body = await request.json()
        messages = body.get("messages", [])
        max_tokens = body.get("max_tokens", 50)
        
        # 메시지를 텍스트로 변환
        if messages:
            user_message = messages[-1].get("content", "")
        else:
            user_message = "안녕하세요"
        
        # 토큰화
        inputs = tokenizer.encode(user_message, return_tensors="pt")
        
        # 생성
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=min(max_tokens, 100),
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # 디코딩
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 입력 텍스트 제거
        if response_text.startswith(user_message):
            response_text = response_text[len(user_message):].strip()
        
        # OpenAI 호환 응답
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "korean-qwen",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text or "안녕하세요! 대체 모델 서버입니다."
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(inputs[0]),
                "completion_tokens": len(response_text.split()),
                "total_tokens": len(inputs[0]) + len(response_text.split())
            }
        }
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"생성 오류: {str(e)}"}
        )

def main():
    print("🚀 대체 한국어 모델 서버 시작")
    print("=" * 40)
    print("💻 CPU 모드로 실행")
    print("🔗 포트: 8000")
    print()
    
    # 환경 설정
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
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
