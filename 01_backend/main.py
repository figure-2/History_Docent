#!/usr/bin/env python3
"""
History Docent API Server
- FastAPI 기반 REST API 서버
- 프론트엔드와 RAG 시스템을 연결하는 브릿지
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from history_docent import HistoryDocent
import uvicorn
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

# FastAPI 앱 초기화
app = FastAPI(
    title="History Docent API",
    description="한국사 RAG 시스템 API 서버",
    version="1.0.0"
)

# CORS 설정 (프론트엔드에서 접근 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:9002",  # Next.js 개발 서버
        "http://127.0.0.1:9002",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# RAG 시스템 인스턴스 (전역 변수)
docent = HistoryDocent()


# 요청/응답 모델
class QueryRequest(BaseModel):
    question: str
    location: str = ""
    language: str = "ko"
    historicalFigurePersona: str = ""
    photoDataUri: str = ""


class QueryResponse(BaseModel):
    answer: str
    sources: list = []
    latency: float = 0.0


# 라이프사이클 이벤트
@app.on_event("startup")
async def startup_event():
    """서버 시작 시 RAG 시스템 초기화"""
    print("=" * 60)
    print("🚀 History Docent API Server 시작")
    print("=" * 60)
    print("📡 API 엔드포인트: http://localhost:8000")
    print("📚 문서: http://localhost:8000/docs")
    print("-" * 60)
    
    try:
        docent.initialize()
        print("✅ 서버 준비 완료!")
    except Exception as e:
        print(f"❌ 초기화 실패: {e}")
        print("⚠️  모델 로드 실패 시 첫 요청에서 다시 시도됩니다.")


@app.on_event("shutdown")
async def shutdown_event():
    """서버 종료 시 정리"""
    print("\n👋 History Docent API Server 종료")


# API 엔드포인트
@app.get("/")
async def root():
    """루트 엔드포인트 (헬스체크)"""
    return {
        "message": "History Docent API Server",
        "status": "running",
        "endpoints": {
            "query": "/api/query",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """헬스체크 엔드포인트"""
    return {
        "status": "healthy",
        "initialized": docent._initialized if hasattr(docent, '_initialized') else False
    }


@app.post("/api/query", response_model=QueryResponse)
async def query_docent(request: QueryRequest):
    """
    질문 처리 엔드포인트
    
    Args:
        request: QueryRequest 객체 (question 필수)
        
    Returns:
        QueryResponse: 생성된 답변과 메타데이터
    """
    try:
        # 입력 검증
        if not request.question or not request.question.strip():
            raise HTTPException(
                status_code=400, 
                detail="질문 내용이 비어있습니다. question 필드를 입력해주세요."
            )
        
        question = request.question.strip()
        
        # 로그 출력
        print(f"\n📩 [요청 수신] 질문: {question[:50]}...")
        if request.language:
            print(f"   언어: {request.language}")
        
        # RAG 시스템 호출 (top_k는 history_docent.py에서 기본값으로 사용됨)
        result = docent.chat(question)
        
        print(f"✅ [처리 완료] 소요 시간: {result['latency']}초")
        
        # 응답 반환
        return QueryResponse(
            answer=result["answer"],
            sources=result.get("sources", []),
            latency=result.get("latency", 0.0)
        )
        
    except HTTPException:
        # HTTP 예외는 그대로 전달
        raise
    except Exception as e:
        # 기타 예외 처리
        error_msg = str(e)
        print(f"❌ [에러 발생] {error_msg}")
        
        import traceback
        traceback.print_exc()
        
        raise HTTPException(
            status_code=500,
            detail=f"서버 오류가 발생했습니다: {error_msg}"
        )


if __name__ == "__main__":
    # 서버 실행
    print("\n" + "=" * 60)
    print("🎯 History Docent API Server 실행")
    print("=" * 60)
    print("📍 주소: http://0.0.0.0:8000")
    print("📖 API 문서: http://localhost:8000/docs")
    print("💡 프론트엔드에서 http://localhost:8000/api/query 로 요청")
    print("=" * 60 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

