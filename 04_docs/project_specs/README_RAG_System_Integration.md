# RAG 시스템 통합 가이드

- **작성 일시:** 2025-11-24 05:30 (KST)
- **목적:** 통합 RAG 시스템 구축 및 프론트엔드 연결 완료

---

## 📋 생성된 파일

### 1. 백엔드 (History_Docent 폴더)

#### `history_docent.py`
- **역할:** 통합 RAG 클래스 (검색 + 생성)
- **위치:** `/home/pencilfoxs/00_new/History_Docent/history_docent.py`
- **기능:**
  - 검색 시스템(RetrievalSystem) 활용
  - LLM 모델(Bllossom-8B)을 통한 답변 생성
  - Multi-turn 지원 준비 (history 변수)

#### `main.py`
- **역할:** FastAPI 서버 (프론트엔드 ↔ RAG 시스템 연결)
- **위치:** `/home/pencilfoxs/00_new/History_Docent/main.py`
- **기능:**
  - REST API 엔드포인트: `POST /api/query`
  - CORS 설정 (프론트엔드 9002 포트 허용)
  - 자동 초기화 및 에러 처리

### 2. 프론트엔드 (3_Docent_frontend_1117 폴더)

#### `src/app/actions.ts` (수정됨)
- **수정 내용:** `getRoyalAnswer` 함수가 FastAPI 백엔드를 호출하도록 변경
- **기존:** Genkit Flow 호출
- **변경 후:** `http://localhost:8000/api/query` 호출

---

## 🚀 실행 방법

### 1. 백엔드 서버 실행

```bash
cd /home/pencilfoxs/00_new/History_Docent
python main.py
```

**예상 출력:**
```
============================================================
🚀 History Docent API Server 시작
============================================================
📡 API 엔드포인트: http://localhost:8000
📚 문서: http://localhost:8000/docs
------------------------------------------------------------
🚀 검색 시스템 초기화 시작...
✅ 검색 시스템 준비 완료!
🤖 LLM 모델 로드 중: MLP-KTLim/llama-3-Korean-Bllossom-8B
✅ 시스템 준비 완료!
✅ 서버 준비 완료!
```

### 2. 프론트엔드 실행 (별도 터미널)

```bash
cd /home/pencilfoxs/00_new/3_Docent_frontend_1117
npm run dev
# 또는
yarn dev
```

**예상 출력:**
```
  ▲ Next.js 15.3.3
  - Local:        http://localhost:9002
```

### 3. 테스트

1. **브라우저:** http://localhost:9002 접속
2. **질문 입력:** 예) "손기정은 누구인가요?"
3. **답변 확인:** RAG 시스템이 생성한 답변 표시

---

## 🔧 의존성 설치 (필요시)

### Python 패키지

```bash
pip install fastapi uvicorn pydantic
```

### Node.js 패키지

프론트엔드는 이미 `package.json`에 모든 의존성이 있으므로 추가 설치 불필요합니다.

---

## 📡 API 명세

### 엔드포인트: `POST /api/query`

#### 요청 (Request)
```json
{
  "question": "손기정은 누구인가요?",
  "location": "",
  "language": "ko",
  "historicalFigurePersona": "",
  "photoDataUri": ""
}
```

#### 응답 (Response)
```json
{
  "answer": "손기정은 1936년 베를린 올림픽에서...",
  "sources": [
    {
      "rank": 1,
      "preview": "손기정은 조선의 마라톤 선수로..."
    }
  ],
  "latency": 7.23
}
```

---

## 🐛 트러블슈팅

### 문제 1: 모델 로드 실패
**증상:** "모델을 찾을 수 없습니다" 에러

**해결:**
- HuggingFace에서 모델 다운로드 확인
- GPU 메모리 확인 (`nvidia-smi`)
- 필요시 CPU 모드로 전환 (자동 처리됨)

### 문제 2: CORS 에러
**증상:** 프론트엔드에서 "CORS policy" 에러

**해결:**
- `main.py`의 `allow_origins`에 프론트엔드 주소가 포함되어 있는지 확인
- 기본값: `["http://localhost:9002"]`

### 문제 3: 검색 결과 없음
**증상:** "관련 정보를 찾을 수 없습니다" 답변

**해결:**
- ChromaDB 경로 확인: `04_VectorDB/chroma_db`
- 청크 파일 확인: `02_Chunking/output/all_chunks.json`
- 검색 시스템 초기화 로그 확인

---

## 🔄 향후 개선 사항

### 2단계: Multi-turn 대화
- `history_docent.py`의 `self.history` 활용
- 프롬프트에 이전 대화 맥락 포함

### 3단계: 데이터베이스 연동
- 대화 기록 영구 저장
- SQLite 또는 PostgreSQL 연동

### 파인튜닝 후 모델 교체
- `history_docent.py`의 `self.model_id`만 변경
- 예: `"MLP-KTLim/llama-3-Korean-Bllossom-8B"` → `"/path/to/finetuned-model"`

---

## ✅ 완성 체크리스트

- [x] 통합 RAG 클래스 작성 (`history_docent.py`)
- [x] FastAPI 서버 작성 (`main.py`)
- [x] 프론트엔드 연결 (`actions.ts` 수정)
- [ ] 백엔드 서버 실행 테스트
- [ ] 프론트엔드에서 질문-답변 테스트
- [ ] 에러 처리 확인

---

**작성자:** AI Assistant  
**최종 업데이트:** 2025-11-24 05:30 (KST)

