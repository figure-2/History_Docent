# 프로젝트 인수인계 보고서 (Handover Report)

- **작성 일시:** 2025-11-22 16:20 (KST)
- **작성자:** AI Assistant & User
- **프로젝트명:** 한국사 RAG 시스템 구축 프로젝트
- **프로젝트 경로:** `/home/pencilfoxs/00_new/History_Docent`

---

## 📋 목차 (Table of Contents)

1. [프로젝트 개요](#1-프로젝트-개요)
2. [완료된 작업 요약](#2-완료된-작업-요약)
3. [현재 진행 상황](#3-현재-진행-상황)
4. [다음 단계 작업 계획](#4-다음-단계-작업-계획)
5. [중요 파일 및 경로](#5-중요-파일-및-경로)
6. [환경 설정 및 의존성](#6-환경-설정-및-의존성)
7. [주의사항 및 트러블슈팅](#7-주의사항-및-트러블슈팅)
8. [참고 자료](#8-참고-자료)

---

## 1. 프로젝트 개요

### 1.1 프로젝트 목적
한국사 관련 PDF 문서(12개)를 기반으로 한 RAG(Retrieval-Augmented Generation) 시스템 구축. 취업 포트폴리오 목적으로 각 단계별 데이터 기반 의사결정 과정을 문서화.

### 1.2 프로젝트 구조
```
History_Docent/
├── 00_PDF_history/              # 원본 PDF 파일 (12개)
├── 01_Data_Preprocessing/        # PDF 파싱 및 전처리
├── 02_Chunking/                  # 문서 청킹 (완료)
├── 03_Embedding/                 # 임베딩 모델 선정 및 생성 (완료)
├── 04_VectorDB/                  # 벡터 DB 구축 (완료)
├── 05_Retrieval_Optimization/    # 검색 전략 최적화 (완료)
├── 06_LLM_Evaluation/            # LLM 모델 선정 (진행 중)
└── HANDOVER_REPORT.md            # 본 인수인계 보고서
```

### 1.3 핵심 기술 스택
- **Chunking:** 구조 인식 청킹 (Upstage Parser 기반)
- **Embedding:** BGE-m3 (1024차원)
- **Vector DB:** ChromaDB
- **Retriever:** BM25 Only (Okt 형태소 분석기)
- **Reranker:** 미사용 (성능 하락 확인)
- **LLM:** 벤치마킹 진행 중 (Gemini API + 4개 로컬 LLM)

---

## 2. 완료된 작업 요약

### 2.1 단계 1: 문서 전처리 및 청킹 ✅

**완료 일시:** 2025-11-22 초반

**주요 작업:**
- 12개 PDF 파일 파싱 (Upstage Parser 사용)
- 구조 인식 청킹 구현 (제목, 본문, 이미지 OCR, 표 구분)
- 하이브리드 청킹 전략 적용 (길이 제한 + 문장 단위 분할)
- 최종 청크 수: **3,719개**

**핵심 결정:**
- 최대 청크 크기: 1000자 (텍스트), 500자 (이미지 OCR)
- 문장 단위 분할 시 100자 overlap 적용

**문서:**
- `02_Chunking/README_Chunking_Log.md`
- `02_Chunking/SPEC_Chunking_Strategy.md`
- `02_Chunking/output/all_chunks.json` (최종 청크 데이터)

---

### 2.2 단계 2: 임베딩 모델 선정 ✅

**완료 일시:** 2025-11-22 10:50

**주요 작업:**
- 7개 임베딩 모델 벤치마크 (500개 → 2000개 질문으로 확장)
- 평가 데이터셋 자동 생성 (Gemini API 활용)
- 정량/정성 평가 병행 (50개 샘플 정성 평가)

**최종 선정 모델:**
- **BGE-m3** (`BAAI/bge-m3`)
- **선정 근거:**
  - Recall@1: 72.8% (2000개 질문 기준)
  - 다국어 지원 및 한국어 성능 우수
  - 1024차원 임베딩

**벤치마크 결과 (2000개 질문):**
| 모델 | Recall@1 | Recall@5 | MRR | 속도 |
|:---|:---:|:---:|:---:|:---:|
| **BGE-m3** | **72.8%** | **94.2%** | **0.812** | 중간 |
| E5-large | 72.0% | 94.0% | 0.807 | 빠름 |
| Ko-SBERT | 68.5% | 92.5% | 0.780 | 빠름 |
| Jina-v3 | 67.2% | 91.8% | 0.770 | 느림 |

**문서:**
- `03_Embedding/README_Embedding_Log.md`
- `03_Embedding/SPEC_Embedding_Strategy.md`
- `03_Embedding/data/korean_history_benchmark_2000.json` (평가 데이터셋)
- `03_Embedding/output/chunks_with_embeddings.json` (임베딩 생성 완료)

---

### 2.3 단계 3: 벡터 DB 구축 ✅

**완료 일시:** 2025-11-22 중반

**주요 작업:**
- ChromaDB 벡터 DB 구축 (3,719개 청크)
- BGE-m3 임베딩 저장 (1024차원)
- 샘플 검색 테스트 완료

**DB 정보:**
- **DB 경로:** `04_VectorDB/chroma_db`
- **Collection 이름:** `korean_history_chunks`
- **총 문서 수:** 3,719개
- **임베딩 차원:** 1024

**문서:**
- `04_VectorDB/README_VectorDB_Log.md`
- `04_VectorDB/SPEC_VectorDB_Strategy.md`

---

### 2.4 단계 4: 검색 전략 최적화 ✅

**완료 일시:** 2025-11-22 15:45

**주요 작업:**
1. **형태소 분석기 선정:** 4개 후보 비교 (Kiwi, Okt, Kkma, Hannanum)
2. **검색 전략 비교:** Vector Only, BM25 Only, Hybrid (Weighted/RRF)
3. **Reranker 검증:** BGE Reranker, Jina Reranker 테스트

**최종 결정:**

#### 4.1 형태소 분석기
- **선정:** Okt (Open Korean Text)
- **선정 근거:** Recall@1 92.0% (최고 성능), 검색 속도 31.95ms (최고)

#### 4.2 검색 전략
- **선정:** **BM25 Only** (Okt 형태소 분석기)
- **선정 근거:**
  - Recall@1: 92.0% (Vector Only 66%보다 26%p 높음)
  - Recall@5: 98.0%
  - MRR: 0.947
  - 평균 검색 시간: 31.0ms

#### 4.3 Reranker
- **결정:** 미사용
- **이유:**
  - BM25 + Reranker: Recall@1 90% (BM25 Only보다 2%p 하락)
  - 검색 시간 8.7배 증가 (31ms → 269ms)
  - 한국사 도메인 특성상 키워드 매칭이 의미 분석보다 중요

**핵심 발견:**
- 하이브리드 검색이 BM25 단독보다 성능이 낮음 (Vector 성능 66%가 하이브리드를 끌어내림)
- LLM-as-a-Judge 재검증으로 Vector의 실제 성능은 72%로 확인 (여전히 BM25보다 낮음)

**문서:**
- `05_Retrieval_Optimization/README_Retrieval_Log.md` (상세 실험 과정)
- `05_Retrieval_Optimization/SPEC_Retrieval_Strategy.md`
- `05_Retrieval_Optimization/hybrid_retriever.py` (구현 코드)

---

## 3. 현재 진행 상황

### 3.1 진행 중인 작업: LLM 모델 선정 🔄

**현재 단계:** 벤치마킹 계획 수립 완료, 실행 대기 중

**계획된 작업:**
1. 5개 LLM 모델 벤치마크 (Gemini API + 4개 로컬 LLM)
2. RAGAS 평가 (Context Recall, Context Precision, Faithfulness, Answer Relevancy)
3. 정량/정성 평가 병행 (최소 20개 샘플)

**테스트 대상 모델:**
1. `gemini-1.5-flash` (API)
2. `beomi/KoAlpaca-Polyglot-12.8B` (로컬, **GPU 메모리 부족으로 제외 예정**)
3. `yanolja/KoSOLAR-10.7B-v0.2` (로컬)
4. `MLP-KTLim/llama-3-Korean-Bllossom-8B` (로컬)
5. `beomi/Llama-3-Open-Ko-8B` (로컬)

**현재 상태:**
- ✅ 벤치마킹 스크립트 작성 완료 (`06_LLM_Evaluation/llm_benchmark.py`)
- ✅ 문서화 완료 (`README_LLM_Log.md`, `SPEC_LLM_Strategy.md`)
- ⏸️ GPU 메모리 부족으로 실행 대기 중
  - 현재 GPU 사용 중: 7.9GB (다른 작업: `generate_data_v2.py`)
  - 사용 가능한 GPU 메모리: 32.5GB
  - 12.8B 모델은 제외, 8B/10.7B 모델만 테스트 가능

**문서:**
- `06_LLM_Evaluation/README_LLM_Log.md` (실험 계획 완료)
- `06_LLM_Evaluation/SPEC_LLM_Strategy.md` (기술 명세서)

---

## 4. 다음 단계 작업 계획

### 4.1 즉시 실행 가능한 작업 (우선순위 높음)

#### 작업 1: LLM 벤치마크 실행
**목적:** 5개 LLM 모델 성능 비교 및 최적 모델 선정

**실행 방법:**
```bash
cd /home/pencilfoxs/00_new/History_Docent/06_LLM_Evaluation

# 전체 모델 테스트 (기본 50개 샘플)
python llm_benchmark.py

# 또는 Gemini만 먼저 테스트 (로컬 모델 스킵)
python llm_benchmark.py --skip-local

# 샘플 수 조정
python llm_benchmark.py -l 20
```

**예상 소요 시간:**
- Gemini API: 약 10-15분 (50개 샘플)
- 로컬 LLM (3개): 각 모델당 약 30-60분 (모델 다운로드 + 추론)
- **총 예상 시간: 약 2-3시간**

**주의사항:**
- GPU 메모리 확인 필요 (현재 다른 작업 사용 중)
- 로컬 모델은 첫 실행 시 HuggingFace에서 다운로드 (시간 소요)
- 12.8B 모델은 GPU 메모리 부족으로 제외

**결과 파일:**
- `06_LLM_Evaluation/output/llm_benchmark_results.csv`
- `06_LLM_Evaluation/output/llm_benchmark_result.md`

**완료 후 작업:**
- `README_LLM_Log.md`의 "3. 검증 결과" 및 "6. 최종 결론" 섹션 업데이트
- 선정된 모델로 최종 RAG 파이프라인 통합

---

#### 작업 2: 최종 RAG 파이프라인 통합
**목적:** 선정된 LLM 모델과 BM25 Retriever를 통합하여 End-to-End 시스템 구축

**구현 내용:**
- BM25 Retriever (Okt) → LLM Generator 통합
- 프롬프트 템플릿 최적화
- 샘플 질문으로 End-to-End 테스트

**예상 소요 시간:** 약 1-2시간

---

### 4.2 선택적 작업 (우선순위 중간)

#### 작업 3: 정성 평가 확대
**목적:** LLM 벤치마크 후 선정된 모델에 대한 상세 정성 평가

**작업 내용:**
- 최소 20개 이상 샘플 직접 확인
- 성공/실패 사례 분석
- 엣지 케이스 확인

**예상 소요 시간:** 약 1시간

---

#### 작업 4: 최종 문서화 정리
**목적:** 전체 프로젝트 문서를 최종 정리 및 검토

**작업 내용:**
- 모든 README 파일 최종 검토
- SPEC 파일 최종 검토
- 프로젝트 README 작성 (전체 개요)

**예상 소요 시간:** 약 1-2시간

---

## 5. 중요 파일 및 경로

### 5.1 핵심 데이터 파일

| 파일 경로 | 설명 | 크기/개수 |
|:---|:---|:---|
| `02_Chunking/output/all_chunks.json` | 최종 청크 데이터 | 3,719개 |
| `03_Embedding/output/chunks_with_embeddings.json` | 임베딩 포함 청크 | 3,719개 (1024차원) |
| `03_Embedding/data/korean_history_benchmark_2000.json` | 평가 데이터셋 | 2,000개 질문 |
| `04_VectorDB/chroma_db/` | ChromaDB 벡터 DB | 3,719개 문서 |

### 5.2 핵심 코드 파일

| 파일 경로 | 설명 |
|:---|:---|
| `02_Chunking/upstage_chunker.py` | 구조 인식 청킹 스크립트 |
| `03_Embedding/generate_embeddings.py` | 임베딩 생성 스크립트 |
| `04_VectorDB/build_vectordb.py` | 벡터 DB 구축 스크립트 |
| `05_Retrieval_Optimization/hybrid_retriever.py` | BM25 리트리버 구현 |
| `06_LLM_Evaluation/llm_benchmark.py` | LLM 벤치마크 스크립트 |

### 5.3 문서 파일

| 파일 경로 | 설명 | 상태 |
|:---|:---|:---|
| `02_Chunking/README_Chunking_Log.md` | 청킹 작업 로그 | ✅ 완료 |
| `02_Chunking/SPEC_Chunking_Strategy.md` | 청킹 기술 명세서 | ✅ 완료 |
| `03_Embedding/README_Embedding_Log.md` | 임베딩 작업 로그 | ✅ 완료 |
| `03_Embedding/SPEC_Embedding_Strategy.md` | 임베딩 기술 명세서 | ✅ 완료 |
| `04_VectorDB/README_VectorDB_Log.md` | 벡터 DB 작업 로그 | ✅ 완료 |
| `04_VectorDB/SPEC_VectorDB_Strategy.md` | 벡터 DB 기술 명세서 | ✅ 완료 |
| `05_Retrieval_Optimization/README_Retrieval_Log.md` | 검색 최적화 로그 | ✅ 완료 |
| `05_Retrieval_Optimization/SPEC_Retrieval_Strategy.md` | 검색 전략 명세서 | ✅ 완료 |
| `06_LLM_Evaluation/README_LLM_Log.md` | LLM 선정 로그 | 🔄 계획 완료 |
| `06_LLM_Evaluation/SPEC_LLM_Strategy.md` | LLM 기술 명세서 | 🔄 계획 완료 |

---

## 6. 환경 설정 및 의존성

### 6.1 하드웨어 환경

**GCP 인스턴스:**
- **GPU:** NVIDIA A100-SXM4-40GB
- **RAM:** 83GB
- **디스크:** 969GB (현재 552GB 사용, 418GB 여유)
- **CUDA:** 12.2

**현재 GPU 상태:**
- 사용 중: 7.9GB (다른 작업: `generate_data_v2.py`, PID 472051)
- 사용 가능: 32.5GB
- **로컬 LLM 테스트 가능:** 8B 모델 3개 + 10.7B 모델 1개

### 6.2 소프트웨어 환경

**Python 버전:**
- Python 3.x (확인 필요)

**주요 라이브러리:**
```txt
# 필수 라이브러리
sentence-transformers
torch
transformers
chromadb
rank-bm25
konlpy
ragas
langchain-google-genai
datasets
tqdm
pandas
numpy
```

**설치 명령어:**
```bash
pip install sentence-transformers torch transformers chromadb rank-bm25 konlpy ragas langchain-google-genai datasets tqdm pandas numpy
```

### 6.3 환경 변수

**필수 환경 변수:**
- `GOOGLE_API_KEY`: Gemini API 키
- **설정 파일:** `/home/pencilfoxs/00_new/.env2`

**확인 방법:**
```bash
cat /home/pencilfoxs/00_new/.env2 | grep GOOGLE_API_KEY
```

---

## 7. 주의사항 및 트러블슈팅

### 7.1 GPU 메모리 관리

**현재 상황:**
- 다른 작업(`generate_data_v2.py`)이 GPU 메모리 7.9GB 사용 중
- 로컬 LLM 테스트 시 GPU 메모리 충분 여부 확인 필요

**해결 방법:**
1. 다른 작업 완료 대기
2. 또는 Gemini API만 먼저 테스트 (`--skip-local` 옵션)
3. GPU 메모리 확인: `nvidia-smi`

**로컬 모델별 예상 메모리:**
- 8B 모델: 약 22.4GB (테스트 가능)
- 10.7B 모델: 약 30.0GB (테스트 가능)
- 12.8B 모델: 약 35.8GB (현재 환경에서 불가능)

---

### 7.2 벤치마크 실행 시 주의사항

**LLM 벤치마크 실행 전:**
1. GPU 메모리 확인 (`nvidia-smi`)
2. API 키 확인 (`.env2` 파일)
3. 평가 데이터셋 경로 확인 (`korean_history_benchmark_2000.json`)
4. ChromaDB 경로 확인 (`04_VectorDB/chroma_db`)

**실행 중:**
- 로컬 모델은 첫 실행 시 HuggingFace에서 다운로드 (시간 소요)
- Gemini API는 Rate Limit 주의 (현재 스크립트에 `time.sleep` 포함)
- 중간 결과 저장 기능 있음 (100개마다 저장)

**실행 후:**
- 결과 파일 확인 (`output/llm_benchmark_results.csv`)
- 오류 발생 시 로그 확인

---

### 7.3 알려진 이슈 및 해결 방법

#### 이슈 1: GPU 메모리 부족
**증상:** 12.8B 모델 로딩 시 OOM (Out of Memory) 에러

**해결:** 12.8B 모델은 제외하고 8B/10.7B 모델만 테스트

---

#### 이슈 2: ChromaDB 차원 불일치
**증상:** `Collection expecting embedding with dimension of 1024, got 384`

**해결:** 쿼리 시 `query_embeddings` 사용 (이미 `build_vectordb.py`에 반영됨)

---

#### 이슈 3: RAGAS 평가 실패
**증상:** RAGAS 메트릭 평가 중 오류 발생

**해결:** 
- 각 메트릭별 try-except 처리 (이미 스크립트에 포함)
- 실패 시 0.0으로 기록하고 계속 진행

---

### 7.4 디버깅 팁

**벤치마크 실행 중 문제 발생 시:**
1. 로그 파일 확인 (터미널 출력)
2. 중간 결과 파일 확인 (`output/` 디렉토리)
3. GPU 메모리 확인 (`nvidia-smi`)
4. Python 프로세스 확인 (`ps aux | grep python`)

**코드 수정이 필요한 경우:**
- 주요 스크립트는 모두 `05_Retrieval_Optimization/` 및 `06_LLM_Evaluation/` 디렉토리에 있음
- 수정 전 백업 권장

---

## 8. 참고 자료

### 8.1 과거 프로젝트 참고

**참고 프로젝트:**
- `/home/pencilfoxs/History_Docent_PJ_gemini`: 과거 프로젝트 (하이브리드 검색 + Reranker 사용)
- `/home/pencilfoxs/History_Docent_PJ`: 초기 프로젝트 (하이브리드 검색 구현 참고)

**차이점:**
- 과거 프로젝트: 하이브리드 검색 + 파인튜닝된 Reranker 사용
- 현재 프로젝트: BM25 Only (데이터 기반으로 최적 전략 선정)

---

### 8.2 외부 문서

**규칙 문서:**
- `/home/pencilfoxs/00_new/rule.md`: 프로젝트 문서화 규칙 (1-36번)

**API 문서:**
- Gemini API: https://ai.google.dev/docs
- RAGAS: https://docs.ragas.io/
- ChromaDB: https://docs.trychroma.com/

---

### 8.3 핵심 결정 사항 요약

| 단계 | 최종 결정 | 주요 근거 |
|:---|:---|:---|
| **청킹** | 하이브리드 청킹 (구조 인식 + 길이 제한) | 긴 청크 문제 해결 |
| **임베딩** | BGE-m3 | Recall@1 72.8% (2000개 질문 기준) |
| **벡터 DB** | ChromaDB | 설치 간편, 로컬 개발에 적합 |
| **형태소 분석기** | Okt | Recall@1 92.0%, 검색 속도 31.95ms |
| **검색 전략** | BM25 Only | Recall@1 92.0%, 모든 하이브리드 전략보다 우수 |
| **Reranker** | 미사용 | 성능 하락(92%→90%), 속도 8.7배 증가 |
| **LLM** | 벤치마킹 진행 중 | (결과 대기 중) |

---

## 9. 다음 작업 시작 가이드

### 9.1 빠른 시작 체크리스트

**작업 시작 전 확인:**
- [ ] GPU 메모리 확인 (`nvidia-smi`)
- [ ] API 키 확인 (`.env2` 파일)
- [ ] 작업 디렉토리 이동 (`cd /home/pencilfoxs/00_new/History_Docent`)
- [ ] 필요한 라이브러리 설치 확인

**LLM 벤치마크 실행:**
```bash
# 1. 작업 디렉토리로 이동
cd /home/pencilfoxs/00_new/History_Docent/06_LLM_Evaluation

# 2. GPU 메모리 확인
nvidia-smi

# 3. 벤치마크 실행 (Gemini만 먼저 테스트)
python llm_benchmark.py --skip-local

# 4. 또는 전체 모델 테스트 (로컬 모델 포함)
python llm_benchmark.py
```

**결과 확인:**
```bash
# 결과 파일 확인
cat 06_LLM_Evaluation/output/llm_benchmark_result.md
```

---

### 9.2 작업 완료 후 필수 작업

**벤치마크 완료 후:**
1. `README_LLM_Log.md` 업데이트
   - "3. 검증 결과" 섹션에 실제 벤치마크 결과 입력
   - "6. 최종 결론" 섹션에 선정 모델 및 근거 입력
2. `SPEC_LLM_Strategy.md` 업데이트
   - 선정된 모델의 상세 스펙 추가
3. 최종 RAG 파이프라인 통합
4. End-to-End 테스트

---

## 10. 연락처 및 추가 정보

**프로젝트 관련 질문:**
- 모든 결정 사항은 각 단계별 `README_*_Log.md` 파일에 상세히 기록되어 있음
- `rule.md` 파일의 규칙(1-36번)에 따라 모든 작업이 문서화됨

**환경 관련 질문:**
- GCP 인스턴스 상태: `nvidia-smi`, `free -h`, `df -h` 명령어로 확인
- 프로세스 확인: `ps aux | grep python`

---

## 11. 마무리

**현재 프로젝트 진행률: 약 85%**

- ✅ 청킹 (100%)
- ✅ 임베딩 (100%)
- ✅ 벡터 DB (100%)
- ✅ 검색 전략 (100%)
- 🔄 LLM 선정 (계획 완료, 실행 대기)

**다음 작업 예상 시간:**
- LLM 벤치마크: 약 2-3시간
- 문서 업데이트: 약 30분
- 최종 통합: 약 1-2시간
- **총 예상 시간: 약 4-6시간**

**성공적인 프로젝트 완료를 기원합니다! 🚀**

---

**작성자:** AI Assistant & User  
**최종 업데이트:** 2025-11-22 16:20 (KST)

