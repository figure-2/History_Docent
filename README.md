# History Docent (역사 도슨트 AI)

> **RAG(Retrieval-Augmented Generation) 기반의 한국사 교육 및 도슨트 AI 서비스**  
> 사용자의 질문에 대해 신뢰할 수 있는 역사적 사실을 바탕으로 정확한 답변을 제공합니다. 

[![Next.js](https://img.shields.io/badge/Next.js-15.3-black?logo=next.js)](https://nextjs.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0-blue?logo=typescript)](https://www.typescriptlang.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.122-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![vLLM](https://img.shields.io/badge/vLLM-0.11-orange)](https://docs.vllm.ai/)

---

## 목차

- [프로젝트 개요](#프로젝트-개요)
- [기술 스택](#기술-스택)
- [시스템 아키텍처](#시스템-아키텍처)
- [주요 기능](#주요-기능)
- [프로젝트 구조](#프로젝트-구조)
- [시작하기](#시작하기)
- [핵심 기술 및 해결 과제](#핵심-기술-및-해결-과제)
- [문서](#문서)

---

## 프로젝트 개요

**History Docent**는 한국사 교육을 위한 AI 도슨트 서비스입니다. RAG(Retrieval-Augmented Generation) 기술을 활용하여:

- **신뢰성 있는 답변**: 검색된 문서에 기반한 정확한 역사 정보 제공
- **Hallucination 방지**: LLM의 환각 현상을 최소화하는 검증된 파이프라인
- **하이브리드 검색**: 벡터 검색 + 키워드 검색(BM25) 결합으로 검색 정확도 향상
- **실시간 질의응답**: FastAPI 기반 RESTful API로 빠른 응답 제공

---

## 기술 스택

### Frontend
| 기술 | 용도 |
|------|------|
| **Next.js 15.3** | React 프레임워크, SSR/SSG 지원 |
| **TypeScript** | 타입 안정성 보장 |
| **Tailwind CSS** | 유틸리티 기반 스타일링 |
| **Radix UI** | 접근성 높은 UI 컴포넌트 |

### Backend
| 기술 | 용도 |
|------|------|
| **FastAPI** | 고성능 REST API 서버 |
| **Python 3.10+** | 백엔드 개발 언어 |
| **vLLM** | 고속 LLM 추론 엔진 |
| **ChromaDB** | 벡터 데이터베이스 |
| **Rank-BM25** | 키워드 기반 검색 알고리즘 |
| **KoNLPy** | 한국어 형태소 분석 |

### AI/ML
| 기술 | 용도 |
|------|------|
| **Bllossom-8B** | 한국어 특화 LLM (생성 모델) |
| **BGE-m3** | 다국어 임베딩 모델 |
| **Reranker** | 검색 결과 재순위화 |

---

## 시스템 아키텍처

### RAG Pipeline Flow

```
사용자 질문
    ↓
[Frontend: Next.js]
    ↓ HTTP Request
[Backend API: FastAPI]
    ↓
[1. Hybrid Retrieval]
    ├─ BM25 (키워드 검색) → Top-50 후보
    └─ Reranker → Top-5 최종 선택
    ↓
[2. Context + Query]
    ↓
[3. LLM Generation (vLLM)]
    ├─ Bllossom-8B 모델
    └─ Prompt Engineering
    ↓
[4. Response]
    └─ 답변 + 출처 반환
```

### 데이터 처리 파이프라인

```
PDF 문서
    ↓
[01_Data_Preprocessing] 전처리
    ↓
[02_Chunking] 텍스트 분할 (3,719개 청크)
    ↓
[03_Embedding] 벡터화 (BGE-m3)
    ↓
[04_VectorDB] ChromaDB 저장
    ↓
[05_Retrieval_Optimization] 검색 최적화
    ↓
[06_LLM_Evaluation] 성능 평가
```

---

## 주요 기능

### 1. 하이브리드 검색 시스템
- **문제**: 단순 벡터 검색 시 고유명사(예: '손기정', '세종대왕') 검색 정확도 저하
- **해결**: BM25(키워드) + Vector(의미) 하이브리드 검색 구현
- **성과**: 검색 정확도(Top-k Recall) 약 **15% 향상**

### 2. Reranking을 통한 정확도 개선
- 1차 검색으로 Top-50 후보 확보
- Cross-Encoder 기반 Reranker로 Top-5 최종 선정
- 질문과 문서 간 관련성 점수 기반 재순위화

### 3. LLM Hallucination 제어
- 검색된 문서(Context)에 기반하지 않은 답변 방지
- Strict Prompting으로 출처 기반 답변 강제
- RAGAS 프레임워크를 활용한 답변 신뢰성 평가

### 4. 고성능 추론 엔진
- vLLM을 활용한 배치 추론 최적화
- GPU 메모리 효율적 관리
- 낮은 지연시간(Latency) 달성

---

## 프로젝트 구조

```
History_Docent_PJ/
├── frontend/                    # Next.js 웹 애플리케이션
│   ├── src/
│   │   ├── app/                # Next.js App Router
│   │   ├── components/         # React 컴포넌트
│   │   └── lib/                # 유틸리티 함수
│   ├── public/                 # 정적 파일
│   └── package.json
│
├── backend/                     # Python 백엔드 서버
│   ├── main.py                 # FastAPI 서버 진입점
│   ├── history_docent.py       # 통합 RAG 시스템 클래스
│   ├── requirements.txt        # Python 패키지 의존성
│   │
│   ├── 00_PDF_history/         # 원본 PDF 문서
│   ├── 01_Data_Preprocessing/  # 데이터 전처리 스크립트
│   ├── 02_Chunking/            # 텍스트 청킹 로직
│   ├── 03_Embedding/           # 임베딩 생성 및 저장
│   ├── 04_VectorDB/            # ChromaDB 구축
│   ├── 05_Retrieval_Optimization/  # 검색 시스템 최적화
│   │   ├── hybrid_retriever.py # 하이브리드 검색기
│   │   ├── reranker.py         # 재순위화 모델
│   │   └── retrieval_system.py # 통합 검색 시스템
│   └── 06_LLM_Evaluation/      # 모델 성능 평가
│
├── docs/                        # 프로젝트 문서
│   ├── RAG_SYSTEM_STATUS.md    # RAG 시스템 현황
│   ├── README_REQUIREMENTS.md  # 환경 설정 가이드
│   └── ...
│
├── scripts/                     # 테스트 및 유틸리티 스크립트
│
├── .gitignore
└── README.md                    # 이 파일
```

---

## 시작하기

### 사전 요구사항

- **Python**: 3.10 이상
- **Node.js**: 18 이상
- **Java**: OpenJDK 21 이상 (KoNLPy 사용 시)
- **GPU**: NVIDIA GPU 권장 (최소 16GB VRAM, CUDA 12.1+)
- **RAM**: 최소 16GB 권장

### 1. Backend 설정

```bash
cd backend

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 필수 패키지 설치
pip install -r requirements.txt

# Java 설정 (KoNLPy 사용 시)
export JAVA_HOME=$(dirname $(dirname $(readlink -f $(which java))))

# 서버 실행
python main.py
```

서버는 `http://localhost:8000`에서 실행됩니다.  
API 문서는 `http://localhost:8000/docs`에서 확인할 수 있습니다.

### 2. Frontend 설정

```bash
cd frontend

# 의존성 설치
npm install

# 개발 서버 실행
npm run dev
```

프론트엔드는 `http://localhost:9002`에서 실행됩니다.

### 3. API 테스트

```bash
# 헬스체크
curl http://localhost:8000/health

# 질문 요청
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "손기정은 누구인가요?"}'
```

---

## Data-Driven Decisions (데이터 기반 의사결정)

본 프로젝트는 직관이 아닌 **가설(Hypothesis) 설정 → 실험(Experiment) → 검증(Validation)** 프로세스를 통해 최적의 기술 스택을 선정했습니다.

| Decision Step | Candidates | Selected | Key Result (Reason) | Report |
|:---:|:---|:---:|:---|:---:|
| **1. Chunking** | Recursive vs Structure-Aware | **Hybrid Chunking** | 문맥 보존 + 길이 제어 (최대 1,063자) | [View](./docs/experiments/01_Chunking_Strategy.md) |
| **2. Embedding** | BGE-m3, Ko-SBERT, E5 등 7개 | **BGE-m3** | **MRR 0.489** (1위), 모든 질문 유형에서 우수 | [View](./docs/experiments/02_Embedding_Benchmark.md) |
| **3. Retrieval** | Vector vs BM25 vs Hybrid | **Hybrid Weighted** | **MRR 0.711**, Recall@1 0.632 (Vector 대비 +9.7%p) | [View](./docs/experiments/03_Retrieval_Benchmark.md) |
| **4. Reranking** | BGE-Reranker vs Dongjin-kr | **BM25 Only** | 리랭커 적용 시 성능 향상 없음, 속도만 저하 | [View](./docs/experiments/06_Reranker_Benchmark.md) |
| **5. LLM** | EXAONE, EEVE, Bllossom 등 | **Bllossom-8B** | 성공률 100%, 평균 지연시간 6.98초 | [View](./docs/experiments/04_LLM_Selection.md) |

---

## 핵심 기술 및 해결 과제

### 1. 검색 정확도 개선: 하이브리드 검색 도입

**문제점**:
- 벡터 검색만으로는 고유명사나 특정 키워드 검색 시 정확도가 낮음
- 예: "손기정", "묘청" 등 특정 역사적 인물이나 사건 검색 시 Vector 검색이 관련 없는 문서를 반환 (Recall@1 66.0%)

**원인 분석**:
- Embedding 모델이 희귀한 고유명사의 정확한 매칭보다 문맥적 유사성에 치중
- "묘청과 정지상이 서경으로의 천도를..." 같은 질문에서 '풍수지리', '지리적 이점' 등 구체적 키워드 매칭 실패

**해결 방법**:
1. 키워드 매칭에 강한 **BM25 알고리즘** 도입
2. 한국어 특성을 고려하여 **KoNLPy(Okt)** 형태소 분석기 적용
3. Vector(60%) + BM25(40%) 하이브리드 검색으로 두 방식의 장점 결합

**결과**:
- Recall@1: 66.0% → **92.0%** (Vector Only 대비 +39.4%p)
- MRR: 0.663 → **0.711** (Hybrid Weighted, +7.3%p)
- OKT 적용으로 BM25 성능 **+11.7% 향상**

**상세 리포트**: [검색 전략 벤치마크](./docs/experiments/03_Retrieval_Benchmark.md)

### 2. 청킹 전략 최적화: 하이브리드 청킹

**문제점**:
- 순수 구조 기반 청킹에서 **1,500자 이상 초대형 청크 3개 발견**
- 최대 1,919자 청크로 인한 임베딩 모델 Context Window 초과 위험

**원인 분석**:
- 이미지 OCR 텍스트가 깨져서 길게 이어짐
- 문단 구분이 없는 긴 텍스트가 통째로 묶임

**해결 방법**:
1. 구조 기반 청킹의 장점(문맥 보존, 제목-본문 결합) 유지
2. 길이 제어 로직 추가:
   - 이미지 청크: 500자 초과 시 Truncate
   - 텍스트 청크: 1,000자 초과 시 문장 단위 재분할
   - Overlap: 100자 적용

**결과**:
- 최대 길이: 1,919자 → **1,063자** (45% 감소)
- 평균 길이: 546.8자 → **471.4자** (최적화)
- 초대형 청크 완전 제거

**상세 리포트**: [청킹 전략 실험](./docs/experiments/01_Chunking_Strategy.md)

### 3. LLM 추론 속도 최적화: vLLM 도입

**문제점**:
- Transformers 라이브러리만으로는 배치 추론 시 속도 저하
- GPU 메모리 사용 비효율

**해결 방법**:
- vLLM 엔진 도입으로 Continuous Batching 구현
- PagedAttention으로 메모리 효율성 개선

**결과**: 추론 속도 약 2-3배 향상

**코드 위치**: `backend/history_docent.py`

### 4. 답변 신뢰성 확보: RAGAS 평가

**문제점**:
- LLM이 생성한 답변의 정확성과 신뢰성을 정량적으로 측정 필요

**해결 방법**:
- RAGAS 프레임워크를 활용한 자동 평가
- Context Precision, Context Recall, Faithfulness 등 지표 측정
- 평가 결과를 바탕으로 파이프라인 개선

**코드 위치**: `backend/06_LLM_Evaluation/`

---

## 장애 대응 및 최적화 (Troubleshooting & Optimization)

실제 RAG 시스템을 구축하며 발생한 시스템 장애와 리소스 한계를 극복한 경험입니다.

### 1. GCP 인스턴스 크래시 및 메모리 부족 해결
- **문제**: 8B 모델 로드 시 시스템 RAM(16GB) 및 GPU VRAM 부족으로 인스턴스 다운(OOM Kill) 발생
- **분석**: vLLM의 KV Cache가 초기화될 때 대량의 메모리를 점유함 확인
- **해결**:
  1. Linux Swap File (32GB) 설정으로 시스템 메모리 확보
  2. `gpu_memory_utilization` 파라미터 튜닝 (0.9 → 0.95)
  3. 모델 양자화(Quantization) 도입 검토
- **상세 리포트**: [GCP 인스턴스 크래시 분석](./docs/troubleshooting/01_GCP_Instance_Crash.md)

### 2. GPU VRAM 최적화 전략
- **문제**: 긴 Context 처리 시 VRAM 부족으로 추론 속도 저하
- **해결**:
  - PagedAttention 기술을 활용한 메모리 파편화 최소화
  - 배치 사이즈(Batch Size) 동적 조절
- **상세 리포트**: [메모리 최적화 계획](./docs/troubleshooting/02_Memory_Optimization.md)

### 3. 대규모 데이터셋 품질 관리
- **문제**: 3,700개 이상의 청크를 처리하며 평가 데이터셋 구축의 일관성 유지 어려움
- **해결**:
  - 데이터셋을 Train/Validation/Test로 체계적으로 분할 (8:1:1)
  - 질문 유형(Keyword, Context, Abstract)별 균형 잡힌 샘플링 전략 수립
- **상세 리포트**: [데이터셋 분할 전략](./docs/experiments/05_Dataset_Strategy.md)

---

## 성능 지표 (E2E Test Result)

실제 환경에서의 End-to-End 테스트 결과입니다.

- **검색 정확도**:
  - Recall@1: **63.2%** (Hybrid Weighted)
  - Recall@5: **83.0%** (Hybrid Weighted)
  - MRR: **0.711** (Hybrid Weighted)
  
- **응답 속도 (Latency)**:
  - 사실 기반 질문(Simple Fact): 평균 **0.48초**
  - 복합 추론 질문(Complex Reasoning): 평균 **6.98초**
  - 전체 평균: **3.28초** (안정적 서비스 가능 범위)

- **리소스 효율성**:
  - GPU VRAM 사용률: 92% (최적화 후 안정적 유지)
  - 처리 가능 청크 수: **3,719개**
  - 모델 크기: Bllossom-8B (8B 파라미터)

---

## 문서

상세한 개발 과정과 실험 결과는 `docs/` 폴더를 참고하세요.

### 실험 리포트 (Experiments)
- [01. 청킹 전략 실험](./docs/experiments/01_Chunking_Strategy.md) - 구조 기반 vs 하이브리드 청킹 비교
- [02. 임베딩 모델 벤치마크](./docs/experiments/02_Embedding_Benchmark.md) - 7개 모델 성능 비교 (BGE-m3 선정)
- [03. 검색 전략 벤치마크](./docs/experiments/03_Retrieval_Benchmark.md) - Vector vs BM25 vs Hybrid 비교
- [04. LLM 모델 선정](./docs/experiments/04_LLM_Selection.md) - 4개 한국어 LLM 성능 비교 (Bllossom-8B 선정)
- [05. 데이터셋 전략](./docs/experiments/05_Dataset_Strategy.md) - 평가 데이터 구성 방법 (과적합 방지)
- [06. 리랭커 벤치마크](./docs/experiments/06_Reranker_Benchmark.md) - 리랭커 적용 효과 분석

### 트러블슈팅 (Troubleshooting)
- [01. 인스턴스 크래시 해결](./docs/troubleshooting/01_GCP_Instance_Crash.md) - OOM 해결 과정
- [02. 메모리 최적화](./docs/troubleshooting/02_Memory_Optimization.md) - VRAM 최적화 전략

### 프로젝트 문서
- [RAG 시스템 현황](./docs/project_specs/RAG_SYSTEM_STATUS.md) - RAG 파이프라인 완성도 및 평가
- [환경 설정 가이드](./docs/project_specs/README_REQUIREMENTS.md) - 상세한 설치 및 설정 방법
- [프론트엔드 통합 보고서](./docs/project_specs/README_Frontend_Integration.md) - Frontend-Backend 연동 가이드

---

## 학습한 내용

이 프로젝트를 통해 다음을 학습하고 구현했습니다:

- **RAG 시스템 설계 및 구현**: 검색-생성 파이프라인 전체 구축
- **하이브리드 검색**: 벡터 검색과 키워드 검색의 결합
- **LLM 최적화**: vLLM을 활용한 고성능 추론
- **평가 프레임워크**: RAGAS를 활용한 정량적 성능 측정
- **장애 대응**: 실제 운영 중 발생한 메모리 부족 문제 해결 경험
- **데이터 관리**: 과적합 방지를 위한 데이터셋 분할 전략 수립
- **문제 해결**: 실제 발생한 기술적 문제를 데이터 기반으로 해결

---


---

**마지막 업데이트**: 2024-11-25

