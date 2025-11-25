# [Embedding] 임베딩 전략 기술 명세서
- **작성 일시:** 2025-11-22 10:50
- **작성자:** Pencilfoxs
- **최종 업데이트:** 2025-11-22 10:50 (2000개 데이터셋 벤치마크 완료, 전체 청크 임베딩 생성 완료)

---

## 1. 개요 (Overview)
본 문서는 한국사 RAG(Retrieval-Augmented Generation) 시스템을 위한 임베딩 모델 선정 결과와 벡터화 전략을 기술한다. 벤치마크 테스트를 통해 선정된 모델을 사용하여, 청킹된 텍스트 데이터를 고차원 벡터로 변환한다.

---

## 2. 선정 모델 스펙 (Selected Model Specification)

### 2.1 모델 정보
- **모델명:** **BAAI/bge-m3** (Beijing Academy of Artificial Intelligence)
- **모델 유형:** Multilingual BERT-based Embedding Model
- **파라미터 수:** 567M (Base 모델 대비 큼)
- **최대 토큰 길이 (Max Sequence Length):** 8192 tokens (긴 문서 처리에 유리)
- **출력 차원 (Embedding Dimension):** 1024 (Dense Embedding)
- **지원 언어:** 100+ 언어 (한국어 포함)
- **HuggingFace 모델 ID:** `BAAI/bge-m3`

### 2.2 주요 특징
- **Dense Retrieval:** 문맥적 의미 기반 검색. Cosine Similarity를 사용하여 유사도 계산.
- **Sparse Retrieval (Lexical):** 키워드 매칭 보완 가능 (본 프로젝트에서는 Dense만 우선 사용).
- **Multi-Lingual:** 다국어 간의 의미 매핑 성능 우수. 한자어, 영어, 한국어 혼재 텍스트 처리에 강점.
- **Instruction Support:** Instruction 프롬프트를 붙일 수 있으나, 본 프로젝트에서는 Instruction 없이 사용 (범용성 확보).

### 2.3 성능 지표 (Benchmark Results)

#### 2.3.1 2000개 데이터셋 (최종 벤치마크)
- **MRR (Mean Reciprocal Rank):** 0.834
- **Recall@1:** 0.754 (75.4%)
- **Recall@3:** 0.902 (90.2%)
- **Recall@5:** 0.933 (93.3%)
- **Latency:** 10.1ms per text (평균)

#### 2.3.2 500개 데이터셋 (초기 벤치마크)
- **MRR:** 0.932
- **Recall@1:** 0.895 (89.5%)
- **Recall@3:** 0.970 (97.0%)
- **Recall@5:** 0.980 (98.0%)
- **Latency:** 20.5ms per text (평균)

**참고:** 2000개 데이터셋에서 성능이 낮아 보이지만, 이는 더 다양한 난이도와 엣지 케이스가 포함되어 평가가 더 엄격해졌기 때문입니다. 두 데이터셋 모두에서 BGE-m3가 1위를 유지하여 성능 안정성이 입증되었습니다.

---

## 3. 임베딩 프로세스 (Embedding Process)

### 3.1 데이터 소스
- **입력 파일:** `02_Chunking/output/all_chunks.json`
- **데이터 형식:** JSON 배열, 각 항목은 다음 구조를 가짐:
  ```json
  {
    "chunk_id": "chk_0001",
    "text": "청크 텍스트 내용...",
    "metadata": {
      "source": "파일명.pdf",
      "page": 1,
      "type": "text" | "table" | "figure"
    }
  }
  ```
- **대상 청크:**
  - `metadata.type == "text"`: 텍스트 청크 (우선 처리)
  - `metadata.type == "table"`: 표 캡션 및 내용
  - `metadata.type == "figure"`: 이미지 OCR 텍스트

### 3.2 전처리 (Preprocessing)
- **특수문자 정제:** 불필요 (모델이 자체 처리)
- **Instruction 접두사:** 사용하지 않음 (범용성 확보)
- **텍스트 길이 제한:** 모델의 최대 토큰 길이(8192) 내에서 처리. 초과 시 자동 Truncation.
- **인코딩:** UTF-8

### 3.3 벡터 생성 (Vectorization)

#### 3.3.1 라이브러리 및 환경
- **라이브러리:** `sentence-transformers` (v2.2.0 이상)
- **백엔드:** PyTorch + CUDA (GPU 가속)
- **디바이스:** `cuda` (사용 가능 시) 또는 `cpu`

#### 3.3.2 모델 로드
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "BAAI/bge-m3",
    device="cuda" if torch.cuda.is_available() else "cpu"
)
```

#### 3.3.3 배치 처리
- **배치 크기 (Batch Size):** 16~32 (GPU 메모리에 맞춰 조정)
- **정규화 (Normalization):** `normalize_embeddings=True` (Cosine Similarity 계산 최적화)
- **예시 코드:**
  ```python
  embeddings = model.encode(
      texts,
      normalize_embeddings=True,
      batch_size=16,
      show_progress_bar=True
  )
  ```

#### 3.3.4 출력 형식
- **벡터 차원:** 1024 (1차원 numpy 배열)
- **데이터 타입:** `numpy.ndarray`, dtype: `float32`
- **정규화:** L2 정규화 적용 (벡터 길이 = 1.0)

### 3.4 저장 (Storage)

#### 3.4.1 임시 저장 (JSON 형식)
- **파일 경로:** `03_Embedding/output/chunks_with_embeddings.json`
- **데이터 구조:**
  ```json
  [
    {
      "chunk_id": "chk_0001",
      "text": "청크 텍스트...",
      "embedding": [0.123, 0.456, ...],  // 1024차원 벡터
      "metadata": {
        "source": "파일명.pdf",
        "page": 1,
        "type": "text"
      }
    }
  ]
  ```

#### 3.4.2 Vector DB 적재 (향후)
- **후보 DB:** ChromaDB, FAISS, Pinecone
- **메타데이터 매핑:** `chunk_id`, `source`, `page`, `type` 등 유지
- **인덱싱:** Cosine Similarity 기반 검색 최적화

---

## 4. 벤치마크 결과 요약 (Benchmark Summary)

### 4.1 테스트 환경
- **데이터셋:** 한국사 관련 질문-답변 2000쌍 (최종), 500쌍 (초기 검증)
- **평가 지표:** MRR, Recall@1, Recall@3, Recall@5, Latency
- **실행 환경:** CUDA 12.8, GPU 가속 사용

### 4.2 성능 비교 (2000개 데이터셋 - 최종)

| Rank | Model | MRR | Recall@1 | Recall@3 | Recall@5 | Latency (ms) |
|---|---|---|---|---|---|---|
| 1 | **BGE-m3** | **0.834** | **0.754** | **0.902** | **0.933** | 10.1 |
| 2 | E5-large | 0.814 | 0.728 | 0.884 | 0.923 | **8.8** |
| 3 | Jina-v3 | 0.804 | 0.709 | 0.888 | 0.920 | 6.9 |
| 4 | Ko-SBERT | 0.587 | 0.496 | 0.634 | 0.691 | **1.8** |

### 4.3 선택 근거
- **2000개 대규모 데이터셋에서 MRR 0.834로 1위 달성**
- **데이터셋 규모에 관계없이 일관된 1위:** 500개(MRR 0.932)와 2000개(MRR 0.834) 모두에서 1위 유지
- 한국어 고유명사, 한자어, 역사적 맥락을 정확히 파악
- 속도(10.1ms)도 실시간 서비스에 무리가 없는 수준

---

## 5. 구현 코드 예시 (Code Example)

### 5.1 전체 청크 임베딩 생성 스크립트
```python
import json
import torch
from sentence_transformers import SentenceTransformer
from pathlib import Path
from tqdm import tqdm

# 설정
CHUNK_FILE = Path("02_Chunking/output/all_chunks.json")
OUTPUT_FILE = Path("03_Embedding/output/chunks_with_embeddings.json")
MODEL_NAME = "BAAI/bge-m3"
BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 모델 로드
print(f"모델 로딩 중: {MODEL_NAME} (Device: {DEVICE})")
model = SentenceTransformer(MODEL_NAME, device=DEVICE)

# 청크 데이터 로드
with open(CHUNK_FILE, 'r', encoding='utf-8') as f:
    chunks = json.load(f)

# 텍스트 추출
texts = [chunk['text'] for chunk in chunks]

# 임베딩 생성
print(f"임베딩 생성 중: {len(texts)}개 청크...")
embeddings = model.encode(
    texts,
    normalize_embeddings=True,
    batch_size=BATCH_SIZE,
    show_progress_bar=True
)

# 결과 저장
results = []
for i, chunk in enumerate(chunks):
    results.append({
        **chunk,
        "embedding": embeddings[i].tolist()  # numpy array를 list로 변환
    })

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"✅ 완료: {OUTPUT_FILE}")
```

---

## 6. 완료 사항 및 향후 계획

### 6.1 완료 사항 ✅
1. ✅ **전체 청크 임베딩 생성 완료:** `all_chunks.json`의 모든 청크(3719개)에 대해 BGE-m3 모델로 임베딩 벡터 생성 완료 (2025-11-22 10:44).
   - **생성 시간:** 약 48초 (CUDA 가속 사용)
   - **출력 파일:** `03_Embedding/output/chunks_with_embeddings.json`
   - **벡터 차원:** 1024차원 (Shape: (3719, 1024))

### 6.2 향후 계획
1. **Vector DB 구축:** ChromaDB 또는 FAISS를 사용하여 벡터 데이터베이스 구축.
2. **검색 모듈 개발:** 사용자 질문을 임베딩하여 벡터 DB에서 유사도 기반 검색 수행.

### 6.2 중장기 계획
1. **하이브리드 검색:** Dense Retrieval(BGE-m3) + Sparse Retrieval(키워드 매칭) 결합.
2. **재랭킹(Re-ranking):** 초기 검색 결과를 Cross-Encoder 모델로 재랭킹하여 정확도 향상.
3. **성능 모니터링:** 실제 사용자 질문에 대한 검색 성능 지표 수집 및 분석.

---

## 7. 참고 자료
- **벤치마크 결과:**
  - `results/benchmark_results_500.csv` (초기 검증)
  - `results/benchmark_results_2000.csv` (최종)
- **벤치마크 코드:** `benchmark_embeddings.py`
- **임베딩 생성 코드:** `generate_embeddings.py`
- **임베딩 결과 파일:** `output/chunks_with_embeddings.json` (3719개 청크)
- **모델 문서:** [HuggingFace - BGE-m3](https://huggingface.co/BAAI/bge-m3)
- **Sentence Transformers 문서:** [Sentence Transformers Documentation](https://www.sbert.net/)

