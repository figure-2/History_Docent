# [Retrieval] 검색 최적화 전략 기술 명세서

- **작성 일시:** 2025-11-22 13:15 (KST)
- **작성자:** Pencilfoxs
- **최종 업데이트:** 2025-11-22 16:15 (최종 전략 확정: BM25 Only, Reranker 미사용)

---

## 1. 개요 (Overview)

본 문서는 한국사 RAG(Retrieval-Augmented Generation) 시스템을 위한 하이브리드 검색(Hybrid Search) 전략과 형태소 분석기 선정 결과를 기술한다. BM25 키워드 검색과 Vector 검색을 결합하여 검색 정확도를 향상시키고, 한국어 특성에 맞는 형태소 분석기를 선정하여 BM25 검색 성능을 최적화한다.

---

## 2. 형태소 분석기 스펙 (Tokenizer Specification)

### 2.1 선정 모델 정보

- **형태소 분석기:** **Okt (Open Korean Text)**
- **라이브러리:** `konlpy.tag.Okt`
- **설치 방법:** `pip install konlpy`
- **개발 기관:** 트위터 데이터 기반으로 개발된 오픈소스 형태소 분석기
- **언어:** 한국어 전용

### 2.2 주요 특징

- **어간 추출(Stemming):** `morphs(text, stem=True)` 옵션으로 다양한 활용형을 정규화
  - 예: "만들었다", "만드니", "만들다" → 모두 "만들다"로 통일
  - BM25 검색에서 검색 매칭률 향상에 기여

- **정규화 기능:** 자동으로 오타나 변형된 표현을 정규화
  - 예: "안녕하세요", "안녕하세여" → 정규화 처리

- **설치 편의성:** Java 기반이지만 `konlpy` 패키지로 간편하게 설치 가능
  - 별도의 C++ 빌드 과정 불필요

### 2.3 성능 지표 (Benchmark Results)

- **Recall@1:** 92.0% (4개 후보 중 1위)
- **Recall@5:** 98.0% (4개 후보 중 공동 1위)
- **인덱싱 속도:** 70.53초 (3,719개 문서 기준)
- **검색 속도:** 31.95ms/query (평균)

### 2.4 사용 방법

```python
from konlpy.tag import Okt

tokenizer = Okt()
tokens = tokenizer.morphs(text, stem=True)  # 어간 추출 활성화
```

**파라미터:**
- `text`: 토큰화할 텍스트 (str)
- `stem`: 어간 추출 여부 (bool, 기본값: False)
  - `True`: 다양한 활용형을 어간으로 정규화 (권장)
  - `False`: 원형 유지

**반환값:**
- `tokens`: 형태소 리스트 (List[str])

### 2.5 BM25 인덱싱 파이프라인

```python
# 1. 문서 토큰화
tokenized_corpus = []
for doc in documents:
    tokens = tokenizer.morphs(doc, stem=True)
    tokenized_corpus.append([t for t in tokens if t.strip()])  # 빈 문자열 제거

# 2. BM25 인덱스 구축
from rank_bm25 import BM25Okapi
bm25_index = BM25Okapi(tokenized_corpus)
```

---

## 3. 검색 전략 (Retrieval Strategy)

### 3.1 최종 선정 전략

**BM25 Only (Okt 형태소 분석기)**를 최종 검색 전략으로 채택.

**선정 근거:**
- 벤치마크 결과: Recall@1 92%, Recall@5 98%, MRR 0.947 (4가지 전략 중 1위)
- 한국사 도메인 특성(고유명사 중심)에 가장 적합
- 하이브리드 검색보다 단독 사용이 더 우수한 성능

### 3.2 비교 실험 결과

4가지 검색 전략을 비교한 결과:

| Strategy | Recall@1 | Recall@5 | MRR | Avg Time |
| :--- | :---: | :---: | :---: | :---: |
| **BM25 Only** | **92.0%** | **98.0%** | **0.947** | 31.2ms |
| Hybrid (RRF) | 80.0% | 96.0% | 0.877 | 57.7ms |
| Hybrid (Weighted) | 70.0% | 94.0% | 0.804 | 76.0ms |
| Vector Only | 66.0% | 96.0% | 0.760 | 29.1ms |

**결론:** BM25 Only가 모든 지표에서 1위를 차지하여 최종 선정.

### 3.3 하이브리드 검색 전략 (참고용 - 현재 미사용)

하이브리드 검색은 **Vector Search (Dense Retrieval)**와 **BM25 (Sparse Retrieval)**를 결합하는 방식이지만, 현재는 BM25 Only가 더 우수한 성능을 보여 사용하지 않음.

**구현 방식:**

1. **가중치 결합 (Weighted Sum):**
   ```
   Hybrid Score = α * Vector_Score + β * BM25_Score
   (α = 0.6, β = 0.4)
   ```

2. **RRF (Reciprocal Rank Fusion):**
   ```
   RRF Score = 1/(k + Rank_Vector) + 1/(k + Rank_BM25)
   (k = 60)
   ```

**미사용 이유:**
- 벤치마크 결과, BM25 Only가 Hybrid 방식보다 우수한 성능을 보임
- Vector Only의 낮은 성능(Recall@1 66%)이 하이브리드 점수를 끌어내림

### 3.3 점수 정규화 (Score Normalization)

두 검색 방식의 점수 스케일이 다르므로, Min-Max Normalization을 적용하여 0-1 범위로 정규화한 후 가중치 결합한다.

```python
def normalize_scores(scores):
    min_score = scores.min()
    max_score = scores.max()
    if max_score == min_score:
        return np.ones_like(scores)
    return (scores - min_score) / (max_score - min_score)

# Vector 점수 정규화
vector_normalized = normalize_scores(vector_scores)

# BM25 점수 정규화
bm25_normalized = normalize_scores(bm25_scores)

# 가중치 결합
hybrid_scores = 0.6 * vector_normalized + 0.4 * bm25_normalized
```

### 3.4 RRF (Reciprocal Rank Fusion) 옵션

가중치 결합 방식 외에 RRF 방식도 지원한다 (선택적 사용).

```python
def apply_rrf(vector_ranks, bm25_ranks, k=60):
    rrf_scores = {}
    for doc_id in all_ids:
        score = 0.0
        if doc_id in vector_ranks:
            score += 1.0 / (k + vector_ranks[doc_id])
        if doc_id in bm25_ranks:
            score += 1.0 / (k + bm25_ranks[doc_id])
        rrf_scores[doc_id] = score
    return rrf_scores
```

**RRF 파라미터:**
- `k`: RRF 상수 (기본값: 60)
  - 일반적으로 60을 사용
  - 값이 작을수록 상위 랭크에 더 높은 가중치 부여

---

## 4. 구현 상세 (Implementation Details)

### 4.1 HybridRetriever 클래스

**파일 위치:** `05_Retrieval_Optimization/hybrid_retriever.py`

**주요 메서드:**

1. `initialize()`: 리트리버 초기화
   - ChromaDB 연결
   - 임베딩 모델(BGE-m3) 로드 (하이브리드 검색용, 현재는 미사용)
   - 형태소 분석기(Okt) 초기화
   - BM25 인덱스 구축

2. `search_bm25_only(query, top_k=10)`: **BM25 검색만 수행 (최종 사용)**
   - Okt 형태소 분석으로 쿼리 토큰화
   - BM25 인덱스에서 점수 계산
   - 상위 K개 결과 반환

3. `search(query, top_k=10)`: 하이브리드 검색 실행 (참고용)
   - Vector 검색 수행
   - BM25 검색 수행
   - 점수 정규화 및 가중치 결합 또는 RRF
   - 상위 K개 결과 반환

4. `search_vector_only(query, top_k=10)`: Vector 검색만 수행 (비교용)

### 4.2 데이터 스키마

**RetrievalResult 데이터 클래스:**

```python
@dataclass
class RetrievalResult:
    chunk_id: str              # 청크 고유 ID
    text: str                  # 청크 텍스트 내용
    metadata: Dict[str, Any]   # 메타데이터 (source, page, type 등)
    vector_score: float        # Vector 검색 점수
    bm25_score: float          # BM25 검색 점수
    hybrid_score: float        # 하이브리드 최종 점수
    rank: int                  # 최종 랭킹 (1부터 시작)
```

### 4.3 검색 파이프라인 플로우

```
1. Query 입력
   ↓
2. Query Embedding 생성 (BGE-m3)
   ↓
3. Vector Search (ChromaDB)
   └─→ Top 2K 후보 추출 (K * 2)
   ↓
4. Query Tokenization (Okt)
   └─→ stem=True로 어간 추출
   ↓
5. BM25 Search
   └─→ Top 2K 후보 추출 (K * 2)
   ↓
6. 점수 정규화 (Min-Max)
   ↓
7. 가중치 결합 (0.6 * Vector + 0.4 * BM25)
   ↓
8. 상위 K개 결과 반환
```

---

## 5. 성능 최적화 (Performance Optimization)

### 5.1 인덱싱 최적화

- **BM25 인덱스:** 초기 로딩 시 1회만 구축 (약 70초 소요)
- **메모리 캐싱:** 인덱스는 메모리에 유지하여 재사용
- **증분 업데이트:** 새 문서 추가 시 인덱스 재구축 필요 (향후 개선 예정)

### 5.2 검색 속도 최적화

- **병렬 처리:** Vector 검색과 BM25 검색을 병렬로 수행 가능 (향후 개선 예정)
- **후보군 제한:** 각 검색 방식에서 Top 2K만 추출하여 하이브리드 점수 계산 비용 절감

### 5.3 메모리 사용량

- **BM25 인덱스:** 약 50MB (3,719개 문서 기준)
- **임베딩 모델:** 약 2GB (BGE-m3, GPU 메모리)
- **총 메모리:** 약 2.1GB (GPU 포함)

---

## 6. 향후 개선 계획 (Future Improvements)

1. **증분 인덱싱:** 새 문서 추가 시 전체 재인덱싱 없이 증분 업데이트
2. **병렬 검색:** Vector와 BM25 검색을 멀티스레드로 병렬 수행
3. **가중치 튜닝:** 실제 사용 데이터로 가중치(α, β) 최적화
4. **RRF 최적화:** RRF 파라미터(k) 튜닝
5. **캐싱 전략:** 자주 검색되는 쿼리 결과 캐싱

---

## 7. 참고 자료 (References)

- **Okt 공식 문서:** [konlpy GitHub](https://github.com/konlpy/konlpy)
- **BM25 알고리즘:** [Okapi BM25 - Wikipedia](https://en.wikipedia.org/wiki/Okapi_BM25)
- **rank_bm25 라이브러리:** [rank-bm25 PyPI](https://pypi.org/project/rank-bm25/)
- **RRF 논문:** "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods" (2009)

