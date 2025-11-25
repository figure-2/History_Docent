# [VectorDB] 벡터 데이터베이스 전략 기술 명세서
- **작성 일시:** 2025-11-22 11:10
- **작성자:** Pencilfoxs
- **최종 업데이트:** 2025-11-22 11:10 (ChromaDB 구축 완료)

---

## 1. 개요 (Overview)
본 문서는 한국사 RAG(Retrieval-Augmented Generation) 시스템을 위한 벡터 데이터베이스 구축 전략을 기술한다. BGE-m3 모델로 생성된 1024차원 임베딩 벡터를 ChromaDB에 저장하고, 고속 유사도 검색을 수행한다.

---

## 2. 선정 벡터 DB 스펙 (Selected Vector DB Specification)

### 2.1 벡터 DB 정보
- **DB 이름:** **ChromaDB** (Open Source)
- **버전:** 1.3.0
- **클라이언트 타입:** PersistentClient (로컬 파일 기반)
- **저장 경로:** `/home/pencilfoxs/00_new/History_Docent/04_VectorDB/chroma_db/`
- **Collection 이름:** `korean_history_chunks`

### 2.2 주요 특징
- **All-in-One:** 벡터, 텍스트, 메타데이터를 한 번에 저장 및 관리
- **로컬 구동:** 서버 설치 없이 Python 라이브러리로 사용 가능
- **메타데이터 필터링:** 내장 필터링 기능으로 고급 검색 지원
- **데이터 영속성:** SQLite 기반으로 재시작 후에도 데이터 유지
- **검색 속도:** 평균 2.37ms (3,719개 문서 기준)

### 2.3 성능 지표 (Performance Metrics)
- **데이터 적재 시간:** 7.13초 (3,719개 청크)
- **검색 속도:** 평균 2.37ms (10회 반복 테스트)
- **임베딩 생성 시간:** 평균 18.80ms (BGE-m3, CUDA 사용)
- **총 검색 시간:** 평균 21.17ms (임베딩 생성 + 검색)

---

## 3. 데이터 구조 (Data Structure)

### 3.1 Collection 스키마
```python
{
    "ids": ["chk_000000", "chk_000001", ...],  # 고유 ID
    "documents": ["텍스트 내용...", ...],      # 원본 텍스트
    "embeddings": [[0.123, 0.456, ...], ...],  # 1024차원 벡터
    "metadatas": [
        {
            "source": "벌거벗은한국사-조선편",
            "page": 1,
            "type": "text"
        },
        ...
    ]
}
```

### 3.2 메타데이터 필드
- **source:** PDF 파일명 (예: "벌거벗은한국사-조선편")
- **page:** 페이지 번호 (정수)
- **type:** 청크 타입 ("text", "figure", "table")

### 3.3 데이터 통계
- **총 문서 수:** 3,719개
- **텍스트 청크:** 3,353개 (90.2%)
- **이미지 OCR:** 346개 (9.3%)
- **표:** 20개 (0.5%)

---

## 4. 구축 프로세스 (Build Process)

### 4.1 데이터 소스
- **입력 파일:** `03_Embedding/output/chunks_with_embeddings.json`
- **데이터 형식:** JSON 배열, 각 항목은 다음 구조를 가짐:
  ```json
  {
    "chunk_id": "chk_000000",
    "text": "청크 텍스트 내용...",
    "embedding": [0.123, 0.456, ...],  // 1024차원 벡터
    "metadata": {
      "source": "파일명.pdf",
      "page": 1,
      "type": "text"
    }
  }
  ```

### 4.2 구축 스크립트
- **스크립트 파일:** `build_vectordb.py`
- **주요 단계:**
  1. 임베딩 JSON 파일 로드
  2. ChromaDB 클라이언트 초기화
  3. Collection 생성
  4. 데이터 변환 (ChromaDB 형식으로)
  5. 배치 적재 (배치 크기: 1000개)

### 4.3 실행 방법
```bash
cd /home/pencilfoxs/00_new/History_Docent/04_VectorDB
python build_vectordb.py
```

---

## 5. 검색 프로세스 (Search Process)

### 5.1 검색 흐름
1. **쿼리 임베딩 생성:** 사용자 질문을 BGE-m3 모델로 1024차원 벡터로 변환
2. **벡터 검색:** ChromaDB에서 Cosine Similarity 기반 유사도 검색
3. **결과 반환:** 상위 K개 문서 반환 (기본값: 5개)

### 5.2 검색 코드 예시
```python
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import torch

# 1. ChromaDB 클라이언트 및 Collection 로드
client = chromadb.PersistentClient(
    path="/home/pencilfoxs/00_new/History_Docent/04_VectorDB/chroma_db",
    settings=Settings(anonymized_telemetry=False)
)
collection = client.get_collection(name="korean_history_chunks")

# 2. BGE-m3 모델 로드 (CUDA 사용)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("BAAI/bge-m3", device=device)

# 3. 쿼리 임베딩 생성
query = "세종대왕이 만든 한글"
query_embedding = model.encode(
    query,
    normalize_embeddings=True,
    show_progress_bar=False
).tolist()

# 4. 검색 실행
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5
)

# 5. 결과 처리
for i, (doc_id, doc, dist, meta) in enumerate(zip(
    results['ids'][0],
    results['documents'][0],
    results['distances'][0],
    results['metadatas'][0]
), 1):
    print(f"{i}. {doc_id} (거리: {dist:.4f})")
    print(f"   소스: {meta['source']}, 페이지: {meta['page']}")
    print(f"   문서: {doc[:100]}...")
```

### 5.3 메타데이터 필터링
```python
# 특정 소스로 필터링
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5,
    where={"source": "벌거벗은한국사-조선편"}
)

# 복합 필터링 (예: 특정 소스 + 특정 타입)
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5,
    where={
        "$and": [
            {"source": "벌거벗은한국사-조선편"},
            {"type": "text"}
        ]
    }
)
```

---

## 6. 성능 최적화 (Performance Optimization)

### 6.1 CUDA 활용
- **임베딩 생성:** BGE-m3 모델이 CUDA를 사용하여 임베딩 생성 속도 향상
- **환경:** GCP 인스턴스, NVIDIA A100-SXM4-40GB GPU
- **효과:** CPU 대비 약 10배 빠른 임베딩 생성 속도

### 6.2 배치 처리
- **적재 시:** 배치 크기 1000개로 나눠서 적재하여 메모리 효율성 향상
- **검색 시:** 단일 쿼리 검색이므로 배치 처리 불필요

### 6.3 검색 속도 최적화
- **인덱싱:** ChromaDB가 자동으로 벡터 인덱스를 생성하여 빠른 검색 보장
- **정규화:** L2 정규화된 벡터 사용으로 Cosine Similarity 계산 최적화

---

## 7. 검증 및 테스트 (Verification & Testing)

### 7.1 검증 스크립트
- **스크립트 파일:** `verify_vectordb.py`
- **테스트 항목:**
  1. 기본 검색 테스트 (다양한 쿼리)
  2. 메타데이터 필터링 테스트
  3. 성능 벤치마크 (10회 반복)
  4. Collection 통계 확인

### 7.2 검증 결과
- **검색 정확도:** 모든 테스트 쿼리에서 관련성 높은 문서가 1위로 검색됨
- **검색 속도:** 평균 2.37ms로 목표(10ms)를 크게 상회
- **필터링 기능:** 메타데이터 필터링 정상 작동 확인

---

## 8. 향후 계획 (Next Steps)

### 8.1 단기 계획
1. **RAG 검색 모듈 개발:** 사용자 질문을 받아 임베딩 생성 → ChromaDB 검색 → 상위 K개 문서 반환하는 모듈 개발
2. **검색 결과 후처리:** 검색된 문서를 LLM에 전달하기 전 전처리 로직 추가

### 8.2 중장기 계획
1. **재랭킹(Re-ranking):** Cross-Encoder 모델을 사용하여 검색 결과 재랭킹하여 정확도 향상
2. **하이브리드 검색:** Dense Retrieval(벡터 검색) + Sparse Retrieval(키워드 검색) 결합
3. **성능 모니터링:** 실제 사용자 질문에 대한 검색 성능 지표 수집 및 분석

---

## 9. 참고 자료
- **구축 스크립트:** `build_vectordb.py`
- **검증 스크립트:** `verify_vectordb.py`
- **구축 로그:** `build_vectordb.log`
- **검증 로그:** `verify_vectordb.log`
- **ChromaDB 문서:** [ChromaDB Documentation](https://docs.trychroma.com/)
- **BGE-m3 모델:** [HuggingFace - BGE-m3](https://huggingface.co/BAAI/bge-m3)

