"""
하이브리드 리트리버 (Hybrid Retriever)
- Vector Search (ChromaDB) + BM25 (Keyword Search) 결합
- 형태소 분석기: Okt (벤치마크 결과 92.0% Recall@1)
- 가중치: Vector 0.6, BM25 0.4
- RRF (Reciprocal Rank Fusion) 옵션 지원
"""
import json
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import chromadb
from chromadb.config import Settings
import torch
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from konlpy.tag import Okt

# -----------------------------------------------------------------------------
# 설정
# -----------------------------------------------------------------------------
BASE_DIR = Path("/home/pencilfoxs/Hackathon/4_History_Docent")
VECTORDB_DIR = BASE_DIR / "04_VectorDB/chroma_db"
COLLECTION_NAME = "korean_history_chunks"
EMBEDDING_MODEL = "BAAI/bge-m3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 하이브리드 검색 가중치
VECTOR_WEIGHT = 0.6
BM25_WEIGHT = 0.4

# RRF 파라미터
RRF_K = 60  # RRF 상수 (일반적으로 60 사용)

# -----------------------------------------------------------------------------
# 데이터 클래스
# -----------------------------------------------------------------------------
@dataclass
class RetrievalResult:
    """검색 결과 데이터 클래스"""
    chunk_id: str
    text: str
    metadata: Dict[str, Any]
    vector_score: float
    bm25_score: float
    hybrid_score: float
    rank: int

# -----------------------------------------------------------------------------
# 하이브리드 리트리버 클래스
# -----------------------------------------------------------------------------
class HybridRetriever:
    """하이브리드 리트리버: Vector + BM25"""
    
    def __init__(
        self,
        vectordb_path: Path = VECTORDB_DIR,
        collection_name: str = COLLECTION_NAME,
        embedding_model: str = EMBEDDING_MODEL,
        device: str = DEVICE,
        vector_weight: float = VECTOR_WEIGHT,
        bm25_weight: float = BM25_WEIGHT,
        use_rrf: bool = False
    ):
        """
        Args:
            vectordb_path: ChromaDB 경로
            collection_name: Collection 이름
            embedding_model: 임베딩 모델 이름
            device: 디바이스 (cuda/cpu)
            vector_weight: Vector 검색 가중치 (기본 0.6)
            bm25_weight: BM25 검색 가중치 (기본 0.4)
            use_rrf: RRF 사용 여부 (기본 False)
        """
        self.vectordb_path = vectordb_path
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        self.device = device
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.use_rrf = use_rrf
        
        # 컴포넌트 초기화
        self.client = None
        self.collection = None
        self.embedding_model = None
        self.tokenizer = None
        self.bm25_index = None
        self.documents = []
        self.doc_ids = []
        self.doc_metadata = []
        
        print(f"✅ HybridRetriever 초기화 완료")
        print(f"   - Vector 가중치: {vector_weight}")
        print(f"   - BM25 가중치: {bm25_weight}")
        print(f"   - RRF 사용: {use_rrf}")
    
    def initialize(self):
        """리트리버 초기화 (ChromaDB 연결, 모델 로드, BM25 인덱스 구축)"""
        print("\n🚀 HybridRetriever 초기화 시작...")
        
        # 1. ChromaDB 연결
        print("   📂 ChromaDB 연결 중...")
        self.client = chromadb.PersistentClient(
            path=str(self.vectordb_path),
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_collection(name=self.collection_name)
        print(f"   ✅ ChromaDB 연결 완료 (Collection: {self.collection_name})")
        
        # 2. 전체 문서 로드
        print("   📄 문서 로드 중...")
        all_data = self.collection.get()
        self.documents = all_data['documents']
        self.doc_ids = all_data['ids']
        self.doc_metadata = all_data['metadatas']
        print(f"   ✅ 문서 {len(self.documents)}개 로드 완료")
        
        # 3. 임베딩 모델 로드
        print(f"   🤖 임베딩 모델 로드 중: {self.embedding_model_name} ({self.device})...")
        self.embedding_model = SentenceTransformer(
            self.embedding_model_name,
            device=self.device
        )
        print("   ✅ 임베딩 모델 로드 완료")
        
        # 4. 형태소 분석기 초기화 (Okt)
        print("   🔤 형태소 분석기 초기화 중 (Okt)...")
        self.tokenizer = Okt()
        print("   ✅ 형태소 분석기 초기화 완료")
        
        # 5. BM25 인덱스 구축
        print("   🧮 BM25 인덱스 구축 중...")
        start_time = time.time()
        tokenized_corpus = []
        for doc in self.documents:
            tokens = self.tokenizer.morphs(doc, stem=True)
            tokenized_corpus.append([t for t in tokens if t.strip()])
        
        self.bm25_index = BM25Okapi(tokenized_corpus)
        indexing_time = time.time() - start_time
        print(f"   ✅ BM25 인덱스 구축 완료 ({indexing_time:.2f}초)")
        
        print("\n✅ HybridRetriever 초기화 완료!\n")
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """점수를 0-1 범위로 정규화 (Min-Max Normalization)"""
        if len(scores) == 0:
            return scores
        min_score = scores.min()
        max_score = scores.max()
        if max_score == min_score:
            return np.ones_like(scores)
        return (scores - min_score) / (max_score - min_score)
    
    def _apply_rrf(self, vector_ranks: Dict[str, int], bm25_ranks: Dict[str, int]) -> Dict[str, float]:
        """Reciprocal Rank Fusion (RRF) 적용"""
        rrf_scores = {}
        all_ids = set(vector_ranks.keys()) | set(bm25_ranks.keys())
        
        for doc_id in all_ids:
            score = 0.0
            if doc_id in vector_ranks:
                score += 1.0 / (RRF_K + vector_ranks[doc_id])
            if doc_id in bm25_ranks:
                score += 1.0 / (RRF_K + bm25_ranks[doc_id])
            rrf_scores[doc_id] = score
        
        return rrf_scores
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        return_scores: bool = False
    ) -> List[RetrievalResult]:
        """
        하이브리드 검색 실행
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 상위 K개 결과
            return_scores: 점수 반환 여부
            
        Returns:
            검색 결과 리스트 (RetrievalResult)
        """
        if self.bm25_index is None:
            raise ValueError("리트리버가 초기화되지 않았습니다. initialize()를 먼저 호출하세요.")
        
        # 1. Vector 검색
        query_embedding = self.embedding_model.encode(
            query,
            normalize_embeddings=True
        ).tolist()
        
        vector_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k * 2  # 더 많이 가져와서 하이브리드 점수 계산
        )
        
        # Vector 점수 딕셔너리 생성
        vector_scores = {}
        vector_ranks = {}
        for i, doc_id in enumerate(vector_results['ids'][0]):
            # ChromaDB는 거리 기반이므로, 거리가 작을수록 높은 점수
            # 거리를 점수로 변환 (1 / (1 + distance))
            distance = 1.0 - vector_results['distances'][0][i]  # cosine similarity
            vector_scores[doc_id] = distance
            vector_ranks[doc_id] = i + 1
        
        # 2. BM25 검색
        tokenized_query = self.tokenizer.morphs(query, stem=True)
        tokenized_query = [t for t in tokenized_query if t.strip()]
        
        bm25_scores_array = self.bm25_index.get_scores(tokenized_query)
        bm25_scores = {}
        bm25_ranks = {}
        
        # 상위 결과만 추출
        top_bm25_indices = np.argsort(bm25_scores_array)[::-1][:top_k * 2]
        for rank, idx in enumerate(top_bm25_indices):
            doc_id = self.doc_ids[idx]
            bm25_scores[doc_id] = float(bm25_scores_array[idx])
            bm25_ranks[doc_id] = rank + 1
        
        # 3. 하이브리드 점수 계산
        if self.use_rrf:
            # RRF 방식
            hybrid_scores = self._apply_rrf(vector_ranks, bm25_ranks)
        else:
            # 가중치 결합 방식
            all_ids = set(vector_scores.keys()) | set(bm25_scores.keys())
            hybrid_scores = {}
            
            # 점수 정규화
            if vector_scores:
                vector_values = np.array([vector_scores.get(doc_id, 0) for doc_id in all_ids])
                vector_normalized = self._normalize_scores(vector_values)
            else:
                vector_normalized = np.zeros(len(all_ids))
            
            if bm25_scores:
                bm25_values = np.array([bm25_scores.get(doc_id, 0) for doc_id in all_ids])
                bm25_normalized = self._normalize_scores(bm25_values)
            else:
                bm25_normalized = np.zeros(len(all_ids))
            
            # 가중치 결합
            for i, doc_id in enumerate(all_ids):
                hybrid_scores[doc_id] = (
                    self.vector_weight * vector_normalized[i] +
                    self.bm25_weight * bm25_normalized[i]
                )
        
        # 4. 상위 K개 결과 추출
        sorted_results = sorted(
            hybrid_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        # 5. 결과 구성
        results = []
        for rank, (doc_id, hybrid_score) in enumerate(sorted_results, 1):
            # 문서 인덱스 찾기
            doc_idx = self.doc_ids.index(doc_id)
            
            result = RetrievalResult(
                chunk_id=doc_id,
                text=self.documents[doc_idx],
                metadata=self.doc_metadata[doc_idx] if self.doc_metadata else {},
                vector_score=vector_scores.get(doc_id, 0.0),
                bm25_score=bm25_scores.get(doc_id, 0.0),
                hybrid_score=hybrid_score,
                rank=rank
            )
            results.append(result)
        
        return results
    
    def search_vector_only(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """Vector 검색만 수행 (비교용)"""
        query_embedding = self.embedding_model.encode(
            query,
            normalize_embeddings=True
        ).tolist()
        
        vector_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        results = []
        for i, doc_id in enumerate(vector_results['ids'][0]):
            doc_idx = self.doc_ids.index(doc_id)
            distance = 1.0 - vector_results['distances'][0][i]
            
            result = RetrievalResult(
                chunk_id=doc_id,
                text=self.documents[doc_idx],
                metadata=self.doc_metadata[doc_idx] if self.doc_metadata else {},
                vector_score=distance,
                bm25_score=0.0,
                hybrid_score=distance,
                rank=i + 1
            )
            results.append(result)
        
        return results
    
    def search_bm25_only(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """BM25 검색만 수행 (비교용)"""
        tokenized_query = self.tokenizer.morphs(query, stem=True)
        tokenized_query = [t for t in tokenized_query if t.strip()]
        
        bm25_scores_array = self.bm25_index.get_scores(tokenized_query)
        top_indices = np.argsort(bm25_scores_array)[::-1][:top_k]
        
        results = []
        for rank, idx in enumerate(top_indices, 1):
            result = RetrievalResult(
                chunk_id=self.doc_ids[idx],
                text=self.documents[idx],
                metadata=self.doc_metadata[idx] if self.doc_metadata else {},
                vector_score=0.0,
                bm25_score=float(bm25_scores_array[idx]),
                hybrid_score=float(bm25_scores_array[idx]),
                rank=rank
            )
            results.append(result)
        
        return results

# -----------------------------------------------------------------------------
# 테스트 코드
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # 리트리버 초기화
    retriever = HybridRetriever(
        vector_weight=0.6,
        bm25_weight=0.4,
        use_rrf=False
    )
    retriever.initialize()
    
    # 테스트 쿼리
    test_queries = [
        "세종대왕이 만든 한글",
        "임진왜란",
        "조선왕조실록",
        "정약용의 실학",
        "고려시대 무역"
    ]
    
    print("=" * 60)
    print("하이브리드 검색 테스트")
    print("=" * 60)
    
    for query in test_queries:
        print(f"\n🔍 쿼리: {query}")
        print("-" * 60)
        
        results = retriever.search(query, top_k=3)
        
        for result in results:
            print(f"\n[Rank {result.rank}] {result.chunk_id}")
            print(f"  Vector Score: {result.vector_score:.4f}")
            print(f"  BM25 Score: {result.bm25_score:.4f}")
            print(f"  Hybrid Score: {result.hybrid_score:.4f}")
            print(f"  Text: {result.text[:100]}...")
            print(f"  Metadata: {result.metadata}")

