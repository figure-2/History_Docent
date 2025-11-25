"""
Reranker 모듈
- 역할: 1차 검색 결과(Candidate Chunks)를 정밀 재순위화
- 모델: BAAI/bge-reranker-v2-m3 (Cross-Encoder)
"""
import torch
from typing import List
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
import sys
from pathlib import Path

# 기존 모듈 임포트
sys.path.append(str(Path(__file__).parent))
from hybrid_retriever import RetrievalResult

class Reranker:
    """Cross-Encoder 기반 Reranker"""
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        
    def initialize(self):
        """모델 및 토크나이저 로드"""
        print(f"🔄 Reranker 초기화 중... (Device: {self.device})")
        try:
            print(f"   모델 로딩: {self.model_name} ...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            self.model.to(self.device)
            self.model.eval()
            print("   ✅ Reranker 로드 완료")
        except Exception as e:
            print(f"   ❌ Reranker 로드 실패: {e}")
            raise

    def rerank(self, query: str, results: List[RetrievalResult], top_k: int = 5) -> List[RetrievalResult]:
        """
        검색 결과를 재순위화 (Reranking)
        
        Args:
            query: 사용자 질문
            results: 1차 검색 결과 리스트
            top_k: 최종 반환할 개수
            
        Returns:
            재순위화된 결과 리스트
        """
        if not results:
            return []
        
        # 입력 쌍 생성 (Query, Document Text)
        pairs = [[query, r.text] for r in results]
        
        # 배치 처리로 추론 (Score 계산)
        scores = []
        batch_size = 16  # GPU 메모리에 따라 조정 가능
        
        with torch.no_grad():
            for i in tqdm(range(0, len(pairs), batch_size), desc="   Reranking", leave=False):
                batch_pairs = pairs[i:i+batch_size]
                
                inputs = self.tokenizer(
                    batch_pairs, 
                    padding=True, 
                    truncation=True, 
                    return_tensors='pt', 
                    max_length=512
                ).to(self.device)
                
                batch_scores = self.model(**inputs, return_dict=True).logits.view(-1,).float()
                scores.extend(batch_scores.cpu().tolist())
        
        # 점수 매핑 및 정렬
        reranked_results = []
        for i, score in enumerate(scores):
            original_result = results[i]
            # metadata에 rerank_score 추가
            if not hasattr(original_result, 'metadata') or original_result.metadata is None:
                original_result.metadata = {}
            original_result.metadata['rerank_score'] = float(score)
            reranked_results.append((original_result, float(score)))
        
        # 점수 내림차순 정렬
        reranked_results.sort(key=lambda x: x[1], reverse=True)
        
        # 상위 K개 반환 및 순위(rank) 갱신
        final_results = []
        for rank, (res, score) in enumerate(reranked_results[:top_k], 1):
            res.rank = rank  # 순위 갱신
            final_results.append(res)
        
        return final_results

# -----------------------------------------------------------------------------
# 테스트 코드
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    from hybrid_retriever import HybridRetriever
    
    # 리트리버 초기화
    retriever = HybridRetriever()
    retriever.initialize()
    
    # Reranker 초기화
    reranker = Reranker()
    reranker.initialize()
    
    # 테스트 쿼리
    test_query = "세종대왕이 만든 한글"
    
    print("\n" + "=" * 60)
    print("Reranker 테스트")
    print("=" * 60)
    print(f"\n🔍 쿼리: {test_query}")
    
    # 1차 검색 (BM25, Top-10)
    print("\n1️⃣ 1차 검색 (BM25, Top-10):")
    candidates = retriever.search_bm25_only(test_query, top_k=10)
    for i, res in enumerate(candidates[:3], 1):
        print(f"   [{i}] {res.chunk_id} (BM25: {res.bm25_score:.4f})")
    
    # 2차 Reranking (Top-5)
    print("\n2️⃣ 2차 Reranking (Top-5):")
    final_results = reranker.rerank(test_query, candidates, top_k=5)
    for res in final_results:
        rerank_score = res.metadata.get('rerank_score', 0)
        print(f"   [{res.rank}] {res.chunk_id} (Rerank: {rerank_score:.4f}, BM25: {res.bm25_score:.4f})")
        print(f"      Text: {res.text[:100]}...")

