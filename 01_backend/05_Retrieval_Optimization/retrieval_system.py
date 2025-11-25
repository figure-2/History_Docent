"""
통합 검색 시스템 (Retrieval System)
- Pipeline: Query -> BM25 Search (Top-50) -> Reranker -> Final Top-K
"""
from hybrid_retriever import HybridRetriever
from reranker import Reranker

class RetrievalSystem:
    """통합 검색 시스템: BM25 + Reranker"""
    
    def __init__(self):
        self.retriever = HybridRetriever()  # 기본값: BM25 Only
        self.reranker = Reranker()
        self.initialized = False
        
    def initialize(self):
        """시스템 초기화"""
        if self.initialized:
            return
            
        print("🚀 검색 시스템 초기화 시작...")
        self.retriever.initialize()
        self.reranker.initialize()
        self.initialized = True
        print("✅ 검색 시스템 준비 완료!")
        
    def search(self, query: str, final_k: int = 5, candidate_k: int = 50) -> list:
        """
        통합 검색 수행
        
        Args:
            query: 사용자 질문
            final_k: 최종 반환할 개수
            candidate_k: 1차 검색에서 가져올 후보군 개수 (기본 50)
            
        Returns:
            재순위화된 최종 검색 결과 리스트
        """
        if not self.initialized:
            self.initialize()
        
        # 1. 1차 검색 (후보군 확보)
        # BM25가 Recall@50은 거의 100%일 것이므로 충분히 많이 가져옴
        candidates = self.retriever.search_bm25_only(query, top_k=candidate_k)
        
        # 2. Reranking
        final_results = self.reranker.rerank(query, candidates, top_k=final_k)
        
        return final_results

# -----------------------------------------------------------------------------
# 테스트 코드
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    system = RetrievalSystem()
    system.initialize()
    
    test_queries = [
        "세종대왕이 만든 한글",
        "임진왜란",
        "조선왕조실록"
    ]
    
    print("\n" + "=" * 60)
    print("통합 검색 시스템 테스트")
    print("=" * 60)
    
    for query in test_queries:
        print(f"\n🔍 쿼리: {query}")
        print("-" * 60)
        
        results = system.search(query, final_k=3)
        
        for res in results:
            rerank_score = res.metadata.get('rerank_score', 0)
            print(f"\n[Rank {res.rank}] {res.chunk_id}")
            print(f"  Rerank Score: {rerank_score:.4f}")
            print(f"  BM25 Score: {res.bm25_score:.4f}")
            print(f"  Text: {res.text[:150]}...")

