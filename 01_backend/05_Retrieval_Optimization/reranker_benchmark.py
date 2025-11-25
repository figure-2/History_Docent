"""
Reranker 성능 비교 벤치마크
- 목적: BM25 Only vs BM25 + Reranker 성능 비교
- 평가 지표: Recall@1, Recall@3, Recall@5, MRR@5
"""
import json
import time
import random
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Any

# 기존 모듈 임포트
import sys
sys.path.append(str(Path(__file__).parent))
from hybrid_retriever import HybridRetriever, RetrievalResult
from reranker import Reranker

# -----------------------------------------------------------------------------
# 설정
# -----------------------------------------------------------------------------
BASE_DIR = Path("/home/pencilfoxs/00_new/History_Docent")
BENCHMARK_DATA = BASE_DIR / "03_Embedding/data/korean_history_benchmark_2000.json"
OUTPUT_REPORT = BASE_DIR / "05_Retrieval_Optimization/reranker_benchmark_result.md"
SAMPLE_SIZE = 50  # 평가용 샘플 수

# -----------------------------------------------------------------------------
# 평가 함수
# -----------------------------------------------------------------------------
def calculate_metrics(results: List[RetrievalResult], gold_chunk_id: str):
    """단일 쿼리에 대한 평가 지표 계산"""
    chunk_ids = [r.chunk_id for r in results]
    
    # Recall@K
    recall_1 = 1 if gold_chunk_id in chunk_ids[:1] else 0
    recall_3 = 1 if gold_chunk_id in chunk_ids[:3] else 0
    recall_5 = 1 if gold_chunk_id in chunk_ids[:5] else 0
    
    # MRR (Mean Reciprocal Rank)
    try:
        rank = chunk_ids.index(gold_chunk_id) + 1
        mrr = 1.0 / rank
    except ValueError:
        mrr = 0.0
        
    return recall_1, recall_3, recall_5, mrr

def run_benchmark(name: str, retriever: HybridRetriever, reranker: Reranker, samples: List[Dict], use_reranker: bool):
    """벤치마크 실행"""
    print(f"\n🚀 [{name}] 평가 중...")
    
    metrics = {
        "recall_1": 0, "recall_3": 0, "recall_5": 0, "mrr": 0, "time": 0
    }
    
    for sample in tqdm(samples, desc=f"   [{name}] 검색", leave=False):
        query = sample['query']
        gold_id = sample['chunk_id']
        
        start_search = time.time()
        
        # 검색 실행
        if use_reranker:
            # BM25 (Top-50) -> Reranker (Top-5)
            candidates = retriever.search_bm25_only(query, top_k=50)
            results = reranker.rerank(query, candidates, top_k=5)
        else:
            # BM25 Only (Top-5)
            results = retriever.search_bm25_only(query, top_k=5)
            
        search_time = time.time() - start_search
        
        # 지표 계산
        r1, r3, r5, mrr = calculate_metrics(results, gold_id)
        
        metrics["recall_1"] += r1
        metrics["recall_3"] += r3
        metrics["recall_5"] += r5
        metrics["mrr"] += mrr
        metrics["time"] += search_time

    # 평균 계산
    count = len(samples)
    final_metrics = {
        "name": name,
        "recall_1": (metrics["recall_1"] / count) * 100,
        "recall_3": (metrics["recall_3"] / count) * 100,
        "recall_5": (metrics["recall_5"] / count) * 100,
        "mrr": (metrics["mrr"] / count),
        "avg_time": (metrics["time"] / count) * 1000  # ms
    }
    
    print(f"   📊 결과: R@1={final_metrics['recall_1']:.1f}%, R@5={final_metrics['recall_5']:.1f}%, MRR={final_metrics['mrr']:.3f}")
    return final_metrics

# -----------------------------------------------------------------------------
# 메인 로직
# -----------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Reranker 성능 비교 벤치마크")
    print("=" * 60)
    
    # 1. 시스템 초기화
    print("\n📂 시스템 초기화 중...")
    retriever = HybridRetriever()
    retriever.initialize()
    
    reranker = Reranker()
    reranker.initialize()
    
    # 2. 데이터 로드 및 샘플링
    print("\n📂 평가 데이터 로드 중...")
    with open(BENCHMARK_DATA, 'r', encoding='utf-8') as f:
        all_samples = json.load(f)
    
    random.seed(42)
    samples = random.sample(all_samples, min(SAMPLE_SIZE, len(all_samples)))
    print(f"   ✅ 평가 데이터: {len(samples)}개 샘플 준비 완료")
    
    # 3. 전략별 평가 실행
    strategies = [
        ("BM25 Only", False),
        ("BM25 + Reranker", True)
    ]
    
    results = []
    for name, use_rerank in strategies:
        res = run_benchmark(name, retriever, reranker, samples, use_rerank)
        results.append(res)
        
    # 4. 리포트 작성
    print("\n" + "="*60)
    print("🏆 최종 벤치마크 결과")
    print("="*60)
    
    with open(OUTPUT_REPORT, 'w', encoding='utf-8') as f:
        f.write("# Reranker 성능 비교 벤치마크\n\n")
        f.write(f"- 평가 일시: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- 샘플 수: {len(samples)}개\n")
        f.write(f"- 전체 문서 수: 3,719개\n\n")
        
        f.write("## 정량 평가 결과 (Quantitative Evaluation)\n\n")
        f.write("| Strategy | Recall@1 | Recall@3 | Recall@5 | MRR | Avg Time (ms) |\n")
        f.write("|---|---|---|---|---|---|\n")
        
        for r in results:
            # 콘솔 출력
            print(f"{r['name']:<25} | R@1: {r['recall_1']:5.1f}% | R@3: {r['recall_3']:5.1f}% | R@5: {r['recall_5']:5.1f}% | MRR: {r['mrr']:.3f} | {r['avg_time']:.1f}ms")
            
            # 파일 쓰기
            f.write(f"| **{r['name']}** | {r['recall_1']:.1f}% | {r['recall_3']:.1f}% | {r['recall_5']:.1f}% | {r['mrr']:.3f} | {r['avg_time']:.1f}ms |\n")
            
        # 승자 선정 (Recall@1 기준)
        winner = max(results, key=lambda x: x['recall_1'])
        f.write(f"\n## 🏆 최종 선정: **{winner['name']}**\n\n")
        f.write(f"- Recall@1: {winner['recall_1']:.1f}%\n")
        f.write(f"- MRR: {winner['mrr']:.3f}\n")
        f.write(f"- Recall@5: {winner['recall_5']:.1f}%\n")
        f.write(f"- 평균 검색 시간: {winner['avg_time']:.1f}ms\n")
        
        # 개선 폭 계산
        if len(results) == 2:
            bm25_only = results[0]
            bm25_rerank = results[1]
            improvement = bm25_rerank['recall_1'] - bm25_only['recall_1']
            f.write(f"\n## 📈 성능 개선\n\n")
            f.write(f"- Recall@1 개선: **+{improvement:.1f}%p** ({bm25_only['recall_1']:.1f}% → {bm25_rerank['recall_1']:.1f}%)\n")
            f.write(f"- MRR 개선: **+{bm25_rerank['mrr'] - bm25_only['mrr']:.3f}** ({bm25_only['mrr']:.3f} → {bm25_rerank['mrr']:.3f})\n")
            f.write(f"- 검색 시간 증가: **+{bm25_rerank['avg_time'] - bm25_only['avg_time']:.1f}ms** ({bm25_only['avg_time']:.1f}ms → {bm25_rerank['avg_time']:.1f}ms)\n")

    print(f"\n💾 결과 리포트 저장 완료: {OUTPUT_REPORT}")

if __name__ == "__main__":
    main()

