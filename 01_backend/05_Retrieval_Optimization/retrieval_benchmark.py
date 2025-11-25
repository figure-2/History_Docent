"""
검색 전략 성능 비교 벤치마크 (Retrieval Strategy Benchmark)
- 목적: 한국사 RAG에 최적화된 검색 조합 선정
- 비교군:
  1. Vector Only (BGE-m3)
  2. BM25 Only (Okt)
  3. Hybrid Weighted (Vector 0.6 + BM25 0.4)
  4. Hybrid RRF (Reciprocal Rank Fusion)
- 평가 지표: Recall@1, Recall@3, Recall@5, MRR@5
"""
import json
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Any
import random

# 기존 모듈 임포트
import sys
sys.path.append(str(Path(__file__).parent))
from hybrid_retriever import HybridRetriever, RetrievalResult

# -----------------------------------------------------------------------------
# 설정
# -----------------------------------------------------------------------------
BASE_DIR = Path("/home/pencilfoxs/00_new/History_Docent")
BENCHMARK_DATA = BASE_DIR / "03_Embedding/data/korean_history_benchmark_2000.json"
OUTPUT_REPORT = BASE_DIR / "05_Retrieval_Optimization/retrieval_benchmark_result.md"
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

def run_strategy(name: str, retriever: HybridRetriever, samples: List[Dict], strategy_type: str):
    """특정 검색 전략 실행 및 평가"""
    print(f"\n🚀 [{name}] 전략 평가 중...")
    
    metrics = {
        "recall_1": 0, "recall_3": 0, "recall_5": 0, "mrr": 0, "time": 0
    }
    
    failure_cases = []  # 실패 케이스 기록
    
    for sample in tqdm(samples, desc=f"   [{name}] 검색", leave=False):
        query = sample['query']
        gold_id = sample['chunk_id']
        
        start_search = time.time()
        
        # 전략에 따른 검색 실행
        try:
            if strategy_type == "vector":
                results = retriever.search_vector_only(query, top_k=5)
            elif strategy_type == "bm25":
                results = retriever.search_bm25_only(query, top_k=5)
            elif strategy_type == "hybrid_weighted":
                # 가중치 방식으로 설정
                retriever.use_rrf = False
                retriever.vector_weight = 0.6
                retriever.bm25_weight = 0.4
                results = retriever.search(query, top_k=5)
            elif strategy_type == "hybrid_rrf":
                # RRF 방식으로 설정
                retriever.use_rrf = True
                results = retriever.search(query, top_k=5)
            else:
                raise ValueError(f"Unknown strategy type: {strategy_type}")
                
            search_time = time.time() - start_search
            
            # 지표 계산
            r1, r3, r5, mrr = calculate_metrics(results, gold_id)
            
            metrics["recall_1"] += r1
            metrics["recall_3"] += r3
            metrics["recall_5"] += r5
            metrics["mrr"] += mrr
            metrics["time"] += search_time
            
            # 실패 케이스 기록 (Recall@1 실패)
            if r1 == 0:
                failure_cases.append({
                    "query": query,
                    "gold_id": gold_id,
                    "top_3_ids": [r.chunk_id for r in results[:3]],
                    "top_3_texts": [r.text[:100] + "..." for r in results[:3]]
                })
                
        except Exception as e:
            print(f"   ⚠️  오류 발생: {e}")
            continue

    # 평균 계산
    count = len(samples)
    final_metrics = {
        "name": name,
        "recall_1": (metrics["recall_1"] / count) * 100,
        "recall_3": (metrics["recall_3"] / count) * 100,
        "recall_5": (metrics["recall_5"] / count) * 100,
        "mrr": (metrics["mrr"] / count),
        "avg_time": (metrics["time"] / count) * 1000,  # ms
        "failure_cases": failure_cases
    }
    
    print(f"   📊 결과: R@1={final_metrics['recall_1']:.1f}%, R@5={final_metrics['recall_5']:.1f}%, MRR={final_metrics['mrr']:.3f}")
    return final_metrics

# -----------------------------------------------------------------------------
# 메인 로직
# -----------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("검색 전략(Retrieval Strategy) 성능 비교 벤치마크")
    print("=" * 60)
    
    # 1. 리트리버 초기화 (한 번만 로드)
    print("\n📂 리트리버 초기화 중...")
    retriever = HybridRetriever()
    retriever.initialize()
    
    # 2. 데이터 로드 및 샘플링
    print("\n📂 평가 데이터 로드 중...")
    with open(BENCHMARK_DATA, 'r', encoding='utf-8') as f:
        all_samples = json.load(f)
    
    # 재현성을 위해 시드 고정
    random.seed(42)
    samples = random.sample(all_samples, min(SAMPLE_SIZE, len(all_samples)))
    print(f"   ✅ 평가 데이터: {len(samples)}개 샘플 준비 완료")
    
    # 3. 전략별 평가 실행
    strategies = [
        ("Vector Only", "vector"),
        ("BM25 Only", "bm25"),
        ("Hybrid (Weighted 0.6:0.4)", "hybrid_weighted"),
        ("Hybrid (RRF)", "hybrid_rrf")
    ]
    
    results = []
    for name, s_type in strategies:
        res = run_strategy(name, retriever, samples, s_type)
        results.append(res)
        
    # 4. 리포트 작성
    print("\n" + "="*60)
    print("🏆 최종 벤치마크 결과")
    print("="*60)
    
    with open(OUTPUT_REPORT, 'w', encoding='utf-8') as f:
        f.write("# 검색 전략 성능 비교 (Retrieval Strategy Benchmark)\n\n")
        f.write(f"- 평가 일시: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- 샘플 수: {len(samples)}개\n")
        f.write(f"- 전체 문서 수: 3,719개\n\n")
        
        f.write("## 정량 평가 결과 (Quantitative Evaluation)\n\n")
        f.write("| Strategy | Recall@1 | Recall@3 | Recall@5 | MRR | Avg Time (ms) |\n")
        f.write("|---|---|---|---|---|---|\n")
        
        for r in results:
            # 콘솔 출력
            print(f"{r['name']:<30} | R@1: {r['recall_1']:5.1f}% | R@3: {r['recall_3']:5.1f}% | R@5: {r['recall_5']:5.1f}% | MRR: {r['mrr']:.3f} | {r['avg_time']:.1f}ms")
            
            # 파일 쓰기
            f.write(f"| **{r['name']}** | {r['recall_1']:.1f}% | {r['recall_3']:.1f}% | {r['recall_5']:.1f}% | {r['mrr']:.3f} | {r['avg_time']:.1f}ms |\n")
            
        # 승자 선정 (Recall@5 기준)
        winner = max(results, key=lambda x: x['recall_5'])
        f.write(f"\n## 🏆 최종 선정: **{winner['name']}**\n\n")
        f.write(f"- Recall@5: {winner['recall_5']:.1f}%\n")
        f.write(f"- MRR: {winner['mrr']:.3f}\n")
        f.write(f"- Recall@1: {winner['recall_1']:.1f}%\n")
        f.write(f"- 평균 검색 시간: {winner['avg_time']:.1f}ms\n")
        
        # 전략별 특징 요약
        f.write("\n## 전략별 특징 분석\n\n")
        for r in results:
            f.write(f"### {r['name']}\n")
            f.write(f"- Recall@1: {r['recall_1']:.1f}%\n")
            f.write(f"- Recall@5: {r['recall_5']:.1f}%\n")
            f.write(f"- MRR: {r['mrr']:.3f}\n")
            f.write(f"- 평균 검색 시간: {r['avg_time']:.1f}ms\n")
            f.write(f"- Recall@1 실패 케이스: {len(r['failure_cases'])}개\n\n")
        
        # 실패 케이스 분석 (상위 5개만)
        f.write("## 실패 케이스 분석 (Failure Analysis)\n\n")
        for r in results:
            if len(r['failure_cases']) > 0:
                f.write(f"### {r['name']} - Recall@1 실패 케이스 (상위 5개)\n\n")
                for i, case in enumerate(r['failure_cases'][:5], 1):
                    f.write(f"#### 케이스 {i}\n")
                    f.write(f"- **쿼리:** {case['query']}\n")
                    f.write(f"- **정답 ID:** {case['gold_id']}\n")
                    f.write(f"- **상위 3개 결과:**\n")
                    for j, (cid, text) in enumerate(zip(case['top_3_ids'], case['top_3_texts']), 1):
                        f.write(f"  {j}. [{cid}] {text}\n")
                    f.write("\n")

    print(f"\n💾 결과 리포트 저장 완료: {OUTPUT_REPORT}")

if __name__ == "__main__":
    main()

