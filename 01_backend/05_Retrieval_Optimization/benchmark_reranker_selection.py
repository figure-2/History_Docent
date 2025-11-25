#!/usr/bin/env python3
"""
리랭커 선정 벤치마크 (Validation Set 사용)
- 목적: Hybrid 검색 결과(Top-50)를 재정렬하여 최적의 리랭커 선정
- 후보군:
  1. BAAI/bge-reranker-v2-m3 (SOTA, 다국어)
  2. Dongjin-kr/ko-reranker (한국어 특화)
  3. BAAI/bge-reranker-large (베이스라인)
  4. cross-encoder/mmarco-mMiniLM-v2-L12-H384-v1 (경량/속도)
  5. maidalun1020/bce-reranker-base_v1 (RAG 전용)
"""

import json
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict
from collections import defaultdict
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi
from konlpy.tag import Okt
import pandas as pd

# -----------------------------------------------------------------------------
# 설정
# -----------------------------------------------------------------------------
BASE_DIR = Path("/home/pencilfoxs/00_new/History_Docent")
BENCHMARK_DATA = BASE_DIR / "03_Embedding/data/validation_set_20.json"
CHUNK_FILE = BASE_DIR / "02_Chunking/output/all_chunks.json"
VECTORDB_DIR = BASE_DIR / "04_VectorDB/chroma_db"
COLLECTION_NAME = "korean_history_chunks"
RESULTS_DIR = BASE_DIR / "05_Retrieval_Optimization/results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 1차 검색 설정 (Hybrid Weighted)
EMBEDDING_MODEL = "BAAI/bge-m3"
TOP_K_RETRIEVAL = 50  # 리랭킹을 위해 넉넉하게 가져옴 (Top-50)
TOP_K_FINAL = 5       # 최종 평가 기준

# 리랭커 후보군
RERANKER_MODELS = [
    "BAAI/bge-reranker-v2-m3",
    "Dongjin-kr/ko-reranker",
    "BAAI/bge-reranker-large",
    "cross-encoder/mmarco-mMiniLM-v2-L12-H384-v1",
    "maidalun1020/bce-reranker-base_v1"
]

# -----------------------------------------------------------------------------
# 데이터 로드 및 1차 검색기(Hybrid) 준비
# -----------------------------------------------------------------------------
# (이전 코드와 동일한 데이터 로드 및 Hybrid 클래스 재사용)
class BM25Retriever:
    def __init__(self, chunk_dict: Dict):
        self.chunk_dict = chunk_dict
        self.chunk_ids = list(chunk_dict.keys())
        self.tokenizer = Okt()
        print("   🧮 BM25 인덱스 구축 중...")
        texts = []
        for chunk_id in tqdm(self.chunk_ids, desc="   BM25 토크나이징", leave=False):
            text = chunk_dict[chunk_id]['text']
            tokens = [t for t in self.tokenizer.morphs(text, stem=True) if t.strip()]
            texts.append(tokens)
        self.bm25 = BM25Okapi(texts)
    
    def search(self, query: str, top_k: int) -> List[Dict]:
        query_tokens = [t for t in self.tokenizer.morphs(query, stem=True) if t.strip()]
        scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [{'chunk_id': self.chunk_ids[i], 'text': self.chunk_dict[self.chunk_ids[i]]['text'], 'score': float(scores[i])} for i in top_indices]

class VectorRetriever:
    def __init__(self, collection, model_name):
        self.collection = collection
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)
        
    def search(self, query: str, top_k: int) -> List[Dict]:
        query_emb = self.model.encode(query, normalize_embeddings=True, show_progress_bar=False).tolist()
        results = self.collection.query(query_embeddings=[query_emb], n_results=top_k)
        ret = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i, chunk_id in enumerate(results['ids'][0]):
                ret.append({
                    'chunk_id': chunk_id,
                    'text': results['documents'][0][i],
                    'score': 1.0 - results['distances'][0][i]
                })
        return ret

class HybridRetriever:
    def __init__(self, bm25, vector):
        self.bm25 = bm25
        self.vector = vector
    
    def search_weighted(self, query: str, top_k: int, v_weight: float = 0.6, b_weight: float = 0.4) -> List[Dict]:
        # Top-K의 2배를 가져와서 정규화
        bm25_res = self.bm25.search(query, top_k * 2)
        vector_res = self.vector.search(query, top_k * 2)
        
        scores = {}
        # 정규화 및 합산 로직 (간소화)
        for res, weight in [(bm25_res, b_weight), (vector_res, v_weight)]:
            if not res: continue
            max_s = max(r['score'] for r in res)
            min_s = min(r['score'] for r in res)
            denom = max_s - min_s if max_s != min_s else 1.0
            for r in res:
                norm = (r['score'] - min_s) / denom
                scores[r['chunk_id']] = scores.get(r['chunk_id'], 0) + weight * norm
                
        sorted_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [{'chunk_id': cid, 'text': self.bm25.chunk_dict[cid]['text'], 'score': sc} for cid, sc in sorted_ids]

# -----------------------------------------------------------------------------
# 벤치마크 실행
# -----------------------------------------------------------------------------
def evaluate_reranker(reranker_name: str, reranker_model, dataset: List[Dict], hybrid_retriever):
    print(f"\n🚀 [{reranker_name}] 평가 중...")
    metrics = {'mrr': 0, 'r1': 0, 'r3': 0, 'r5': 0, 'time': 0}
    type_metrics = defaultdict(lambda: {'mrr': 0, 'r1': 0, 'count': 0})
    failure_cases = []
    
    for item in tqdm(dataset, desc=f"   [{reranker_name.split('/')[-1]}]", leave=False):
        query = item['query']
        gold_id = item['chunk_id']
        q_type = item['type']
        
        # 1. Hybrid로 Top-50 검색 (베이스라인)
        start_time = time.time()
        candidates = hybrid_retriever.search_weighted(query, top_k=TOP_K_RETRIEVAL)
        
        # 2. Reranking
        if candidates:
            # (Query, Document) 쌍 생성
            pairs = [[query, doc['text']] for doc in candidates]
            # 점수 계산
            scores = reranker_model.predict(pairs)
            
            # 점수 기준 재정렬
            for i, doc in enumerate(candidates):
                doc['rerank_score'] = float(scores[i])
            
            reranked = sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)
        else:
            reranked = []
            
        search_time = time.time() - start_time
        
        # 3. 평가 (Top-5 기준)
        top_ids = [doc['chunk_id'] for doc in reranked[:5]]
        
        mrr = 0
        if gold_id in top_ids:
            mrr = 1.0 / (top_ids.index(gold_id) + 1)
        
        r1 = 1 if gold_id in top_ids[:1] else 0
        r3 = 1 if gold_id in top_ids[:3] else 0
        r5 = 1 if gold_id in top_ids[:5] else 0
        
        metrics['mrr'] += mrr
        metrics['r1'] += r1
        metrics['r3'] += r3
        metrics['r5'] += r5
        metrics['time'] += search_time
        
        # 유형별 통계
        type_metrics[q_type]['mrr'] += mrr
        type_metrics[q_type]['r1'] += r1
        type_metrics[q_type]['count'] += 1
        
        # 실패 분석 (Top-1 기준)
        if r1 == 0:
            failure_cases.append({
                'query': query,
                'gold_id': gold_id,
                'top_1_id': top_ids[0] if top_ids else None,
                'type': q_type
            })

    n = len(dataset)
    result = {
        'name': reranker_name,
        'MRR': metrics['mrr'] / n,
        'Recall@1': metrics['r1'] / n,
        'Recall@3': metrics['r3'] / n,
        'Recall@5': metrics['r5'] / n,
        'Latency(ms)': (metrics['time'] / n) * 1000,
        'failure_cases': failure_cases[:5]
    }
    
    for qt, data in type_metrics.items():
        if data['count'] > 0:
            result[f'{qt}_MRR'] = data['mrr'] / data['count']
            result[f'{qt}_R@1'] = data['r1'] / data['count']
            
    return result

def main():
    print("=" * 80)
    print(f"🚀 리랭커 선정 벤치마크 (Validation Set, Top-{TOP_K_RETRIEVAL} Reranking)")
    print("=" * 80)
    
    # 데이터 로드
    with open(BENCHMARK_DATA, 'r') as f: dataset = json.load(f)
    with open(CHUNK_FILE, 'r') as f: chunks = json.load(f)
    chunk_dict = {c['chunk_id']: c for c in chunks}
    
    # Hybrid 준비
    bm25 = BM25Retriever(chunk_dict)
    client = chromadb.PersistentClient(path=str(VECTORDB_DIR))
    vector = VectorRetriever(client.get_collection(COLLECTION_NAME), EMBEDDING_MODEL)
    hybrid = HybridRetriever(bm25, vector)
    
    results = []
    
    # 0. 리랭커 없는 Hybrid 성능 (기준점)
    print("\n📊 [Baseline] Hybrid Weighted Only 평가 중...")
    baseline_metrics = {'mrr': 0, 'r1': 0, 'r3': 0, 'r5': 0, 'time': 0}
    baseline_type_metrics = defaultdict(lambda: {'mrr': 0, 'r1': 0, 'count': 0})
    
    for item in tqdm(dataset, desc="   [Baseline]", leave=False):
        query = item['query']
        gold_id = item['chunk_id']
        q_type = item['type']
        
        start_time = time.time()
        candidates = hybrid.search_weighted(query, top_k=TOP_K_RETRIEVAL)
        search_time = time.time() - start_time
        
        top_ids = [doc['chunk_id'] for doc in candidates[:5]]
        mrr = 1.0 / (top_ids.index(gold_id) + 1) if gold_id in top_ids else 0.0
        r1 = 1 if gold_id in top_ids[:1] else 0
        r3 = 1 if gold_id in top_ids[:3] else 0
        r5 = 1 if gold_id in top_ids[:5] else 0
        
        baseline_metrics['mrr'] += mrr
        baseline_metrics['r1'] += r1
        baseline_metrics['r3'] += r3
        baseline_metrics['r5'] += r5
        baseline_metrics['time'] += search_time
        
        baseline_type_metrics[q_type]['mrr'] += mrr
        baseline_type_metrics[q_type]['r1'] += r1
        baseline_type_metrics[q_type]['count'] += 1
    
    n = len(dataset)
    baseline_result = {
        'name': 'Hybrid Weighted (Baseline)',
        'MRR': baseline_metrics['mrr'] / n,
        'Recall@1': baseline_metrics['r1'] / n,
        'Recall@3': baseline_metrics['r3'] / n,
        'Recall@5': baseline_metrics['r5'] / n,
        'Latency(ms)': (baseline_metrics['time'] / n) * 1000
    }
    for qt, data in baseline_type_metrics.items():
        if data['count'] > 0:
            baseline_result[f'{qt}_MRR'] = data['mrr'] / data['count']
            baseline_result[f'{qt}_R@1'] = data['r1'] / data['count']
    
    results.append(baseline_result)
    print(f"   👉 MRR: {baseline_result['MRR']:.3f}, R@1: {baseline_result['Recall@1']:.3f}, Latency: {baseline_result['Latency(ms)']:.1f}ms")
    
    # 각 리랭커 평가
    for model_name in RERANKER_MODELS:
        try:
            print(f"\n📥 모델 로드: {model_name}")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"   🖥️  사용 디바이스: {device}")
            if device == "cuda":
                print(f"   GPU 이름: {torch.cuda.get_device_name(0)}")
            # CrossEncoder 로드
            reranker = CrossEncoder(model_name, device=device, automodel_args={"torch_dtype": torch.float16})
            
            res = evaluate_reranker(model_name, reranker, dataset, hybrid)
            results.append(res)
            
            print(f"   👉 MRR: {res['MRR']:.3f}, R@1: {res['Recall@1']:.3f}, Latency: {res['Latency(ms)']:.1f}ms")
            print(f"      (Abstract MRR: {res.get('abstract_MRR', 0):.3f})")
            
        except Exception as e:
            print(f"   ❌ 모델 평가 실패 ({model_name}): {e}")

    # 결과 저장
    df = pd.DataFrame(results)
    save_path = RESULTS_DIR / "reranker_selection_results.csv"
    df.to_csv(save_path, index=False)
    print(f"\n💾 결과 저장 완료: {save_path}")
    
    # 최종 리포트 출력
    print("\n" + "="*80)
    print("🏆 리랭커 벤치마크 최종 결과")
    print("="*80)
    print(df[['name', 'MRR', 'Recall@1', 'abstract_MRR', 'Latency(ms)']].sort_values('MRR', ascending=False).to_string())

if __name__ == "__main__":
    main()
