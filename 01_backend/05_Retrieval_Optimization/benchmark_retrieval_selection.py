#!/usr/bin/env python3
"""
리트리버 선정 벤치마크 (Validation Set 사용)
- 목적: 한국사 RAG에 최적화된 리트리버 전략 선정
- 비교군:
  1. BM25 Only (키워드 기반)
  2. Vector Only (BGE-m3 임베딩)
  3. Hybrid Weighted (Vector + BM25 가중치)
  4. Hybrid RRF (Reciprocal Rank Fusion)
- 평가 지표: MRR, Recall@1, Recall@3, Recall@5
- 데이터셋: Validation Set (2,223개) - 과적합 방지
"""

import json
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Any
from collections import defaultdict
import torch
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi
from konlpy.tag import Okt
import re

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

# 임베딩 모델
EMBEDDING_MODEL = "BAAI/bge-m3"

# -----------------------------------------------------------------------------
# 데이터 로드
# -----------------------------------------------------------------------------
def load_benchmark_data():
    """벤치마크 데이터 로드"""
    print(f"📂 벤치마크 데이터 로드: {BENCHMARK_DATA}")
    with open(BENCHMARK_DATA, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"   ✅ {len(data)}개 질문 로드 완료")
    return data

def load_chunks():
    """전체 청크 데이터 로드"""
    print(f"📂 청크 데이터 로드: {CHUNK_FILE}")
    with open(CHUNK_FILE, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    # chunk_id를 키로 하는 딕셔너리 생성
    chunk_dict = {chunk['chunk_id']: chunk for chunk in chunks}
    print(f"   ✅ {len(chunk_dict)}개 청크 로드 완료")
    return chunk_dict

def load_vectordb():
    """ChromaDB 벡터 데이터베이스 로드"""
    print(f"📂 벡터 DB 로드: {VECTORDB_DIR}")
    client = chromadb.PersistentClient(
        path=str(VECTORDB_DIR),
        settings=Settings(anonymized_telemetry=False)
    )
    collection = client.get_collection(name=COLLECTION_NAME)
    print(f"   ✅ 벡터 DB 로드 완료 ({collection.count()}개 문서)")
    return collection

# -----------------------------------------------------------------------------
# 리트리버 클래스
# -----------------------------------------------------------------------------
class BM25Retriever:
    """BM25 키워드 기반 리트리버 (OKT 형태소 분석기 사용)"""
    def __init__(self, dataset: List[Dict], chunk_dict: Dict):
        self.chunk_dict = chunk_dict
        self.chunk_ids = list(chunk_dict.keys())
        
        # OKT 형태소 분석기 초기화
        print("   🔤 OKT 형태소 분석기 초기화 중...")
        self.tokenizer = Okt()
        
        # BM25용 텍스트 준비 (OKT 형태소 분석)
        print("   🧮 BM25 인덱스 구축 중 (OKT 토크나이징)...")
        texts = []
        for chunk_id in tqdm(self.chunk_ids, desc="   토크나이징", leave=False):
            text = chunk_dict[chunk_id]['text']
            # OKT 형태소 분석 (어간 추출 포함)
            tokens = self.tokenizer.morphs(text, stem=True)
            # 빈 토큰 제거
            tokens = [t for t in tokens if t.strip()]
            texts.append(tokens)
        
        self.bm25 = BM25Okapi(texts)
        print("   ✅ BM25 인덱스 구축 완료 (OKT 사용)")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """BM25 검색"""
        # 쿼리 토크나이징 (OKT 사용)
        query_tokens = self.tokenizer.morphs(query, stem=True)
        query_tokens = [t for t in query_tokens if t.strip()]
        
        # BM25 점수 계산
        scores = self.bm25.get_scores(query_tokens)
        
        # 상위 K개 선택
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            chunk_id = self.chunk_ids[idx]
            results.append({
                'chunk_id': chunk_id,
                'text': self.chunk_dict[chunk_id]['text'],
                'score': float(scores[idx])
            })
        
        return results

class VectorRetriever:
    """벡터 기반 리트리버 (BGE-m3)"""
    def __init__(self, collection, model_name: str = EMBEDDING_MODEL):
        self.collection = collection
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   📝 임베딩 모델 로드 중: {model_name} ({self.device})")
        self.model = SentenceTransformer(model_name, device=self.device)
        print("   ✅ 임베딩 모델 로드 완료")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """벡터 검색"""
        # 쿼리 임베딩 생성
        query_embedding = self.model.encode(
            query,
            normalize_embeddings=True,
            show_progress_bar=False
        ).tolist()
        
        # ChromaDB 검색
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # 결과 변환
        ret = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i, chunk_id in enumerate(results['ids'][0]):
                ret.append({
                    'chunk_id': chunk_id,
                    'text': results['documents'][0][i],
                    'score': 1.0 - results['distances'][0][i]  # 거리를 점수로 변환
                })
        
        return ret

class HybridRetriever:
    """하이브리드 리트리버 (BM25 + Vector)"""
    def __init__(self, bm25_retriever: BM25Retriever, vector_retriever: VectorRetriever):
        self.bm25 = bm25_retriever
        self.vector = vector_retriever
    
    def search_weighted(self, query: str, top_k: int = 5, 
                       vector_weight: float = 0.6, bm25_weight: float = 0.4) -> List[Dict]:
        """가중치 기반 하이브리드 검색"""
        # 각각 검색
        bm25_results = self.bm25.search(query, top_k=top_k * 2)
        vector_results = self.vector.search(query, top_k=top_k * 2)
        
        # 점수 정규화 및 결합
        scores = {}
        
        # BM25 점수 정규화
        if bm25_results:
            max_bm25 = max(r['score'] for r in bm25_results)
            min_bm25 = min(r['score'] for r in bm25_results)
            bm25_range = max_bm25 - min_bm25 if max_bm25 != min_bm25 else 1.0
            
            for r in bm25_results:
                chunk_id = r['chunk_id']
                normalized = (r['score'] - min_bm25) / bm25_range if bm25_range > 0 else 0.5
                scores[chunk_id] = scores.get(chunk_id, 0) + bm25_weight * normalized
        
        # Vector 점수 정규화
        if vector_results:
            max_vector = max(r['score'] for r in vector_results)
            min_vector = min(r['score'] for r in vector_results)
            vector_range = max_vector - min_vector if max_vector != min_vector else 1.0
            
            for r in vector_results:
                chunk_id = r['chunk_id']
                normalized = (r['score'] - min_vector) / vector_range if vector_range > 0 else 0.5
                scores[chunk_id] = scores.get(chunk_id, 0) + vector_weight * normalized
        
        # 상위 K개 선택
        sorted_chunks = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        results = []
        for chunk_id, score in sorted_chunks:
            # 텍스트 가져오기
            text = self.bm25.chunk_dict.get(chunk_id, {}).get('text', '')
            results.append({
                'chunk_id': chunk_id,
                'text': text,
                'score': score
            })
        
        return results
    
    def search_rrf(self, query: str, top_k: int = 5, k: int = 60) -> List[Dict]:
        """RRF (Reciprocal Rank Fusion) 기반 하이브리드 검색"""
        # 각각 검색
        bm25_results = self.bm25.search(query, top_k=k)
        vector_results = self.vector.search(query, top_k=k)
        
        # RRF 점수 계산
        rrf_scores = {}
        
        # BM25 순위 기반 RRF 점수
        for rank, r in enumerate(bm25_results, 1):
            chunk_id = r['chunk_id']
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1.0 / (k + rank)
        
        # Vector 순위 기반 RRF 점수
        for rank, r in enumerate(vector_results, 1):
            chunk_id = r['chunk_id']
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1.0 / (k + rank)
        
        # 상위 K개 선택
        sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        results = []
        for chunk_id, score in sorted_chunks:
            text = self.bm25.chunk_dict.get(chunk_id, {}).get('text', '')
            results.append({
                'chunk_id': chunk_id,
                'text': text,
                'score': score
            })
        
        return results

# -----------------------------------------------------------------------------
# 평가 함수
# -----------------------------------------------------------------------------
def calculate_metrics(results: List[Dict], gold_chunk_id: str):
    """단일 쿼리에 대한 평가 지표 계산"""
    chunk_ids = [r['chunk_id'] for r in results]
    
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

def evaluate_retriever(name: str, retriever, dataset: List[Dict], 
                      search_func, search_kwargs: Dict = None):
    """리트리버 평가"""
    print(f"\n🚀 [{name}] 평가 중...")
    
    if search_kwargs is None:
        search_kwargs = {}
    
    metrics = {
        'recall_1': 0, 'recall_3': 0, 'recall_5': 0, 
        'mrr': 0, 'time': 0
    }
    
    type_metrics = defaultdict(lambda: {
        'recall_1': 0, 'recall_3': 0, 'recall_5': 0, 'mrr': 0, 'count': 0
    })
    
    failure_cases = []
    
    for item in tqdm(dataset, desc=f"   [{name}] 검색", leave=False):
        query = item['query']
        gold_id = item['chunk_id']
        q_type = item.get('type', 'unknown')
        
        start_time = time.time()
        results = search_func(query, **search_kwargs)
        search_time = time.time() - start_time
        
        r1, r3, r5, mrr = calculate_metrics(results, gold_id)
        
        metrics['recall_1'] += r1
        metrics['recall_3'] += r3
        metrics['recall_5'] += r5
        metrics['mrr'] += mrr
        metrics['time'] += search_time
        
        # 질문 유형별 집계
        type_metrics[q_type]['recall_1'] += r1
        type_metrics[q_type]['recall_3'] += r3
        type_metrics[q_type]['recall_5'] += r5
        type_metrics[q_type]['mrr'] += mrr
        type_metrics[q_type]['count'] += 1
        
        # 실패 케이스 기록
        if r1 == 0:
            failure_cases.append({
                'query': query,
                'gold_id': gold_id,
                'top_1_id': results[0]['chunk_id'] if results else None,
                'type': q_type
            })
    
    n = len(dataset)
    result = {
        'name': name,
        'MRR': metrics['mrr'] / n,
        'Recall@1': metrics['recall_1'] / n,
        'Recall@3': metrics['recall_3'] / n,
        'Recall@5': metrics['recall_5'] / n,
        'Latency(ms)': (metrics['time'] / n) * 1000,
        'failure_cases': failure_cases[:10]  # 상위 10개만 저장
    }
    
    # 질문 유형별 메트릭 추가
    for q_type, type_data in type_metrics.items():
        count = type_data['count']
        if count > 0:
            result[f'{q_type}_MRR'] = type_data['mrr'] / count
            result[f'{q_type}_R@1'] = type_data['recall_1'] / count
            result[f'{q_type}_R@3'] = type_data['recall_3'] / count
            result[f'{q_type}_R@5'] = type_data['recall_5'] / count
    
    return result

# -----------------------------------------------------------------------------
# 메인 실행
# -----------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("🚀 리트리버 선정 벤치마크 시작 (Validation Set)")
    print("=" * 80)
    
    # 데이터 로드
    dataset = load_benchmark_data()
    chunk_dict = load_chunks()
    collection = load_vectordb()
    
    # 질문 유형별 통계
    type_counts = defaultdict(int)
    for item in dataset:
        type_counts[item.get('type', 'unknown')] += 1
    print("\n📊 질문 유형별 분포:")
    for q_type, count in sorted(type_counts.items()):
        print(f"   - {q_type}: {count}개 ({count/len(dataset)*100:.1f}%)")
    
    # 리트리버 초기화
    print("\n🔧 리트리버 초기화 중...")
    bm25_retriever = BM25Retriever(dataset, chunk_dict)
    vector_retriever = VectorRetriever(collection)
    hybrid_retriever = HybridRetriever(bm25_retriever, vector_retriever)
    
    # 평가 실행
    results = []
    
    # 1. BM25 Only
    result_bm25 = evaluate_retriever(
        "BM25 Only",
        bm25_retriever,
        dataset,
        bm25_retriever.search,
        {'top_k': 5}
    )
    results.append(result_bm25)
    
    # 2. Vector Only
    result_vector = evaluate_retriever(
        "Vector Only",
        vector_retriever,
        dataset,
        vector_retriever.search,
        {'top_k': 5}
    )
    results.append(result_vector)
    
    # 3. Hybrid Weighted (0.6 Vector + 0.4 BM25)
    result_hybrid_weighted = evaluate_retriever(
        "Hybrid Weighted (0.6V+0.4B)",
        hybrid_retriever,
        dataset,
        hybrid_retriever.search_weighted,
        {'top_k': 5, 'vector_weight': 0.6, 'bm25_weight': 0.4}
    )
    results.append(result_hybrid_weighted)
    
    # 4. Hybrid RRF
    result_hybrid_rrf = evaluate_retriever(
        "Hybrid RRF",
        hybrid_retriever,
        dataset,
        hybrid_retriever.search_rrf,
        {'top_k': 5, 'k': 60}
    )
    results.append(result_hybrid_rrf)
    
    # 결과 출력
    print("\n" + "=" * 80)
    print("🏆 리트리버 선정 벤치마크 결과")
    print("=" * 80)
    
    import pandas as pd
    df = pd.DataFrame(results)
    df = df.sort_values(by='MRR', ascending=False)
    
    # 주요 지표만 출력
    main_cols = ['name', 'MRR', 'Recall@1', 'Recall@3', 'Recall@5', 'Latency(ms)']
    print("\n📊 전체 성능 비교:")
    print(df[main_cols].to_string(index=False))
    
    # 질문 유형별 결과
    type_cols = [col for col in df.columns if any(t in col for t in ['keyword', 'context', 'abstract'])]
    if type_cols:
        print("\n📈 질문 유형별 성능:")
        type_df = df[['name'] + type_cols]
        print(type_df.to_string(index=False))
    
    # 결과 저장
    csv_path = RESULTS_DIR / "retrieval_selection_validation_set.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n📁 결과 저장: {csv_path}")
    
    json_path = RESULTS_DIR / "retrieval_selection_validation_set.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"📁 상세 결과 저장: {json_path}")
    
    print("\n" + "=" * 80)
    print("✅ 벤치마크 완료!")
    print("=" * 80)

if __name__ == "__main__":
    main()

