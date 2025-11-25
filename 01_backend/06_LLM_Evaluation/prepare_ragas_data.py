#!/usr/bin/env python3
"""
RAGAS 평가를 위한 데이터 준비 스크립트
- 목적: 검색된 컨텍스트 정보를 CSV에 추가
- 입력: llm_selected_model_full_test.csv
- 출력: llm_selected_model_full_test_with_contexts.csv
- 특징: 50개씩 배치 처리, 재개 기능, 백그라운드 실행 지원
"""

import json
import pandas as pd
import torch
import gc
import os
import sys
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Set
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from konlpy.tag import Okt
import chromadb
from datetime import datetime

# -----------------------------------------------------------------------------
# 설정
# -----------------------------------------------------------------------------
BASE_DIR = Path("/home/pencilfoxs/00_new/History_Docent")
INPUT_CSV = BASE_DIR / "06_LLM_Evaluation/results/llm_selected_model_full_test.csv"
OUTPUT_CSV = BASE_DIR / "06_LLM_Evaluation/results/llm_selected_model_full_test_with_contexts.csv"
CHUNK_FILE = BASE_DIR / "02_Chunking/output/all_chunks.json"
VECTORDB_DIR = BASE_DIR / "04_VectorDB/chroma_db"
COLLECTION_NAME = "korean_history_chunks"
RESULTS_DIR = BASE_DIR / "06_LLM_Evaluation/results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 선정된 모델
EMBEDDING_MODEL = "BAAI/bge-m3"
RERANKER_MODEL = "Dongjin-kr/ko-reranker"

# 검색 설정
TOP_K_RETRIEVE = 3  # RAGAS 평가에 필요한 최종 검색 문서 수
CANDIDATE_K = 50  # 1차 검색 후보군 개수
BATCH_SIZE = 50  # 배치 크기

# -----------------------------------------------------------------------------
# 검색 시스템 클래스 (test_selected_model.py에서 가져옴)
# -----------------------------------------------------------------------------
class BM25Retriever:
    def __init__(self, chunk_dict):
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
    
    def search(self, query: str, top_k: int) -> list:
        query_tokens = [t for t in self.tokenizer.morphs(query, stem=True) if t.strip()]
        scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [{'chunk_id': self.chunk_ids[i], 'text': self.chunk_dict[self.chunk_ids[i]]['text'], 'score': float(scores[i])} for i in top_indices]

class VectorRetriever:
    def __init__(self, collection, model_name):
        self.collection = collection
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)
        
    def search(self, query: str, top_k: int) -> list:
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
        self.chunk_dict = bm25.chunk_dict
    
    def search_weighted(self, query: str, top_k: int, v_weight: float = 0.6, b_weight: float = 0.4) -> list:
        bm25_res = self.bm25.search(query, top_k * 2)
        vector_res = self.vector.search(query, top_k * 2)
        
        scores = {}
        for res, weight in [(bm25_res, b_weight), (vector_res, v_weight)]:
            if not res: continue
            max_s = max(r['score'] for r in res) if res else 1.0
            min_s = min(r['score'] for r in res) if res else 0.0
            denom = max_s - min_s if max_s != min_s else 1.0
            for r in res:
                norm = (r['score'] - min_s) / denom if denom > 0 else 0.5
                if r['chunk_id'] not in scores:
                    scores[r['chunk_id']] = {'chunk_id': r['chunk_id'], 'text': r['text'], 'score': 0.0}
                scores[r['chunk_id']]['score'] += weight * norm
                
        sorted_items = sorted(scores.values(), key=lambda x: x['score'], reverse=True)
        return sorted_items[:top_k]

class Reranker:
    def __init__(self, model_name):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CrossEncoder(
            model_name, 
            device=self.device,
            automodel_args={"torch_dtype": torch.float16}
        )
    
    def rerank(self, query: str, candidates: list, top_k: int) -> list:
        if not candidates:
            return []
        
        pairs = [[query, item['text']] for item in candidates]
        scores = self.model.predict(pairs)
        
        reranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [item for item, _ in reranked[:top_k]]

# -----------------------------------------------------------------------------
# 메인 실행
# -----------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("RAGAS 평가용 데이터 준비 시작")
    print("=" * 60)
    
    # 1. 데이터 로드
    print(f"\n📂 CSV 파일 로드: {INPUT_CSV}")
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {INPUT_CSV}")
    
    df = pd.read_csv(INPUT_CSV)
    print(f"   총 {len(df)}개 질문 로드 완료")
    
    # 재개 기능: 이미 contexts가 있다면 건너뛰기
    if OUTPUT_CSV.exists():
        print(f"\n🔄 기존 결과 파일 발견: {OUTPUT_CSV}")
        existing_df = pd.read_csv(OUTPUT_CSV)
        if 'contexts' in existing_df.columns:
            processed_ids = set(existing_df['query_id'].tolist())
            df_to_process = df[~df['query_id'].isin(processed_ids)].copy()
            print(f"   이미 처리됨: {len(processed_ids)}개")
            print(f"   남은 작업: {len(df_to_process)}개")
            
            if len(df_to_process) == 0:
                print("\n✅ 모든 작업이 이미 완료되었습니다!")
                return
            
            # 기존 결과에 새 결과를 추가하기 위해 기존 DataFrame 준비
            base_df = existing_df.copy()
        else:
            df_to_process = df.copy()
            base_df = None
    else:
        df_to_process = df.copy()
        base_df = None
    
    # 2. 청크 데이터 로드
    print(f"\n📚 청크 데이터 로드: {CHUNK_FILE}")
    with open(CHUNK_FILE, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    chunk_dict = {chunk['chunk_id']: chunk for chunk in chunks}
    print(f"   총 {len(chunk_dict)}개 청크 로드 완료")
    
    # 3. 검색 시스템 초기화
    print(f"\n🛠️  검색 시스템 초기화 중...")
    
    # BM25
    bm25_retriever = BM25Retriever(chunk_dict)
    
    # Vector DB
    print("   📍 Vector DB 연결 중...")
    client = chromadb.PersistentClient(path=str(VECTORDB_DIR))
    collection = client.get_collection(COLLECTION_NAME)
    vector_retriever = VectorRetriever(collection, EMBEDDING_MODEL)
    
    # Hybrid
    hybrid_retriever = HybridRetriever(bm25_retriever, vector_retriever)
    
    # Reranker
    print("   🔄 Reranker 로드 중...")
    reranker = Reranker(RERANKER_MODEL)
    
    print("   ✅ 검색 시스템 준비 완료!")
    
    # 4. 각 질문에 대해 검색 수행 및 contexts 추가
    print(f"\n🔍 검색 수행 중 (Top-{TOP_K_RETRIEVE} 문서)...")
    
    contexts_list = []
    results_list = []
    
    for idx, row in tqdm(df_to_process.iterrows(), total=len(df_to_process), desc="   검색 진행"):
        query = row['query']
        
        try:
            # Hybrid 검색 (후보군 많이 가져오기)
            candidates = hybrid_retriever.search_weighted(query, top_k=CANDIDATE_K, v_weight=0.6, b_weight=0.4)
            
            # Reranking
            final_results = reranker.rerank(query, candidates, top_k=TOP_K_RETRIEVE)
            
            # 텍스트만 추출하여 리스트로 변환
            contexts = [item['text'] for item in final_results]
            contexts_list.append(contexts)
            
            # 결과 저장용 데이터 준비
            result_row = row.to_dict()
            result_row['contexts'] = contexts
            results_list.append(result_row)
            
        except Exception as e:
            print(f"\n   ⚠️  질문 {row['query_id']} 처리 중 오류: {str(e)}")
            # 빈 contexts로 대체
            contexts_list.append([])
            result_row = row.to_dict()
            result_row['contexts'] = []
            results_list.append(result_row)
        
        # 메모리 정리 (배치마다)
        if (idx + 1) % BATCH_SIZE == 0:
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # 5. DataFrame에 contexts 컬럼 추가
    print(f"\n💾 결과 저장 중...")
    new_df = pd.DataFrame(results_list)
    
    # 기존 결과와 병합
    if base_df is not None:
        final_df = pd.concat([base_df, new_df], ignore_index=True)
        # 중복 제거 (query_id 기준)
        final_df = final_df.drop_duplicates(subset=['query_id'], keep='last')
    else:
        final_df = new_df
    
    # contexts를 JSON 문자열로 변환 (CSV 저장을 위해)
    final_df['contexts'] = final_df['contexts'].apply(lambda x: json.dumps(x, ensure_ascii=False))
    
    # 저장
    final_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
    
    print(f"   ✅ 완료! 총 {len(final_df)}개 질문에 contexts 추가됨")
    print(f"   💾 저장 위치: {OUTPUT_CSV}")
    
    print("\n" + "=" * 60)
    print("데이터 준비 완료!")
    print("=" * 60)

if __name__ == "__main__":
    main()

