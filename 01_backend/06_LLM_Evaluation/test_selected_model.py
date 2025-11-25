#!/usr/bin/env python3
"""
선정된 LLM 모델 (Bllossom-8B) 전체 Validation Set 테스트
- 목적: 선정된 모델의 전체 성능 검증
- 데이터셋: validation_set_20.json (전체 2,223개)
- 기능: 중간 저장, 재개 기능, 백그라운드 실행 지원
"""

import json
import time
import torch
import gc
import pandas as pd
import os
import sys
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Set
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from konlpy.tag import Okt
import numpy as np
import chromadb
from datetime import datetime

# -----------------------------------------------------------------------------
# 설정
# -----------------------------------------------------------------------------
BASE_DIR = Path("/home/pencilfoxs/00_new/History_Docent")
BENCHMARK_DATA = BASE_DIR / "03_Embedding/data/validation_set_20.json"
CHUNK_FILE = BASE_DIR / "02_Chunking/output/all_chunks.json"
VECTORDB_DIR = BASE_DIR / "04_VectorDB/chroma_db"
COLLECTION_NAME = "korean_history_chunks"
RESULTS_DIR = BASE_DIR / "06_LLM_Evaluation/results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 선정된 모델
SELECTED_MODEL = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
EMBEDDING_MODEL = "BAAI/bge-m3"
RERANKER_MODEL = "Dongjin-kr/ko-reranker"

# 중간 저장 설정
CHECKPOINT_INTERVAL = 50  # 50개마다 중간 저장
SAVE_PATH = RESULTS_DIR / "llm_selected_model_full_test.csv"
LOG_PATH = RESULTS_DIR / "full_test_progress.log"
PROGRESS_PATH = RESULTS_DIR / "full_test_progress.json"

# -----------------------------------------------------------------------------
# RAG 클래스 (기존과 동일)
# -----------------------------------------------------------------------------
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
        bm25_res = self.bm25.search(query, top_k * 2)
        vector_res = self.vector.search(query, top_k * 2)
        
        scores = {}
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
# RAG 파이프라인
# -----------------------------------------------------------------------------
def setup_rag_pipeline():
    print("🛠️ RAG 파이프라인(Retriever + Reranker) 초기화 중...")
    with open(CHUNK_FILE, 'r') as f: chunks = json.load(f)
    chunk_dict = {c['chunk_id']: c for c in chunks}
    
    bm25 = BM25Retriever(chunk_dict)
    client = chromadb.PersistentClient(path=str(VECTORDB_DIR))
    vector = VectorRetriever(client.get_collection(COLLECTION_NAME), EMBEDDING_MODEL)
    hybrid = HybridRetriever(bm25, vector)
    
    reranker = CrossEncoder(RERANKER_MODEL, device="cuda", automodel_args={"torch_dtype": torch.float16})
    
    return hybrid, reranker

def get_rag_context(query, hybrid, reranker, top_k=3):
    candidates = hybrid.search_weighted(query, top_k=50)
    if not candidates: return "관련 문서를 찾을 수 없습니다."
        
    pairs = [[query, doc['text']] for doc in candidates]
    scores = reranker.predict(pairs)
    for i, doc in enumerate(candidates):
        doc['rerank_score'] = float(scores[i])
    
    reranked = sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)[:top_k]
    return "\n\n".join([f"문서 {i+1}: {doc['text']}" for i, doc in enumerate(reranked)])

def get_prompt(query, context):
    return f"""당신은 한국사 전문가입니다. 아래 [참고 문서]를 바탕으로 [질문]에 대해 정확하고 상세하게 답변해주세요.
문서에 없는 내용은 지어내지 말고, 정보가 부족하면 부족하다고 말해주세요.

[참고 문서]
{context}

[질문]
{query}

[답변]
"""

def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()

def log_message(message: str, log_file: Path = LOG_PATH):
    """로그 메시지를 파일과 stdout에 출력"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] {message}\n"
    
    # 파일에 로그 기록
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(log_line)
    
    # stdout에도 출력 (백그라운드 실행 시 nohup.out에 기록됨)
    print(message, flush=True)

def load_progress() -> Dict:
    """진행 상황 로드"""
    if PROGRESS_PATH.exists():
        try:
            with open(PROGRESS_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {"completed_ids": [], "last_checkpoint": None}
    return {"completed_ids": [], "last_checkpoint": None}

def save_progress(completed_ids: Set[str], checkpoint_time: str):
    """진행 상황 저장"""
    progress = {
        "completed_ids": list(completed_ids),
        "last_checkpoint": checkpoint_time,
        "total_completed": len(completed_ids)
    }
    with open(PROGRESS_PATH, 'w', encoding='utf-8') as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)

def load_existing_results() -> pd.DataFrame:
    """기존 결과 로드"""
    if SAVE_PATH.exists():
        try:
            df = pd.read_csv(SAVE_PATH)
            log_message(f"📂 기존 결과 로드: {len(df)}개 응답")
            return df
        except Exception as e:
            log_message(f"⚠️ 기존 결과 로드 실패: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

def save_results(results: List[Dict], append: bool = False):
    """결과 저장 (중간 저장 지원)"""
    df = pd.DataFrame(results)
    if append and SAVE_PATH.exists():
        # 기존 결과와 병합
        existing_df = load_existing_results()
        if len(existing_df) > 0:
            # 중복 제거 (query_id 기준)
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=['query_id'], keep='last')
            df = combined_df
    
    df.to_csv(SAVE_PATH, index=False)
    log_message(f"💾 결과 저장 완료: {len(df)}개 응답 (파일: {SAVE_PATH})")

# -----------------------------------------------------------------------------
# 메인 함수
# -----------------------------------------------------------------------------
def main():
    print("=" * 80)
    print(f"🚀 선정된 모델 전체 Validation Set 테스트: {SELECTED_MODEL}")
    print("=" * 80)
    
    # 데이터 로드
    print("\n📂 데이터셋 로드 중...")
    with open(BENCHMARK_DATA, 'r') as f: dataset = json.load(f)
    print(f"✅ 총 {len(dataset)}개 질문 로드 완료")
    
    # RAG 파이프라인 설정
    hybrid, reranker = setup_rag_pipeline()
    
    # RAG Context 생성
    print("\n⚙️  RAG Context 생성 중...")
    for item in tqdm(dataset, desc="Context 생성"):
        item['rag_context'] = get_rag_context(item['query'], hybrid, reranker)
    
    del hybrid, reranker
    clear_gpu()
    
    # 모델 로드
    print(f"\n📥 모델 로드 중: {SELECTED_MODEL}")
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(SELECTED_MODEL, trust_remote_code=True, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        SELECTED_MODEL, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True, token=hf_token
    )
    
    # 추론 실행
    print(f"\n🚀 추론 시작 (총 {len(dataset)}개 질문)...")
    results = []
    
    for item in tqdm(dataset, desc="Generating"):
        prompt = get_prompt(item['query'], item['rag_context'])
        start_time = time.time()
        
        try:
            messages = [{"role": "user", "content": prompt}]
            inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
        except:
            inputs = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
        
        outputs = model.generate(
            inputs, max_new_tokens=512, temperature=0.1, do_sample=True,
            pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id
        )
        response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        
        results.append({
            "model": SELECTED_MODEL,
            "query_id": item.get('id', f"q_{len(results)}"),
            "query": item['query'],
            "response": response.strip(),
            "latency": time.time() - start_time,
            "type": item['type'],
            "chunk_id": item.get('chunk_id', ''),
            "gold_text": item.get('gold_text', '')
        })
    
    # 결과 저장
    df = pd.DataFrame(results)
    save_path = RESULTS_DIR / "llm_selected_model_full_test.csv"
    df.to_csv(save_path, index=False)
    
    # 통계 출력
    print("\n" + "=" * 80)
    print("📊 테스트 결과 요약")
    print("=" * 80)
    print(f"총 질문 수: {len(results)}개")
    print(f"평균 지연시간: {df['latency'].mean():.2f}초")
    print(f"질문 유형별 분포:")
    print(df['type'].value_counts())
    print(f"\n💾 결과 저장: {save_path}")
    
    del model, tokenizer
    clear_gpu()

if __name__ == "__main__":
    main()
