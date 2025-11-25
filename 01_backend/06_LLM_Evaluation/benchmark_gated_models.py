#!/usr/bin/env python3
"""
Gated Repo 모델만 실행하는 벤치마크 스크립트
- LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct
- google/gemma-2-27b-it
"""

import json
import time
import torch
import gc
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from konlpy.tag import Okt
from dotenv import load_dotenv
import os
import sys
import chromadb

# RAG 클래스들 (benchmark_llm_selection.py와 동일)
BASE_DIR = Path("/home/pencilfoxs/00_new/History_Docent")
BENCHMARK_DATA = BASE_DIR / "03_Embedding/data/validation_set_20.json"
CHUNK_FILE = BASE_DIR / "02_Chunking/output/all_chunks.json"
VECTORDB_DIR = BASE_DIR / "04_VectorDB/chroma_db"
COLLECTION_NAME = "korean_history_chunks"
RESULTS_DIR = BASE_DIR / "06_LLM_Evaluation/results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_MODEL = "BAAI/bge-m3"
RERANKER_MODEL = "Dongjin-kr/ko-reranker"
SAMPLE_SIZE = 50

# Gated Repo 모델만
GATED_MODELS = [
    "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct",
    "google/gemma-2-27b-it"
]

# RAG 클래스들 (benchmark_llm_selection.py에서 복사)
class BM25Retriever:
    def __init__(self, chunk_dict: Dict):
        self.chunk_dict = chunk_dict
        self.chunk_ids = list(chunk_dict.keys())
        self.tokenizer = Okt()
        print("   🧮 BM25 인덱스 구축 중...")
        texts = []
        for chunk_id in tqdm(self.chunk_ids, desc="   BM25 토크나이징"):
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

def load_env():
    load_dotenv("/home/pencilfoxs/00_new/.env2")
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
        try:
            from huggingface_hub import login
            login(token=hf_token, add_to_git_credential=False)
            print("✅ HuggingFace 토큰 로드 완료 (Gated Repo 접근 가능)")
        except Exception as e:
            print(f"⚠️ HuggingFace 로그인 실패: {e}")

def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()

def get_prompt(query, context):
    return f"""당신은 한국사 전문가입니다. 아래 [참고 문서]를 바탕으로 [질문]에 대해 정확하고 상세하게 답변해주세요.
문서에 없는 내용은 지어내지 말고, 정보가 부족하면 부족하다고 말해주세요.

[참고 문서]
{context}

[질문]
{query}

[답변]
"""

def generate_local(model_name, dataset):
    print(f"📥 모델 로드 중: {model_name}")
    clear_gpu()
    results = []
    try:
        hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            token=hf_token,
            trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            token=hf_token,
            device_map="auto", 
            torch_dtype=torch.float16, 
            trust_remote_code=True
        )
        
        for item in tqdm(dataset, desc=f"   Generating ({model_name.split('/')[-1]})"):
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
                "model": model_name, "query_id": item['id'], "query": item['query'],
                "response": response.strip(), "latency": time.time() - start_time, "type": item['type']
            })
            
        del model, tokenizer
        clear_gpu()
        return results
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        clear_gpu()
        return []

def main():
    load_env()
    print("📂 데이터셋 준비 및 RAG Context 생성 중...")
    with open(BENCHMARK_DATA, 'r') as f: full_data = json.load(f)
    
    test_data = []
    counts = {"keyword": 0, "context": 0, "abstract": 0}
    target = SAMPLE_SIZE // 3
    for item in full_data:
        q_type = item['type']
        if counts[q_type] < target + (1 if q_type == 'abstract' and SAMPLE_SIZE % 3 != 0 else 0):
            item['id'] = f"bench_{len(test_data)}"
            test_data.append(item)
            counts[q_type] += 1
        if len(test_data) >= SAMPLE_SIZE: break
            
    hybrid, reranker = setup_rag_pipeline()
    print("⚙️  RAG Context 생성 중...")
    for item in tqdm(test_data):
        item['rag_context'] = get_rag_context(item['query'], hybrid, reranker)
    del hybrid, reranker
    clear_gpu()
    
    all_results = []
    
    for model_name in GATED_MODELS:
        print(f"\n🚀 벤치마크 시작: {model_name}")
        res = generate_local(model_name, test_data)
        all_results.extend(res)
        
        # 기존 결과와 병합
        existing_file = RESULTS_DIR / "llm_benchmark_responses.csv"
        if existing_file.exists():
            df_existing = pd.read_csv(existing_file)
            df_new = pd.DataFrame(all_results)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.to_csv(existing_file, index=False)
            print(f"💾 기존 결과에 추가 저장 완료")
        else:
            df = pd.DataFrame(all_results)
            df.to_csv(existing_file, index=False)
            print(f"💾 결과 저장 완료")
        
        all_results = []  # 다음 모델을 위해 초기화
            
    print(f"\n✅ Gated Repo 모델 벤치마크 완료!")

if __name__ == "__main__":
    main()

