#!/usr/bin/env python3
"""
LLM 모델 선정 벤치마크 (Validation Set 50개 샘플 사용)
- 목적: RAG 파이프라인(리트리버+리랭커)을 통과한 Context를 기반으로 답변 생성 품질 비교
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
import google.generativeai as genai
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from konlpy.tag import Okt
from dotenv import load_dotenv
import os
import sys
import random
import chromadb

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

EMBEDDING_MODEL = "BAAI/bge-m3"
RERANKER_MODEL = "Dongjin-kr/ko-reranker"
SAMPLE_SIZE = 50

MODELS = {
    "Small": [
        "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct",
        "yanolja/EEVE-Korean-10.8B-v1.0",
        "MLP-KTLim/llama-3-Korean-Bllossom-8B",
        "beomi/Llama-3-Open-Ko-8B"
    ],
    "Medium": [
        "Qwen/Qwen2.5-14B-Instruct",
        "google/gemma-2-27b-it",
        "Qwen/Qwen2.5-32B-Instruct"
    ],
    "API": [
        "gemini-2.5-flash"
    ]
}

# -----------------------------------------------------------------------------
# RAG 클래스 정의 (의존성 문제 해결을 위해 직접 정의)
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# 유틸리티
# -----------------------------------------------------------------------------
def load_env():
    load_dotenv("/home/pencilfoxs/00_new/.env2")
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    
    # HuggingFace 토큰 설정 (Gated Repo 접근용)
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

# -----------------------------------------------------------------------------
# 추론 함수
# -----------------------------------------------------------------------------
def generate_local(model_name, dataset):
    print(f"📥 모델 로드 중: {model_name}")
    clear_gpu()
    results = []
    try:
        # HuggingFace 토큰 사용 (Gated Repo 접근)
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
        clear_gpu()
        return []

def generate_api(model_name, dataset):
    print(f"🌐 API 호출 중: {model_name}")
    try:
        model = genai.GenerativeModel(model_name)
        model.generate_content("test")
    except:
        print(f"⚠️ {model_name} 실패. 'gemini-1.5-flash'로 대체")
        model_name = "gemini-1.5-flash"
        model = genai.GenerativeModel(model_name)

    results = []
    for item in tqdm(dataset):
        prompt = get_prompt(item['query'], item['rag_context'])
        start_time = time.time()
        try:
            response = model.generate_content(prompt)
            text = response.text.strip()
        except Exception as e:
            text = f"Error: {e}"
        
        results.append({
            "model": model_name, "query_id": item['id'], "query": item['query'],
            "response": text, "latency": time.time() - start_time, "type": item['type']
        })
        time.sleep(1)
    return results

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
    all_models = MODELS["Small"] + MODELS["Medium"] + MODELS["API"]
    
    # 기존 결과 로드 (있는 경우)
    save_path = RESULTS_DIR / "llm_benchmark_responses.csv"
    if save_path.exists():
        try:
            existing_df = pd.read_csv(save_path)
            existing_models = existing_df['model'].unique().tolist()
            print(f"📂 기존 결과 발견: {len(existing_df)}개 응답 ({', '.join(existing_models)})")
            all_results = existing_df.to_dict('records')
        except:
            print("⚠️ 기존 결과 파일 읽기 실패, 새로 시작")
    
    for model_name in all_models:
        # 이미 실행된 모델은 건너뛰기
        if any(model_name in str(r.get('model', '')) for r in all_results):
            print(f"\n⏭️  [{model_name}] 이미 실행됨, 건너뜀")
            continue
            
        print(f"\n🚀 벤치마크 시작: {model_name}")
        try:
            if "gemini" in model_name.lower():
                res = generate_api(model_name, test_data)
            else:
                res = generate_local(model_name, test_data)
            
            if res:
                all_results.extend(res)
                # 각 모델 실행 후 즉시 저장 (크래시 방지)
                df = pd.DataFrame(all_results)
                df.to_csv(save_path, index=False)
                print(f"   💾 중간 저장 완료 (총 {len(df)}개 응답)")
        except Exception as e:
            print(f"   ❌ 모델 실행 중 에러: {e}")
            # 에러가 나도 지금까지 결과는 저장
            if all_results:
                df = pd.DataFrame(all_results)
                df.to_csv(save_path, index=False)
                print(f"   💾 에러 발생 전까지 결과 저장 (총 {len(df)}개 응답)")
            continue
            
    # 최종 저장
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(save_path, index=False)
        print(f"\n💾 벤치마크 완료! 결과 저장됨: {save_path}")
        print(f"   총 {len(df)}개 응답, 모델: {df['model'].unique()}")

if __name__ == "__main__":
    main()
