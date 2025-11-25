#!/usr/bin/env python3
"""
균형잡힌 QA 데이터셋으로 임베딩 모델 벤치마크 평가
- 새로운 데이터셋(11,140개)으로 7개 임베딩 모델 재평가
- 질문 유형별(Keyword, Context, Abstract) 성능 분석 포함
"""

import time
import json
import torch
import numpy as np
import pandas as pd
import requests
import os
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

# -----------------------------------------------------------------------------
# 설정 (Configuration)
# -----------------------------------------------------------------------------
DATA_DIR = Path("/home/pencilfoxs/00_new/History_Docent/03_Embedding/data")
# ✅ Validation Set 사용 (과적합 방지 - 모델 선정 단계)
# ⚠️ Test Set은 최종 평가에만 사용!
BENCHMARK_FILE = DATA_DIR / "validation_set_20.json"
RESULTS_DIR = Path("/home/pencilfoxs/00_new/History_Docent/03_Embedding/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# API 키 로드
ENV_FILE = Path("/home/pencilfoxs/00_new/.env2")
if ENV_FILE.exists():
    with open(ENV_FILE, 'r') as f:
        for line in f:
            if line.startswith("GOOGLE_API_KEY="):
                os.environ["GOOGLE_API_KEY"] = line.split("=", 1)[1].strip()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# ✅ 평가할 모델 리스트 (7개 후보)
MODELS = {
    # 1. Multilingual 강자
    "BGE-m3": "BAAI/bge-m3",
    
    # 2. 최신 SOTA (Instruction 기반)
    "Jina-v3": "jinaai/jina-embeddings-v3",
    
    # 3. 성능 좋은 대형 모델
    "GTE-large": "Alibaba-NLP/gte-large-en-v1.5",
    
    # 4. 꾸준히 성능 좋은 모델
    "E5-large": "intfloat/multilingual-e5-large",
    
    # 5. 한국어 특화 (Baseline)
    "Ko-SBERT": "jhgan/ko-sbert-nli", 
    
    # 6. Google Open Source (Gemma 기반)
    "EmbeddingGemma": "google/embedding-gemma-2b-en", 
    
    # 7. Google API (최신)
    "Gemini-API": "models/text-embedding-004" 
}

# -----------------------------------------------------------------------------
# 평가 함수 (Evaluation Functions)
# -----------------------------------------------------------------------------

def load_benchmark_data():
    """벤치마크 데이터 로드"""
    if not BENCHMARK_FILE.exists():
        raise FileNotFoundError(f"데이터셋을 찾을 수 없습니다: {BENCHMARK_FILE}")
    with open(BENCHMARK_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_gemini_embeddings(texts, model_name="models/text-embedding-004"):
    """Gemini REST API를 사용한 임베딩 생성"""
    url = f"https://generativelanguage.googleapis.com/v1beta/{model_name}:batchEmbedContents?key={GOOGLE_API_KEY}"
    headers = {"Content-Type": "application/json"}
    
    all_embeddings = []
    batch_size = 50  # API 한도 고려
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        payload = {
            "requests": [{"model": model_name, "content": {"parts": [{"text": text}]}} for text in batch_texts]
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            if response.status_code != 200:
                print(f"Error API: {response.text}")
                all_embeddings.extend([np.zeros(768) for _ in batch_texts])
                continue
                
            result = response.json()
            embeddings = [e['values'] for e in result['embeddings']]
            all_embeddings.extend(embeddings)
            time.sleep(0.5)  # Rate limit 방지
        except Exception as e:
            print(f"Gemini API Error: {e}")
            all_embeddings.extend([np.zeros(768) for _ in batch_texts])
            
    return np.array(all_embeddings)

def get_embeddings(model_name, model_path, texts, device="cuda" if torch.cuda.is_available() else "cpu"):
    """모델별 임베딩 생성 함수"""
    print(f"   Creating embeddings for {len(texts)} texts with {model_name}...")
    start_time = time.time()
    
    # 1. Google Gemini API
    if "Gemini-API" in model_name:
        embeddings = get_gemini_embeddings(texts, model_path)
    
    # 2. Jina (Trust Remote Code 필요)
    elif "Jina" in model_name:
        model = SentenceTransformer(model_path, trust_remote_code=True, device=device)
        embeddings = model.encode(texts, normalize_embeddings=True, batch_size=8)
        
    # 3. GTE (Trust Remote Code 필요)
    elif "GTE" in model_name:
        model = SentenceTransformer(model_path, trust_remote_code=True, device=device)
        embeddings = model.encode(texts, normalize_embeddings=True)
    
    # 4. E5 (Prefix 필요)
    elif "E5" in model_name:
        model = SentenceTransformer(model_path, device=device)
        embeddings = model.encode(texts, normalize_embeddings=True)
    
    # 5. Embedding Gemma
    elif "Gemma" in model_name:
        try:
            model = SentenceTransformer(model_path, trust_remote_code=True, device=device)
            embeddings = model.encode(texts, normalize_embeddings=True, batch_size=4)
        except Exception as e:
            print(f"Gemma 로드 실패: {e}")
            return np.zeros((len(texts), 768)), 0
    
    # 6. 일반 모델 (BGE, Ko-SBERT 등)
    else:
        model = SentenceTransformer(model_path, device=device)
        embeddings = model.encode(texts, normalize_embeddings=True)
        
    elapsed = time.time() - start_time
    speed = elapsed / len(texts) * 1000  # ms per text
    return np.array(embeddings), speed

def evaluate_model(model_name, model_path, dataset, device="cuda" if torch.cuda.is_available() else "cpu"):
    """모델 평가 함수"""
    queries = [item['query'] for item in dataset]
    golds = [item['gold_text'] for item in dataset]
    types = [item.get('type', 'unknown') for item in dataset]
    
    # E5 모델일 경우 Prefix 추가 처리
    if "E5" in model_name:
        q_texts = [f"query: {q}" for q in queries]
        c_texts = [f"passage: {g}" for g in golds]
    else:
        q_texts = queries
        c_texts = golds
    
    # 1. 임베딩 생성
    print(f"   📝 질문 임베딩 생성 중...")
    q_embs, q_speed = get_embeddings(model_name, model_path, q_texts, device)
    print(f"   📝 문서 임베딩 생성 중...")
    c_embs, c_speed = get_embeddings(model_name, model_path, c_texts, device)
    
    # 2. 유사도 계산 (질문-문서 매트릭스)
    print(f"   🔍 유사도 계산 중...")
    similarities = cosine_similarity(q_embs, c_embs)
    
    # 3. 지표 계산 (전체)
    mrr_sum = 0
    hits_1 = 0
    hits_3 = 0
    hits_5 = 0
    
    # 질문 유형별 지표
    type_metrics = defaultdict(lambda: {'mrr_sum': 0, 'hits_1': 0, 'hits_3': 0, 'hits_5': 0, 'count': 0})
    
    n = len(queries)
    for i in range(n):
        target_idx = i  # i번째 질문의 정답은 i번째 문서
        
        # 유사도 내림차순 정렬
        sorted_indices = np.argsort(similarities[i])[::-1]
        
        # 순위 (1-based)
        rank = np.where(sorted_indices == target_idx)[0][0] + 1
        
        mrr_sum += 1.0 / rank
        if rank <= 1: hits_1 += 1
        if rank <= 3: hits_3 += 1
        if rank <= 5: hits_5 += 1
        
        # 질문 유형별 집계
        q_type = types[i]
        type_metrics[q_type]['mrr_sum'] += 1.0 / rank
        type_metrics[q_type]['count'] += 1
        if rank <= 1: type_metrics[q_type]['hits_1'] += 1
        if rank <= 3: type_metrics[q_type]['hits_3'] += 1
        if rank <= 5: type_metrics[q_type]['hits_5'] += 1
    
    # 전체 메트릭
    metrics = {
        "Model": model_name,
        "MRR": round(mrr_sum / n, 3),
        "Recall@1": round(hits_1 / n, 3),
        "Recall@3": round(hits_3 / n, 3),
        "Recall@5": round(hits_5 / n, 3),
        "Latency(ms)": round((q_speed + c_speed) / 2, 1)
    }
    
    # 질문 유형별 메트릭 추가
    for q_type, type_data in type_metrics.items():
        count = type_data['count']
        if count > 0:
            metrics[f"{q_type}_MRR"] = round(type_data['mrr_sum'] / count, 3)
            metrics[f"{q_type}_R@1"] = round(type_data['hits_1'] / count, 3)
            metrics[f"{q_type}_R@3"] = round(type_data['hits_3'] / count, 3)
            metrics[f"{q_type}_R@5"] = round(type_data['hits_5'] / count, 3)
    
    return metrics

# -----------------------------------------------------------------------------
# 메인 실행
# -----------------------------------------------------------------------------
def main():
    print("🚀 [Korean History Docent] 임베딩 모델 벤치마크 시작 (Validation Set)")
    print(f"📂 데이터셋: {BENCHMARK_FILE}")
    print(f"📊 평가 모델 수: {len(MODELS)}개")
    print("⚠️  Validation Set 사용: 모델 선정 단계 (과적합 방지)")
    print("=" * 80)
    
    try:
        dataset = load_benchmark_data()
    except Exception as e:
        print(f"❌ 데이터셋 로드 실패: {e}")
        return
    
    print(f"📊 평가 데이터 수: {len(dataset)}개")
    
    # 질문 유형별 통계
    type_counts = defaultdict(int)
    for item in dataset:
        type_counts[item.get('type', 'unknown')] += 1
    print("📊 질문 유형별 분포:")
    for q_type, count in sorted(type_counts.items()):
        print(f"   - {q_type}: {count}개 ({count/len(dataset)*100:.1f}%)")
    print("=" * 80)
    
    results = []
    # GPU 0번에 할당
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(0)  # GPU 0번 명시적 설정
        print(f"🖥️  사용 디바이스: {device} (GPU 0번)")
        print(f"   GPU 이름: {torch.cuda.get_device_name(0)}")
        print(f"   GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print(f"🖥️  사용 디바이스: {device}")
    print("")
    
    for name, path in tqdm(MODELS.items(), desc="모델 평가 진행 중"):
        print(f"\n{'='*80}")
        print(f"🔍 평가 중: {name}")
        print(f"{'='*80}")
        try:
            metrics = evaluate_model(name, path, dataset, device)
            results.append(metrics)
            print(f"\n   ✅ {name} 평가 완료:")
            print(f"      MRR: {metrics['MRR']}, Recall@1: {metrics['Recall@1']}")
            print(f"      Latency: {metrics['Latency(ms)']}ms")
        except Exception as e:
            print(f"\n   ❌ {name} 평가 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
    
    # 결과 출력
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values(by="MRR", ascending=False)
        
        print("\n" + "=" * 80)
        print("🏆 [최종 벤치마크 결과] 🏆")
        print("=" * 80)
        print(df.to_markdown(index=False))
        
        # 질문 유형별 결과도 출력
        print("\n" + "=" * 80)
        print("📊 [질문 유형별 성능 분석]")
        print("=" * 80)
        type_columns = [col for col in df.columns if any(t in col for t in ['keyword', 'context', 'abstract'])]
        if type_columns:
            type_df = df[['Model'] + type_columns]
            print(type_df.to_markdown(index=False))
        
        # 결과 저장
        csv_path = RESULTS_DIR / "benchmark_results_validation_set.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"\n📁 결과 저장 완료: {csv_path}")
        
        # JSON으로도 저장 (상세 정보 포함)
        json_path = RESULTS_DIR / "benchmark_results_validation_set.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"📁 상세 결과 저장 완료: {json_path}")
    else:
        print("\n❌ 생성된 결과가 없습니다.")

if __name__ == "__main__":
    main()


