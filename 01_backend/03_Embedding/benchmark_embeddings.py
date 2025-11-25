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

# -----------------------------------------------------------------------------
# 설정 (Configuration)
# -----------------------------------------------------------------------------
DATA_DIR = Path("/home/pencilfoxs/00_new/History_Docent/03_Embedding/data")
# ✅ 최종 고품질 데이터셋 2000개 사용
BENCHMARK_FILE = DATA_DIR / "korean_history_benchmark_2000.json"
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

# ✅ 평가할 모델 리스트 (7 Candidates)
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
    # 주의: Transformers 최신 버전 필요, 메모리 부족시 제외 가능
    "EmbeddingGemma": "google/embedding-gemma-2b-en", 
    
    # 7. Google API (최신)
    "Gemini-API": "models/text-embedding-004" 
}

# -----------------------------------------------------------------------------
# 평가 함수 (Evaluation Functions)
# -----------------------------------------------------------------------------

def load_benchmark_data():
    if not BENCHMARK_FILE.exists():
        raise FileNotFoundError(f"데이터셋을 찾을 수 없습니다: {BENCHMARK_FILE}")
    with open(BENCHMARK_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_gemini_embeddings(texts, model_name="models/text-embedding-004"):
    """Gemini REST API를 사용한 임베딩 생성"""
    url = f"https://generativelanguage.googleapis.com/v1beta/{model_name}:batchEmbedContents?key={GOOGLE_API_KEY}"
    headers = {"Content-Type": "application/json"}
    
    all_embeddings = []
    batch_size = 50 # API 한도 고려
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        payload = {
            "requests": [{"model": model_name, "content": {"parts": [{"text": text}]}} for text in batch_texts]
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            if response.status_code != 200:
                print(f"Error API: {response.text}")
                # 에러 시 0으로 채움 (중단 방지)
                all_embeddings.extend([np.zeros(768) for _ in batch_texts])
                continue
                
            result = response.json()
            embeddings = [e['values'] for e in result['embeddings']]
            all_embeddings.extend(embeddings)
            time.sleep(0.5) # Rate limit 방지
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
        # Jina v3는 task 지정 가능 (retrieval.query / retrieval.passage)
        # 여기서는 쿼리와 문서 구분 없이 단순히 encode (또는 분기 처리 가능)
        embeddings = model.encode(texts, normalize_embeddings=True, batch_size=8)
        
    # 3. GTE (Trust Remote Code 필요)
    elif "GTE" in model_name:
        model = SentenceTransformer(model_path, trust_remote_code=True, device=device)
        embeddings = model.encode(texts, normalize_embeddings=True)

    # 4. E5 (Prefix 필요)
    elif "E5" in model_name:
        model = SentenceTransformer(model_path, device=device)
        # E5는 query와 passage에 prefix가 붙어야 성능이 좋음.
        # 벤치마크 구조상 texts가 쿼리인지 문서인지 구분하여 넘기면 좋으나,
        # 여기서는 단순화를 위해 일괄 처리하거나, 호출하는 쪽에서 처리해야 함.
        # (이 함수는 범용이므로, 입력된 텍스트 그대로 임베딩)
        embeddings = model.encode(texts, normalize_embeddings=True)

    # 5. Embedding Gemma
    elif "Gemma" in model_name:
        # SentenceTransformer 지원 여부 확인 필요, 미지원시 HF Transformers 사용
        # 여기서는 SentenceTransformer로 시도하되 안되면 예외 처리
        try:
            model = SentenceTransformer(model_path, trust_remote_code=True, device=device)
            embeddings = model.encode(texts, normalize_embeddings=True, batch_size=4) # 메모리 주의
        except Exception as e:
            print(f"Gemma 로드 실패 (Transformers로 시도 필요): {e}")
            return np.zeros((len(texts), 768)), 0

    # 6. 일반 모델 (BGE, Ko-SBERT 등)
    else:
        model = SentenceTransformer(model_path, device=device)
        embeddings = model.encode(texts, normalize_embeddings=True)
        
    elapsed = time.time() - start_time
    speed = elapsed / len(texts) * 1000 # ms per text
    return np.array(embeddings), speed

def evaluate_model(model_name, model_path, dataset):
    queries = [item['query'] for item in dataset]
    golds = [item['gold_text'] for item in dataset]
    
    # E5 모델일 경우 Prefix 추가 처리
    if "E5" in model_name:
        q_texts = [f"query: {q}" for q in queries]
        c_texts = [f"passage: {g}" for g in golds]
    else:
        q_texts = queries
        c_texts = golds
    
    # 1. 임베딩 생성
    q_embs, q_speed = get_embeddings(model_name, model_path, q_texts)
    c_embs, c_speed = get_embeddings(model_name, model_path, c_texts)
    
    # 2. 유사도 계산
    similarities = cosine_similarity(q_embs, c_embs)
    
    # 3. 지표 계산
    mrr_sum = 0
    hits_1 = 0
    hits_3 = 0
    hits_5 = 0
    
    n = len(queries)
    for i in range(n):
        target_idx = i # i번째 질문의 정답은 i번째 문서
        
        # 유사도 내림차순 정렬
        sorted_indices = np.argsort(similarities[i])[::-1]
        
        # 순위 (1-based)
        rank = np.where(sorted_indices == target_idx)[0][0] + 1
        
        mrr_sum += 1.0 / rank
        if rank <= 1: hits_1 += 1
        if rank <= 3: hits_3 += 1
        if rank <= 5: hits_5 += 1
        
    metrics = {
        "Model": model_name,
        "MRR": round(mrr_sum / n, 3),
        "Recall@1": round(hits_1 / n, 3),
        "Recall@3": round(hits_3 / n, 3),
        "Recall@5": round(hits_5 / n, 3),
        "Latency(ms)": round((q_speed + c_speed) / 2, 1)
    }
    
    return metrics

# -----------------------------------------------------------------------------
# 메인 실행
# -----------------------------------------------------------------------------
def main():
    print("🚀 [Korean History Docent] 임베딩 모델 벤치마크 시작")
    print(f"📂 데이터셋: {BENCHMARK_FILE}")
    
    try:
        dataset = load_benchmark_data()
    except Exception as e:
        print(f"❌ 데이터셋 로드 실패: {e}")
        return

    # 테스트용 (너무 많으면 50개만 먼저 해보기 가능)
    # dataset = dataset[:50] 
    print(f"📊 평가 데이터 수: {len(dataset)}개")
    
    results = []
    
    for name, path in tqdm(MODELS.items(), desc="모델 평가 진행 중"):
        try:
            metrics = evaluate_model(name, path, dataset)
            results.append(metrics)
            print(f"\n   ✅ {name}: MRR={metrics['MRR']}, R@1={metrics['Recall@1']} (Lat: {metrics['Latency(ms)']}ms)")
        except Exception as e:
            print(f"\n   ❌ {name} 평가 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
    
    # 결과 출력
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values(by="MRR", ascending=False)
        
        print("\n🏆 [최종 벤치마크 결과] 🏆")
        print(df.to_markdown(index=False))
        
        csv_path = RESULTS_DIR / "benchmark_results_2000.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n📁 결과 저장 완료: {csv_path}")
    else:
        print("\n❌ 생성된 결과가 없습니다.")

if __name__ == "__main__":
    main()