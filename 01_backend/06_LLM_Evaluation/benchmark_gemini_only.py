#!/usr/bin/env python3
"""
Gemini API만 실행하는 벤치마크 스크립트
"""

import json
import time
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import google.generativeai as genai
from dotenv import load_dotenv
import os

BASE_DIR = Path("/home/pencilfoxs/00_new/History_Docent")
BENCHMARK_DATA = BASE_DIR / "03_Embedding/data/validation_set_20.json"
RESULTS_DIR = BASE_DIR / "06_LLM_Evaluation/results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_SIZE = 50

def load_env():
    load_dotenv("/home/pencilfoxs/00_new/.env2")
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def get_prompt(query, context):
    return f"""당신은 한국사 전문가입니다. 아래 [참고 문서]를 바탕으로 [질문]에 대해 정확하고 상세하게 답변해주세요.
문서에 없는 내용은 지어내지 말고, 정보가 부족하면 부족하다고 말해주세요.

[참고 문서]
{context}

[질문]
{query}

[답변]
"""

def load_existing_responses():
    """기존에 생성된 RAG Context가 있는지 확인"""
    existing_file = RESULTS_DIR / "llm_benchmark_responses.csv"
    if existing_file.exists():
        df = pd.read_csv(existing_file)
        # 이미 RAG context가 있는 데이터 찾기
        if 'rag_context' in df.columns or any('rag_context' in str(v) for v in df.values):
            print("✅ 기존 RAG Context 발견")
            return df
    return None

def get_rag_context_from_existing(query_id, existing_df):
    """기존 결과에서 같은 query_id의 rag_context 찾기"""
    if existing_df is not None:
        # query_id로 매칭 시도
        matches = existing_df[existing_df['query_id'] == query_id]
        if len(matches) > 0:
            # 같은 모델의 rag_context가 있다면 사용
            return matches.iloc[0].get('rag_context', None)
    return None

def generate_gemini(model_name, dataset, existing_df=None):
    print(f"🌐 Gemini API 호출 중: {model_name}")
    results = []
    
    # 모델명 시도 (2.5 -> 2.0 -> 1.5 순서)
    model_names_to_try = ["gemini-2.0-flash-exp", "gemini-1.5-flash"]
    if "2.5" in model_name.lower():
        model_names_to_try.insert(0, "gemini-2.5-flash")
    
    model = None
    actual_model_name = None
    
    for try_name in model_names_to_try:
        try:
            model = genai.GenerativeModel(try_name)
            # 테스트 호출
            model.generate_content("test")
            actual_model_name = try_name
            print(f"✅ {try_name} 사용 가능")
            break
        except Exception as e:
            print(f"⚠️ {try_name} 실패: {e}")
            continue
    
    if model is None:
        print("❌ 모든 Gemini 모델 시도 실패")
        return []

    for item in tqdm(dataset, desc=f"   Generating ({actual_model_name})"):
        # RAG Context 가져오기 (기존 결과에서 또는 새로 생성)
        rag_context = item.get('rag_context', '')
        if not rag_context and existing_df is not None:
            rag_context = get_rag_context_from_existing(item['id'], existing_df)
        
        if not rag_context:
            rag_context = "관련 문서를 찾을 수 없습니다."
        
        prompt = get_prompt(item['query'], rag_context)
        
        start_time = time.time()
        try:
            response = model.generate_content(prompt)
            text = response.text.strip()
        except Exception as e:
            text = f"Error: {e}"
            print(f"   ⚠️ 에러 발생: {e}")
        
        end_time = time.time()
        
        results.append({
            "model": actual_model_name,
            "query_id": item['id'],
            "query": item['query'],
            "response": text,
            "latency": end_time - start_time,
            "type": item['type'],
            "rag_context": rag_context[:200] + "..." if len(rag_context) > 200 else rag_context
        })
        time.sleep(0.5)  # Rate limit 방지
        
    return results

def main():
    load_env()
    
    print("📂 데이터셋 준비 중...")
    with open(BENCHMARK_DATA, 'r') as f: full_data = json.load(f)
    
    # 기존 결과 확인
    existing_df = load_existing_responses()
    
    # 데이터셋 준비 (기존에 RAG context가 있다면 사용)
    test_data = []
    counts = {"keyword": 0, "context": 0, "abstract": 0}
    target = SAMPLE_SIZE // 3
    
    for item in full_data:
        q_type = item['type']
        if counts[q_type] < target + (1 if q_type == 'abstract' and SAMPLE_SIZE % 3 != 0 else 0):
            item['id'] = f"bench_{len(test_data)}"
            
            # 기존 결과에서 rag_context 찾기
            if existing_df is not None:
                existing_context = get_rag_context_from_existing(item['id'], existing_df)
                if existing_context:
                    item['rag_context'] = existing_context
            
            test_data.append(item)
            counts[q_type] += 1
        if len(test_data) >= SAMPLE_SIZE:
            break
    
    # RAG Context가 없는 경우, 간단히 gold_text 사용 (빠른 테스트용)
    for item in test_data:
        if 'rag_context' not in item or not item['rag_context']:
            item['rag_context'] = item.get('gold_text', '관련 문서를 찾을 수 없습니다.')
    
    print(f"✅ {len(test_data)}개 샘플 준비 완료")
    
    # Gemini 실행
    results = generate_gemini("gemini-2.5-flash", test_data, existing_df)
    
    if results:
        # 기존 결과와 병합
        if existing_df is not None:
            df_new = pd.DataFrame(results)
            df_combined = pd.concat([existing_df, df_new], ignore_index=True)
            df_combined.to_csv(RESULTS_DIR / "llm_benchmark_responses.csv", index=False)
            print(f"💾 기존 결과에 추가 저장 완료 (총 {len(df_combined)}개)")
        else:
            df = pd.DataFrame(results)
            df.to_csv(RESULTS_DIR / "llm_benchmark_responses.csv", index=False)
            print(f"💾 결과 저장 완료: {len(df)}개")
    else:
        print("❌ 결과가 없습니다.")

if __name__ == "__main__":
    main()

