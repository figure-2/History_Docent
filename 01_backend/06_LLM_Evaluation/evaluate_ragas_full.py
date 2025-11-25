#!/usr/bin/env python3
"""
RAGAS 평가 전체 실행 스크립트
- 목적: 전체 Validation Set (2,223개)에 대한 RAGAS 지표 측정
- 지표: Context Recall, Context Precision, Faithfulness, Answer Relevancy
- 특징: 50개씩 배치 처리, 재개 기능, 백그라운드 실행 지원
"""

import os
import pandas as pd
import time
import json
from pathlib import Path
from tqdm import tqdm
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    context_recall,
    context_precision,
    faithfulness,
    answer_relevancy,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from datetime import datetime

# 1. 환경 변수 로드 (API Key)
env_path = Path("/home/pencilfoxs/00_new/.env2")
load_dotenv(env_path)

google_api_key = os.getenv("GOOGLE_AI_STUDIO_API_KEY")
if not google_api_key:
    raise ValueError("GOOGLE_AI_STUDIO_API_KEY not found in .env2 file")

os.environ["GOOGLE_API_KEY"] = google_api_key

# 2. 설정
BASE_DIR = Path("/home/pencilfoxs/00_new/History_Docent")
INPUT_FILE = BASE_DIR / "06_LLM_Evaluation/results/llm_selected_model_full_test_with_contexts.csv"
OUTPUT_FILE = BASE_DIR / "06_LLM_Evaluation/results/ragas_evaluation_results.csv"
LOG_FILE = BASE_DIR / "06_LLM_Evaluation/results/ragas_evaluation.log"
PROGRESS_FILE = BASE_DIR / "06_LLM_Evaluation/results/ragas_evaluation_progress.json"

BATCH_SIZE = 50
MODEL_NAME = "gemini-2.0-flash"  # 심판관 모델 (RPD 무제한)

# 3. 심판관(Judge) 모델 설정
print("=" * 60)
print("RAGAS 평가 시스템 초기화")
print("=" * 60)

print(f"\n🔧 Judge 모델 설정: {MODEL_NAME}")
llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    temperature=0,  # 평가는 일관성 있게
    top_p=1,
)
ragas_llm = LangchainLLMWrapper(llm)

# Embeddings 설정 (로컬 모델 사용, OpenAI API 불필요)
print(f"\n🔧 Embeddings 모델 설정: BAAI/bge-m3")
ragas_embeddings = HuggingFaceEmbeddings(model="BAAI/bge-m3")

# 4. 평가 지표 설정
print(f"\n📊 평가 지표 설정:")
print("   1. Context Recall (검색 재현율)")
print("   2. Context Precision (검색 정밀도)")
print("   3. Faithfulness (신뢰성)")
print("   4. Answer Relevancy (답변 적절성)")

metrics = [
    context_recall,
    context_precision,
    faithfulness,
    answer_relevancy,
]

# LLM 설정 적용
for metric in metrics:
    metric.llm = ragas_llm

print("   ✅ 지표 설정 완료")

# 5. 데이터 로드 및 전처리
print(f"\n📂 데이터 로드: {INPUT_FILE}")

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"입력 파일을 찾을 수 없습니다: {INPUT_FILE}\n"
        f"먼저 prepare_ragas_data.py를 실행하여 contexts를 추가해주세요."
    )

df = pd.read_csv(INPUT_FILE)
print(f"   총 {len(df)}개 질문 로드 완료")

# 필수 컬럼 확인
required_columns = ['query', 'response', 'contexts', 'gold_text']
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    raise ValueError(
        f"필수 컬럼이 없습니다: {missing_columns}\n"
        f"현재 컬럼: {list(df.columns)}"
    )

# contexts가 JSON 문자열인지 확인 및 변환
if df['contexts'].dtype == 'object':
    try:
        # JSON 문자열을 리스트로 변환
        df['contexts'] = df['contexts'].apply(
            lambda x: json.loads(x) if isinstance(x, str) and x.startswith('[') else (x if isinstance(x, list) else [])
        )
    except:
        # 파싱 실패 시 빈 리스트로 처리
        df['contexts'] = df['contexts'].apply(lambda x: [] if not isinstance(x, list) else x)

# RAGAS가 요구하는 컬럼명으로 변경
df_ragas = df.copy()
df_ragas.rename(columns={
    'query': 'question',
    'response': 'answer',
    'gold_text': 'ground_truth'
}, inplace=True)

# contexts를 리스트 형식으로 확실히 변환 (RAGAS는 리스트의 리스트 필요)
df_ragas['contexts'] = df_ragas['contexts'].apply(
    lambda x: x if isinstance(x, list) else ([x] if isinstance(x, str) and x.strip() else [])
)

print("   ✅ 데이터 전처리 완료")

# 6. 재개(Resume) 기능 구현
if OUTPUT_FILE.exists():
    print(f"\n🔄 기존 결과 파일 발견: {OUTPUT_FILE}")
    existing_df = pd.read_csv(OUTPUT_FILE)
    processed_ids = set(existing_df['query_id'].tolist())
    df_to_process = df_ragas[~df_ragas['query_id'].isin(processed_ids)].copy()
    print(f"   이미 처리됨: {len(processed_ids)}개")
    print(f"   남은 작업: {len(df_to_process)}개")
else:
    print(f"\n🆕 새로운 평가 시작")
    df_to_process = df_ragas.copy()
    # 결과 파일 헤더 생성
    result_df_template = pd.DataFrame(columns=[
        'query_id', 'question', 'answer', 
        'context_recall', 'context_precision', 'faithfulness', 'answer_relevancy'
    ])
    result_df_template.to_csv(OUTPUT_FILE, index=False)
    print(f"   결과 파일 생성: {OUTPUT_FILE}")

if len(df_to_process) == 0:
    print("\n✅ 모든 평가가 이미 완료되었습니다!")
    exit(0)

print(f"\n📊 처리 대상: 총 {len(df_ragas)}개 중 {len(df_to_process)}개 남음")

# 7. 배치 단위 평가 실행
print(f"\n🚀 배치 평가 시작 (배치 크기: {BATCH_SIZE})")
print(f"   총 {len(df_to_process)}개 질문, {((len(df_to_process) + BATCH_SIZE - 1) // BATCH_SIZE)}개 배치")

total_batches = (len(df_to_process) + BATCH_SIZE - 1) // BATCH_SIZE
start_time = time.time()

for batch_idx in range(0, len(df_to_process), BATCH_SIZE):
    batch_num = (batch_idx // BATCH_SIZE) + 1
    batch_df = df_to_process.iloc[batch_idx:batch_idx + BATCH_SIZE].copy()
    
    print(f"\n📦 배치 {batch_num}/{total_batches} 처리 중... ({len(batch_df)}개 질문)")
    
    try:
        # RAGAS 데이터셋 변환
        # contexts가 리스트의 리스트여야 함 (각 질문마다 여러 문서)
        ragas_data = {
            'question': batch_df['question'].tolist(),
            'answer': batch_df['answer'].tolist(),
            'contexts': batch_df['contexts'].tolist(),
            'ground_truth': batch_df['ground_truth'].tolist(),
        }
        ragas_dataset = Dataset.from_dict(ragas_data)
        
        # 평가 실행
        print("   🔍 RAGAS 평가 실행 중...")
        results = evaluate(
            dataset=ragas_dataset,
            metrics=metrics,
            llm=ragas_llm,
            embeddings=ragas_embeddings,
            raise_exceptions=False,
        )
        
        # 결과를 DataFrame으로 변환
        results_df = results.to_pandas()
        
        # 원본 정보(query_id)와 결합
        results_df['query_id'] = batch_df['query_id'].values
        
        # RAGAS 결과 컬럼명을 우리 형식으로 변경
        # RAGAS는 user_input, response를 사용하므로 이를 question, answer로 변경
        if 'user_input' in results_df.columns:
            results_df['question'] = results_df['user_input']
        if 'response' in results_df.columns:
            results_df['answer'] = results_df['response']
        
        # 컬럼명 정리
        output_columns = ['query_id', 'question', 'answer']
        metric_columns = ['context_recall', 'context_precision', 'faithfulness', 'answer_relevancy']
        
        for col in metric_columns:
            if col in results_df.columns:
                output_columns.append(col)
        
        # 존재하는 컬럼만 선택
        available_columns = [col for col in output_columns if col in results_df.columns]
        results_df_output = results_df[available_columns].copy()
        
        # 파일에 추가 (Append)
        results_df_output.to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
        
        # 진행 상황 저장
        progress = {
            'batch_num': batch_num,
            'total_batches': total_batches,
            'processed': min(batch_num * BATCH_SIZE, len(df_to_process)),
            'total': len(df_to_process),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        with open(PROGRESS_FILE, 'w') as f:
            json.dump(progress, f, indent=2)
        
        # 평균 점수 출력
        if metric_columns:
            available_metrics = [col for col in metric_columns if col in results_df.columns]
            if available_metrics:
                avg_scores = results_df[available_metrics].mean()
                print(f"   📊 배치 평균 점수:")
                for metric, score in avg_scores.items():
                    print(f"      - {metric}: {score:.4f}")
        
        # 속도 정보
        elapsed = time.time() - start_time
        avg_time_per_batch = elapsed / batch_num
        remaining_batches = total_batches - batch_num
        estimated_remaining = avg_time_per_batch * remaining_batches
        
        print(f"   ⏱️  경과 시간: {elapsed/60:.1f}분, 예상 남은 시간: {estimated_remaining/60:.1f}분")
        
        # 속도 조절 (너무 빠르면 잠시 대기)
        time.sleep(1)
        
    except Exception as e:
        print(f"   ❌ 배치 {batch_num} 처리 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # 에러 로그 기록
        with open(LOG_FILE, 'a') as f:
            f.write(f"\n[ERROR] Batch {batch_num} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{str(e)}\n")
            f.write(traceback.format_exc() + "\n")
        
        # 다음 배치로 넘어감
        continue

# 8. 최종 요약
print("\n" + "=" * 60)
print("✅ 평가 완료!")
print("=" * 60)

if OUTPUT_FILE.exists():
    final_df = pd.read_csv(OUTPUT_FILE)
    print(f"\n📊 최종 결과 요약:")
    print(f"   총 평가 완료: {len(final_df)}개")
    
    metric_columns = ['context_recall', 'context_precision', 'faithfulness', 'answer_relevancy']
    available_metrics = [col for col in metric_columns if col in final_df.columns]
    
    if available_metrics:
        print(f"\n   전체 평균 점수:")
        for metric in available_metrics:
            avg_score = final_df[metric].mean()
            print(f"      - {metric}: {avg_score:.4f}")
    
    print(f"\n💾 결과 파일: {OUTPUT_FILE}")
else:
    print("\n⚠️  결과 파일이 생성되지 않았습니다.")

print("\n" + "=" * 60)
