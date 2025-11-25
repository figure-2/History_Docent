#!/usr/bin/env python3
"""
RAGAS 평가 결과 분석 및 Data Leakage 검증 스크립트
- 질문 유형별(키워드/문맥/추상) RAGAS 점수 분석
- Answer Relevancy 메트릭 디버깅
"""

import pandas as pd
import os
import time
import traceback
from pathlib import Path
from dotenv import load_dotenv
from ragas import evaluate
from ragas.metrics import answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from datasets import Dataset

# HuggingFaceEmbeddings 래퍼 (embed_query 메서드 추가)
class HuggingFaceEmbeddingsWrapper:
    """HuggingFaceEmbeddings를 RAGAS가 기대하는 인터페이스로 래핑"""
    def __init__(self, model_name):
        self.base_embeddings = HuggingFaceEmbeddings(model=model_name)
    
    def embed_query(self, text: str):
        """embed_text를 embed_query로 매핑"""
        return self.base_embeddings.embed_text(text)
    
    def embed_documents(self, texts: list):
        """여러 텍스트 임베딩"""
        return [self.base_embeddings.embed_text(text) for text in texts]

# ============================================================================
# 1. 질문 유형별 RAGAS 점수 분석 (Data Leakage 검증)
# ============================================================================
print("=" * 70)
print("📊 1. 질문 유형별 RAGAS 점수 분석 (Data Leakage 검증)")
print("=" * 70)

base_dir = Path("/home/pencilfoxs/00_new/History_Docent/06_LLM_Evaluation/results")
ragas_file = base_dir / "ragas_evaluation_results.csv"
meta_file = base_dir / "llm_selected_model_full_test.csv"

if not ragas_file.exists():
    print(f"❌ Error: {ragas_file} 파일을 찾을 수 없습니다.")
    exit(1)

if not meta_file.exists():
    print(f"❌ Error: {meta_file} 파일을 찾을 수 없습니다.")
    exit(1)

# 데이터 로드
print(f"\n📂 데이터 로드 중...")
ragas_df = pd.read_csv(ragas_file)
meta_df = pd.read_csv(meta_file)

print(f"  - RAGAS 결과: {len(ragas_df)}행")
print(f"  - 메타데이터: {len(meta_df)}행")

# 데이터 병합
if 'query_id' in ragas_df.columns and 'query_id' in meta_df.columns:
    merged_df = pd.merge(
        ragas_df[['query_id', 'context_recall', 'context_precision', 'faithfulness', 'answer_relevancy']],
        meta_df[['query_id', 'type', 'chunk_id']],
        on='query_id',
        how='left'
    )
    
    print(f"  - 병합 후: {len(merged_df)}행")
    print(f"\n  📊 질문 유형별 분포:")
    type_counts = merged_df['type'].value_counts()
    print(type_counts.to_string())
    
    # 유형별 통계 계산
    metrics = ['context_recall', 'context_precision', 'faithfulness']
    
    print("\n" + "-" * 70)
    print("📈 유형별 평균 점수 비교 (Data Leakage 검증)")
    print("-" * 70)
    
    results_summary = []
    
    for metric in metrics:
        if metric in merged_df.columns:
            print(f"\n[{metric.upper()}]")
            print("-" * 70)
            
            # 전체 통계
            valid_all = merged_df[merged_df[metric].notna()]
            if len(valid_all) > 0:
                overall_mean = valid_all[metric].mean()
                overall_std = valid_all[metric].std()
                overall_count = len(valid_all)
                print(f"전체 평균: {overall_mean:.4f} (std: {overall_std:.4f}, n={overall_count})")
            
            # 유형별 통계
            valid_df = merged_df[merged_df[metric].notna()]
            if len(valid_df) > 0:
                type_stats = valid_df.groupby('type')[metric].agg(['count', 'mean', 'std', 'min', 'max'])
                print(type_stats.to_string())
                
                # 유형별 평균 점수 추출
                for q_type in ['keyword', 'context', 'abstract']:
                    if q_type in type_stats.index:
                        mean_score = type_stats.loc[q_type, 'mean']
                        count = int(type_stats.loc[q_type, 'count'])
                        results_summary.append({
                            'metric': metric,
                            'type': q_type,
                            'mean': mean_score,
                            'count': count
                        })
    
    # 유형별 점수 비교 요약 (핵심 검증)
    print("\n" + "=" * 70)
    print("🔍 핵심 검증: Abstract vs Keyword 점수 차이 (Data Leakage 검증)")
    print("=" * 70)
    
    summary_df = pd.DataFrame(results_summary)
    if len(summary_df) > 0:
        # 피벗 테이블 생성
        pivot_df = summary_df.pivot(index='metric', columns='type', values='mean')
        print("\n유형별 평균 점수:")
        print(pivot_df.to_string())
        
        # Abstract vs Keyword 비교 (핵심 검증)
        print("\n" + "-" * 70)
        print("💡 Data Leakage 검증 결과:")
        print("-" * 70)
        
        for metric in metrics:
            if metric in pivot_df.index:
                keyword_mean = pivot_df.loc[metric, 'keyword'] if 'keyword' in pivot_df.columns else None
                abstract_mean = pivot_df.loc[metric, 'abstract'] if 'abstract' in pivot_df.columns else None
                
                if keyword_mean is not None and abstract_mean is not None:
                    diff = keyword_mean - abstract_mean
                    diff_pct = (diff / keyword_mean) * 100 if keyword_mean > 0 else 0
                    
                    print(f"\n[{metric}]")
                    print(f"  Keyword 평균:  {keyword_mean:.4f}")
                    print(f"  Abstract 평균: {abstract_mean:.4f}")
                    print(f"  차이: {diff:.4f} ({diff_pct:+.1f}%)")
                    
                    if diff > 0.15:  # 15% 이상 차이
                        print(f"  ⚠️  경고: Abstract가 Keyword보다 현저히 낮습니다!")
                        print(f"      → Data Leakage 가능성 높음 (검색 DB에 정답이 직접 포함됨)")
                        print(f"      → 새로운 질문(Unseen Query)에 약할 가능성")
                    elif diff > 0.05:  # 5-15% 차이
                        print(f"  ⚠️  주의: Abstract가 Keyword보다 약간 낮습니다.")
                        print(f"      → 일부 Data Leakage 가능성, Unseen Query 테스트 권장")
                    else:  # 차이 < 5%
                        print(f"  ✅ 양호: Abstract와 Keyword 점수가 비슷합니다.")
                        print(f"      → 시스템이 새로운 질문에도 잘 작동할 가능성")
    
    # Data Leakage 해석
    print("\n" + "=" * 70)
    print("📋 Data Leakage 검증 해석 가이드")
    print("=" * 70)
    print("""
    💡 해석 방법:
    
    1. Keyword (30%): 고유명사/사실 기반 질문
       - 검색 DB에 정확히 매칭될 가능성 높음
       - 예: "손기정의 제자는 누구인가?"
    
    2. Abstract (30%): 핵심 키워드를 생략한 추상적 질문
       - 검색 난이도 높음 (Unseen Query와 유사)
       - 예: "그 올림픽에서 1등 한 유명한 사람이 제일 아끼던 제자는..."
    
    3. Context (40%): 문맥/인과관계 질문 (중간 난이도)
    
    🔍 검증 기준:
    - Abstract 점수가 Keyword보다 현저히 낮다면 → Data Leakage 존재
    - Abstract 점수가 Keyword와 비슷하다면 → 시스템이 잘 학습됨
    
    📌 면접관 답변 전략:
    "네, 맞습니다. Synthetic Dataset의 한계입니다. 하지만 Abstract 유형은 
    키워드를 제거하여 'Unseen Query'와 유사한 환경을 시뮬레이션했습니다. 
    실제 Unseen Query 테스트를 위해 외부 질문셋(수능 기출 등)을 추가할 계획입니다."
    """)
    
    # chunk_id 기반 분석 (같은 청크에서 생성된 질문들의 점수)
    print("\n" + "-" * 70)
    print("🔍 청크별 질문 생성 분석")
    print("-" * 70)
    
    chunk_question_count = merged_df.groupby('chunk_id').size()
    print(f"  - 총 청크 수: {chunk_question_count.shape[0]}")
    print(f"  - 청크당 평균 질문 수: {chunk_question_count.mean():.2f}")
    print(f"  - 최대 질문 수: {chunk_question_count.max()}")
    print(f"  - 최소 질문 수: {chunk_question_count.min()}")
    
    # 같은 청크에서 생성된 질문들의 점수 분포
    multi_question_chunks = chunk_question_count[chunk_question_count > 1].index
    if len(multi_question_chunks) > 0:
        print(f"\n  - 여러 질문이 생성된 청크 수: {len(multi_question_chunks)}")
        for metric in metrics:
            if metric in merged_df.columns:
                multi_chunk_scores = merged_df[merged_df['chunk_id'].isin(multi_question_chunks) & merged_df[metric].notna()][metric]
                single_chunk_scores = merged_df[~merged_df['chunk_id'].isin(multi_question_chunks) & merged_df[metric].notna()][metric]
                if len(multi_chunk_scores) > 0 and len(single_chunk_scores) > 0:
                    print(f"\n  [{metric}]")
                    print(f"    - 여러 질문 청크 평균: {multi_chunk_scores.mean():.4f} (n={len(multi_chunk_scores)})")
                    print(f"    - 단일 질문 청크 평균: {single_chunk_scores.mean():.4f} (n={len(single_chunk_scores)})")
    
else:
    print("❌ Error: query_id 컬럼이 없습니다.")

# ============================================================================
# 2. Answer Relevancy 디버깅 (Minimal Test)
# ============================================================================
print("\n" + "=" * 70)
print("🐛 2. Answer Relevancy 디버깅 (Minimal Test)")
print("=" * 70)

# 환경 변수 로드
env_path = Path("/home/pencilfoxs/00_new/.env2")
load_dotenv(env_path)
google_api_key = os.getenv("GOOGLE_AI_STUDIO_API_KEY")
if not google_api_key:
    google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    print("❌ Error: GOOGLE_AI_STUDIO_API_KEY 또는 GOOGLE_API_KEY를 찾을 수 없습니다.")
    exit(1)

os.environ["GOOGLE_API_KEY"] = google_api_key

try:
    # 모델 설정 (evaluate_ragas_full.py와 동일)
    print("\n🔧 모델 설정 중 (evaluate_ragas_full.py와 동일한 설정)...")
    print("  - LLM: Gemini-2.0-flash (temperature=0, top_p=1)")
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0, top_p=1)
    ragas_llm = LangchainLLMWrapper(llm)
    
    print("  - Embeddings: BAAI/bge-m3 (래퍼 적용)")
    # RAGAS 호환성을 위한 래퍼 사용
    base_embeddings = HuggingFaceEmbeddings(model="BAAI/bge-m3")
    ragas_embeddings = HuggingFaceEmbeddingsWrapper("BAAI/bge-m3")
    
    # AnswerRelevancy 설정 (명시적으로 embeddings 설정)
    print("  - AnswerRelevancy 메트릭 설정...")
    answer_relevancy.embeddings = ragas_embeddings
    answer_relevancy.llm = ragas_llm
    
    print("  ✅ 설정 완료")
    
    # 샘플 데이터 (한국어)
    print("\n📝 샘플 데이터 준비...")
    sample_data = {
        'question': ["이순신 장군은 어느 시대 사람인가요?"],
        'answer': ["이순신 장군은 조선 시대의 장군입니다."],
        'contexts': [["이순신은 조선 중기의 무신이다."]],
        'ground_truth': ["이순신 장군은 조선 시대 사람입니다."]
    }
    dataset = Dataset.from_dict(sample_data)
    
    print("  - 질문: 이순신 장군은 어느 시대 사람인가요?")
    print("  - 답변: 이순신 장군은 조선 시대의 장군입니다.")
    
    print("\n⏳ 평가 실행 중... (약 10-30초 소요)")
    print("  💡 Answer Relevancy 작동 원리:")
    print("     1. LLM이 답변(Answer)을 보고 가상 질문(Generated Question)을 생성")
    print("     2. 원래 질문(User Question)과 가상 질문 사이의 코사인 유사도 계산")
    print("     3. 이때 임베딩 모델(BAAI/bge-m3)이 필요")
    
    start_time = time.time()
    
    result = evaluate(
        dataset=dataset,
        metrics=[answer_relevancy],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        raise_exceptions=True
    )
    
    elapsed = time.time() - start_time
    print(f"\n✅ 성공! (소요 시간: {elapsed:.2f}초)")
    
    result_df = result.to_pandas()
    print("\n📊 결과:")
    print(result_df.to_string())
    
    relevancy_score = result_df['answer_relevancy'].iloc[0] if 'answer_relevancy' in result_df.columns else None
    if relevancy_score is not None and not pd.isna(relevancy_score):
        print(f"\n✅ Answer Relevancy 점수: {relevancy_score:.4f}")
        print("\n💡 해결책:")
        print("  - 임베딩 모델 설정이 정상적으로 작동합니다.")
        print("  - 전체 평가에서 Answer Relevancy가 NaN인 이유는:")
        print("    1. evaluate_ragas_full.py에서 metric.embeddings를 명시적으로 설정하지 않음")
        print("    2. 대량 데이터 처리 중 타임아웃 또는 오류")
        print("    3. 한국어 텍스트 처리 문제")
        print("\n  📌 권장 수정사항:")
        print("    evaluate_ragas_full.py의 line 81-83에 다음 추가:")
        print("    for metric in metrics:")
        print("        metric.llm = ragas_llm")
        print("        metric.embeddings = ragas_embeddings  # ← 추가 필요")
    else:
        print("\n❌ Answer Relevancy 점수가 여전히 NaN입니다.")
        print("  - 추가 디버깅이 필요합니다.")
        print("  - 임베딩 모델 로딩 확인 필요")
        
except Exception as e:
    print("\n❌ 오류 발생:")
    print(str(e))
    traceback.print_exc()
    print("\n💡 디버깅 정보:")
    print("  - API 키가 올바르게 설정되었는지 확인")
    print("  - 네트워크 연결 상태 확인")
    print("  - RAGAS 라이브러리 버전 확인")

print("\n" + "=" * 70)
print("✅ 분석 완료")
print("=" * 70)

