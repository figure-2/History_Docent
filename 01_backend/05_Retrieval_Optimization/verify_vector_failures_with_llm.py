"""
[Vector 검색 실패 원인 심층 분석 - LLM-as-a-Judge]
- 목적: ID 매칭 실패가 진짜 실패인지, 유사 정답(Semantic Match)인지 LLM으로 판별
- 대상: Vector Only 전략에서 실패한(Recall@1=0) 케이스들
- 방법: Gemini API를 사용하여 검색된 텍스트가 질문에 답할 수 있는지 판단
"""
import json
import time
import requests
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Any
import os
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv("/home/pencilfoxs/00_new/.env2")

# -----------------------------------------------------------------------------
# 설정
# -----------------------------------------------------------------------------
BASE_DIR = Path("/home/pencilfoxs/00_new/History_Docent")
BENCHMARK_DATA = BASE_DIR / "03_Embedding/data/korean_history_benchmark_2000.json"
BENCHMARK_RESULT = BASE_DIR / "05_Retrieval_Optimization/retrieval_benchmark_result.md"
OUTPUT_REPORT = BASE_DIR / "05_Retrieval_Optimization/vector_failure_verification_report.md"

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent"

# 기존 모듈 임포트
import sys
sys.path.append(str(Path(__file__).parent))
from hybrid_retriever import HybridRetriever

# -----------------------------------------------------------------------------
# LLM 판정 함수
# -----------------------------------------------------------------------------
def evaluate_with_llm(query: str, retrieved_text: str, max_retries: int = 3) -> Dict[str, Any]:
    """
    LLM을 사용하여 검색된 텍스트가 질문에 답할 수 있는지 판단
    
    Returns:
        {
            "can_answer": bool,  # 질문에 답할 수 있는가?
            "confidence": str,   # "HIGH", "MEDIUM", "LOW"
            "reasoning": str,    # 판단 근거
            "raw_response": str # LLM 원본 응답
        }
    """
    prompt = f"""당신은 엄격한 평가자입니다. 아래 [제공된 텍스트]가 [질문]에 대한 명확하고 충분한 정답을 포함하고 있는지 판단하세요.

[질문]: {query}

[제공된 텍스트]: {retrieved_text[:2000]}  # 텍스트가 너무 길면 잘라냄

**판단 기준:**
1. 제공된 텍스트만으로 질문에 대한 답을 할 수 있는가?
2. 답이 명확하고 구체적인가? (모호하거나 추측성 답변은 NO)
3. 질문의 핵심 키워드나 개념이 텍스트에 포함되어 있는가?

**응답 형식 (JSON):**
{{
    "can_answer": true/false,
    "confidence": "HIGH"/"MEDIUM"/"LOW",
    "reasoning": "판단 근거를 한 문장으로 설명"
}}

YES 또는 NO만 답하지 말고, 반드시 위 JSON 형식으로만 답하세요."""

    payload = {
        "contents": [{
            "parts": [{
                "text": prompt
            }]
        }],
        "generationConfig": {
            "temperature": 0.1,  # 낮은 온도로 일관성 확보
            "maxOutputTokens": 200
        }
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"{GEMINI_API_URL}?key={GOOGLE_API_KEY}",
                json=payload,
                headers=headers,
                timeout=30
            )
            
            if response.status_code != 200:
                raise Exception(f"API 호출 실패: {response.status_code} - {response.text[:200]}")
            
            result = response.json()
            if 'candidates' in result and len(result['candidates']) > 0:
                text_response = result['candidates'][0]['content']['parts'][0]['text'].strip()
                
                # JSON 추출 (마크다운 코드 블록 제거)
                if text_response.startswith("```"):
                    text_response = text_response.replace("```json", "").replace("```", "").strip()
                
                try:
                    llm_result = json.loads(text_response)
                    return {
                        "can_answer": llm_result.get("can_answer", False),
                        "confidence": llm_result.get("confidence", "LOW"),
                        "reasoning": llm_result.get("reasoning", ""),
                        "raw_response": text_response
                    }
                except json.JSONDecodeError:
                    # JSON 파싱 실패 시 텍스트에서 추출 시도
                    if "can_answer" in text_response.lower() or "true" in text_response.lower():
                        return {
                            "can_answer": True,
                            "confidence": "MEDIUM",
                            "reasoning": "LLM 응답 파싱 실패, 텍스트 기반 추정",
                            "raw_response": text_response
                        }
                    else:
                        return {
                            "can_answer": False,
                            "confidence": "MEDIUM",
                            "reasoning": "LLM 응답 파싱 실패, 텍스트 기반 추정",
                            "raw_response": text_response
                        }
            else:
                raise Exception(f"응답 형식 오류: {result}")
                
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            else:
                return {
                    "can_answer": False,
                    "confidence": "LOW",
                    "reasoning": f"API 호출 실패: {str(e)}",
                    "raw_response": ""
                }
    
    return {
        "can_answer": False,
        "confidence": "LOW",
        "reasoning": "최대 재시도 횟수 초과",
        "raw_response": ""
    }

# -----------------------------------------------------------------------------
# 메인 로직
# -----------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Vector 검색 실패 케이스 LLM 재평가")
    print("=" * 60)
    
    # 1. 벤치마크 결과에서 Vector Only 실패 케이스 추출
    print("\n📂 벤치마크 결과 로드 중...")
    
    # 벤치마크를 다시 실행하여 실패 케이스 수집
    retriever = HybridRetriever()
    retriever.initialize()
    
    with open(BENCHMARK_DATA, 'r', encoding='utf-8') as f:
        all_samples = json.load(f)
    
    import random
    random.seed(42)
    samples = random.sample(all_samples, min(50, len(all_samples)))
    
    print(f"   ✅ 평가 데이터: {len(samples)}개 샘플")
    
    # 2. Vector Only 검색 실행 및 실패 케이스 수집
    print("\n🔍 Vector Only 검색 실행 중...")
    failure_cases = []
    
    for sample in tqdm(samples, desc="   검색 중"):
        query = sample['query']
        gold_id = sample['chunk_id']
        
        results = retriever.search_vector_only(query, top_k=5)
        
        # Recall@1 실패 케이스만 수집
        if results and results[0].chunk_id != gold_id:
            failure_cases.append({
                "query": query,
                "gold_id": gold_id,
                "gold_text": sample.get('gold_text', ''),
                "retrieved_id": results[0].chunk_id,
                "retrieved_text": results[0].text,
                "retrieved_rank": 1
            })
    
    print(f"\n   📊 Vector Only 실패 케이스: {len(failure_cases)}개")
    
    if len(failure_cases) == 0:
        print("   ✅ 실패 케이스가 없습니다. Vector 검색이 완벽합니다!")
        return
    
    # 3. LLM으로 각 실패 케이스 재평가
    print(f"\n🤖 LLM 재평가 시작 (총 {len(failure_cases)}개)...")
    print("   ⚠️  Gemini API 호출로 인해 시간이 소요될 수 있습니다.\n")
    
    verified_results = []
    semantic_matches = 0  # 의미적으로는 정답인 케이스
    
    for i, case in enumerate(tqdm(failure_cases, desc="   LLM 평가"), 1):
        query = case['query']
        retrieved_text = case['retrieved_text']
        
        # LLM 판정
        llm_result = evaluate_with_llm(query, retrieved_text)
        
        verified_results.append({
            **case,
            "llm_can_answer": llm_result["can_answer"],
            "llm_confidence": llm_result["confidence"],
            "llm_reasoning": llm_result["reasoning"],
            "llm_raw_response": llm_result["raw_response"]
        })
        
        if llm_result["can_answer"]:
            semantic_matches += 1
        
        # API Rate Limit 방지
        time.sleep(0.5)
    
    # 4. 보정된 점수 계산
    original_failures = len(failure_cases)
    semantic_successes = semantic_matches
    corrected_failures = original_failures - semantic_successes
    
    original_recall_1 = ((50 - original_failures) / 50) * 100
    corrected_recall_1 = ((50 - corrected_failures) / 50) * 100
    improvement = corrected_recall_1 - original_recall_1
    
    # 5. 리포트 작성
    print("\n" + "="*60)
    print("📊 LLM 재평가 결과")
    print("="*60)
    print(f"\n원본 Recall@1: {original_recall_1:.1f}% (실패: {original_failures}개)")
    print(f"보정 Recall@1: {corrected_recall_1:.1f}% (실제 실패: {corrected_failures}개)")
    print(f"개선 폭: +{improvement:.1f}%p")
    print(f"\n의미적 정답 (Semantic Match): {semantic_successes}개")
    print(f"진짜 실패 (True Failure): {corrected_failures}개")
    
    with open(OUTPUT_REPORT, 'w', encoding='utf-8') as f:
        f.write("# Vector 검색 실패 케이스 LLM 재평가 리포트\n\n")
        f.write(f"- 평가 일시: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- 평가 대상: Vector Only 실패 케이스 {len(failure_cases)}개\n\n")
        
        f.write("## 📊 요약 결과\n\n")
        f.write(f"- **원본 Recall@1:** {original_recall_1:.1f}% (실패: {original_failures}개)\n")
        f.write(f"- **보정 Recall@1:** {corrected_recall_1:.1f}% (실제 실패: {corrected_failures}개)\n")
        f.write(f"- **개선 폭:** +{improvement:.1f}%p\n")
        f.write(f"- **의미적 정답 (Semantic Match):** {semantic_successes}개\n")
        f.write(f"- **진짜 실패 (True Failure):** {corrected_failures}개\n\n")
        
        f.write("## 🔍 상세 분석\n\n")
        
        # 의미적 정답 케이스
        f.write("### ✅ 의미적 정답 (Semantic Match) 케이스\n\n")
        semantic_cases = [r for r in verified_results if r["llm_can_answer"]]
        for i, case in enumerate(semantic_cases[:10], 1):  # 상위 10개만
            f.write(f"#### 케이스 {i}\n")
            f.write(f"- **질문:** {case['query']}\n")
            f.write(f"- **정답 ID:** {case['gold_id']}\n")
            f.write(f"- **검색된 ID:** {case['retrieved_id']}\n")
            f.write(f"- **LLM 판정:** ✅ 답변 가능 (신뢰도: {case['llm_confidence']})\n")
            f.write(f"- **판단 근거:** {case['llm_reasoning']}\n")
            f.write(f"- **검색된 텍스트 (일부):** {case['retrieved_text'][:200]}...\n\n")
        
        # 진짜 실패 케이스
        f.write("### ❌ 진짜 실패 (True Failure) 케이스\n\n")
        true_failures = [r for r in verified_results if not r["llm_can_answer"]]
        for i, case in enumerate(true_failures[:10], 1):  # 상위 10개만
            f.write(f"#### 케이스 {i}\n")
            f.write(f"- **질문:** {case['query']}\n")
            f.write(f"- **정답 ID:** {case['gold_id']}\n")
            f.write(f"- **검색된 ID:** {case['retrieved_id']}\n")
            f.write(f"- **LLM 판정:** ❌ 답변 불가 (신뢰도: {case['llm_confidence']})\n")
            f.write(f"- **판단 근거:** {case['llm_reasoning']}\n")
            f.write(f"- **검색된 텍스트 (일부):** {case['retrieved_text'][:200]}...\n\n")
        
        # 결론
        f.write("## 💡 결론\n\n")
        if corrected_recall_1 >= 85:
            f.write("**Vector 검색의 실제 성능은 보정 후 85% 이상으로, 충분히 우수합니다.**\n\n")
            f.write("- ID 매칭 실패의 상당 부분이 '의미적 정답'이었음\n")
            f.write("- Vector 검색은 실제로 관련 문서를 잘 찾고 있음\n")
            f.write("- 향후 Hybrid 검색의 잠재력은 여전히 유효할 수 있음\n")
        else:
            f.write("**Vector 검색의 실제 성능도 보정 후에도 상대적으로 낮습니다.**\n\n")
            f.write("- ID 매칭 실패의 대부분이 '진짜 실패'였음\n")
            f.write("- Vector 모델(bge-m3)이 한국사 고유명사 학습이 부족할 가능성\n")
            f.write("- BM25 선택이 확실히 옳았음\n")
    
    print(f"\n💾 상세 리포트 저장 완료: {OUTPUT_REPORT}")

if __name__ == "__main__":
    main()

