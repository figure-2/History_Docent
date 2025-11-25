#!/usr/bin/env python3
"""
균형 잡힌 QA 데이터셋 생성 스크립트 (Robust Version)
- 모든 청크에 대해 Keyword, Context, Abstract 3가지 유형 질문 생성
- Rate Limit (429) 방어: 지수 백오프(Exponential Backoff)
- 10개 단위 자동 저장 (데이터 손실 방지)
- 이어하기 기능 (중단 후 재실행 시 이어서 진행)
- 백그라운드 실행 지원 (nohup)
"""

import json
import time
import os
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import requests
from datetime import datetime

# -----------------------------------------------------------------------------
# 설정 (Configuration)
# -----------------------------------------------------------------------------
ENV_FILE = Path("/home/pencilfoxs/00_new/.env2")
API_KEY = None

# API Key 로드 (공백 처리 포함)
if ENV_FILE.exists():
    with open(ENV_FILE, 'r') as f:
        for line in f:
            if "GOOGLE_API_KEY_2" in line and "=" in line:
                API_KEY = line.split("=", 1)[1].strip()
                break

if not API_KEY:
    print("⚠️ Error: GOOGLE_API_KEY_2 not found in .env2")
    exit(1)

GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent"

# 경로 설정
BASE_DIR = Path("/home/pencilfoxs/00_new/History_Docent")
CHUNK_FILE = BASE_DIR / "02_Chunking/output/all_chunks.json"
OUTPUT_DIR = BASE_DIR / "06_LLM_Evaluation"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "balanced_qa_dataset_test.jsonl"  # 테스트용 JSONL 포맷
LOG_FILE = OUTPUT_DIR / "generation_progress_test.log"
STATS_FILE = OUTPUT_DIR / "generation_stats_test.json"

# 실행 설정
MAX_WORKERS = 5  # 429 방지를 위해 워커 수 조절
SAVE_INTERVAL = 10  # 10개 단위 저장
BASE_DELAY = 0.5  # 기본 API 호출 간 딜레이 (초)
MAX_RETRIES = 5  # 최대 재시도 횟수

# -----------------------------------------------------------------------------
# 프롬프트 템플릿 (Prompt Templates)
# -----------------------------------------------------------------------------

# 1. 키워드형 (BM25 유리)
PROMPT_KEYWORD = """
당신은 '한국사 시험을 준비하는 학생'입니다.
주어진 텍스트에서 중요한 **사실, 연도, 인물, 사건명**을 확인하는 단답형 질문을 **하나만** 만드세요.

[규칙]
1. 핵심 고유명사(키워드)를 반드시 포함해서 질문하세요.
2. 명확한 정답이 나오도록 구체적으로 물어보세요.
3. **반드시 질문 하나만** 출력하세요. (번호나 여러 질문 금지)
4. 예: "이성계가 위화도 회군을 단행한 년도는 언제인가요?"

[텍스트]
{text}

[질문 생성 (질문 하나만 한 문장으로 출력)]
"""

# 2. 문맥/스토리형 (Hybrid 유리)
PROMPT_CONTEXT = """
당신은 '역사 이야기를 듣는 관람객'입니다.
주어진 텍스트의 **인과관계, 이유, 배경**에 대해 물어보는 질문을 **하나만** 만드세요.

[규칙]
1. 단순 사실보다는 "왜?", "어떻게?", "그 결과는?" 위주로 질문하세요.
2. 문맥을 이해해야 답할 수 있는 질문을 만드세요.
3. **반드시 질문 하나만** 출력하세요. (번호나 여러 질문 금지)
4. 예: "이성계가 요동 정벌을 반대하고 결국 회군하게 된 결정적인 이유는 무엇인가요?"

[텍스트]
{text}

[질문 생성 (질문 하나만 한 문장으로 출력)]
"""

# 3. 추상/풀어쓰기형 (Vector 유리)
PROMPT_ABSTRACT = """
당신은 '역사 용어가 잘 기억나지 않는 일반인'입니다.
주어진 텍스트의 내용을 물어보되, **핵심 고유명사를 쓰지 말고 풀어서** 질문하세요.

[규칙]
1. **절대 본문의 핵심 고유명사(인물명, 사건명 등)를 직접 쓰지 마세요.**
2. "그거 있잖아", "그 사람", "그 사건" 처럼 대명사나 묘사를 사용하세요.
3. 예: "비 많이 온다고 군대 돌려서 왕 쫓아낸 그 사건이 뭐예요?" (위화도 회군 언급 X)

[텍스트]
{text}

[질문 생성 (질문만 한 문장으로)]
"""

# -----------------------------------------------------------------------------
# 유틸리티 함수
# -----------------------------------------------------------------------------

def log_message(message):
    """로그 파일에 메시지 기록"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}\n"
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_entry)
    print(message)  # 콘솔에도 출력

def call_gemini_api(payload, retry_count=0):
    """지수 백오프(Exponential Backoff) 적용된 API 호출"""
    headers = {"Content-Type": "application/json"}
    
    try:
        # 기본 딜레이
        time.sleep(BASE_DELAY + random.uniform(0, 0.2))
        
        response = requests.post(
            f"{GEMINI_API_URL}?key={API_KEY}",
            json=payload,
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        
        # 429 (Too Many Requests) 또는 500번대 에러 시 재시도
        if response.status_code == 429 or response.status_code >= 500:
            if retry_count < MAX_RETRIES:
                # 지수 백오프: 2^retry_count 초 + 랜덤 추가
                wait_time = (2 ** retry_count) + random.uniform(0, 1)
                log_message(f"[Retry {retry_count + 1}/{MAX_RETRIES}] Status {response.status_code}. Waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
                return call_gemini_api(payload, retry_count + 1)
            else:
                log_message(f"[Fail] Max retries reached. Status: {response.status_code}")
                return None
        
        # 기타 에러 (400, 404 등)
        log_message(f"[Error] Status {response.status_code}: {response.text[:200]}")
        return None

    except requests.exceptions.Timeout:
        if retry_count < MAX_RETRIES:
            wait_time = (2 ** retry_count) + random.uniform(0, 1)
            log_message(f"[Timeout Retry {retry_count + 1}/{MAX_RETRIES}] Waiting {wait_time:.1f}s...")
            time.sleep(wait_time)
            return call_gemini_api(payload, retry_count + 1)
        return None
    except Exception as e:
        if retry_count < MAX_RETRIES:
            wait_time = (2 ** retry_count)
            log_message(f"[Exception Retry {retry_count + 1}/{MAX_RETRIES}] {str(e)[:100]}. Waiting {wait_time:.1f}s...")
            time.sleep(wait_time)
            return call_gemini_api(payload, retry_count + 1)
        log_message(f"[Fail] Exception: {str(e)[:200]}")
        return None

def clean_question(question):
    """생성된 질문 전처리 (번호 제거, 마크다운 제거 등)"""
    question = question.strip()
    
    # 마크다운 코드 블록 제거
    if question.startswith("```"):
        question = question.replace("```json", "").replace("```", "").strip()
    
    # 여러 질문이 생성된 경우 첫 번째 질문만 추출
    lines = question.split('\n')
    first_question = lines[0]
    
    # "1. ", "2. " 같은 번호 제거
    if first_question.strip() and first_question.strip()[0].isdigit():
        parts = first_question.split('.', 1)
        if len(parts) > 1:
            first_question = parts[1].strip()
        else:
            # "1)" 형식 처리
            parts = first_question.split(')', 1)
            if len(parts) > 1:
                first_question = parts[1].strip()
    
    return first_question.strip()

def generate_single_question(chunk, q_type):
    """단일 질문 생성"""
    text = chunk['text']
    chunk_id = chunk['chunk_id']
    
    # 프롬프트 선택
    if q_type == 'keyword':
        prompt_text = PROMPT_KEYWORD.format(text=text)
        temperature = 0.5
    elif q_type == 'context':
        prompt_text = PROMPT_CONTEXT.format(text=text)
        temperature = 0.6
    else:  # abstract
        prompt_text = PROMPT_ABSTRACT.format(text=text)
        temperature = 0.8  # 추상형은 창의성 필요
    
    payload = {
        "contents": [{
            "parts": [{
                "text": prompt_text
            }]
        }],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": 150,
        }
    }
    
    api_result = call_gemini_api(payload)
    
    if api_result and 'candidates' in api_result and api_result['candidates']:
        question = api_result['candidates'][0]['content']['parts'][0]['text'].strip()
        question = clean_question(question)
        
        if question and len(question) > 10:  # 최소 길이 검증
            return {
                "chunk_id": chunk_id,
                "type": q_type,
                "question": question,
                "source_text": text,
                "source_metadata": chunk.get('metadata', {}),
                "generated_at": datetime.now().isoformat()
            }
    
    return None

def process_chunk(chunk):
    """청크 처리: 3가지 유형 모두 생성"""
    results = []
    
    # 모든 청크에 대해 3가지 유형 모두 생성
    for q_type in ['keyword', 'context', 'abstract']:
        result = generate_single_question(chunk, q_type)
        if result:
            results.append(result)
        else:
            # 실패해도 계속 진행 (다른 유형은 시도)
            pass
    
    return results

def load_processed_chunks():
    """이미 처리된 청크 ID 목록 로드 (이어하기 기능)"""
    processed_chunks = set()
    
    if OUTPUT_FILE.exists():
        try:
            with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            processed_chunks.add(data['chunk_id'])
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            log_message(f"[Warning] Failed to load processed chunks: {e}")
    
    return processed_chunks

def save_questions(questions, stats):
    """질문들을 JSONL 파일에 저장 (Append 모드)"""
    if not questions:
        return
    
    try:
        with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
            for item in questions:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        
        # 통계 업데이트
        stats['total_saved'] += len(questions)
        # 모든 질문의 타입별로 카운트
        for q in questions:
            stats['by_type'][q['type']] = stats['by_type'].get(q['type'], 0) + 1
        
        # 통계 파일 저장
        with open(STATS_FILE, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        log_message(f"💾 Saved {len(questions)} questions (Total: {stats['total_saved']})")
    except Exception as e:
        log_message(f"[Error] Failed to save questions: {e}")

def main():
    """메인 실행 함수"""
    log_message("=" * 70)
    log_message("🧪 TEST MODE: Balanced QA Dataset Generation (30 chunks only)")
    log_message("=" * 70)
    
    # 1. 이어하기 (Resume) 로직
    processed_chunks = load_processed_chunks()
    log_message(f"📂 Resume: {len(processed_chunks)} chunks already processed")
    
    # 2. 청크 로드 및 필터링
    log_message(f"📂 Loading chunks from: {CHUNK_FILE}")
    with open(CHUNK_FILE, 'r', encoding='utf-8') as f:
        all_chunks = json.load(f)
    
    log_message(f"   ✅ Total chunks: {len(all_chunks)}")
    
    target_chunks = [c for c in all_chunks if c['chunk_id'] not in processed_chunks]
    
    # 🧪 테스트 모드: 30개로 제한
    TEST_LIMIT = 30
    if len(target_chunks) > TEST_LIMIT:
        target_chunks = target_chunks[:TEST_LIMIT]
        log_message(f"   🧪 TEST MODE: Limited to {TEST_LIMIT} chunks")
    
    log_message(f"   📊 Remaining chunks: {len(target_chunks)}")
    
    if not target_chunks:
        log_message("✅ All chunks already processed!")
        return
    
    # 3. 통계 초기화
    stats = {
        'started_at': datetime.now().isoformat(),
        'total_chunks': len(all_chunks),
        'processed_chunks': len(processed_chunks),
        'remaining_chunks': len(target_chunks),
        'total_saved': 0,
        'by_type': {'keyword': 0, 'context': 0, 'abstract': 0}
    }
    
    # 4. 실행
    buffer = []
    completed_count = 0
    
    log_message(f"🔄 Starting generation (Workers: {MAX_WORKERS}, Save Interval: {SAVE_INTERVAL})")
    
    try:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(process_chunk, chunk): chunk for chunk in target_chunks}
            
            for future in tqdm(as_completed(futures), total=len(target_chunks), desc="   Progress"):
                chunk = futures[future]
                result_list = future.result()
                
                if result_list:
                    buffer.extend(result_list)
                    completed_count += 1
                
                # 10개 단위 저장
                if len(buffer) >= SAVE_INTERVAL:
                    save_questions(buffer, stats)
                    buffer = []  # 버퍼 비우기
                
                # 주기적으로 진행 상황 로그
                if completed_count % 50 == 0:
                    log_message(f"📊 Progress: {completed_count}/{len(target_chunks)} chunks processed, {stats['total_saved']} questions saved")
        
        # 남은 버퍼 저장
        if buffer:
            save_questions(buffer, stats)
        
        # 최종 통계
        stats['completed_at'] = datetime.now().isoformat()
        stats['final_saved'] = stats['total_saved']
        
        with open(STATS_FILE, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        log_message("=" * 70)
        log_message("✅ Generation Completed!")
        log_message(f"   Total questions saved: {stats['total_saved']}")
        log_message(f"   By type: Keyword={stats['by_type'].get('keyword', 0)}, "
                   f"Context={stats['by_type'].get('context', 0)}, "
                   f"Abstract={stats['by_type'].get('abstract', 0)}")
        log_message("=" * 70)
        
    except KeyboardInterrupt:
        log_message("\n⚠️ Interrupted by user. Saving remaining buffer...")
        if buffer:
            save_questions(buffer, stats)
        log_message("💾 Progress saved. You can resume by running the script again.")
    except Exception as e:
        log_message(f"❌ Fatal error: {e}")
        if buffer:
            save_questions(buffer, stats)
        raise

if __name__ == "__main__":
    main()

