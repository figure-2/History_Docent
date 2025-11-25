import json
import time
import os
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import requests

# -----------------------------------------------------------------------------
# 설정 (Configuration)
# -----------------------------------------------------------------------------
# API 설정
ENV_FILE = Path("/home/pencilfoxs/00_new/.env2")
API_KEY = None

if ENV_FILE.exists():
    with open(ENV_FILE, 'r') as f:
        for line in f:
            # 공백 포함 가능성 고려
            if "GOOGLE_API_KEY_2" in line and "=" in line:
                API_KEY = line.split("=", 1)[1].strip()
                break

if not API_KEY:
    print("⚠️ Error: GOOGLE_API_KEY_2 not found in .env2")
    exit(1)

# Gemini 2.0 Flash (Experimental) - 지시 이행 능력이 좋고 빠름
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent"

# 경로 설정
BASE_DIR = Path("/home/pencilfoxs/00_new/History_Docent")
CHUNK_FILE = BASE_DIR / "02_Chunking/output/all_chunks.json"
OUTPUT_DIR = BASE_DIR / "06_LLM_Evaluation"
OUTPUT_DIR.mkdir(exist_ok=True)

# 생성 설정
SAMPLE_SIZE = 60   # 샘플 테스트용 (유형별 약 20개)
MAX_WORKERS = 10   # 병렬 처리

# -----------------------------------------------------------------------------
# 프롬프트 템플릿 (Prompt Templates)
# -----------------------------------------------------------------------------

# 1. 키워드형 (BM25 유리) - 30%
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

# 2. 문맥/스토리형 (Hybrid 유리) - 40%
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

# 3. 추상/풀어쓰기형 (Vector 유리) - 30%
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

def generate_question(chunk, index, q_type=None):
    """Gemini API를 호출하여 유형별 질문 생성"""
    text = chunk['text']
    
    # q_type이 지정되지 않았으면 랜덤 선택 (가중치: 30%, 40%, 30%)
    if q_type is None:
        q_type = random.choices(
            ['keyword', 'context', 'abstract'], 
            weights=[0.3, 0.4, 0.3],
            k=1
        )[0]
    
    # 프롬프트 선택
    if q_type == 'keyword':
        prompt_text = PROMPT_KEYWORD.format(text=text)
        temperature = 0.5
    elif q_type == 'context':
        prompt_text = PROMPT_CONTEXT.format(text=text)
        temperature = 0.6
    else: # abstract
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
    
    headers = {"Content-Type": "application/json"}
    
    try:
        time.sleep(random.uniform(0.3, 0.6)) # Rate Limit 방지 (429 에러 방지)
        
        response = requests.post(
            f"{GEMINI_API_URL}?key={API_KEY}",
            json=payload,
            headers=headers,
            timeout=30
        )
        
        if response.status_code != 200:
            print(f"   ⚠️ API Error (Status {response.status_code}) for chunk {chunk['chunk_id']}")
            return None
            
        result = response.json()
        
        if 'candidates' in result and result['candidates']:
            question = result['candidates'][0]['content']['parts'][0]['text'].strip()
            # 마크다운 코드 블록 제거
            if question.startswith("```"):
                question = question.replace("```json", "").replace("```", "").strip()
            
            # 여러 질문이 생성된 경우 첫 번째 질문만 추출
            # "1. 질문" 또는 "질문\n2. 질문" 형식 처리
            lines = question.split('\n')
            first_question = lines[0]
            # "1. ", "2. " 같은 번호 제거
            if first_question.strip().startswith(('1.', '2.', '3.', '4.', '5.')):
                first_question = first_question.split('.', 1)[1].strip()
            question = first_question.strip()
            
            return {
                "id": f"q_{index:05d}_{q_type}",
                "chunk_id": chunk['chunk_id'],
                "question": question,
                "type": q_type,
                "source_text": text,
                "source_metadata": chunk.get('metadata', {})
            }
        else:
            print(f"   ⚠️ No candidates in response for chunk {chunk['chunk_id']}")
            return None
            
    except Exception as e:
        print(f"   ⚠️ Exception for chunk {chunk['chunk_id']}: {e}")
        return None

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="균형 잡힌 QA 데이터셋 생성")
    parser.add_argument("--mode", choices=["sample", "full"], default="sample",
                       help="sample: 샘플 테스트 (60개), full: 전체 청크 대상 (3,719개)")
    parser.add_argument("--output", type=str, default=None,
                       help="출력 파일명 (기본값: balanced_qa_dataset_{mode}.json)")
    
    args = parser.parse_args()
    
    print(f"🚀 Balanced QA Dataset Generation (API Key: ...{API_KEY[-4:]})")
    print(f"   - Mode: {args.mode}")
    print(f"   - Types: Keyword(30%), Context(40%), Abstract(30%)")
    
    # 1. 청크 로드
    print(f"\n📂 Loading chunks from: {CHUNK_FILE}")
    with open(CHUNK_FILE, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    print(f"   ✅ Total chunks: {len(chunks)}")
    
    # 2. 샘플링 또는 전체 선택
    if args.mode == "sample":
        selected_chunks = random.sample(chunks, min(SAMPLE_SIZE, len(chunks)))
        print(f"   📊 Sample size: {len(selected_chunks)} chunks")
    else:  # full
        selected_chunks = chunks
        print(f"   📊 Full mode: {len(selected_chunks)} chunks (1 question per chunk)")
    
    # 3. 출력 파일명 결정
    if args.output:
        output_file = OUTPUT_DIR / args.output
    else:
        output_file = OUTPUT_DIR / f"balanced_qa_dataset_{args.mode}.json"
    
    dataset = []
    
    # 4. 병렬 처리
    print(f"\n🔄 Generating questions...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(generate_question, chunk, i): i for i, chunk in enumerate(selected_chunks)}
        
        for future in tqdm(as_completed(futures), total=len(selected_chunks), desc="   Progress"):
            result = future.result()
            if result:
                dataset.append(result)
    
    # 5. 저장
    print(f"\n💾 Saving to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
        
    print(f"✅ Saved {len(dataset)} questions ({len(dataset)/len(selected_chunks)*100:.1f}% success rate)")
    
    # 6. 유형별 통계
    print("\n--- [Type Statistics] ---")
    type_counts = {}
    for item in dataset:
        q_type = item['type']
        type_counts[q_type] = type_counts.get(q_type, 0) + 1
    
    for q_type in ['keyword', 'context', 'abstract']:
        count = type_counts.get(q_type, 0)
        pct = (count / len(dataset) * 100) if dataset else 0
        print(f"   {q_type:10s}: {count:4d} ({pct:5.1f}%)")
    
    # 7. 유형별 미리보기 (검증)
    print("\n--- [Type-based Preview] ---")
    for q_type in ['keyword', 'context', 'abstract']:
        samples = [d for d in dataset if d['type'] == q_type][:2]
        if samples:
            print(f"\n[Type: {q_type.upper()}]")
            for s in samples:
                print(f"  Q: {s['question']}")
                print(f"  Context: {s['source_text'][:60]}...")
                print()

if __name__ == "__main__":
    main()

