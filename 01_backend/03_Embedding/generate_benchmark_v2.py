import json
import random
import time
import os
import requests
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# -----------------------------------------------------------------------------
# 설정 (Configuration)
# -----------------------------------------------------------------------------
BASE_DIR = Path("/home/pencilfoxs/00_new/History_Docent")
CHUNK_FILE = BASE_DIR / "02_Chunking/output/all_chunks.json"
OUTPUT_DIR = BASE_DIR / "03_Embedding/data"

# ✅ 기존 파일(500개)과 최종 파일(2000개) 경로
EXISTING_FILE = OUTPUT_DIR / "korean_history_benchmark_500.json"
OUTPUT_FILE = OUTPUT_DIR / "korean_history_benchmark_2000.json"

TARGET_SIZE = 2000  # 목표 질문 개수

# API 키 로드 (.env2 파일에서 직접 읽기)
ENV_FILE = Path("/home/pencilfoxs/00_new/.env2")
API_KEY = None
if ENV_FILE.exists():
    with open(ENV_FILE, 'r') as f:
        for line in f:
            if line.startswith("GOOGLE_API_KEY="):
                API_KEY = line.split("=", 1)[1].strip()
                break

# .env2에서 못 찾았으면 환경 변수에서 가져오기
if not API_KEY:
    API_KEY = os.getenv("GOOGLE_API_KEY")

# -----------------------------------------------------------------------------
# Gemini API 설정 (REST API 직접 호출)
# -----------------------------------------------------------------------------
if not API_KEY:
    print("⚠️ Error: GOOGLE_API_KEY 환경 변수가 설정되지 않았습니다.")
    exit(1)

# REST API 엔드포인트 (사용 가능한 모델: gemini-2.5-flash 사용)
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"

# -----------------------------------------------------------------------------
# 프롬프트 템플릿
# -----------------------------------------------------------------------------
PROMPT_TEMPLATE = """
너는 한국사 전문가이자 수능 출제 위원이다.
아래 제공된 [역사 문서 조각]을 읽고, 그 내용을 바탕으로 고품질의 질문 1개를 생성하라.

[역사 문서 조각]
{chunk_text}

[조건]
1. 질문 유형은 다음 중 하나를 무작위로 선택하라:
   - 사실 확인 (Fact): 연도, 인물, 사건명 등을 묻는 질문
   - 인과 추론 (Reasoning): 사건의 원인, 결과, 의도를 묻는 질문
   - 복합 이해 (Complex): 여러 정보를 종합해야 답할 수 있는 질문
2. 정답이 반드시 위 문서 조각 안에 포함되어야 한다.
3. 질문은 한국어로 자연스럽게 작성하라.
4. 출력 형식은 오직 JSON 포맷만 출력하라. (마크다운 backticks 없이)

[출력 예시]
{{
  "query": "세종대왕이 훈민정음을 창제한 주된 목적은 무엇인가?",
  "type": "Reasoning",
  "difficulty": "Medium"
}}
"""

# -----------------------------------------------------------------------------
# 메인 로직
# -----------------------------------------------------------------------------
def generate_questions():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. 기존 데이터 로드 (이어하기)
    benchmark_dataset = []
    used_chunk_ids = set()
    
    # 500개 파일이 있으면 먼저 로드
    if EXISTING_FILE.exists():
        print(f"📂 기존 데이터 로드 중: {EXISTING_FILE}")
        with open(EXISTING_FILE, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
            benchmark_dataset.extend(existing_data)
            for item in existing_data:
                used_chunk_ids.add(item['chunk_id'])
        print(f"   ✅ 기존 {len(benchmark_dataset)}개 질문 로드 완료.")
    
    # 이미 2000개 넘으면 종료
    current_count = len(benchmark_dataset)
    if current_count >= TARGET_SIZE:
        print(f"🎉 이미 목표 개수({TARGET_SIZE}개)를 달성했습니다. 생성을 건너뜁니다.")
        return

    needed_count = TARGET_SIZE - current_count
    print(f"🚀 추가 생성 필요 개수: {needed_count}개")
    
    # 2. 청크 데이터 로드
    print(f"📂 청크 파일 로딩 중: {CHUNK_FILE}")
    with open(CHUNK_FILE, 'r', encoding='utf-8') as f:
        all_chunks = json.load(f)
    
    # 3. 유효 청크 필터링 & 중복 제거
    valid_chunks = []
    for c in all_chunks:
        # 텍스트 길이 충분하고 + 아직 질문 안 만든 청크
        if (len(c['text']) > 100 and 
            c['metadata']['type'] == 'text' and 
            c['chunk_id'] not in used_chunk_ids):
            valid_chunks.append(c)
            
    print(f"   - 사용 가능한 후보 청크: {len(valid_chunks)}개")
    
    if len(valid_chunks) < needed_count:
        print(f"⚠️ 주의: 남은 청크({len(valid_chunks)}개)가 목표({needed_count}개)보다 적습니다. 전부 사용합니다.")
        selected_chunks = valid_chunks
    else:
        selected_chunks = random.sample(valid_chunks, needed_count)
    
    print(f"🚀 신규 질문 생성 시작 (목표: {len(selected_chunks)}개)...")
    
    # 병렬 처리 함수
    def generate_single_question(i_offset, chunk):
        try:
            # REST API 직접 호출
            payload = {
                "contents": [{
                    "parts": [{
                        "text": PROMPT_TEMPLATE.format(chunk_text=chunk['text'])
                    }]
                }]
            }
            
            headers = {
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                f"{GEMINI_API_URL}?key={API_KEY}",
                json=payload,
                headers=headers,
                timeout=30
            )
            
            if response.status_code != 200:
                raise Exception(f"API 호출 실패: {response.status_code} - {response.text[:200]}")
            
            result = response.json()
            
            # 응답에서 텍스트 추출
            if 'candidates' in result and len(result['candidates']) > 0:
                text_response = result['candidates'][0]['content']['parts'][0]['text'].strip()
            else:
                raise Exception(f"응답 형식 오류: {result}")
            
            # 마크다운 코드 블록 제거 로직
            if text_response.startswith("```"):
                text_response = text_response.replace("```json", "").replace("```", "").strip()
            
            # JSON 파싱 시도
            try:
                q_data = json.loads(text_response)
            except json.JSONDecodeError as je:
                raise Exception(f"JSON 파싱 실패: {je}")
            
            # 데이터셋 엔트리 생성
            # ID는 기존 개수 + 현재 인덱스로 생성
            entry = {
                "id": f"q_{i_offset:05d}",  # 5자리로 늘림 (q_00001)
                "chunk_id": chunk['chunk_id'],
                "query": q_data.get("query", ""),
                "type": q_data.get("type", "General"),
                "difficulty": q_data.get("difficulty", "Medium"),
                "gold_text": chunk['text'],
                "source": chunk['metadata']['source']
            }
            return (entry, None)
        except Exception as e:
            return (None, str(e))
    
    # 병렬 처리 실행 (속도 위해 워커 10개)
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {}
        # 인덱스는 (기존 개수 + 0) 부터 시작
        start_idx = current_count
        
        for i, chunk in enumerate(selected_chunks):
            real_idx = start_idx + i
            futures[executor.submit(generate_single_question, real_idx, chunk)] = real_idx
            
        for future in tqdm(as_completed(futures), total=len(selected_chunks), desc="질문 생성 중"):
            idx = futures[future]
            entry, error = future.result()
            
            if entry:
                benchmark_dataset.append(entry)
            else:
                if error:
                    print(f"⚠️ [Index {idx}] 생성 실패: {error}")
                
            # 중간 저장 (100개마다)
            if len(benchmark_dataset) % 100 == 0 and len(benchmark_dataset) > 0:
                with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                    json.dump(benchmark_dataset, f, ensure_ascii=False, indent=2)
                print(f"   💾 중간 저장 완료 ({len(benchmark_dataset)}개)")

    # 4. 최종 저장
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(benchmark_dataset, f, ensure_ascii=False, indent=2)
        
    print(f"\n✅ 벤치마크 데이터셋 확장 완료!")
    print(f"   총 질문: {len(benchmark_dataset)}개")
    print(f"   저장 경로: {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_questions()

