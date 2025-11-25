import json
from pathlib import Path

# 설정
CHUNK_FILE = Path("/home/pencilfoxs/00_new/History_Docent/02_Chunking/output/all_chunks.json")
MIN_LENGTH = 1500

def analyze_long_chunks():
    """1500자 이상인 청크를 찾아서 내용과 구조를 분석"""
    
    with open(CHUNK_FILE, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    long_chunks = []
    for chunk in chunks:
        text = chunk.get("text", "")
        if len(text) >= MIN_LENGTH:
            long_chunks.append({
                "chunk_id": chunk.get("chunk_id", "UNKNOWN"),
                "length": len(text),
                "source": chunk.get("metadata", {}).get("source", "UNKNOWN"),
                "page": chunk.get("metadata", {}).get("page", 0),
                "type": chunk.get("metadata", {}).get("type", "unknown"),
                "text": text
            })
    
    print("="*80)
    print(f"🔍 1500자 이상인 청크 분석 (총 {len(long_chunks)}개)")
    print("="*80)
    
    for idx, chunk in enumerate(long_chunks, 1):
        print(f"\n{'='*80}")
        print(f"[{idx}] 청크 ID: {chunk['chunk_id']}")
        print(f"    길이: {chunk['length']:,}자")
        print(f"    출처: {chunk['source']}")
        print(f"    페이지: {chunk['page']}")
        print(f"    타입: {chunk['type']}")
        print(f"\n    [내용 미리보기 - 처음 500자]")
        print(f"    {chunk['text'][:500]}...")
        print(f"\n    [내용 미리보기 - 중간 500자]")
        mid_start = len(chunk['text']) // 2 - 250
        mid_end = len(chunk['text']) // 2 + 250
        print(f"    ...{chunk['text'][mid_start:mid_end]}...")
        print(f"\n    [내용 미리보기 - 마지막 500자]")
        print(f"    ...{chunk['text'][-500:]}")
        
        # 문맥 분석: 제목 패턴이 여러 개 있는지 확인
        import re
        heading_patterns = [
            r'\[[^\]]+\]',  # [제목] 패턴
            r'\d+[\.\)]\s+',  # 1. 2) 같은 번호 패턴
            r'제\d+장',  # 제1장 패턴
            r'제\d+절',  # 제1절 패턴
        ]
        
        heading_count = 0
        found_headings = []
        for pattern in heading_patterns:
            matches = re.findall(pattern, chunk['text'])
            if matches:
                heading_count += len(matches)
                found_headings.extend(matches[:3])  # 처음 3개만 저장
        
        print(f"\n    [문맥 분석]")
        print(f"    - 발견된 제목/구분 패턴: {heading_count}개")
        if found_headings:
            print(f"    - 예시: {found_headings}")
        
        # 문장 개수 확인
        sentences = re.split(r'[.!?]\s+', chunk['text'])
        print(f"    - 문장 개수: 약 {len(sentences)}개")
        
        # 줄바꿈 개수 확인 (문단 구분 여부)
        newlines = chunk['text'].count('\n')
        print(f"    - 줄바꿈 개수: {newlines}개")
        
        if heading_count > 3 or newlines > 10:
            print(f"    ⚠️  경고: 여러 제목/문단이 합쳐진 것으로 보입니다!")
        else:
            print(f"    ✅ 단일 주제의 긴 문단으로 보입니다.")

if __name__ == "__main__":
    analyze_long_chunks()

