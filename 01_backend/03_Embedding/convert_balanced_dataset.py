#!/usr/bin/env python3
"""
균형잡힌 QA 데이터셋을 임베딩 벤치마크 형식으로 변환
- 새로운 데이터셋(11,140개)을 평가용 형식으로 변환
- 질문 유형별로도 분리하여 평가 가능하도록 구성
"""

import json
from pathlib import Path
from collections import defaultdict

# 경로 설정
QA_DATASET = Path("/home/pencilfoxs/00_new/History_Docent/06_LLM_Evaluation/balanced_qa_dataset_full.jsonl")
CHUNK_FILE = Path("/home/pencilfoxs/00_new/History_Docent/02_Chunking/output/all_chunks.json")
OUTPUT_DIR = Path("/home/pencilfoxs/00_new/History_Docent/03_Embedding/data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_chunks():
    """청크 파일 로드"""
    with open(CHUNK_FILE, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    return {chunk['chunk_id']: chunk for chunk in chunks}

def convert_dataset():
    """QA 데이터셋을 벤치마크 형식으로 변환"""
    chunks = load_chunks()
    
    # 전체 데이터셋
    all_benchmark = []
    # 질문 유형별 데이터셋
    by_type = defaultdict(list)
    
    with open(QA_DATASET, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            data = json.loads(line)
            chunk_id = data.get('chunk_id')
            question = data.get('question')
            q_type = data.get('type')
            
            if not chunk_id or not question:
                continue
            
            # 청크 찾기
            chunk = chunks.get(chunk_id)
            if not chunk:
                continue
            
            # 벤치마크 형식으로 변환
            benchmark_item = {
                'query': question,
                'gold_text': chunk.get('text', ''),
                'chunk_id': chunk_id,
                'type': q_type,
                'metadata': chunk.get('metadata', {})
            }
            
            all_benchmark.append(benchmark_item)
            by_type[q_type].append(benchmark_item)
    
    # 저장
    # 1. 전체 데이터셋
    output_file = OUTPUT_DIR / "korean_history_benchmark_balanced_11140.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_benchmark, f, ensure_ascii=False, indent=2)
    print(f"✅ 전체 데이터셋 저장: {output_file} ({len(all_benchmark)}개)")
    
    # 2. 질문 유형별 데이터셋
    for q_type, items in by_type.items():
        type_file = OUTPUT_DIR / f"korean_history_benchmark_balanced_{q_type}_{len(items)}.json"
        with open(type_file, 'w', encoding='utf-8') as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        print(f"✅ {q_type} 데이터셋 저장: {type_file} ({len(items)}개)")
    
    # 3. 통계 출력
    print("\n📊 데이터셋 통계:")
    print(f"  전체: {len(all_benchmark)}개")
    for q_type, items in by_type.items():
        print(f"  {q_type}: {len(items)}개 ({len(items)/len(all_benchmark)*100:.1f}%)")
    
    return all_benchmark, by_type

if __name__ == "__main__":
    print("🔄 균형잡힌 QA 데이터셋 변환 시작...")
    convert_dataset()
    print("✅ 변환 완료!")


