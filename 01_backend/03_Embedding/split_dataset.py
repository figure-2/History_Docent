#!/usr/bin/env python3
"""
데이터셋 분할 스크립트 (과적합 방지)
- Chunk ID 기준으로 분할하여 데이터 누수 방지
- Train(60%) / Validation(20%) / Test(20%) 분할
- 질문 유형별 비율 유지
"""

import json
import random
from pathlib import Path
from collections import defaultdict

# -----------------------------------------------------------------------------
# 설정
# -----------------------------------------------------------------------------
DATA_DIR = Path("/home/pencilfoxs/00_new/History_Docent/03_Embedding/data")
INPUT_FILE = DATA_DIR / "korean_history_benchmark_balanced_11140.json"
SEED = 42  # 재현 가능성을 위한 고정 시드

# 분할 비율
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2

# -----------------------------------------------------------------------------
# 통계 출력 함수
# -----------------------------------------------------------------------------
def print_stats(name, data):
    """데이터셋 통계 출력"""
    print(f"\n📊 {name} 통계:")
    print(f"  총 개수: {len(data)}개")
    
    # 질문 유형별 분포
    type_counts = defaultdict(int)
    for item in data:
        type_counts[item.get('type', 'unknown')] += 1
    
    total = len(data)
    if total > 0:
        for q_type in ['keyword', 'context', 'abstract']:
            count = type_counts[q_type]
            if count > 0:
                print(f"  - {q_type:<8}: {count:>4}개 ({count/total*100:.1f}%)")

# -----------------------------------------------------------------------------
# 메인 함수
# -----------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("🚀 데이터셋 분할 시작 (Chunk ID 기준 - 데이터 누수 방지)")
    print("=" * 80)
    
    # 1. 데이터 로드
    print(f"\n📂 데이터 로드 중: {INPUT_FILE}")
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"데이터셋 파일을 찾을 수 없습니다: {INPUT_FILE}")
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✅ 총 데이터 로드: {len(data)}개")
    
    # 2. Chunk ID 별로 데이터 그룹화
    # ⚠️ 중요: 같은 지문(Chunk)에서 나온 질문들은 반드시 같은 세트에 있어야 함
    # 이렇게 하지 않으면 모델이 학습 데이터에서 본 지문을 평가 데이터에서도 보게 되어 데이터 누수 발생!
    print("\n🔍 Chunk ID 기준으로 그룹화 중...")
    chunk_groups = defaultdict(list)
    for item in data:
        chunk_id = item.get('chunk_id', 'unknown')
        chunk_groups[chunk_id].append(item)
    
    chunk_ids = list(chunk_groups.keys())
    print(f"✅ 총 고유 청크 수: {len(chunk_ids)}개")
    
    # 청크별 질문 수 확인
    chunk_sizes = [len(chunk_groups[cid]) for cid in chunk_ids]
    print(f"   - 청크당 평균 질문 수: {sum(chunk_sizes)/len(chunk_sizes):.1f}개")
    print(f"   - 최소 질문 수: {min(chunk_sizes)}개")
    print(f"   - 최대 질문 수: {max(chunk_sizes)}개")
    
    # 3. 셔플 및 분할 (6:2:2)
    print(f"\n🎲 Random Seed 고정: {SEED} (재현 가능성 보장)")
    random.seed(SEED)
    random.shuffle(chunk_ids)
    
    n_chunks = len(chunk_ids)
    n_train = int(n_chunks * TRAIN_RATIO)
    n_val = int(n_chunks * VAL_RATIO)
    # 나머지는 test
    
    train_chunks = chunk_ids[:n_train]
    val_chunks = chunk_ids[n_train:n_train+n_val]
    test_chunks = chunk_ids[n_train+n_val:]
    
    print(f"\n📊 분할 결과:")
    print(f"  - Train 청크: {len(train_chunks)}개 ({len(train_chunks)/n_chunks*100:.1f}%)")
    print(f"  - Validation 청크: {len(val_chunks)}개 ({len(val_chunks)/n_chunks*100:.1f}%)")
    print(f"  - Test 청크: {len(test_chunks)}개 ({len(test_chunks)/n_chunks*100:.1f}%)")
    
    # 4. 데이터셋 생성
    print("\n📝 데이터셋 생성 중...")
    train_set = []
    val_set = []
    test_set = []
    
    for cid in train_chunks:
        train_set.extend(chunk_groups[cid])
    for cid in val_chunks:
        val_set.extend(chunk_groups[cid])
    for cid in test_chunks:
        test_set.extend(chunk_groups[cid])
    
    print(f"  ✅ Train Set: {len(train_set)}개")
    print(f"  ✅ Validation Set: {len(val_set)}개")
    print(f"  ✅ Test Set: {len(test_set)}개")
    
    # 5. 결과 저장
    output_files = {
        "train": DATA_DIR / "train_set_60.json",
        "validation": DATA_DIR / "validation_set_20.json",
        "test": DATA_DIR / "test_set_20.json"
    }
    
    print("\n💾 파일 저장 중...")
    with open(output_files["train"], 'w', encoding='utf-8') as f:
        json.dump(train_set, f, ensure_ascii=False, indent=2)
    print(f"  ✅ {output_files['train'].name}")
        
    with open(output_files["validation"], 'w', encoding='utf-8') as f:
        json.dump(val_set, f, ensure_ascii=False, indent=2)
    print(f"  ✅ {output_files['validation'].name}")
        
    with open(output_files["test"], 'w', encoding='utf-8') as f:
        json.dump(test_set, f, ensure_ascii=False, indent=2)
    print(f"  ✅ {output_files['test'].name}")
    
    # 6. 검증 통계 출력
    print("\n" + "=" * 80)
    print("📊 분할 결과 검증")
    print("=" * 80)
    print_stats("전체 데이터 (원본)", data)
    print("-" * 80)
    print_stats(f"Train Set ({len(train_set)}개)", train_set)
    print_stats(f"Validation Set ({len(val_set)}개)", val_set)
    print_stats(f"Test Set ({len(test_set)}개)", test_set)
    print("=" * 80)
    
    # 7. 데이터 누수 검증
    print("\n🔍 데이터 누수 검증 중...")
    train_chunk_ids = set(train_chunks)
    val_chunk_ids = set(val_chunks)
    test_chunk_ids = set(test_chunks)
    
    # 겹치는 청크가 있는지 확인
    train_val_overlap = train_chunk_ids & val_chunk_ids
    train_test_overlap = train_chunk_ids & test_chunk_ids
    val_test_overlap = val_chunk_ids & test_chunk_ids
    
    if train_val_overlap or train_test_overlap or val_test_overlap:
        print("  ❌ 경고: 청크가 겹칩니다! 데이터 누수 위험!")
        if train_val_overlap:
            print(f"     Train-Val 겹침: {len(train_val_overlap)}개")
        if train_test_overlap:
            print(f"     Train-Test 겹침: {len(train_test_overlap)}개")
        if val_test_overlap:
            print(f"     Val-Test 겹침: {len(val_test_overlap)}개")
    else:
        print("  ✅ 완벽! 모든 세트가 독립적입니다. 데이터 누수 없음!")
    
    print("\n" + "=" * 80)
    print("✅ 데이터셋 분할 완료!")
    print("=" * 80)
    print("\n📋 사용 가이드:")
    print("  1. 임베딩 모델 선정 → validation_set_20.json 사용")
    print("  2. 임베딩 모델 파인튜닝 → train_set_60.json (학습), validation_set_20.json (검증)")
    print("  3. 리트리버 평가 → validation_set_20.json 사용")
    print("  4. 리랭커 파인튜닝 → train_set_60.json (학습), validation_set_20.json (검증)")
    print("  5. 최종 평가 → test_set_20.json 사용 (한 번만!)")
    print("\n⚠️  주의: Test Set은 모든 개발/튜닝 완료 후 최종 평가에만 사용하세요!")
    print("=" * 80)

if __name__ == "__main__":
    main()

