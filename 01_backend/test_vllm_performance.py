#!/usr/bin/env python3
"""
vLLM 성능 벤치마크 테스트
- 여러 질문으로 테스트하여 평균 속도 확인
"""
import sys
import os
import time
import statistics

# 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), "05_Retrieval_Optimization"))

from history_docent import HistoryDocent

# 테스트 질문 리스트 (다양한 유형 포함)
TEST_QUERIES = [
    # 키워드형 (간단한 사실 질문)
    "손기정 선수는 어떤 올림픽에서 금메달을 땄나요?",
    "세종대왕이 만든 한글은 언제 완성되었나요?",
    "임진왜란은 몇 년도에 발생했나요?",
    
    # 문맥형 (인과관계 질문)
    "손기정이 서윤복을 특별히 아끼고 훈련을 도운 이유는 무엇이었을까요?",
    "세종이 휘빈 김씨의 압승술에 대해 분노한 배경에는 어떤 가치관이 작용했나요?",
    
    # 추상형 (키워드 없이 묘사)
    "그 올림픽에서 1등 한 유명한 사람이 제일 아끼던 제자는 누구인가요?",
    "한글을 만든 왕이 왜 그렇게 중요한 업적을 남겼나요?",
    
    # 복잡한 질문
    "조선 시대에 왕실의 여성으로서 모범을 보여야 하는 세자빈이 행한 압승술은 어떤 문제였나요?",
    "을사늑약 이후 경상도에서 항일 의병을 이끈 주요 의병장은 누구인가요?",
]

def run_benchmark():
    """벤치마크 테스트 실행"""
    print("=" * 70)
    print("🚀 vLLM 성능 벤치마크 테스트")
    print("=" * 70)
    print(f"\n📊 테스트 질문 수: {len(TEST_QUERIES)}개")
    print(f"질문 유형: 키워드형, 문맥형, 추상형, 복잡한 질문 포함\n")
    
    # 시스템 초기화
    print("🔧 시스템 초기화 중...")
    docent = HistoryDocent()
    docent.initialize()
    print("✅ 초기화 완료!\n")
    
    # 테스트 실행
    results = []
    latencies = []
    
    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"\n{'='*70}")
        print(f"테스트 {i}/{len(TEST_QUERIES)}")
        print(f"{'='*70}")
        print(f"질문: {query}")
        print("-" * 70)
        
        try:
            start_time = time.time()
            result = docent.chat(query)
            elapsed = time.time() - start_time
            
            latencies.append(result['latency'])
            results.append({
                'query': query,
                'answer': result['answer'],
                'latency': result['latency'],
                'sources_count': len(result['sources'])
            })
            
            print(f"\n✅ 성공!")
            print(f"답변: {result['answer'][:100]}..." if len(result['answer']) > 100 else f"답변: {result['answer']}")
            print(f"소요 시간: {result['latency']}초")
            print(f"출처 수: {len(result['sources'])}개")
            
        except Exception as e:
            print(f"\n❌ 오류 발생: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # 결과 요약
    print("\n" + "=" * 70)
    print("📊 벤치마크 결과 요약")
    print("=" * 70)
    
    if latencies:
        print(f"\n✅ 성공한 테스트: {len(latencies)}/{len(TEST_QUERIES)}개")
        print(f"\n⏱️  속도 통계:")
        print(f"  - 평균: {statistics.mean(latencies):.2f}초")
        print(f"  - 중앙값: {statistics.median(latencies):.2f}초")
        print(f"  - 최소: {min(latencies):.2f}초")
        print(f"  - 최대: {max(latencies):.2f}초")
        if len(latencies) > 1:
            print(f"  - 표준편차: {statistics.stdev(latencies):.2f}초")
        
        print(f"\n📈 개선 결과:")
        old_avg = 33  # 기존 평균 30~36초의 중간값
        new_avg = statistics.mean(latencies)
        improvement = old_avg / new_avg
        print(f"  - 이전 평균: {old_avg}초")
        print(f"  - 현재 평균: {new_avg:.2f}초")
        print(f"  - 개선율: 약 {improvement:.1f}배 빠름")
        
        print(f"\n🎯 목표 달성 여부:")
        target_max = 5.0  # 목표: 3~5초
        if new_avg <= target_max:
            print(f"  ✅ 목표 달성! (목표: {target_max}초 이하, 실제: {new_avg:.2f}초)")
        else:
            print(f"  ⚠️  목표 미달 (목표: {target_max}초 이하, 실제: {new_avg:.2f}초)")
        
        # 질문별 상세 결과
        print(f"\n📋 질문별 상세 결과:")
        print("-" * 70)
        for i, res in enumerate(results, 1):
            print(f"\n[{i}] {res['query'][:50]}...")
            print(f"    시간: {res['latency']:.2f}초 | 출처: {res['sources_count']}개")
    else:
        print("\n❌ 성공한 테스트가 없습니다.")
    
    print("\n" + "=" * 70)
    print("✅ 벤치마크 완료")
    print("=" * 70)

if __name__ == "__main__":
    run_benchmark()

