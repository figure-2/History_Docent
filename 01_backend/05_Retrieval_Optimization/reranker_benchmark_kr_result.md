# 한국어 특화 Reranker 성능 비교 벤치마크

- 평가 일시: 2025-11-22 15:44:53
- 샘플 수: 50개
- 전체 문서 수: 3,719개
- Reranker 모델: jinaai/jina-reranker-v2-base-multilingual
- 하이브리드 검색: BM25 (0.4) + E5 Vector (0.6)

## 정량 평가 결과 (Quantitative Evaluation)

| Strategy | Recall@1 | Recall@3 | Recall@5 | MRR | Avg Time (ms) |
|---|---|---|---|---|---|
| **BM25 Only** | 92.0% | 98.0% | 98.0% | 0.947 | 30.6ms |
| **BM25 + E5 Vector Hybrid + Korean Reranker (Jina v2)** | 86.0% | 94.0% | 94.0% | 0.897 | 217.4ms |

## 🏆 최종 선정: **BM25 Only**
- Recall@1: 92.0%
- MRR: 0.947
- Recall@5: 98.0%
- 평균 검색 시간: 30.6ms

## 📈 성능 개선 분석

### BM25 Only vs BM25 + E5 Vector Hybrid + Korean Reranker
- Recall@1 개선: **-6.0%p** (92.0% → 86.0%)
- MRR 개선: **-0.050** (0.947 → 0.897)
- 검색 시간 증가: **+186.8ms** (30.6ms → 217.4ms)

## 💡 결론

**하이브리드 검색 + Reranker가 BM25 Only보다 성능이 하락했습니다.**
- Recall@1이 -6.0%p 하락하여 BM25 Only를 유지하는 것이 최선입니다.
- 한국사 도메인에서는 키워드 매칭(BM25)이 의미 기반 검색 + 재순위화 조합보다 더 효과적임을 확인했습니다.
