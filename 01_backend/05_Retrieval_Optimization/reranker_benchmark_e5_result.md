# E5-large + Reranker 성능 비교 벤치마크

- 평가 일시: 2025-11-22 15:20:19
- 샘플 수: 50개
- 전체 문서 수: 3,719개

## 정량 평가 결과 (Quantitative Evaluation)

| Strategy | Recall@1 | Recall@3 | Recall@5 | MRR | Avg Time (ms) |
|---|---|---|---|---|---|
| **BM25 Only** | 92.0% | 98.0% | 98.0% | 0.947 | 30.4ms |
| **E5 Vector Only** | 72.0% | 88.0% | 88.0% | 0.787 | 28.9ms |
| **E5 Hybrid (BM25 0.4 + Vector 0.6)** | 86.0% | 90.0% | 90.0% | 0.880 | 59.3ms |
| **E5 Hybrid + Reranker** | 88.0% | 94.0% | 96.0% | 0.904 | 319.4ms |

## 🏆 최종 선정: **BM25 Only**
- Recall@1: 92.0%
- MRR: 0.947
- Recall@5: 98.0%
- 평균 검색 시간: 30.4ms

## 📈 성능 개선 분석

### BM25 Only vs E5 Hybrid + Reranker
- Recall@1 개선: **-4.0%p** (92.0% → 88.0%)

### E5 Hybrid vs E5 Hybrid + Reranker
- Recall@1 개선: **+2.0%p** (86.0% → 88.0%)
