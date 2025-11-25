# [LLM] 답변 생성 모델 선정 로그

- **작성 일시:** 2025-11-22 16:10 (KST)
- **작성자:** AI Assistant & User
- **최종 업데이트:** 2025-11-22 16:10 (LLM 벤치마킹 계획 수립 완료)

---

## 1. 가설 (Hypothesis)

### 1.1 초기 가정

한국사 RAG 시스템의 최종 답변 생성 단계에서, **로컬 LLM 모델과 Gemini API 중 어떤 모델이 가장 우수한 성능을 보일 것**인지 데이터 기반으로 선정해야 합니다.

**초기 가설:**
1. **Gemini API**는 Google의 최신 모델로, 한국어 이해도가 높고 빠른 응답 속도를 제공할 것으로 예상됩니다.
2. **로컬 LLM 모델들**은 과거 프로젝트(`History_Docent_PJ_gemini`)에서 검증된 모델들이며, 특히 `beomi/KoAlpaca-Polyglot-12.8B`는 과거 프로젝트에서 최종 선정된 모델입니다.
3. **한국어 특화 모델들**(KoSOLAR, Llama-3-Korean 등)은 한국사 도메인에서 더 나은 성능을 보일 가능성이 있습니다.

### 1.2 비교 실험의 필요성

과거 프로젝트(`History_Docent_PJ_gemini`)에서는 `beomi/KoAlpaca-Polyglot-12.8B`를 최종 선정했지만, 이는 **과거 프로젝트의 특정 조건(하이브리드 검색 + 파인튜닝된 Reranker)**에서의 결과입니다. 현재 프로젝트는 **BM25 Only 검색 전략**을 사용하므로, 다른 LLM 모델이 더 적합할 수 있습니다.

따라서 **현재 프로젝트의 검색 전략(BM25 Only)과 평가 데이터셋(`korean_history_benchmark_2000.json`)**을 기준으로 객관적인 비교 실험이 필요합니다.

---

## 2. 실험 설계 (Experiment Design)

### 2.1 목표 (Goal)

한국사 RAG 시스템의 답변 생성 단계에 가장 적합한 LLM 모델을 **데이터 기반으로 선정**한다.

### 2.2 후보군 선정 이유 (Why Candidates)

#### 2.2.1 Gemini API
- **선정 이유:**
  - Google의 최신 생성형 AI 모델로, 한국어 성능이 검증됨
  - API 방식으로 빠른 응답 속도와 안정성 제공
  - 유료 서비스이지만, 프로젝트 목적(포트폴리오)상 사용 가능
- **모델명:** `gemini-1.5-flash`

#### 2.2.2 로컬 LLM 모델들 (4개)

**1. beomi/KoAlpaca-Polyglot-12.8B**
- **선정 이유:**
  - 과거 프로젝트(`History_Docent_PJ_gemini`)에서 최종 선정된 모델
  - 한국어 데이터로 학습된 고성능 언어 모델
  - 복잡한 지시를 잘 따르고 유창한 한국어 답변 생성 능력이 뛰어남
- **특징:** 12.8B 파라미터, KoAlpaca 프롬프트 형식 지원

**2. yanolja/KoSOLAR-10.7B-v0.2**
- **선정 이유:**
  - Upstage SOLAR 기반으로 파인튜닝된 한국어 모델
  - SOLAR 아키텍처의 우수한 성능
  - 한국어 특화 파인튜닝으로 한국사 도메인에 적합할 가능성
- **특징:** 10.7B 파라미터

**3. MLP-KTLim/llama-3-Korean-Bllossom-8B**
- **선정 이유:**
  - Meta의 Llama 3 기반 한국어 모델
  - 최신 아키텍처 활용
  - 8B 파라미터로 상대적으로 가벼워 빠른 추론 속도 기대
- **특징:** 8B 파라미터, Llama 3 기반

**4. beomi/Llama-3-Open-Ko-8B**
- **선정 이유:**
  - Llama 3 기반의 또 다른 한국어 모델
  - 범용 성능이 우수한 모델
  - 8B 파라미터로 메모리 효율적
- **특징:** 8B 파라미터, Llama 3 기반

### 2.3 비교 기준 (Criteria)

#### 2.3.1 평가 지표 (RAGAS Metrics)

**정량 평가 (Quantitative Evaluation):**
1. **Context Recall:** 검색된 컨텍스트가 정답을 포함하는 비율 (0~1, 높을수록 좋음)
2. **Context Precision:** 검색된 컨텍스트의 관련성 (0~1, 높을수록 좋음)
3. **Faithfulness:** 답변이 컨텍스트에 기반하는 정도, 환각(Hallucination) 방지 (0~1, 높을수록 좋음)
4. **Answer Relevancy:** 답변이 질문에 적절한 정도 (0~1, 높을수록 좋음)
5. **평균 점수 (Average Score):** 위 4개 메트릭의 평균값

**정성 평가 (Qualitative Evaluation):**
- 최소 20개 이상의 샘플을 직접 확인하여 성공/실패 사례 분석
- 엣지 케이스(복잡한 추론 질문, 고유명사 포함 질문 등)에서의 성능 확인

#### 2.3.2 실험 규모

- **평가 데이터셋:** `korean_history_benchmark_2000.json`
- **평가 샘플 수:** 50개 (초기 벤치마크), 필요 시 확대
- **검색 전략:** BM25 Only (Top-5 문서)
- **프롬프트 형식:** KoAlpaca 형식 (`### instruction`, `### input`, `### response`)

#### 2.3.3 실행 환경

- **GPU:** NVIDIA A100-SXM4-40GB
- **메모리:** 83GB RAM
- **CUDA:** 12.2
- **Python:** 3.x
- **주요 라이브러리:** `ragas`, `langchain-google-genai`, `transformers`, `torch`

### 2.4 실험 방법론

**과거 프로젝트 참고:**
- `History_Docent_PJ_gemini/7_Evaluation/run_llm_ragas_evaluation.py`의 평가 방식을 참고하여 구현
- RAGAS 라이브러리를 사용한 자동화된 평가 파이프라인 구축
- Gemini API를 심판관(Judge) LLM으로 사용하여 객관적 평가

**RAG 파이프라인:**
1. **Retrieval:** BM25 Only 검색 (Top-5 문서)
2. **Prompt Generation:** KoAlpaca 형식 프롬프트 생성
3. **Answer Generation:** LLM이 답변 생성
4. **Evaluation:** RAGAS 메트릭으로 자동 평가

---

## 3. 검증 결과 (Validation)

### 3.1 GPU 메모리 분석 (2025-11-22 16:02)

**현재 GCP 인스턴스 상태:**
- **GPU:** NVIDIA A100-SXM4-40GB
- **사용 가능한 GPU 메모리:** 32.5GB (총 40GB 중 7.9GB 사용 중)
- **다른 작업 실행 중:** `generate_data_v2.py` (PID 472051, GPU 메모리 7.9GB 사용)

**모델별 예상 GPU 메모리 사용량 (float16 기준):**

| 모델명 | 파라미터 | 예상 메모리 | 현재 여유 | 테스트 가능 여부 |
|:---|:---:|:---:|:---:|:---:|
| **beomi/KoAlpaca-Polyglot-12.8B** | 12.8B | **35.8GB** | 32.5GB | ❌ **부족** |
| **yanolja/KoSOLAR-10.7B-v0.2** | 10.7B | **30.0GB** | 32.5GB | ✅ **가능** |
| **MLP-KTLim/llama-3-Korean-Bllossom-8B** | 8.0B | **22.4GB** | 32.5GB | ✅ **가능** |
| **beomi/Llama-3-Open-Ko-8B** | 8.0B | **22.4GB** | 32.5GB | ✅ **가능** |

**결론:**
- **8B 모델 3개 + 10.7B 모델 1개:** 테스트 가능 (총 4개)
- **12.8B 모델:** GPU 메모리 부족으로 테스트 불가능 (현재 환경에서는 제외)

### 3.2 과거 프로젝트 분석 결과

**분석 대상:** `/home/pencilfoxs/History_Docent_PJ_gemini`

**주요 발견:**
1. **로컬 LLM 로드 방법:**
   - `AutoModelForCausalLM.from_pretrained()` 사용
   - `torch_dtype=torch.float16` 또는 `torch.bfloat16` 사용 (메모리 절약)
   - `device_map="auto"` 사용 (GPU 자동 할당)
   - `HuggingFacePipeline`로 LangChain 래퍼 생성 (RAGAS 호환)

2. **프롬프트 형식:**
   - KoAlpaca 공식 프롬프트 형식 사용 (`### instruction`, `### input`, `### response`)
   - "역사 전문가 AI" 역할 부여
   - "자료에 없는 내용은 답변하지 말 것" 명시 (환각 방지)

3. **평가 방법론:**
   - RAGAS 라이브러리 사용
   - 4개 메트릭: Context Recall, Context Precision, Faithfulness, Answer Relevancy
   - Gemini API를 심판관(Judge) LLM으로 사용

4. **최종 선정 모델:**
   - `beomi/KoAlpaca-Polyglot-12.8B` (과거 프로젝트)

### 3.3 벤치마킹 스크립트 작성 완료

**작성된 파일:**
- `/home/pencilfoxs/00_new/History_Docent/06_LLM_Evaluation/llm_benchmark.py`

**주요 기능:**
1. 5개 모델 자동 테스트 (Gemini API + 4개 로컬 LLM)
2. RAGAS 자동 평가 (4개 메트릭)
3. 결과 CSV 및 Markdown 리포트 자동 생성
4. GPU 메모리 부족 모델 자동 스킵

**실행 방법:**
```bash
# 전체 모델 테스트 (기본 50개 샘플)
python llm_benchmark.py

# Gemini만 테스트 (로컬 모델 스킵)
python llm_benchmark.py --skip-local

# 샘플 수 조정
python llm_benchmark.py -l 20
```

### 3.4 정량 평가 결과

**⚠️ 아직 실행 전 상태입니다.**

실제 벤치마크 실행 후 다음 정보를 업데이트할 예정:
- 각 모델별 RAGAS 4개 메트릭 점수
- 평균 점수 및 순위
- 평가 소요 시간

### 3.5 정성 평가 결과

**⚠️ 아직 실행 전 상태입니다.**

실제 벤치마크 실행 후 다음 정보를 업데이트할 예정:
- 최소 20개 이상 샘플의 성공/실패 사례 분석
- 엣지 케이스에서의 성능 확인
- 모델별 답변 품질 비교

---

## 4. 의사결정 (Conclusion & Pivot)

### 4.1 계획 수정 사항

**원래 계획:**
- 5개 모델 모두 테스트 (Gemini API + 4개 로컬 LLM)

**수정된 계획:**
- **12.8B 모델 제외:** GPU 메모리 부족으로 현재 환경에서 테스트 불가능
- **4개 모델 테스트:** Gemini API + 3개 로컬 LLM (10.7B, 8B×2)

**수정 근거:**
- 현재 GPU 메모리 여유: 32.5GB
- 12.8B 모델 예상 메모리: 35.8GB (부족)
- 다른 작업이 GPU 메모리 7.9GB 사용 중

### 4.2 다음 단계

1. **벤치마크 실행:**
   - 4개 모델 (Gemini API + 3개 로컬 LLM)에 대해 RAGAS 평가 실행
   - 50개 샘플로 초기 벤치마크 수행

2. **결과 분석:**
   - 정량 평가 결과 비교
   - 정성 평가 (최소 20개 샘플) 수행
   - 실패 케이스 분석

3. **최종 선정:**
   - 4개 메트릭의 평균 점수 기준으로 최고 성능 모델 선정
   - 정성 평가 결과를 종합하여 최종 결정

### 4.3 향후 개선 계획

1. **12.8B 모델 테스트:**
   - 다른 작업 완료 후 GPU 메모리가 충분해지면 테스트 가능
   - 또는 더 작은 샘플 수로 테스트

2. **프롬프트 엔지니어링:**
   - 선정된 모델에 최적화된 프롬프트 형식 탐색
   - Few-shot 예시 추가 등

3. **성능 최적화:**
   - 선정된 모델의 추론 속도 최적화
   - 배치 처리 등

---

## 5. 참고 자료

### 5.1 과거 프로젝트 분석
- `/home/pencilfoxs/History_Docent_PJ_gemini/7_Evaluation/run_llm_ragas_evaluation.py`
- `/home/pencilfoxs/History_Docent_PJ_gemini/6_Integrated_RAG/run_rag.py`
- `/home/pencilfoxs/History_Docent_PJ_gemini/config.yaml`

### 5.2 평가 데이터셋
- `/home/pencilfoxs/00_new/History_Docent/03_Embedding/data/korean_history_benchmark_2000.json`

### 5.3 관련 문서
- `README_Retrieval_Log.md`: BM25 Only 검색 전략 선정 과정
- `README_Embedding_Log.md`: 임베딩 모델 선정 과정

---

## 6. 최종 결론 (Final Conclusion)

**⚠️ 아직 벤치마크 실행 전 상태입니다.**

벤치마크 실행 후 다음 정보를 업데이트할 예정:

### 6.1 최종 선택과 채택 근거

- **최종 선정 모델:** (벤치마크 실행 후 업데이트)
- **선정 근거:**
  - 정량 평가 결과 (RAGAS 4개 메트릭 평균 점수)
  - 정성 평가 결과 (최소 20개 샘플 분석)
  - 실패 분석 결과 (어떤 케이스에 약한지)
- **성능 비교 요약:** (모델별 상세 비교표)

### 6.2 핵심 교훈 (Key Takeaways)

*(벤치마크 실행 후 업데이트)*

### 6.3 향후 개선 방향

*(벤치마크 실행 후 업데이트)*

