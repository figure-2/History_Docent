# [LLM] 답변 생성 모델 기술 명세서

- **작성 일시:** 2025-11-23 14:00 (KST)
- **작성자:** AI Assistant & User
- **최종 업데이트:** 2025-11-23 14:00 (벤치마크 완료, 최종 모델 선정 반영)

---

## 1. 개요 (Overview)

본 문서는 한국사 RAG 시스템의 답변 생성(Generation) 단계에서 사용할 LLM 모델의 기술 명세서입니다. 4개 로컬 LLM 모델을 벤치마크하여 최적의 모델을 선정한 결과를 기술합니다.

**최종 선정 모델:** `MLP-KTLim/llama-3-Korean-Bllossom-8B`

**벤치마크 완료 모델:**
- ✅ `MLP-KTLim/llama-3-Korean-Bllossom-8B` (최종 선정)
- ✅ `beomi/Llama-3-Open-Ko-8B`
- ✅ `yanolja/EEVE-Korean-10.8B-v1.0`
- ✅ `Qwen/Qwen2.5-14B-Instruct`

---

## 2. 선정된 모델 스펙 (Selected Model Specification)

### 2.1 최종 선정 모델

**모델명:** `MLP-KTLim/llama-3-Korean-Bllossom-8B`

**타입:** 로컬 (HuggingFace)

**기본 정보:**
- **파라미터:** 8.0B
- **기반 모델:** Meta Llama 3
- **한국어 특화:** Llama 3 기반 한국어 파인튜닝 모델
- **라이선스:** Apache 2.0 (상업적 사용 가능)

**하드웨어 요구사항:**
- **GPU 메모리:** float16 기준 약 16GB (실제 사용: 약 14-15GB)
- **CPU 메모리:** 최소 8GB
- **디스크:** 모델 파일 약 15GB

**성능 지표 (벤치마크 결과):**
- **성공률:** 100.0% (49/49)
- **평균 지연시간:** 6.98초
- **질문 유형별 지연시간:**
  - Keyword: 2.32초
  - Context: 10.63초
  - Abstract: 7.95초
- **평균 응답 길이:** 285자

**배포 가능성:** ⭐⭐⭐⭐⭐

---

### 2.2 비교 모델 스펙

#### 2.2.1 beomi/Llama-3-Open-Ko-8B

**기본 정보:**
- **파라미터:** 8.0B
- **성능 지표:**
  - 성공률: 100.0% (49/49)
  - 평균 지연시간: 8.40초
  - 평균 응답 길이: 342자

**특징:**
- Bllossom과 유사한 성능
- 일부 프롬프트 포함 문제 발견 (개선 필요)

#### 2.2.2 Qwen/Qwen2.5-14B-Instruct

**기본 정보:**
- **파라미터:** 14.0B
- **성능 지표:**
  - 성공률: 100.0% (49/49)
  - 평균 지연시간: 14.41초
  - 평균 응답 길이: 369자

**특징:**
- 가장 상세하고 구조화된 답변 생성
- 속도가 느려 실시간 서비스에는 부적합

#### 2.2.3 yanolja/EEVE-Korean-10.8B-v1.0

**기본 정보:**
- **파라미터:** 10.8B
- **성능 지표:**
  - 성공률: 98.0% (48/49)
  - 평균 지연시간: 93.59초
  - 평균 응답 길이: 1983자

**특징:**
- 영어 응답 생성 문제로 한국어 RAG 시스템에 부적합
- 속도가 매우 느려 실용성 낮음

---

## 3. 기술 구현 상세 (Technical Implementation)

### 3.1 RAG 파이프라인 구성

**전체 파이프라인:**
1. **1차 검색:** Hybrid Weighted (Vector 60% + BM25 40%)
2. **2차 리랭킹:** Cross-Encoder Reranker (Dongjin-kr/ko-reranker)
3. **3차 생성:** LLM 답변 생성 (선정된 모델)

**코드 위치:** `benchmark_llm_selection.py`

---

### 3.2 모델 로드 방법

**코드 위치:** `benchmark_llm_selection.py`의 `generate_local` 함수

```python
def generate_local(model_name, dataset):
    # GPU 메모리 정리
    torch.cuda.empty_cache()
    gc.collect()
    
    # HuggingFace 토큰 (Gated Repo 접근용)
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    
    # 토크나이저 및 모델 로드
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        trust_remote_code=True, 
        token=hf_token
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        token=hf_token
    )
    
    # 추론 실행
    # ...
    
    # 메모리 정리
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()
```

**핵심 설정:**
- `torch_dtype=torch.float16`: 메모리 절약 (약 50% 감소)
- `device_map="auto"`: GPU 자동 할당
- `trust_remote_code=True`: 커스텀 코드 실행 허용
- `token=hf_token`: Gated Repo 접근을 위한 토큰

**메모리 관리:**
- 각 모델 실행 후 명시적 메모리 해제 (`del model, tokenizer`)
- `torch.cuda.empty_cache()` 호출로 GPU 캐시 정리
- `gc.collect()` 호출로 Python 가비지 컬렉션

---

### 3.3 프롬프트 생성 함수

**코드 위치:** `benchmark_llm_selection.py`의 `get_prompt` 함수

```python
def get_prompt(query, context):
    return f"""당신은 한국사 전문가입니다. 아래 [참고 문서]를 바탕으로 [질문]에 대해 정확하고 상세하게 답변해주세요.
문서에 없는 내용은 지어내지 말고, 정보가 부족하면 부족하다고 말해주세요.

[참고 문서]
{context}

[질문]
{query}

[답변]
"""
```

**프롬프트 설계 원칙:**
1. **역할 부여:** "한국사 전문가"로 명확한 역할 정의
2. **제약 조건:** "문서에 없는 내용은 지어내지 말 것" 명시 (할루시네이션 방지)
3. **투명성:** "정보가 부족하면 부족하다고 말해주세요" (정보 부족 시 정직한 답변)
4. **구조화:** [참고 문서], [질문], [답변] 섹션으로 명확한 구조

**프롬프트 특징:**
- KoAlpaca 형식 미사용 (일반 텍스트 프롬프트)
- 간결하고 명확한 지시사항
- RAG Context를 명확히 구분하여 제공

---

### 3.4 RAG Context 생성

**코드 위치:** `benchmark_llm_selection.py`의 `get_rag_context` 함수

```python
def get_rag_context(query, hybrid, reranker, top_k=3):
    # 1. Hybrid 검색 (Top-50 후보)
    candidates = hybrid.search_weighted(query, top_k=50)
    if not candidates: 
        return "관련 문서를 찾을 수 없습니다."
    
    # 2. Reranker로 재정렬
    pairs = [[query, doc['text']] for doc in candidates]
    scores = reranker.predict(pairs)
    for i, doc in enumerate(candidates):
        doc['rerank_score'] = float(scores[i])
    
    # 3. Top-K 선택
    reranked = sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)[:top_k]
    
    # 4. Context 포맷팅
    return "\n\n".join([f"문서 {i+1}: {doc['text']}" for i, doc in enumerate(reranked)])
```

**RAG 파이프라인 구성:**
- **1차 검색:** Hybrid Weighted (Vector 60% + BM25 40%)
  - Vector: BAAI/bge-m3 (선정된 임베딩 모델)
  - BM25: Okt 형태소 분석기 사용
- **2차 리랭킹:** Dongjin-kr/ko-reranker (선정된 리랭커)
- **최종 Context:** Top-3 문서 제공

**Context 형식:**
```
문서 1: [첫 번째 문서 내용]

문서 2: [두 번째 문서 내용]

문서 3: [세 번째 문서 내용]
```

---

### 3.5 답변 생성 파이프라인

**코드 위치:** `benchmark_llm_selection.py`의 `generate_local` 함수 내부

```python
# 1. 프롬프트 생성
prompt = get_prompt(item['query'], item['rag_context'])

# 2. 토큰화 (Chat Template 시도)
try:
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, 
        tokenize=True, 
        add_generation_prompt=True, 
        return_tensors="pt"
    ).to("cuda")
except:
    # Chat Template 미지원 시 일반 토큰화
    inputs = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

# 3. 생성
outputs = model.generate(
    inputs,
    max_new_tokens=512,
    temperature=0.1,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id
)

# 4. 디코딩
response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
```

**생성 파라미터:**
- `max_new_tokens=512`: 최대 생성 토큰 수
- `temperature=0.1`: 낮은 온도로 일관된 답변 생성
- `do_sample=True`: 샘플링 활성화
- `pad_token_id=tokenizer.eos_token_id`: 패딩 토큰 설정

**Chat Template 처리:**
- 모델이 Chat Template을 지원하는 경우 자동 적용
- 미지원 시 일반 프롬프트로 처리

---

## 4. 실행 환경 및 요구사항

### 4.1 하드웨어 요구사항

**GPU:**
- **최소:** NVIDIA A100-SXM4-40GB (또는 동등한 성능)
- **실제 사용:** GCP 인스턴스 (A100 40GB)
- **메모리:** 선정된 모델(Bllossom-8B) 기준 약 14-15GB 사용

**RAM:**
- **최소:** 32GB
- **권장:** 64GB 이상

**디스크:**
- **모델 다운로드용:** 각 모델당 약 15-20GB
- **총 필요 공간:** 약 100GB (4개 모델 + 임베딩 모델 + 리랭커)

### 4.2 소프트웨어 요구사항

**Python 버전:**
- Python 3.8 이상

**주요 라이브러리:**
```txt
transformers>=4.30.0
torch>=2.0.0
sentence-transformers>=2.2.0
chromadb>=0.4.0
rank-bm25>=0.2.2
konlpy>=0.6.0
google-generativeai>=0.3.0
pandas>=1.5.0
numpy>=1.24.0
tqdm>=4.65.0
python-dotenv>=1.0.0
```

**CUDA:**
- CUDA 12.2 이상
- cuDNN 8.0 이상

### 4.3 환경 변수

**필수:**
- `HUGGINGFACEHUB_API_TOKEN`: HuggingFace API 토큰 (Gated Repo 접근용)
- `GOOGLE_API_KEY`: Gemini API 키 (API 모델 테스트용, 선택사항)

**설정 파일:**
- `.env` 파일에 환경 변수 저장

---

## 5. 실행 방법

### 5.1 벤치마크 실행

**기본 실행:**
```bash
cd /home/pencilfoxs/00_new/History_Docent/06_LLM_Evaluation
python benchmark_llm_selection.py
```

**실행 과정:**
1. RAG 파이프라인 초기화 (Retriever + Reranker)
2. 각 모델별로 순차 실행:
   - 모델 로드
   - 49개 질문에 대한 답변 생성
   - 결과 저장
   - 메모리 정리
3. 중간 저장: 각 모델 완료 후 `llm_benchmark_responses.csv`에 저장

**출력 결과:**
- `results/llm_benchmark_responses.csv`: 모든 모델의 응답 결과
- `results/llm_model_comparison.json`: 모델별 성능 비교 데이터
- `results/llm_qualitative_samples.json`: 정성 평가 샘플

### 5.2 선정된 모델 테스트

**전체 Validation Set 테스트:**
```bash
python test_selected_model.py
```

**테스트 데이터:**
- 전체 Validation Set (2,223개 질문)
- RAG Context 자동 생성
- 결과 저장: `results/llm_selected_model_full_test.csv`

---

## 6. 성능 벤치마크 결과

### 6.1 정량 평가 결과

**종합 비교표:**

| 모델 | 성공률 | 평균 지연시간 | Keyword 지연 | Context 지연 | Abstract 지연 | 응답길이 | 배포가능성 |
|------|--------|---------------|--------------|--------------|---------------|----------|------------|
| **Bllossom-8B** | **100.0%** | **6.98초** | 2.32초 | 10.63초 | 7.95초 | 285자 | ⭐⭐⭐⭐⭐ |
| **Open-Ko-8B** | **100.0%** | 8.40초 | 4.97초 | 9.86초 | 10.24초 | 342자 | ⭐⭐⭐⭐⭐ |
| **Qwen-14B** | **100.0%** | 14.41초 | 6.18초 | 20.90초 | 16.04초 | 369자 | ⭐⭐⭐⭐ |
| **EEVE-10.8B** | 98.0% | 93.59초 | 98.53초 | 88.72초 | 93.81초 | 1983자 | ⭐⭐⭐ |

**선정 모델 상세 성능:**
- **성공률:** 100.0% (49/49) - 모든 질문 유형에서 완벽한 성공률
- **평균 지연시간:** 6.98초 - 4개 모델 중 가장 빠름
- **질문 유형별 성능:**
  - Keyword: 2.32초 (16개 질문)
  - Context: 10.63초 (16개 질문)
  - Abstract: 7.95초 (17개 질문)
- **응답 길이:** 평균 285자 (적절한 길이)

### 6.2 정성 평가 결과

**모델별 평가:**

| 모델 | 답변 정확성 | 답변 완성도 | 할루시네이션 | 한국어 자연스러움 | 종합 평가 |
|------|------------|------------|-------------|----------------|----------|
| **Bllossom-8B** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Open-Ko-8B** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Qwen-14B** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **EEVE-10.8B** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ |

**선정 모델 특징:**
- ✅ **정확성:** Context 기반 정확한 답변 생성
- ✅ **완성도:** 질문에 대한 충분한 정보 제공
- ✅ **할루시네이션 방지:** Context에 없는 내용 생성하지 않음
- ✅ **한국어 자연스러움:** 자연스러운 한국어 표현

---

## 7. 성능 최적화 방안

### 7.1 메모리 최적화

**현재 적용:**
1. **float16 사용:** 모델 로딩 시 `torch_dtype=torch.float16` 사용
2. **명시적 메모리 해제:** 각 모델 실행 후 `del model, tokenizer` 및 `torch.cuda.empty_cache()`
3. **순차 실행:** 모델을 동시에 로드하지 않고 순차적으로 실행

**추가 최적화 가능:**
1. **Quantization:** INT8 양자화 고려 (품질 저하 가능성)
2. **모델 최적화:** ONNX 변환 등 고려
3. **배치 처리:** 가능한 경우 배치 단위로 처리

### 7.2 속도 최적화

**현재 적용:**
1. **KV Cache 활용:** `model.generate()` 내부에서 자동 활용
2. **적절한 max_new_tokens:** 512 토큰으로 제한

**추가 최적화 가능:**
1. **Flash Attention:** 지원되는 경우 활성화
2. **Tensor Parallelism:** 멀티 GPU 환경에서 활용
3. **모델 최적화:** TensorRT 등 활용

### 7.3 품질 최적화

**현재 적용:**
1. **프롬프트 엔지니어링:** 역할 기반 프롬프트 사용
2. **Temperature 조정:** 0.1로 낮은 온도 설정 (일관된 답변)

**추가 최적화 가능:**
1. **Few-shot 예시:** 복잡한 질문에 대해 예시 추가
2. **프롬프트 개선:** 선정된 모델에 최적화된 프롬프트 탐색
3. **후처리:** 생성된 답변의 품질 검증 및 개선

---

## 8. 배포 고려사항

### 8.1 프로덕션 환경

**권장 사양:**
- **GPU:** NVIDIA A100 40GB 이상
- **메모리:** 64GB RAM 이상
- **디스크:** SSD 500GB 이상

**배포 방식:**
- **로컬 배포:** GPU 서버에 직접 배포
- **컨테이너화:** Docker 컨테이너로 배포 (권장)
- **API 서버:** FastAPI 또는 Flask로 API 서버 구축

### 8.2 모니터링

**필수 모니터링 항목:**
1. **성능 지표:**
   - 응답 생성 시간
   - 성공률
   - GPU 메모리 사용량
2. **품질 지표:**
   - 사용자 피드백
   - 답변 정확도
   - 할루시네이션 발생률

### 8.3 확장성

**수평 확장:**
- 멀티 GPU 환경에서 Tensor Parallelism 활용
- 로드 밸런싱을 통한 여러 인스턴스 운영

**수직 확장:**
- 더 큰 GPU 메모리 활용
- 모델 양자화로 메모리 사용량 감소

---

## 9. 향후 개선 계획

### 9.1 추가 평가

1. **전체 Validation Set 테스트:**
   - 현재 49개 샘플로 평가 완료
   - 전체 2,223개 샘플로 확장 테스트 예정

2. **RAGAS 평가:**
   - Answer Relevance, Faithfulness, Context Recall 등 정량 평가
   - LLM-as-a-Judge 평가 추가

### 9.2 프롬프트 엔지니어링

1. **프롬프트 최적화:**
   - 선정된 모델에 최적화된 프롬프트 탐색
   - Few-shot 예시 추가

2. **템플릿 개선:**
   - 질문 유형별 맞춤 프롬프트 템플릿
   - 동적 프롬프트 생성

### 9.3 성능 모니터링

1. **실제 서비스 환경:**
   - 프로덕션 환경에서의 성능 모니터링
   - 사용자 피드백 수집 및 분석

2. **지속적 개선:**
   - A/B 테스트를 통한 프롬프트 개선
   - 모델 업데이트 및 재평가

---

## 10. 참고 자료

### 10.1 관련 문서

- `README_LLM_Selection_Log.md`: LLM 모델 선정 벤치마크 로그
- `README_Retrieval_Selection_Log.md`: 리트리버 선정 로그
- `SPEC_Retrieval_Strategy.md`: 리트리버 전략 기술 명세서

### 10.2 평가 데이터셋

- `validation_set_20.json`: Validation Set (2,223개 질문)
- `results/llm_benchmark_responses.csv`: 벤치마크 결과
- `results/llm_model_comparison.json`: 모델 비교 데이터

### 10.3 코드 파일

- `benchmark_llm_selection.py`: LLM 벤치마크 스크립트
- `test_selected_model.py`: 선정된 모델 전체 테스트 스크립트

---

**작성자:** AI Assistant  
**최종 업데이트:** 2025-11-23 14:00  
**상태:** ✅ 벤치마크 완료, 최종 모델 선정 반영
