#!/usr/bin/env python3
"""
History Docent - 통합 RAG 시스템
- 검색(Retrieval): BM25 + Reranker (05_Retrieval_Optimization 모듈 사용)
- 생성(Generation): Bllossom-8B LLM
"""
import sys
import os
import time
import torch

# 상위 폴더(05_Retrieval_Optimization)에서 모듈을 가져오기 위해 경로 추가 (import 전에 실행)
sys.path.append(os.path.join(os.path.dirname(__file__), "05_Retrieval_Optimization"))

from retrieval_system import RetrievalSystem
from vllm import LLM, SamplingParams  # transformers 대신 vLLM 사용

class HistoryDocent:
    def __init__(self):
        self.retrieval_system = RetrievalSystem()
        self.model_id = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
        self.model = None
        self.history = []
        
        # vLLM Sampling Parameters 설정
        self.sampling_params = SamplingParams(
            temperature=0.1,  # 창의성 조절 (낮을수록 일관됨)
            top_p=0.9,
            max_tokens=512,
            stop=["<|end_of_text|>", "<|eot_id|>"]  # 종료 토큰 설정
        )

    def initialize(self):
        """시스템 초기화: 검색 시스템 및 vLLM 모델 로드"""
        if self.model is not None:
            print("ℹ️ 모델이 이미 로드되었습니다.")
            return

        print("🚀 History Docent 시스템 초기화 중...")
        
        # 1. 검색 시스템 초기화
        self.retrieval_system.initialize()
        
        # 2. vLLM 모델 로드
        print(f"🤖 vLLM 모델 로드 중: {self.model_id}")
        # GPU 메모리 점유율을 0.6으로 제한 (검색 시스템도 GPU 사용하므로 여유 확보)
        self.model = LLM(
            model=self.model_id, 
            dtype="float16",
            gpu_memory_utilization=0.6,  # 0.9 → 0.6으로 낮춤 (다른 프로세스 고려)
            tensor_parallel_size=1  # 단일 GPU 사용
        )
        print("✅ 시스템 준비 완료!")

    def generate_prompt(self, query: str, contexts: list) -> str:
        """RAG 프롬프트 생성"""
        # contexts는 RetrievalResult 객체 리스트이므로 속성으로 접근
        context_str = "\n\n".join([f"- {ctx.text}" for ctx in contexts])
        
        # Bllossom 모델 프롬프트 포맷 준수
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

당신은 한국사 전문가 'History Docent'입니다. 아래 [검색 결과]를 바탕으로 질문에 답변해주세요.
[검색 결과]에 없는 내용은 지어내지 말고, 모른다고 답변하세요.

[검색 결과]
{context_str}<|eot_id|><|start_header_id|>user<|end_header_id|>

{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        return prompt

    def chat(self, query: str) -> dict:
        """사용자 질문에 대한 답변 생성 (RAG)"""
        if not self.model:
            self.initialize()
            
        start_time = time.time()
        
        # 1. 문서 검색
        print(f"🔍 검색 수행 중: {query}")
        search_results = self.retrieval_system.search(query, final_k=3)
        
        # 2. 프롬프트 생성
        prompt = self.generate_prompt(query, search_results)
        
        # 3. 답변 생성 (vLLM)
        print("🤔 답변 생성 중...")
        outputs = self.model.generate([prompt], self.sampling_params)
        generated_text = outputs[0].outputs[0].text.strip()
        
        elapsed_time = time.time() - start_time
        print(f"⏱️ 총 소요 시간: {elapsed_time:.2f}초")

        return {
            "answer": generated_text,
            "sources": [res.text[:100] + "..." for res in search_results],
            "latency": round(elapsed_time, 2)
        }

if __name__ == "__main__":
    # 간단한 테스트
    docent = HistoryDocent()
    docent.initialize()
    
    test_query = "손기정 선수는 어떤 올림픽에서 금메달을 땄나요?"
    result = docent.chat(test_query)
    
    print("\n" + "="*50)
    print(f"질문: {test_query}")
    print(f"답변: {result['answer']}")
    print(f"시간: {result['latency']}초")
    print("="*50)

