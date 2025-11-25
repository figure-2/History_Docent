"""
형태소 분석기 성능 비교 벤치마크 (Tokenizer Benchmark)
- 대상: Kiwi, Okt, Kkma, Hannanum (4개)
- 목적: 한국사 RAG 시스템의 BM25 검색에 가장 적합한 토크나이저 선정
- 평가 지표: Recall@1, Recall@5, 인덱싱 속도, 검색 속도
"""
import time
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from rank_bm25 import BM25Okapi
import chromadb
from chromadb.config import Settings
import random

# 형태소 분석기 라이브러리 임포트
try:
    from kiwipiepy import Kiwi
    KIWI_AVAILABLE = True
except ImportError:
    print("❌ kiwipiepy 설치 필요: pip install kiwipiepy")
    KIWI_AVAILABLE = False

try:
    from konlpy.tag import Okt
    OKT_AVAILABLE = True
except ImportError:
    print("❌ konlpy 설치 필요: pip install konlpy")
    OKT_AVAILABLE = False

try:
    from konlpy.tag import Kkma
    KKMA_AVAILABLE = True
except ImportError:
    print("❌ konlpy 설치 필요: pip install konlpy")
    KKMA_AVAILABLE = False

try:
    from konlpy.tag import Hannanum
    HANNANUM_AVAILABLE = True
except ImportError:
    print("❌ konlpy 설치 필요: pip install konlpy")
    HANNANUM_AVAILABLE = False

# -----------------------------------------------------------------------------
# 설정
# -----------------------------------------------------------------------------
BASE_DIR = Path("/home/pencilfoxs/00_new/History_Docent")
VECTORDB_DIR = BASE_DIR / "04_VectorDB/chroma_db"
COLLECTION_NAME = "korean_history_chunks"
BENCHMARK_DATA = BASE_DIR / "03_Embedding/data/korean_history_benchmark_2000.json"
OUTPUT_REPORT = BASE_DIR / "05_Retrieval_Optimization/tokenizer_benchmark_result.md"

SAMPLE_SIZE = 50  # 평가용 샘플 수

# -----------------------------------------------------------------------------
# 토크나이저 래퍼 클래스
# -----------------------------------------------------------------------------
class TokenizerWrapper:
    def __init__(self, name):
        self.name = name
        if name == "Kiwi":
            self.processor = Kiwi()
        elif name == "Okt":
            self.processor = Okt()
        elif name == "Kkma":
            self.processor = Kkma()
        elif name == "Hannanum":
            self.processor = Hannanum()
            
    def tokenize(self, text):
        """텍스트를 토큰화하여 리스트 반환"""
        if self.name == "Kiwi":
            # 명사, 동사, 형용사 등 실질 형태소 추출
            tokens = [token.form for token in self.processor.tokenize(text)]
            # 빈 문자열 제거
            return [t for t in tokens if t.strip()]
        elif self.name == "Okt":
            # Okt는 morphs 사용 (어간 추출 포함)
            tokens = self.processor.morphs(text, stem=True)
            return [t for t in tokens if t.strip()]
        elif self.name == "Kkma":
            # Kkma는 morphs 사용
            tokens = self.processor.morphs(text)
            return [t for t in tokens if t.strip()]
        elif self.name == "Hannanum":
            # Hannanum은 morphs 사용
            tokens = self.processor.morphs(text)
            return [t for t in tokens if t.strip()]
        return []

# -----------------------------------------------------------------------------
# 벤치마크 함수
# -----------------------------------------------------------------------------
def run_benchmark(tokenizer_name, documents, doc_ids, samples):
    print(f"\n🚀 [{tokenizer_name}] 벤치마크 시작...")
    tokenizer = TokenizerWrapper(tokenizer_name)
    
    # 1. 인덱싱 속도 측정
    print(f"   📝 토큰화 진행 중...")
    start_time = time.time()
    tokenized_corpus = []
    for doc in tqdm(documents, desc=f"   [{tokenizer_name}] 토큰화", leave=False):
        tokens = tokenizer.tokenize(doc)
        tokenized_corpus.append(tokens)
    
    print(f"   🧮 BM25 인덱싱 중...")
    bm25 = BM25Okapi(tokenized_corpus)
    indexing_time = time.time() - start_time
    print(f"   ⏱️  인덱싱 소요 시간: {indexing_time:.2f}초")
    
    # 2. 검색 성능 측정
    hits_1 = 0
    hits_5 = 0
    search_times = []
    
    print(f"   🔍 검색 평가 진행 중...")
    for sample in tqdm(samples, desc=f"   [{tokenizer_name}] 검색", leave=False):
        query = sample['query']
        gold_id = sample['chunk_id']
        
        start_search = time.time()
        tokenized_query = tokenizer.tokenize(query)
        scores = bm25.get_scores(tokenized_query)
        
        # 상위 5개 추출
        top_n_indices = np.argsort(scores)[::-1][:5]
        top_ids = [doc_ids[i] for i in top_n_indices]
        
        search_times.append(time.time() - start_search)
        
        if gold_id in top_ids[:1]:
            hits_1 += 1
        if gold_id in top_ids:
            hits_5 += 1
            
    avg_search_time = sum(search_times) / len(search_times) if search_times else 0
    recall_1 = hits_1 / len(samples) * 100 if samples else 0
    recall_5 = hits_5 / len(samples) * 100 if samples else 0
    
    print(f"   📊 결과: Recall@1={recall_1:.1f}%, Recall@5={recall_5:.1f}%")
    print(f"   ⚡ 평균 검색 시간: {avg_search_time*1000:.2f}ms")
    
    return {
        "name": tokenizer_name,
        "indexing_time": indexing_time,
        "recall_1": recall_1,
        "recall_5": recall_5,
        "avg_search_time": avg_search_time * 1000
    }

# -----------------------------------------------------------------------------
# 메인 로직
# -----------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("형태소 분석기 BM25 성능 비교 벤치마크")
    print("=" * 60)
    
    # 1. 데이터 로드 (문서 & 평가 데이터)
    print("\n📂 데이터 로딩 중...")
    
    # ChromaDB에서 전체 문서 로드
    client = chromadb.PersistentClient(
        path=str(VECTORDB_DIR), 
        settings=Settings(anonymized_telemetry=False)
    )
    collection = client.get_collection(name=COLLECTION_NAME)
    all_data = collection.get()
    documents = all_data['documents']
    doc_ids = all_data['ids']
    print(f"   ✅ 문서 {len(documents)}개 로드 완료")
    
    # 평가 데이터 로드 (50개 샘플링)
    with open(BENCHMARK_DATA, 'r', encoding='utf-8') as f:
        all_samples = json.load(f)
    random.seed(42)
    samples = random.sample(all_samples, min(SAMPLE_SIZE, len(all_samples)))
    print(f"   ✅ 평가용 샘플 {len(samples)}개 준비 완료")
    
    # 2. 비교 대상 설정 (4개)
    candidates = []
    if KIWI_AVAILABLE:
        candidates.append("Kiwi")
    if OKT_AVAILABLE:
        candidates.append("Okt")
    if KKMA_AVAILABLE:
        candidates.append("Kkma")
    if HANNANUM_AVAILABLE:
        candidates.append("Hannanum")
    
    if not candidates:
        print("❌ 사용 가능한 형태소 분석기가 없습니다.")
        return
    
    print(f"\n📋 비교 대상 ({len(candidates)}개): {', '.join(candidates)}")
    
    results = []
    
    # 3. 벤치마크 실행
    for name in candidates:
        try:
            res = run_benchmark(name, documents, doc_ids, samples)
            results.append(res)
        except Exception as e:
            print(f"   ❌ {name} 실행 실패: {e}")
            import traceback
            traceback.print_exc()
            
    # 4. 결과 리포트 작성
    print("\n" + "="*60)
    print("🏆 최종 벤치마크 결과")
    print("="*60)
    print(f"{'Tokenizer':<10} | {'Recall@1':<10} | {'Recall@5':<10} | {'Indexing(s)':<12} | {'Search(ms)':<10}")
    print("-" * 65)
    
    with open(OUTPUT_REPORT, 'w', encoding='utf-8') as f:
        f.write("# 형태소 분석기 BM25 성능 비교\n\n")
        f.write(f"- 평가 일시: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- 평가 샘플 수: {len(samples)}개\n")
        f.write(f"- 전체 문서 수: {len(documents)}개\n\n")
        f.write("| Tokenizer | Recall@1 | Recall@5 | Indexing(s) | Search(ms) |\n")
        f.write("|---|---|---|---|---|\n")
        
        for r in results:
            line = f"{r['name']:<10} | {r['recall_1']:.1f}%     | {r['recall_5']:.1f}%     | {r['indexing_time']:.2f}s       | {r['avg_search_time']:.2f}ms"
            print(line)
            f.write(f"| **{r['name']}** | {r['recall_1']:.1f}% | {r['recall_5']:.1f}% | {r['indexing_time']:.2f}s | {r['avg_search_time']:.2f}ms |\n")
        
        # 승자 선정
        if len(results) >= 2:
            winner = max(results, key=lambda x: x['recall_1'])
            f.write(f"\n## 🏆 최종 선정: **{winner['name']}**\n\n")
            f.write(f"- Recall@1 기준 최고 성능: {winner['recall_1']:.1f}%\n")
            f.write(f"- Recall@5: {winner['recall_5']:.1f}%\n")
            f.write(f"- 인덱싱 시간: {winner['indexing_time']:.2f}초\n")
            f.write(f"- 평균 검색 시간: {winner['avg_search_time']:.2f}ms\n")
            
            print(f"\n🏆 최종 선정: {winner['name']} (Recall@1: {winner['recall_1']:.1f}%)")
            
    print(f"\n💾 결과 저장 완료: {OUTPUT_REPORT}")

if __name__ == "__main__":
    main()

