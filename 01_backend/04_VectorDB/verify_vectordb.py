"""
ChromaDB 벡터 데이터베이스 검증 스크립트
- 다양한 쿼리로 검색 성능 테스트
- 메타데이터 필터링 테스트
- 성능 측정
"""
import chromadb
from chromadb.config import Settings
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer
import time

# -----------------------------------------------------------------------------
# 설정
# -----------------------------------------------------------------------------
BASE_DIR = Path("/home/pencilfoxs/00_new/History_Docent")
VECTORDB_DIR = BASE_DIR / "04_VectorDB/chroma_db"
COLLECTION_NAME = "korean_history_chunks"
MODEL_NAME = "BAAI/bge-m3"

# 테스트 쿼리 목록
TEST_QUERIES = [
    "세종대왕이 만든 한글",
    "임진왜란",
    "조선왕조실록",
    "정약용의 실학",
    "고려시대 무역"
]

# -----------------------------------------------------------------------------
# 메인 로직
# -----------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("ChromaDB 벡터 데이터베이스 검증 테스트")
    print("=" * 60)
    
    # 1. ChromaDB 클라이언트 및 Collection 로드
    print(f"\n📂 ChromaDB 로드 중...")
    client = chromadb.PersistentClient(
        path=str(VECTORDB_DIR),
        settings=Settings(anonymized_telemetry=False)
    )
    
    collection = client.get_collection(name=COLLECTION_NAME)
    count = collection.count()
    print(f"   ✅ Collection 로드 완료: {count}개 문서")
    
    # 2. BGE-m3 모델 로드 (CUDA 사용)
    print(f"\n🤖 BGE-m3 모델 로드 중 (CUDA 사용)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(MODEL_NAME, device=device)
    print(f"   ✅ 모델 로드 완료 (Device: {device})")
    
    # 3. 기본 검색 테스트
    print(f"\n" + "=" * 60)
    print("1. 기본 검색 테스트")
    print("=" * 60)
    
    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"\n[{i}/{len(TEST_QUERIES)}] 쿼리: '{query}'")
        
        # 쿼리 임베딩 생성
        start_time = time.time()
        query_embedding = model.encode(
            query,
            normalize_embeddings=True,
            show_progress_bar=False
        ).tolist()
        embed_time = time.time() - start_time
        
        # 검색 실행
        start_time = time.time()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5
        )
        search_time = time.time() - start_time
        
        print(f"   ⏱️  임베딩 생성: {embed_time*1000:.2f}ms, 검색: {search_time*1000:.2f}ms")
        print(f"   📊 검색 결과: {len(results['ids'][0])}개 문서")
        
        if results['ids'][0]:
            top_id = results['ids'][0][0]
            top_doc = results['documents'][0][0]
            top_dist = results['distances'][0][0]
            top_meta = results['metadatas'][0][0]
            
            print(f"   🥇 1위:")
            print(f"      - ID: {top_id}")
            print(f"      - 거리: {top_dist:.4f}")
            print(f"      - 소스: {top_meta.get('source', 'N/A')}")
            print(f"      - 페이지: {top_meta.get('page', 'N/A')}")
            print(f"      - 문서 일부: {top_doc[:150]}...")
    
    # 4. 메타데이터 필터링 테스트
    print(f"\n" + "=" * 60)
    print("2. 메타데이터 필터링 테스트")
    print("=" * 60)
    
    test_query = "세종대왕"
    query_embedding = model.encode(
        test_query,
        normalize_embeddings=True,
        show_progress_bar=False
    ).tolist()
    
    # 특정 소스로 필터링
    print(f"\n쿼리: '{test_query}'")
    print("필터: source == '벌거벗은한국사-조선편'")
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        where={"source": "벌거벗은한국사-조선편"}
    )
    
    print(f"   검색 결과: {len(results['ids'][0])}개 문서")
    if results['ids'][0]:
        for i, (doc_id, doc, dist, meta) in enumerate(zip(
            results['ids'][0],
            results['documents'][0],
            results['distances'][0],
            results['metadatas'][0]
        ), 1):
            print(f"   {i}. {doc_id} (거리: {dist:.4f}, 페이지: {meta.get('page', 'N/A')})")
    
    # 5. 성능 벤치마크
    print(f"\n" + "=" * 60)
    print("3. 성능 벤치마크")
    print("=" * 60)
    
    num_tests = 10
    embed_times = []
    search_times = []
    
    print(f"\n{num_tests}회 반복 테스트 중...")
    for i in range(num_tests):
        query = f"테스트 쿼리 {i}"
        
        # 임베딩 생성 시간 측정
        start = time.time()
        query_embedding = model.encode(
            query,
            normalize_embeddings=True,
            show_progress_bar=False
        ).tolist()
        embed_times.append(time.time() - start)
        
        # 검색 시간 측정
        start = time.time()
        collection.query(
            query_embeddings=[query_embedding],
            n_results=10
        )
        search_times.append(time.time() - start)
    
    avg_embed = sum(embed_times) / len(embed_times) * 1000
    avg_search = sum(search_times) / len(search_times) * 1000
    
    print(f"\n   📊 평균 성능:")
    print(f"      - 임베딩 생성: {avg_embed:.2f}ms")
    print(f"      - 벡터 검색: {avg_search:.2f}ms")
    print(f"      - 총 소요 시간: {avg_embed + avg_search:.2f}ms")
    
    # 6. Collection 통계
    print(f"\n" + "=" * 60)
    print("4. Collection 통계")
    print("=" * 60)
    
    # 메타데이터별 문서 수 집계
    all_data = collection.get()
    sources = {}
    types = {}
    
    for meta in all_data['metadatas']:
        source = meta.get('source', 'unknown')
        doc_type = meta.get('type', 'unknown')
        sources[source] = sources.get(source, 0) + 1
        types[doc_type] = types.get(doc_type, 0) + 1
    
    print(f"\n   📚 소스별 문서 수:")
    for source, count in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"      - {source}: {count}개")
    
    print(f"\n   📄 타입별 문서 수:")
    for doc_type, count in sorted(types.items(), key=lambda x: -x[1]):
        print(f"      - {doc_type}: {count}개")
    
    print(f"\n" + "=" * 60)
    print("✅ 검증 테스트 완료!")
    print("=" * 60)

if __name__ == "__main__":
    main()

