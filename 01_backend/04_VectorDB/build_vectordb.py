"""
ChromaDB 벡터 데이터베이스 구축 스크립트
- 기존에 생성된 임베딩 데이터를 ChromaDB에 적재
- CUDA 환경에서 최적화된 처리
"""
import json
import chromadb
from chromadb.config import Settings
from pathlib import Path
from tqdm import tqdm
import time
import torch
from sentence_transformers import SentenceTransformer

# -----------------------------------------------------------------------------
# 설정
# -----------------------------------------------------------------------------
BASE_DIR = Path("/home/pencilfoxs/00_new/History_Docent")
EMBEDDING_FILE = BASE_DIR / "03_Embedding/output/chunks_with_embeddings.json"
VECTORDB_DIR = BASE_DIR / "04_VectorDB/chroma_db"
COLLECTION_NAME = "korean_history_chunks"

# -----------------------------------------------------------------------------
# 메인 로직
# -----------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("ChromaDB 벡터 데이터베이스 구축 시작")
    print("=" * 60)
    
    # 1. 임베딩 데이터 로드
    print(f"\n📂 임베딩 파일 로딩 중: {EMBEDDING_FILE}")
    start_time = time.time()
    
    with open(EMBEDDING_FILE, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    load_time = time.time() - start_time
    print(f"   ✅ {len(chunks)}개 청크 로드 완료 (소요 시간: {load_time:.2f}초)")
    
    # 2. 데이터 구조 확인 및 검증
    print(f"\n🔍 데이터 구조 검증 중...")
    sample = chunks[0]
    print(f"   - 텍스트 길이: {len(sample['text'])}자")
    print(f"   - 임베딩 차원: {len(sample['embedding'])}차원")
    print(f"   - 메타데이터: {sample['metadata']}")
    print(f"   - 청크 ID: {sample.get('chunk_id', 'N/A')}")
    
    # 3. ChromaDB 클라이언트 초기화
    print(f"\n🗄️  ChromaDB 클라이언트 초기화 중...")
    VECTORDB_DIR.mkdir(parents=True, exist_ok=True)
    
    client = chromadb.PersistentClient(
        path=str(VECTORDB_DIR),
        settings=Settings(
            anonymized_telemetry=False,  # 텔레메트리 비활성화
            allow_reset=True
        )
    )
    
    # 4. Collection 생성 또는 기존 Collection 사용
    print(f"\n📚 Collection 생성/로드 중: '{COLLECTION_NAME}'")
    
    try:
        # 기존 Collection이 있으면 삭제하고 재생성 (깨끗한 상태로 시작)
        try:
            client.delete_collection(name=COLLECTION_NAME)
            print(f"   기존 Collection 삭제 완료")
        except:
            pass  # 없으면 무시
        
        collection = client.create_collection(
            name=COLLECTION_NAME,
            metadata={"description": "한국사 RAG 시스템용 벡터 데이터베이스"}
        )
        print(f"   ✅ 새 Collection 생성 완료")
    except Exception as e:
        print(f"   ⚠️  Collection 생성 오류: {e}")
        collection = client.get_collection(name=COLLECTION_NAME)
        print(f"   ✅ 기존 Collection 로드 완료")
    
    # 5. 데이터 준비 (ChromaDB 형식으로 변환)
    print(f"\n🔄 데이터 변환 중...")
    start_time = time.time()
    
    ids = []
    documents = []
    embeddings = []
    metadatas = []
    
    for chunk in tqdm(chunks, desc="데이터 변환"):
        # ID: chunk_id가 있으면 사용, 없으면 인덱스 기반 생성
        chunk_id = chunk.get('chunk_id', f"chk_{len(ids):06d}")
        ids.append(chunk_id)
        
        # Document: 텍스트
        documents.append(chunk['text'])
        
        # Embedding: 벡터 리스트
        embeddings.append(chunk['embedding'])
        
        # Metadata: 메타데이터 (ChromaDB는 dict만 허용)
        metadata = chunk.get('metadata', {})
        # ChromaDB는 메타데이터 값이 str, int, float만 허용
        clean_metadata = {}
        for k, v in metadata.items():
            if isinstance(v, (str, int, float)):
                clean_metadata[k] = v
            else:
                clean_metadata[k] = str(v)  # 다른 타입은 문자열로 변환
        
        metadatas.append(clean_metadata)
    
    convert_time = time.time() - start_time
    print(f"   ✅ 변환 완료 (소요 시간: {convert_time:.2f}초)")
    
    # 6. ChromaDB에 배치 적재
    print(f"\n💾 ChromaDB에 데이터 적재 중...")
    start_time = time.time()
    
    # ChromaDB는 배치 크기 제한이 있으므로, 1000개씩 나눠서 추가
    BATCH_SIZE = 1000
    total_batches = (len(ids) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for i in tqdm(range(0, len(ids), BATCH_SIZE), desc="배치 적재", total=total_batches):
        batch_ids = ids[i:i+BATCH_SIZE]
        batch_documents = documents[i:i+BATCH_SIZE]
        batch_embeddings = embeddings[i:i+BATCH_SIZE]
        batch_metadatas = metadatas[i:i+BATCH_SIZE]
        
        collection.add(
            ids=batch_ids,
            documents=batch_documents,
            embeddings=batch_embeddings,
            metadatas=batch_metadatas
        )
    
    load_time = time.time() - start_time
    print(f"   ✅ 적재 완료 (소요 시간: {load_time:.2f}초)")
    
    # 7. 검증: Collection 통계 확인
    print(f"\n📊 Collection 통계 확인 중...")
    count = collection.count()
    print(f"   - 저장된 문서 수: {count}개")
    
    # 샘플 검색 테스트 (BGE-m3 모델로 쿼리 임베딩 생성)
    print(f"\n🔍 샘플 검색 테스트 중...")
    test_query = "세종대왕"
    
    # BGE-m3 모델 로드 (CUDA 사용)
    print(f"   BGE-m3 모델 로딩 중 (CUDA 사용)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("BAAI/bge-m3", device=device)
    
    # 쿼리 임베딩 생성
    query_embedding = model.encode(
        test_query,
        normalize_embeddings=True,
        show_progress_bar=False
    ).tolist()
    
    print(f"   쿼리 임베딩 생성 완료 (차원: {len(query_embedding)})")
    
    # 임베딩으로 검색
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )
    
    print(f"   테스트 쿼리: '{test_query}'")
    print(f"   검색 결과: {len(results['ids'][0])}개 문서 발견")
    if results['ids'][0]:
        print(f"   - 1위 문서 ID: {results['ids'][0][0]}")
        print(f"   - 1위 문서 일부: {results['documents'][0][0][:100]}...")
        print(f"   - 1위 문서 거리: {results['distances'][0][0]:.4f}")
    
    # 8. 완료 요약
    total_time = time.time() - start_time
    print(f"\n" + "=" * 60)
    print("✅ ChromaDB 벡터 데이터베이스 구축 완료!")
    print("=" * 60)
    print(f"📁 DB 경로: {VECTORDB_DIR}")
    print(f"📚 Collection 이름: {COLLECTION_NAME}")
    print(f"📊 저장된 문서 수: {count}개")
    print(f"⏱️  총 소요 시간: {load_time:.2f}초")
    print("=" * 60)

if __name__ == "__main__":
    main()

