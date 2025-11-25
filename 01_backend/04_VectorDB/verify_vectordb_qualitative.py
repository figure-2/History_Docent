"""
ChromaDB 정성 평가 스크립트 (Qualitative Analysis)
- 전체 데이터셋에서 50개 샘플을 추출하여 심층 분석 수행
- 검색 결과의 품질(Relevance)을 육안으로 확인할 수 있도록 상세 로그 출력
"""
import chromadb
from chromadb.config import Settings
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer
import json
import random
import time

# -----------------------------------------------------------------------------
# 설정
# -----------------------------------------------------------------------------
BASE_DIR = Path("/home/pencilfoxs/00_new/History_Docent")
VECTORDB_DIR = BASE_DIR / "04_VectorDB/chroma_db"
COLLECTION_NAME = "korean_history_chunks"
MODEL_NAME = "BAAI/bge-m3"
BENCHMARK_DATA = BASE_DIR / "03_Embedding/data/korean_history_benchmark_2000.json"
OUTPUT_REPORT = BASE_DIR / "04_VectorDB/qualitative_analysis_report.txt"

SAMPLE_SIZE = 50

# -----------------------------------------------------------------------------
# 메인 로직
# -----------------------------------------------------------------------------
def main():
    print("=" * 60)
    print(f"ChromaDB 정성 평가 시작 (샘플 {SAMPLE_SIZE}개)")
    print("=" * 60)
    
    # 1. 데이터셋 로드 및 샘플링
    print(f"\n📂 벤치마크 데이터셋 로드 중: {BENCHMARK_DATA}")
    with open(BENCHMARK_DATA, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"   총 {len(data)}개 데이터 중 {SAMPLE_SIZE}개 샘플링...")
    
    # 소스별로 골고루 섞어서 뽑기 위해 셔플 후 선택
    random.seed(42)  # 재현성을 위해 시드 고정
    random.shuffle(data)
    samples = data[:SAMPLE_SIZE]
    
    # 2. ChromaDB 및 모델 로드
    print(f"\n📂 ChromaDB 및 모델 로드 중...")
    client = chromadb.PersistentClient(
        path=str(VECTORDB_DIR),
        settings=Settings(anonymized_telemetry=False)
    )
    collection = client.get_collection(name=COLLECTION_NAME)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(MODEL_NAME, device=device)
    print(f"   ✅ 로드 완료 (Device: {device})")
    
    # 3. 평가 실행 및 리포트 작성
    print(f"\n🚀 정성 평가 실행 중...")
    
    with open(OUTPUT_REPORT, 'w', encoding='utf-8') as report:
        report.write(f"# ChromaDB 정성 평가 보고서\n")
        report.write(f"- 일시: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.write(f"- 모델: {MODEL_NAME}\n")
        report.write(f"- 샘플 수: {SAMPLE_SIZE}개\n")
        report.write("-" * 80 + "\n\n")
        
        success_count = 0
        rank1_count = 0
        rank3_count = 0
        rank5_count = 0
        
        for i, sample in enumerate(samples, 1):
            query = sample['query']
            gold_chunk_id = sample['chunk_id']
            
            # 쿼리 임베딩
            query_embedding = model.encode(query, normalize_embeddings=True).tolist()
            
            # 검색 (Top 5)
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=5
            )
            
            # 결과 분석
            retrieved_ids = results['ids'][0]
            distances = results['distances'][0]
            documents = results['documents'][0]
            metadatas = results['metadatas'][0]
            
            is_success = gold_chunk_id in retrieved_ids
            if is_success:
                success_count += 1
                rank = retrieved_ids.index(gold_chunk_id) + 1
                if rank == 1:
                    rank1_count += 1
                if rank <= 3:
                    rank3_count += 1
                if rank <= 5:
                    rank5_count += 1
                status_icon = "✅ 성공"
            else:
                rank = "X"
                status_icon = "❌ 실패"
                
            # 터미널 출력 (진행 상황)
            print(f"[{i}/{SAMPLE_SIZE}] {status_icon} (Rank: {rank}) - {query[:40]}...")

            # 리포트 작성
            report.write(f"## Case {i}: {status_icon} (Rank: {rank})\n")
            report.write(f"**Q:** {query}\n")
            report.write(f"**정답 ID:** {gold_chunk_id}\n\n")
            
            report.write(f"**검색 결과 (Top 3):**\n")
            for j in range(3):  # 상위 3개만 상세 출력
                if j >= len(retrieved_ids):
                    break
                
                rid = retrieved_ids[j]
                dist = distances[j]
                doc = documents[j].replace("\n", " ")[:150]  # 줄바꿈 제거 및 길이 제한
                source = metadatas[j].get('source', 'Unknown')
                page = metadatas[j].get('page', 'N/A')
                
                match_mark = "👈 정답" if rid == gold_chunk_id else ""
                report.write(f"{j+1}. [{rid}] (Dist: {dist:.4f}) [{source}, p.{page}] - {doc}... {match_mark}\n")
            
            report.write("\n" + "-" * 40 + "\n\n")
            
        # 최종 요약
        accuracy = (success_count / SAMPLE_SIZE) * 100
        recall1 = (rank1_count / SAMPLE_SIZE) * 100
        recall3 = (rank3_count / SAMPLE_SIZE) * 100
        recall5 = (rank5_count / SAMPLE_SIZE) * 100
        
        summary = f"\n# 최종 요약\n"
        summary += f"- 총 샘플: {SAMPLE_SIZE}개\n"
        summary += f"- Recall@1: {rank1_count}개 ({recall1:.1f}%)\n"
        summary += f"- Recall@3: {rank3_count}개 ({recall3:.1f}%)\n"
        summary += f"- Recall@5: {success_count}개 ({accuracy:.1f}%)\n"
        report.write(summary)
        print("\n" + summary)

    print(f"\n💾 보고서 저장 완료: {OUTPUT_REPORT}")

if __name__ == "__main__":
    main()

