import json
import torch
from sentence_transformers import SentenceTransformer
from pathlib import Path
from tqdm import tqdm
import numpy as np

# -----------------------------------------------------------------------------
# 설정
# -----------------------------------------------------------------------------
BASE_DIR = Path("/home/pencilfoxs/00_new/History_Docent")
CHUNK_FILE = BASE_DIR / "02_Chunking/output/all_chunks.json"
OUTPUT_DIR = BASE_DIR / "03_Embedding/output"
OUTPUT_FILE = OUTPUT_DIR / "chunks_with_embeddings.json"

# 선정된 모델
MODEL_NAME = "BAAI/bge-m3"
BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------------------------------------------------------
# 메인 로직
# -----------------------------------------------------------------------------
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 모델 로드
    print(f"📦 모델 로딩 중: {MODEL_NAME} (Device: {DEVICE})")
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    print("✅ 모델 로드 완료")
    
    # 청크 데이터 로드
    print(f"\n📂 청크 파일 로딩 중: {CHUNK_FILE}")
    with open(CHUNK_FILE, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    print(f"   총 {len(chunks)}개 청크 로드 완료")
    
    # 텍스트 추출
    texts = [chunk['text'] for chunk in chunks]
    
    # 임베딩 생성
    print(f"\n🚀 임베딩 생성 중: {len(texts)}개 청크...")
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=BATCH_SIZE,
        show_progress_bar=True
    )
    
    print(f"✅ 임베딩 생성 완료 (Shape: {embeddings.shape})")
    
    # 결과 저장
    print(f"\n💾 결과 저장 중...")
    results = []
    for i, chunk in enumerate(chunks):
        results.append({
            **chunk,
            "embedding": embeddings[i].tolist()  # numpy array를 list로 변환
        })
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 저장 완료: {OUTPUT_FILE}")
    print(f"   총 {len(results)}개 청크의 임베딩이 저장되었습니다.")

if __name__ == "__main__":
    main()

