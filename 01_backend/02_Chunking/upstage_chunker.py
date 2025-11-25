import json
import re
from pathlib import Path
from tqdm import tqdm
from bs4 import BeautifulSoup

# -----------------------------------------------------------------------------
# 설정 (Environment Config)
# -----------------------------------------------------------------------------
BASE_DIR = Path("/home/pencilfoxs/00_new/History_Docent")
INPUT_DIR = BASE_DIR / "01_Data_Preprocessing"
OUTPUT_DIR = BASE_DIR / "02_Chunking/output"
MAX_CHUNK_SIZE = 1000
MAX_FIGURE_SIZE = 500  # 이미지 OCR 최대 길이 (깨진 텍스트 방지)
MAX_TEXT_SIZE = 1000   # 텍스트 청크 최대 길이 (임베딩 모델 제한 고려)

# -----------------------------------------------------------------------------
# 유틸리티 함수 (Utility)
# -----------------------------------------------------------------------------
def clean_text(text):
    if not text: return ""
    return re.sub(r'\s+', ' ', text).strip()

def split_long_text(text, max_size=MAX_TEXT_SIZE, overlap=100):
    """
    긴 텍스트를 문장 단위로 재분할 (Gemini 방식 적용)
    - 문장 구분자: . ? ! \n\n
    - overlap을 적용하여 문맥 보존
    """
    if len(text) <= max_size:
        return [text]
    
    # 문장 구분자로 분할
    sentences = re.split(r'([.!?]\s+|\n\n)', text)
    # 구분자와 문장을 다시 결합
    parts = []
    for i in range(0, len(sentences), 2):
        if i + 1 < len(sentences):
            parts.append(sentences[i] + sentences[i+1])
        else:
            parts.append(sentences[i])
    
    chunks = []
    current_chunk = ""
    
    for part in parts:
        # 현재 청크에 추가했을 때 길이 확인
        if len(current_chunk) + len(part) <= max_size:
            current_chunk += part
        else:
            # 현재 청크 저장
            if current_chunk:
                chunks.append(current_chunk)
            # overlap 적용: 이전 청크의 마지막 부분을 가져옴
            if overlap > 0 and chunks:
                overlap_text = chunks[-1][-overlap:] if len(chunks[-1]) > overlap else chunks[-1]
                current_chunk = overlap_text + part
            else:
                current_chunk = part
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks if chunks else [text]

def parse_json_structure(json_path):
    """
    [전략: 구조 기반 청킹]
    - 단순 텍스트 나열이 아닌, Heading을 추적하여 문맥(Context)을 주입함.
    - 이미지/표는 별도 타입으로 분류하여 검색 정확도 향상.
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {json_path}: {e}")
        return []

    elements = data.get("elements", []) if isinstance(data, dict) else data
    chunks = []
    
    current_heading = "무제"
    buffer_text = ""
    buffer_page = 0
    source_name = json_path.stem.split("_")[0] # 파일명 간소화

    for element in elements:
        category = element.get("category", "")
        page_num = element.get("page", 0)
        
        # content 필드 구조: {"html": "...", "text": "...", "markdown": "..."}
        content_dict = element.get("content", {})
        if isinstance(content_dict, dict):
            html_content = content_dict.get("html", "")
            text_content = content_dict.get("text", "")
        else:
            # fallback: 직접 html, text 필드 확인
            html_content = element.get("html", "")
            text_content = element.get("text", "")

        # 1. HTML 태그 내 텍스트/OCR 추출
        if not text_content and html_content:
            soup = BeautifulSoup(html_content, "html.parser")
            if category == "figure":
                img = soup.find("img")
                if img and img.get("alt"):
                    text_content = img['alt']
            else:
                text_content = soup.get_text(strip=True)
        
        text_content = clean_text(text_content)
        if not text_content: continue

        # 2. 청킹 로직 (Heading Tracking)
        if category in ["header", "footer", "page_number"]:
            continue # 노이즈 제거

        elif category.startswith("heading"):
            # 이전 문맥 저장
            if buffer_text:
                chunks.append({
                    "text": f"[{current_heading}] {buffer_text}",
                    "metadata": {"source": source_name, "page": buffer_page, "type": "text"}
                })
                buffer_text = ""
            current_heading = text_content
            buffer_page = page_num

        elif category in ["paragraph", "list", "caption"]:
            if not buffer_text: buffer_page = page_num
            
            if buffer_text: buffer_text += " " + text_content
            else: buffer_text = text_content
            
            if len(buffer_text) >= MAX_CHUNK_SIZE:
                chunks.append({
                    "text": f"[{current_heading}] {buffer_text}",
                    "metadata": {"source": source_name, "page": buffer_page, "type": "text"}
                })
                buffer_text = ""

        elif category in ["table", "figure"]:
            # 독립 청크 생성
            if buffer_text:
                chunks.append({
                    "text": f"[{current_heading}] {buffer_text}",
                    "metadata": {"source": source_name, "page": buffer_page, "type": "text"}
                })
                buffer_text = ""
            
            # 이미지 OCR 텍스트 길이 제한 (깨진 텍스트 방지)
            if category == "figure" and len(text_content) > MAX_FIGURE_SIZE:
                # 너무 긴 이미지 OCR은 앞부분만 사용 (깨진 텍스트일 가능성 높음)
                text_content = text_content[:MAX_FIGURE_SIZE] + "..."
            
            chunks.append({
                "text": f"[{'표' if category=='table' else '이미지'}] [{current_heading}] {text_content}",
                "metadata": {"source": source_name, "page": page_num, "type": category}
            })

    # 잔여 버퍼 처리
    if buffer_text:
        chunks.append({
            "text": f"[{current_heading}] {buffer_text}",
            "metadata": {"source": source_name, "page": buffer_page, "type": "text"}
        })

    # 최종 검증: 1000자 이상인 텍스트 청크는 재분할
    final_chunks = []
    for chunk in chunks:
        if chunk["metadata"]["type"] == "text" and len(chunk["text"]) > MAX_TEXT_SIZE:
            # 긴 텍스트를 문장 단위로 재분할
            heading_match = re.match(r'^\[([^\]]+)\]\s*(.*)$', chunk["text"])
            if heading_match:
                heading = heading_match.group(1)
                body_text = heading_match.group(2)
                split_bodies = split_long_text(body_text, MAX_TEXT_SIZE, overlap=100)
                
                for split_body in split_bodies:
                    final_chunks.append({
                        "text": f"[{heading}] {split_body}",
                        "metadata": chunk["metadata"].copy()
                    })
            else:
                # 제목이 없는 경우 그냥 분할
                split_texts = split_long_text(chunk["text"], MAX_TEXT_SIZE, overlap=100)
                for split_text in split_texts:
                    final_chunks.append({
                        "text": split_text,
                        "metadata": chunk["metadata"].copy()
                    })
        else:
            final_chunks.append(chunk)

    return final_chunks

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_chunks = []
    
    # 대상 파일 탐색 (재귀)
    target_files = list(INPUT_DIR.glob("**/data/*.json"))
    print(f"Target JSON Files: {len(target_files)}")

    chunk_cnt = 0
    for json_file in tqdm(target_files):
        if "document_analysis" in json_file.name: continue
        
        file_chunks = parse_json_structure(json_file)
        for ch in file_chunks:
            ch["chunk_id"] = f"chk_{chunk_cnt:06d}"
            all_chunks.append(ch)
            chunk_cnt += 1

    output_path = OUTPUT_DIR / "all_chunks.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    
    print(f"Chunking Complete. Saved to {output_path}")

if __name__ == "__main__":
    main()
