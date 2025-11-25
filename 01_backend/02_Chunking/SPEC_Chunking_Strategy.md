# 기술 명세서: 하이브리드 청킹 전략 (Hybrid Chunking Strategy)

**작성 일시:** 2025-11-22 07:14 (초기 작성)  
**최종 수정일:** 2025-11-22 07:52 (하이브리드 방식 반영)

---

## 1. 개요

### 목적
Upstage Document API로 분석된 JSON 파일에서 구조 정보(제목, 문단, 이미지, 표)를 보존하면서, 임베딩 모델의 Context Window 제한을 고려한 안전한 길이로 청크를 생성한다.

### 핵심 원칙
- **문맥 보존 (Context Preservation)**: 제목과 본문을 결합하여 검색 정확도 향상
- **타입 분리 (Type Isolation)**: 텍스트, 이미지, 표를 별도로 처리하여 멀티모달 검색 지원
- **메타데이터 주입**: 출처, 페이지, 타입 정보를 포함하여 추적 가능성 확보
- **길이 제어 (Length Control)**: 구조 기반 청킹의 부작용(초대형 청크)을 2차 분할로 해결

---

## 2. 데이터 스키마

### 입력 데이터 (Input)

**Upstage JSON 구조:**
```json
{
  "elements": [
    {
      "category": "heading1",
      "html": "<h1>태조 이성계의 건국</h1>",
      "text": "태조 이성계의 건국",
      "page": 5
    },
    {
      "category": "paragraph",
      "html": "<p>1392년에 조선이 건국되었습니다.</p>",
      "text": "1392년에 조선이 건국되었습니다.",
      "page": 5
    },
    {
      "category": "figure",
      "html": "<figure><img alt='조선 팔도 지도...' /></figure>",
      "page": 6
    }
  ]
}
```

### 출력 데이터 (Output)

**청크 JSON 구조:**
```json
[
  {
    "chunk_id": "chk_000001",
    "text": "[태조 이성계의 건국] 1392년에 조선이 건국되었습니다.",
    "metadata": {
      "source": "벌거벗은한국사-조선편",
      "page": 5,
      "type": "text"
    }
  },
  {
    "chunk_id": "chk_000002",
    "text": "[이미지] [태조 이성계의 건국] 조선 팔도 지도...",
    "metadata": {
      "source": "벌거벗은한국사-조선편",
      "page": 6,
      "type": "figure"
    }
  }
]
```

---

## 3. 알고리즘 상세

### 3.1. Heading 추적 (Context Tracking)

**로직:**
```python
current_heading = "무제"  # 초기값

for element in elements:
    if element.category.startswith("heading"):
        current_heading = element.text  # 제목 갱신
    
    elif element.category == "paragraph":
        chunk_text = f"[{current_heading}] {element.text}"  # 제목 + 본문
```

**효과:**
- "세종대왕"으로 검색하면, 제목에 "세종대왕"이 포함된 모든 본문이 검색됨
- 문맥이 끊기지 않아 LLM이 더 정확한 답변 생성 가능

### 3.2. 길이 제어 (Length Control) - **하이브리드 핵심**

**1차 청킹 (구조 기반):**
- 기본적으로 구조 기반 청킹 수행
- Heading 추적, 이미지/표 분리 등 구조 보존

**2차 분할 (길이 기반):**
- **텍스트 청크**: 1,000자 초과 시 문장 단위 재분할
  ```python
  if len(chunk["text"]) > MAX_TEXT_SIZE:
      split_bodies = split_long_text(body_text, MAX_TEXT_SIZE, overlap=100)
  ```
- **이미지 OCR**: 500자 초과 시 Truncate (노이즈 방지)
  ```python
  if category == "figure" and len(text_content) > MAX_FIGURE_SIZE:
      text_content = text_content[:MAX_FIGURE_SIZE] + "..."
  ```

**Overlap 적용:**
- 재분할 시 이전 청크의 마지막 100자를 다음 청크 앞에 붙임
- 목적: 문맥 단절 방지, 검색 정확도 유지

### 3.3. 멀티모달 처리

**이미지 (Figure):**
```python
if category == "figure":
    img = soup.find("img")
    ocr_text = img.get("alt", "")  # OCR 결과 추출
    chunk = {
        "text": f"[이미지] [{current_heading}] {ocr_text}",
        "metadata": {"type": "figure"}
    }
```

**표 (Table):**
```python
if category == "table":
    table_text = soup.get_text()  # 표 내용 텍스트화
    chunk = {
        "text": f"[표] [{current_heading}] {table_text}",
        "metadata": {"type": "table"}
    }
```

---

## 4. 하이브리드 청킹 알고리즘 상세

### 4.1. split_long_text() 함수

**목적:** 1,000자를 초과하는 텍스트를 문장 단위로 재분할

**로직:**
```python
def split_long_text(text, max_size=1000, overlap=100):
    # 1. 문장 구분자로 분할: . ? ! \n\n
    sentences = re.split(r'([.!?]\s+|\n\n)', text)
    
    # 2. 구분자와 문장을 다시 결합
    parts = []
    for i in range(0, len(sentences), 2):
        if i + 1 < len(sentences):
            parts.append(sentences[i] + sentences[i+1])
        else:
            parts.append(sentences[i])
    
    # 3. max_size를 넘지 않도록 청크 생성
    chunks = []
    current_chunk = ""
    
    for part in parts:
        if len(current_chunk) + len(part) <= max_size:
            current_chunk += part
        else:
            if current_chunk:
                chunks.append(current_chunk)
            # Overlap 적용: 이전 청크의 마지막 부분을 가져옴
            if overlap > 0 and chunks:
                overlap_text = chunks[-1][-overlap:] if len(chunks[-1]) > overlap else chunks[-1]
                current_chunk = overlap_text + part
            else:
                current_chunk = part
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks if chunks else [text]
```

**특징:**
- 문장 단위 분할로 자연스러운 경계 유지
- Overlap으로 문맥 보존
- Gemini 프로젝트의 RecursiveCharacterTextSplitter와 유사한 로직

### 4.2. 최종 검증 단계

**처리 흐름:**
```python
# 1. 구조 기반 청킹 수행
chunks = parse_json_structure(json_file)

# 2. 길이 제어 적용
final_chunks = []
for chunk in chunks:
    if chunk["metadata"]["type"] == "text" and len(chunk["text"]) > MAX_TEXT_SIZE:
        # 제목 추출
        heading_match = re.match(r'^\[([^\]]+)\]\s*(.*)$', chunk["text"])
        if heading_match:
            heading = heading_match.group(1)
            body_text = heading_match.group(2)
            # 재분할
            split_bodies = split_long_text(body_text, MAX_TEXT_SIZE, overlap=100)
            # 제목과 함께 재조합
            for split_body in split_bodies:
                final_chunks.append({
                    "text": f"[{heading}] {split_body}",
                    "metadata": chunk["metadata"].copy()
                })
    else:
        final_chunks.append(chunk)
```

---

## 5. 구현 세부사항

### 4.1. 파일 탐색 전략

**재귀 탐색:**
```python
target_files = list(INPUT_DIR.glob("**/data/*.json"))
```

- `**/data/*.json`: 모든 하위 폴더의 `data` 디렉토리 내 JSON 파일 탐색
- 예외 처리: `document_analysis_results.json` 같은 메타데이터 파일 제외

### 4.2. 텍스트 정제 (Text Cleaning)

**정규식 패턴:**
```python
re.sub(r'\s+', ' ', text).strip()  # 연속된 공백을 하나로 통합
```

**목적:** 불필요한 공백 제거로 청크 크기 최적화

### 4.3. HTML 파싱

**BeautifulSoup 활용:**
```python
soup = BeautifulSoup(html_content, "html.parser")
text = soup.get_text(strip=True)  # 태그 제거 후 텍스트만 추출
```

**이미지 처리:**
```python
img = soup.find("img")
ocr_text = img.get("alt", "")  # alt 속성에서 OCR 텍스트 추출
```

---

## 6. 주요 파라미터 (Parameters)

| Parameter | Value | Description |
| :--- | :--- | :--- |
| `MAX_TEXT_SIZE` | 1,000 | 텍스트 청크의 최대 허용 길이 (글자 수) |
| `MAX_FIGURE_SIZE` | 500 | 이미지 OCR 텍스트의 최대 허용 길이 |
| `OVERLAP_SIZE` | 100 | 재분할 시 청크 간 중복되는 글자 수 |
| `SPLIT_SEPARATORS` | `\n\n`, `.`, `?`, `!` | 문장 분할 기준 |

**설정 근거:**
- `MAX_TEXT_SIZE = 1,000`: 대부분의 임베딩 모델(OpenAI, BERT 등)이 512~8,000 토큰 범위에서 동작하므로, 한글 기준 약 1,000자면 안전함
- `MAX_FIGURE_SIZE = 500`: 이미지 OCR은 보통 짧은 설명이므로, 500자 초과는 노이즈일 가능성이 높음
- `OVERLAP_SIZE = 100`: 문맥 보존과 청크 수 증가 사이의 균형점

---

## 7. 성능 최적화

### 5.1. 메모리 관리

- **스트리밍 처리**: 파일을 하나씩 처리하여 메모리 사용량 제어
- **배치 저장**: 모든 청크를 메모리에 올리지 않고 즉시 JSON에 추가 (현재는 메모리 저장 후 일괄 저장)

### 5.2. 처리 속도

- **병렬 처리 가능성**: 여러 JSON 파일을 동시에 처리 (현재는 순차 처리)
- **캐싱**: 동일 파일 재처리 방지

---

## 8. 예외 처리

### 6.1. 파일 읽기 실패
```python
try:
    data = json.load(f)
except Exception as e:
    print(f"Error reading {json_path}: {e}")
    return []  # 빈 리스트 반환하여 스킵
```

### 6.2. 구조 불일치
- `elements` 필드가 없는 경우: 빈 리스트 반환
- `category` 필드가 없는 경우: `paragraph`로 기본값 설정

### 6.3. HTML 파싱 실패
- BeautifulSoup 파싱 에러 시: 원본 `text` 필드 사용 (fallback)

---

## 9. 검증 방법

### 9.1. 자동 검증 스크립트 (verify_chunks.py)

**기능:**
- 총 청크 수, 평균/최대/최소 길이 계산
- 타입별 분포 확인
- 이상치 탐지 (너무 짧은/긴 청크)
- 파일별 분포 확인

**실행:**
```bash
python verify_chunks.py
```

**출력 예시:**
```
📊 청킹 데이터 품질 리포트
1. 총 청크 수: 3,719
2. 빈 텍스트(Empty) 수: 0개
3. 소스(Source) 누락 수: 0개
...
✅ 너무 긴 청크 없음
🎉 [PASS] 데이터 구조 무결성 검증 통과!
```

### 9.2. 정성 검증

### 7.1. 정량 검증
- 총 청크 수 확인
- 평균 청크 길이 계산
- 타입별 분포 확인 (text/image/table)

- 샘플 청크 20개 이상 직접 확인
- 제목과 본문이 함께 저장되는지 검증
- 이미지 OCR 텍스트가 포함되는지 확인
- 재분할된 청크의 문맥이 자연스러운지 확인

---

## 10. 기술적 차별점 (Technical Edge)

### 10.1. Context Injection
단순 분할이 아닌, 문서의 계층 구조(Hierarchy)를 반영하여 청크마다 소제목을 달아줌으로써 검색 시 **Self-Contained** 특성을 강화함.

### 10.2. Fail-Safe Design
구조 인식이 실패하여 텍스트가 뭉쳐도, 길이 기반 강제 분할 로직이 작동하여 시스템 안정성을 보장함.

### 10.3. Hybrid Approach
구조 기반 청킹의 장점(문맥 보존)과 길이 제어의 장점(안정성)을 결합하여 최적의 성능을 달성함.

---

---

**작성자:** AI Assistant  
**최종 수정일:** 2025-11-22 07:52

