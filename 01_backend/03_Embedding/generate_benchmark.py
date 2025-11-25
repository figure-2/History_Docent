import json
import random
from pathlib import Path

# 설정
OUTPUT_DIR = Path("/home/pencilfoxs/00_new/History_Docent/03_Embedding/data")
OUTPUT_FILE = OUTPUT_DIR / "korean_history_benchmark.json"

def generate_dataset():
    """
    한국사 RAG 성능 평가를 위한 벤치마크 데이터셋 생성
    (Golden Dataset: 질문 - 정답 텍스트 매핑)
    """
    
    # 4가지 유형의 데이터셋 정의
    dataset = [
        # ---------------------------------------------------------
        # Type 1: 단답형 의미 유사도 (Entity Matching)
        # ---------------------------------------------------------
        {
            "id": "q_001",
            "type": "entity_matching",
            "query": "태조 이성계",
            "gold_text": "조선의 건국자이자 초대 국왕이다. 위화도 회군을 통해 권력을 잡았다."
        },
        {
            "id": "q_002",
            "type": "entity_matching",
            "query": "삼국통일",
            "gold_text": "신라의 문무왕이 당나라 세력을 몰아내고 676년에 달성한 업적이다."
        },
        {
            "id": "q_003",
            "type": "entity_matching",
            "query": "훈민정음",
            "gold_text": "세종대왕이 백성을 가르치는 바른 소리라는 뜻으로 창제한 문자이다."
        },
        
        # ---------------------------------------------------------
        # Type 2: 지명·왕·연도 매칭 (Fact Checking)
        # ---------------------------------------------------------
        {
            "id": "q_004",
            "type": "fact_matching",
            "query": "한성",
            "gold_text": "조선의 수도이자 현재의 서울로, 태조가 천도하였다."
        },
        {
            "id": "q_005",
            "type": "fact_matching",
            "query": "임진왜란 발생 연도",
            "gold_text": "1592년에 일본의 침략으로 시작되어 7년간 지속된 전쟁이다."
        },
        {
            "id": "q_006",
            "type": "fact_matching",
            "query": "정조의 업적",
            "gold_text": "수원 화성 축조, 규장각 설치, 장용영 창설 등의 개혁 정치를 펼쳤다."
        },

        # ---------------------------------------------------------
        # Type 3: 개념-설명 검색 (Concept Explanation)
        # ---------------------------------------------------------
        {
            "id": "q_007",
            "type": "concept_search",
            "query": "대동법이란 무엇인가?",
            "gold_text": "광해군 때 경기도에서 처음 실시된 것으로, 특산물 대신 쌀이나 동전으로 세금을 내게 한 제도이다."
        },
        {
            "id": "q_008",
            "type": "concept_search",
            "query": "고려시대 과거제도",
            "gold_text": "광종 때 후주 출신 쌍기의 건의로 도입되어 유교적 소양을 갖춘 관리를 선발하였다."
        },
        
        # ---------------------------------------------------------
        # Type 4: 장문 문단 검색 (Context Retrieval)
        # ---------------------------------------------------------
        {
            "id": "q_009",
            "type": "context_retrieval",
            "query": "조선 후기 사회의 신분제 변동 양상",
            "gold_text": "양반의 수가 급증하고 상민과 노비의 수가 감소하였다. 공명첩 발급과 족보 위조 등이 성행하며 신분제가 동요하였다."
        },
        {
            "id": "q_010",
            "type": "context_retrieval",
            "query": "일제 강점기 무단 통치 시기의 특징",
            "gold_text": "1910년대 헌병 경찰 제도를 실시하여 한국인의 모든 정치 활동을 금지하고 즉결 처분권을 행사하였다."
        }
    ]

    # 데이터셋 확장 (임시로 2배로 늘림 - 실제론 더 다양한 데이터 필요)
    # 실제 구축 시에는 LLM을 이용해 200~500개를 생성해야 함.
    extended_dataset = dataset * 2 

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(extended_dataset, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 벤치마크 데이터셋 생성 완료: {len(extended_dataset)}개 샘플")
    print(f"   저장 경로: {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_dataset()
