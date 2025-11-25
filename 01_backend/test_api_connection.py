#!/usr/bin/env python3
"""
API 연결 테스트 스크립트
- FastAPI 서버가 정상적으로 응답하는지 확인
"""
import requests
import json
import time

API_URL = "http://localhost:8000/api/query"

def test_api():
    """API 연결 테스트"""
    print("=" * 60)
    print("🧪 FastAPI 서버 연결 테스트")
    print("=" * 60)
    
    # 1. 헬스체크
    print("\n1️⃣ 헬스체크 테스트...")
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ 서버가 정상적으로 실행 중입니다.")
            print(f"   응답: {response.json()}")
        else:
            print(f"⚠️ 서버가 실행 중이지만 상태 확인 실패: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("❌ 서버에 연결할 수 없습니다.")
        print("   💡 해결 방법: 'python3 main.py' 명령어로 서버를 먼저 실행하세요.")
        return False
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return False
    
    # 2. API 엔드포인트 테스트
    print("\n2️⃣ API 엔드포인트 테스트...")
    test_question = "손기정 선수는 어떤 올림픽에서 금메달을 땄나요?"
    
    print(f"   질문: {test_question}")
    print("   요청 전송 중...")
    
    try:
        start_time = time.time()
        response = requests.post(
            API_URL,
            json={"question": test_question},
            timeout=15  # vLLM이 빠르므로 15초면 충분
        )
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n✅ 응답 수신 성공! (요청-응답 시간: {elapsed:.2f}초)")
            print(f"   서버 처리 시간: {data.get('latency', 0)}초")
            print(f"   답변: {data.get('answer', '')[:100]}...")
            print(f"   출처 수: {len(data.get('sources', []))}개")
            return True
        else:
            print(f"❌ 서버 오류: {response.status_code}")
            print(f"   응답: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ 요청 시간 초과 (15초)")
        print("   💡 서버가 vLLM 모델 로딩 중일 수 있습니다. 조금 더 기다린 후 다시 시도하세요.")
        return False
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return False

if __name__ == "__main__":
    success = test_api()
    print("\n" + "=" * 60)
    if success:
        print("✅ 모든 테스트 통과! 프론트엔드와 연동할 준비가 되었습니다.")
    else:
        print("⚠️ 일부 테스트 실패. 서버 상태를 확인해주세요.")
    print("=" * 60)

