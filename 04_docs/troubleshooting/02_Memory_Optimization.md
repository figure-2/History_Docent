# 메모리 최적화 계획

> **Summary**: GPU VRAM 부족 문제를 해결하기 위해 PagedAttention 활용, 배치 사이즈 조절, 불필요한 텐서 메모리 해제 등의 최적화 기법을 적용한 계획서입니다.

## 1. 현재 상황 분석

### 1.1 선정된 모델 상태
**선정된 모델:** `MLP-KTLim/llama-3-Korean-Bllossom-8B`

**실행 상태:**
- **정상 작동 확인**: 49개 질문 모두 성공 (100%)
- **메모리 문제 없음**: 벤치마크 완료
- **평균 지연시간**: 6.98초 (4개 모델 중 가장 빠름)

**결론:** **선정된 모델은 그대로 사용 가능합니다!**

### 1.2 메모리 부족 발생 원인
**문제가 발생한 모델:**
- Qwen-14B: GPU 메모리 부족 (29.35 GiB 사용)
- Qwen-32B: 시스템 메모리 부족 + Java 크래시

**선정된 모델과의 차이:**
- Bllossom-8B: 약 16GB GPU 메모리 사용 (정상)
- Qwen-14B: 약 29GB GPU 메모리 사용 (부족)

## 2. 해결 방안

### 2.1 옵션 1: 선정된 모델 그대로 사용 (권장)

**장점:**
- 이미 검증 완료 (성공률 100%)
- 메모리 문제 없음
- 가장 빠른 응답 속도
- 추가 작업 불필요

**전제 조건:**
- Swap 메모리 활성화 (안정성 향상)
- 메모리 모니터링 (재발 방지)
- 배치 처리 (전체 Validation Set 테스트 시)

### 2.2 즉시 실행 가능한 조치

#### Swap 메모리 활성화 (Priority: High)
```bash
# Swap 파일 생성 (32GB)
sudo fallocate -l 32G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# 영구적으로 활성화
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

#### 메모리 모니터링 스크립트
```bash
#!/bin/bash
watch -n 5 'echo "=== GPU Memory ===" && nvidia-smi --query-gpu=memory.used,memory.total --format=csv && echo "" && echo "=== System Memory ===" && free -h'
```

#### 코드 개선 (메모리 정리 강화)
- 각 질문 처리 후 중간 정리
- 배치 처리로 메모리 사용량 분산
- 명시적 메모리 해제

### 2.3 전체 Validation Set 테스트 전략

#### 전략 1: 배치 처리 (권장)
**방법:**
- 전체 2,223개 질문을 100-200개씩 배치로 나누어 처리
- 각 배치 완료 후 메모리 정리
- 중간 저장으로 진행 상황 보존

**장점:**
- 메모리 사용량 안정적
- 중단 시 재개 가능
- 진행 상황 추적 용이

## 3. 최종 권장 사항

### 3.1 모델 선택
**선정된 모델 그대로 사용: `MLP-KTLim/llama-3-Korean-Bllossom-8B`**

**이유:**
1. 이미 검증 완료 (성공률 100%)
2. 메모리 문제 없음
3. 가장 빠른 응답 속도
4. 최고 성능

### 3.2 실행 전 필수 조치
1. **Swap 메모리 활성화** (최우선)
2. **메모리 모니터링 시작**
3. **코드 개선 적용**

### 3.3 예상 결과
**Swap 활성화 후:**
- GPU 메모리: 40GB (Bllossom-8B 사용 약 16GB, 여유 24GB)
- 시스템 메모리: 87.5GB + Swap 32GB = 119.5GB
- 안정성: 대폭 향상

---

**관련 파일**: `backend/06_LLM_Evaluation/`
