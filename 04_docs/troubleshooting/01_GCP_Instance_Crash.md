# GCP 인스턴스 크래시 분석 및 해결

> **Summary**: vLLM 모델 로드 중 GCP 인스턴스가 다운되는 현상을 분석하고, Swap 메모리 설정과 모델 최적화를 통해 해결한 과정을 기록했습니다.

## 1. 사건 개요

### 발생 시점
- **작업 내용**: 전체 Validation Set 테스트 실행 중
- **증상**: GCP 인스턴스 강제 종료, SSH 연결 끊김

## 2. 원인 분석

### 2.1 주요 원인: 메모리 부족 (OOM - Out of Memory)

**GPU 메모리 부족:**
```
Error: CUDA out of memory. Tried to allocate 1.45 GiB. 
GPU 0 has a total capacity of 39.39 GiB of which 537.25 MiB is free. 
Process 206315 has 29.35 GiB memory in use.
```

**시스템 메모리 부족:**
```
Memory: 4k page, physical 87518944k(1594644k free), swap 0k(0 free)
```
- 시스템 메모리: 87.5GB 총량
- 여유 메모리: 1.59GB (약 1.8%)
- **Swap 메모리: 0 (스왑 없음)**

### 2.2 문제점
1. **Swap 메모리 부재**: 메모리 부족 시 대체 공간 없음
2. **여러 모델 동시 실행**: Qwen-14B와 Qwen-32B 모델이 동시에 메모리 사용
3. **대용량 모델 실행**: Qwen-32B 모델은 40GB GPU로는 부족

## 3. 해결 방안

### 3.1 즉시 조치

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

**효과:**
- 메모리 부족 시 Swap으로 대체
- OOM Killer 발생 가능성 대폭 감소
- 시스템 안정성 향상

#### 메모리 모니터링
```bash
watch -n 5 'free -h && nvidia-smi --query-gpu=memory.used,memory.total --format=csv'
```

#### 프로세스 순차 실행 강화
- 한 번에 하나의 모델만 실행
- 각 모델 실행 후 완전한 메모리 정리 확인

### 3.2 코드 개선

#### 메모리 정리 강화
```python
def clear_gpu():
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
```

#### 메모리 사용량 체크
```python
def check_memory():
    if torch.cuda.is_available():
        gpu_free = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
        if gpu_free < 5:  # 5GB 미만이면 경고
            log_message(f"Warning: GPU 메모리 부족: {gpu_free:.2f}GB 여유")
            return False
    return True
```

## 4. 결론

### 4.1 종료 원인
**주요 원인:** 메모리 부족 (OOM)
- GPU 메모리 부족 (31.89 GiB / 40 GiB 사용)
- 시스템 메모리 부족 (85.9 GB / 87.5 GB 사용)
- Swap 메모리 부재 (0)

### 4.2 즉시 조치 필요
1. **Swap 메모리 활성화** (최우선)
2. **메모리 모니터링 시스템 구축**
3. **코드 개선 (메모리 정리 강화)**

### 4.3 예상 효과
- Swap 활성화 후 메모리 부족 시 대체 공간 제공
- OOM Killer 발생 가능성 대폭 감소
- 시스템 안정성 향상

---

**관련 파일**: `backend/06_LLM_Evaluation/`
