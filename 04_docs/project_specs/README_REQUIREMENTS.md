# 환경 설정 가이드 (Requirements)

## 개요

이 문서는 History Docent 백엔드 서버를 실행하기 위한 환경 설정 가이드를 제공합니다.

---

## Python 환경

### Python 버전
- **최소 요구사항**: Python 3.10 이상
- **권장 버전**: Python 3.11 이상
- **현재 환경**: Python 3.13.9

### 가상환경 생성 (권장)
```bash
# 가상환경 생성
python3 -m venv venv

# 가상환경 활성화
source venv/bin/activate

# 비활성화
deactivate
```

---

## 패키지 설치

### 기본 설치
```bash
cd backend
pip install -r requirements.txt
```

### 단계별 설치 (문제 발생 시)

#### 1. 웹 프레임워크
```bash
pip install fastapi==0.122.0 uvicorn[standard]==0.38.0 pydantic==2.12.3
```

#### 2. PyTorch (CUDA 지원)
```bash
# CUDA 12.1 버전
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu121
```

#### 3. Transformers 및 관련 패키지
```bash
pip install transformers==4.57.2 accelerate==1.12.0 sentencepiece==0.2.1 huggingface-hub==0.36.0
```

#### 4. vLLM
```bash
pip install vllm==0.11.2
```

#### 5. 벡터 DB 및 임베딩
```bash
pip install chromadb==1.3.5 sentence-transformers==5.1.2
```

#### 6. 검색 시스템
```bash
pip install rank-bm25==0.2.2 konlpy==0.6.0
```

#### 7. 데이터 처리
```bash
pip install numpy==2.2.6 pandas==2.3.3 requests==2.32.5 protobuf==6.33.1
```

---

## Java 설치 (KoNLPy 필수)

KoNLPy는 한국어 형태소 분석을 위해 Java가 필요합니다.

### Ubuntu/Debian
```bash
# Java 설치
sudo apt-get update
sudo apt-get install -y default-jdk

# JAVA_HOME 환경 변수 설정
export JAVA_HOME=$(dirname $(dirname $(readlink -f $(which java))))
echo "export JAVA_HOME=\$(dirname \$(dirname \$(readlink -f \$(which java))))" >> ~/.bashrc

# 확인
java -version
echo $JAVA_HOME
```

### 확인
```bash
python3 -c "from konlpy.tag import Okt; okt = Okt(); print('KoNLPy 작동 확인')"
```

---

## GPU 설정 (권장)

### NVIDIA GPU 드라이버 설치

#### Ubuntu 24.04
```bash
# 드라이버 설치
sudo apt-get update
sudo apt-get install -y ubuntu-drivers-common
sudo apt-get install -y nvidia-driver-535-server

# 재부팅 후 확인
nvidia-smi
```

### CUDA 설치 확인
```bash
# PyTorch CUDA 확인
python3 -c "import torch; print('CUDA 사용 가능:', torch.cuda.is_available()); print('CUDA 버전:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"
```

---

## 패키지 설치 확인

### 필수 패키지 확인
```bash
python3 << 'EOF'
packages = [
    'torch', 'chromadb', 'transformers', 'vllm',
    'sentence_transformers', 'rank_bm25', 'konlpy',
    'fastapi', 'uvicorn', 'pydantic', 'numpy'
]

for pkg in packages:
    try:
        __import__(pkg)
        print(f"OK: {pkg}")
    except ImportError:
        print(f"FAIL: {pkg} - 설치 필요")
EOF
```

---

## 서버 실행

### 환경 변수 설정
```bash
# JAVA_HOME 설정 (KoNLPy 사용 시)
export JAVA_HOME=$(dirname $(dirname $(readlink -f $(which java))))

# 백엔드 서버 실행
cd backend
python3 main.py
```

### 백그라운드 실행
```bash
# JAVA_HOME 포함하여 실행
export JAVA_HOME=$(dirname $(dirname $(readlink -f $(which java))))
nohup env JAVA_HOME=$JAVA_HOME python3 main.py > backend.log 2>&1 &
```

---

## 시스템 요구사항

### 최소 사양
- **CPU**: 4 cores 이상
- **RAM**: 8GB 이상
- **디스크**: 50GB 이상 여유 공간
- **GPU**: 없음 (CPU 모드 가능, 매우 느림)

### 권장 사양 (GPU 사용)
- **CPU**: 8 cores 이상
- **RAM**: 32GB 이상
- **VRAM**: 16GB 이상 (Tesla T4 또는 동급 이상)
- **디스크**: 100GB 이상 여유 공간

---

## 문제 해결

### KoNLPy 오류: "No JVM shared library file"
```bash
# Java 설치 확인
java -version

# JAVA_HOME 설정 확인
echo $JAVA_HOME

# libjvm.so 찾기
find /usr/lib/jvm -name "libjvm.so" 2>/dev/null
```

### GPU 인식 안 됨
```bash
# NVIDIA 드라이버 확인
nvidia-smi

# PyTorch CUDA 확인
python3 -c "import torch; print(torch.cuda.is_available())"

# 드라이버 재로드
sudo modprobe nvidia
```

### 메모리 부족 오류
- GPU 메모리 부족 시: `history_docent.py`의 `gpu_memory_utilization` 값 조정
- 시스템 메모리 부족 시: 모델을 더 작은 버전으로 변경 또는 CPU 모드 사용

---

## 추가 리소스

- **ChromaDB 문서**: https://docs.trychroma.com/
- **vLLM 문서**: https://docs.vllm.ai/
- **FastAPI 문서**: https://fastapi.tiangolo.com/
- **KoNLPy 문서**: https://konlpy.org/

---

**마지막 업데이트**: 2024-11-25

