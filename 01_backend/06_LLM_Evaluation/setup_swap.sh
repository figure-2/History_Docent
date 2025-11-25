#!/bin/bash
# Swap 메모리 활성화 스크립트

echo "=========================================="
echo "💾 Swap 메모리 활성화"
echo "=========================================="
echo ""

# 현재 Swap 상태 확인
echo "📊 현재 Swap 상태:"
swapon --show
echo ""

# Swap 파일 크기 설정 (기본 32GB)
SWAP_SIZE=${1:-32}
SWAP_FILE="/swapfile"

# 이미 Swap이 활성화되어 있는지 확인
if swapon --show | grep -q "$SWAP_FILE"; then
    echo "✅ Swap이 이미 활성화되어 있습니다."
    swapon --show
    exit 0
fi

# Swap 파일이 이미 존재하는지 확인
if [ -f "$SWAP_FILE" ]; then
    echo "⚠️  Swap 파일이 이미 존재합니다: $SWAP_FILE"
    read -p "기존 파일을 사용하시겠습니까? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "기존 파일 삭제 후 새로 생성합니다..."
        sudo swapoff "$SWAP_FILE" 2>/dev/null
        sudo rm -f "$SWAP_FILE"
    else
        echo "기존 Swap 파일을 활성화합니다..."
        sudo swapon "$SWAP_FILE"
        if [ $? -eq 0 ]; then
            echo "✅ Swap 활성화 완료!"
            swapon --show
            exit 0
        else
            echo "❌ Swap 활성화 실패. 새로 생성합니다..."
            sudo rm -f "$SWAP_FILE"
        fi
    fi
fi

# Swap 파일 생성
echo "📝 ${SWAP_SIZE}GB Swap 파일 생성 중..."
sudo fallocate -l ${SWAP_SIZE}G "$SWAP_FILE"

if [ $? -ne 0 ]; then
    echo "❌ fallocate 실패. dd 명령으로 시도합니다..."
    sudo dd if=/dev/zero of="$SWAP_FILE" bs=1G count=$SWAP_SIZE status=progress
fi

# 권한 설정
echo "🔐 권한 설정 중..."
sudo chmod 600 "$SWAP_FILE"

# Swap 포맷
echo "💾 Swap 포맷 중..."
sudo mkswap "$SWAP_FILE"

# Swap 활성화
echo "🚀 Swap 활성화 중..."
sudo swapon "$SWAP_FILE"

if [ $? -eq 0 ]; then
    echo "✅ Swap 활성화 완료!"
    echo ""
    echo "📊 현재 Swap 상태:"
    swapon --show
    echo ""
    
    # 영구적으로 활성화 (fstab에 추가)
    if ! grep -q "$SWAP_FILE" /etc/fstab; then
        echo "💾 영구적으로 활성화하기 위해 /etc/fstab에 추가합니다..."
        echo "$SWAP_FILE none swap sw 0 0" | sudo tee -a /etc/fstab
        echo "✅ /etc/fstab에 추가 완료!"
    else
        echo "✅ /etc/fstab에 이미 등록되어 있습니다."
    fi
    
    echo ""
    echo "=========================================="
    echo "✅ Swap 메모리 활성화 완료!"
    echo "=========================================="
    echo ""
    echo "📊 메모리 상태:"
    free -h
    echo ""
    echo "💡 재부팅 후에도 자동으로 활성화됩니다."
else
    echo "❌ Swap 활성화 실패!"
    exit 1
fi

