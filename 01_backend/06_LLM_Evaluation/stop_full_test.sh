#!/bin/bash
# 전체 Validation Set 테스트 중지 스크립트

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/results/full_test.pid"

if [ ! -f "$PID_FILE" ]; then
    echo "❌ PID 파일을 찾을 수 없습니다. 프로세스가 실행 중이지 않을 수 있습니다."
    exit 1
fi

PID=$(cat "$PID_FILE")

if ! ps -p "$PID" > /dev/null 2>&1; then
    echo "⚠️  프로세스가 실행 중이지 않습니다 (PID: $PID)"
    rm -f "$PID_FILE"
    exit 1
fi

echo "🛑 프로세스 중지 중... (PID: $PID)"
kill "$PID"

# 프로세스 종료 대기
for i in {1..10}; do
    if ! ps -p "$PID" > /dev/null 2>&1; then
        echo "✅ 프로세스가 정상적으로 종료되었습니다."
        rm -f "$PID_FILE"
        exit 0
    fi
    sleep 1
done

# 강제 종료
echo "⚠️  정상 종료 실패, 강제 종료 시도..."
kill -9 "$PID" 2>/dev/null
rm -f "$PID_FILE"
echo "✅ 프로세스가 강제 종료되었습니다."

