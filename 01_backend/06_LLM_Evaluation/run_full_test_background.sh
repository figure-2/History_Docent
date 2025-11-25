#!/bin/bash
# 전체 Validation Set 테스트 백그라운드 실행 스크립트
# SSH 종료 후에도 계속 실행됨

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 로그 파일 설정
LOG_DIR="$SCRIPT_DIR/results"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/full_test_background.log"
NOHUP_OUT="$SCRIPT_DIR/nohup_full_test.out"
PID_FILE="$LOG_DIR/full_test.pid"

# Python 스크립트 경로
PYTHON_SCRIPT="$SCRIPT_DIR/test_selected_model.py"

echo "=========================================="
echo "🚀 전체 Validation Set 테스트 백그라운드 실행"
echo "=========================================="
echo "스크립트: $PYTHON_SCRIPT"
echo "로그 파일: $LOG_FILE"
echo "출력 파일: $NOHUP_OUT"
echo "PID 파일: $PID_FILE"
echo ""

# 이미 실행 중인지 확인
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "⚠️  이미 실행 중인 프로세스가 있습니다 (PID: $OLD_PID)"
        echo "   중지하려면: kill $OLD_PID"
        echo "   또는: ./stop_full_test.sh"
        exit 1
    else
        echo "🧹 이전 PID 파일 정리 중..."
        rm -f "$PID_FILE"
    fi
fi

# 환경 변수 확인
if [ -z "$HUGGINGFACEHUB_API_TOKEN" ]; then
    echo "⚠️  HUGGINGFACEHUB_API_TOKEN 환경 변수가 설정되지 않았습니다."
    echo "   .env2 파일을 로드하거나 환경 변수를 설정해주세요."
    if [ -f "$SCRIPT_DIR/../.env2" ]; then
        echo "   .env2 파일 발견, 로드 시도 중..."
        export $(cat "$SCRIPT_DIR/../.env2" | grep -v '^#' | xargs)
    fi
fi

# nohup으로 백그라운드 실행
echo "📝 백그라운드 실행 시작..."
nohup python3 "$PYTHON_SCRIPT" > "$NOHUP_OUT" 2>&1 &
NEW_PID=$!

# PID 저장
echo "$NEW_PID" > "$PID_FILE"

echo "✅ 프로세스 시작됨 (PID: $NEW_PID)"
echo ""
echo "📊 모니터링 방법:"
echo "   1. 로그 확인: tail -f $LOG_FILE"
echo "   2. 출력 확인: tail -f $NOHUP_OUT"
echo "   3. 진행 상황: cat $LOG_DIR/full_test_progress.json"
echo "   4. 프로세스 확인: ps -p $NEW_PID"
echo ""
echo "🛑 중지 방법:"
echo "   kill $NEW_PID"
echo "   또는: ./stop_full_test.sh"
echo ""
echo "💡 SSH를 종료해도 프로세스는 계속 실행됩니다."

