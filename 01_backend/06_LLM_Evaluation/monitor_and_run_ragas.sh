#!/bin/bash
# 전처리 완료 모니터링 및 RAGAS 평가 자동 실행 스크립트

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_FILE="$SCRIPT_DIR/results/llm_selected_model_full_test_with_contexts.csv"
INPUT_FILE="$SCRIPT_DIR/results/llm_selected_model_full_test.csv"
LOG_FILE="$SCRIPT_DIR/nohup_prepare_ragas.out"
PREPARE_PID_FILE="$SCRIPT_DIR/results/prepare_pid.txt"

echo "=========================================="
echo "🔍 전처리 완료 모니터링 시작"
echo "=========================================="
echo ""

# 전처리 프로세스 PID 확인
PREPARE_PID=$(ps aux | grep "prepare_ragas_data.py" | grep -v grep | awk '{print $2}' | head -1)

if [ -z "$PREPARE_PID" ]; then
    echo "⚠️  전처리 프로세스를 찾을 수 없습니다."
    echo "전처리가 이미 완료되었을 수 있습니다."
else
    echo "✅ 전처리 프로세스 실행 중 (PID: $PREPARE_PID)"
    echo "$PREPARE_PID" > "$PREPARE_PID_FILE"
fi

# 전처리 완료까지 대기
TOTAL_QUESTIONS=$(python3 << EOF
import pandas as pd
df = pd.read_csv("$INPUT_FILE")
print(len(df))
EOF
)

echo "📊 총 질문 수: $TOTAL_QUESTIONS개"
echo ""
echo "⏳ 전처리 완료 대기 중..."
echo ""

CHECK_INTERVAL=10  # 10초마다 확인
MAX_WAIT=3600      # 최대 1시간 대기
ELAPSED=0

while [ $ELAPSED -lt $MAX_WAIT ]; do
    # 프로세스 확인
    if [ ! -z "$PREPARE_PID" ]; then
        if ! ps -p "$PREPARE_PID" > /dev/null 2>&1; then
            echo "✅ 전처리 프로세스 종료됨"
            break
        fi
    fi
    
    # 결과 파일 확인
    if [ -f "$OUTPUT_FILE" ]; then
        COMPLETED=$(python3 << EOF
import pandas as pd
try:
    df = pd.read_csv("$OUTPUT_FILE")
    if 'contexts' in df.columns:
        non_empty = df['contexts'].notna().sum()
        print(non_empty)
    else:
        print(0)
except:
    print(0)
EOF
)
        
        if [ "$COMPLETED" -ge "$TOTAL_QUESTIONS" ]; then
            echo "✅ 전처리 완료! ($COMPLETED/$TOTAL_QUESTIONS개 완료)"
            break
        else
            PERCENTAGE=$((COMPLETED * 100 / TOTAL_QUESTIONS))
            echo "⏳ 진행 중: $COMPLETED/$TOTAL_QUESTIONS개 ($PERCENTAGE%) - $(date '+%H:%M:%S')"
        fi
    fi
    
    sleep $CHECK_INTERVAL
    ELAPSED=$((ELAPSED + CHECK_INTERVAL))
done

# 전처리 완료 확인
if [ -f "$OUTPUT_FILE" ]; then
    FINAL_COMPLETED=$(python3 << EOF
import pandas as pd
try:
    df = pd.read_csv("$OUTPUT_FILE")
    if 'contexts' in df.columns:
        print(len(df))
    else:
        print(0)
except:
    print(0)
EOF
)
    
    if [ "$FINAL_COMPLETED" -ge "$TOTAL_QUESTIONS" ]; then
        echo ""
        echo "=========================================="
        echo "✅ 전처리 완료 확인!"
        echo "=========================================="
        echo ""
        
        # 2단계 RAGAS 평가 시작
        echo "🚀 2단계 RAGAS 평가 시작..."
        echo ""
        
        cd "$SCRIPT_DIR"
        nohup python3 evaluate_ragas_full.py > nohup_ragas_evaluation.out 2>&1 &
        RAGAS_PID=$!
        
        echo "✅ RAGAS 평가 백그라운드 실행 시작"
        echo "   프로세스 ID: $RAGAS_PID"
        echo "   로그 파일: nohup_ragas_evaluation.out"
        echo ""
        echo "📊 모니터링 명령어:"
        echo "   tail -f nohup_ragas_evaluation.out"
        echo ""
        echo "💾 PID 저장: results/ragas_pid.txt"
        echo "$RAGAS_PID" > "$SCRIPT_DIR/results/ragas_pid.txt"
        
        echo "=========================================="
    else
        echo "⚠️  전처리가 완전히 완료되지 않았습니다. ($FINAL_COMPLETED/$TOTAL_QUESTIONS)"
    fi
else
    echo "⚠️  결과 파일이 생성되지 않았습니다."
fi

