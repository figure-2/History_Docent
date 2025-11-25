#!/bin/bash
# RAGAS 평가 상태 확인 스크립트

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_FILE="$SCRIPT_DIR/results/ragas_evaluation_results.csv"
LOG_FILE="$SCRIPT_DIR/nohup_ragas_evaluation.out"
PROGRESS_FILE="$SCRIPT_DIR/results/ragas_evaluation_progress.json"
PID_FILE="$SCRIPT_DIR/results/ragas_pid.txt"

echo "=========================================="
echo "📊 RAGAS 평가 상태 확인"
echo "=========================================="
echo ""

# 프로세스 확인
if [ -f "$PID_FILE" ]; then
    RAGAS_PID=$(cat "$PID_FILE")
else
    RAGAS_PID=$(ps aux | grep "evaluate_ragas_full.py" | grep -v grep | awk '{print $2}' | head -1)
fi

if [ ! -z "$RAGAS_PID" ]; then
    echo "✅ 프로세스 실행 중 (PID: $RAGAS_PID)"
    
    # CPU/메모리 사용량 확인
    CPU_MEM=$(ps -p "$RAGAS_PID" -o %cpu,%mem,etime --no-headers 2>/dev/null)
    if [ ! -z "$CPU_MEM" ]; then
        echo "   CPU/메모리: $CPU_MEM"
    fi
else
    echo "❌ 프로세스가 실행 중이지 않습니다."
fi

echo ""

# 진행 상황 확인
if [ -f "$PROGRESS_FILE" ]; then
    echo "📁 진행 상황:"
    cat "$PROGRESS_FILE" | python3 -m json.tool 2>/dev/null || cat "$PROGRESS_FILE"
else
    echo "⚠️  진행 상황 파일이 아직 생성되지 않았습니다."
fi

echo ""

# 결과 파일 확인
if [ -f "$OUTPUT_FILE" ]; then
    echo "📁 결과 파일:"
    python3 << EOF
import pandas as pd
from pathlib import Path

output_file = Path("$OUTPUT_FILE")

if output_file.exists():
    df_output = pd.read_csv(output_file)
    
    print(f"   파일: {output_file}")
    print(f"   평가 완료: {len(df_output)}개")
    
    # 지표 컬럼 확인
    metric_columns = ['context_recall', 'context_precision', 'faithfulness', 'answer_relevancy']
    available_metrics = [col for col in metric_columns if col in df_output.columns]
    
    if available_metrics:
        print(f"\n   현재 평균 점수:")
        for metric in available_metrics:
            avg_score = df_output[metric].mean()
            print(f"      - {metric}: {avg_score:.4f}")
EOF
else
    echo "⚠️  결과 파일이 아직 생성되지 않았습니다."
fi

echo ""

# 로그 파일 확인
if [ -f "$LOG_FILE" ]; then
    echo "📝 최근 로그 (마지막 15줄):"
    tail -n 15 "$LOG_FILE" 2>/dev/null || echo "   로그 파일 읽기 실패"
else
    echo "⚠️  로그 파일이 없습니다."
fi

echo ""
echo "=========================================="

