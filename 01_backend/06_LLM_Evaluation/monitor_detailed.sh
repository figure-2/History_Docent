#!/bin/bash
# RAGAS 평가 상세 모니터링 스크립트

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$SCRIPT_DIR/nohup_ragas_evaluation.out"
OUTPUT_FILE="$SCRIPT_DIR/results/ragas_evaluation_results.csv"
PROGRESS_FILE="$SCRIPT_DIR/results/ragas_evaluation_progress.json"

echo "=========================================="
echo "📊 RAGAS 평가 상세 모니터링"
echo "=========================================="
echo "시간: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 1. 프로세스 상태
echo "1️⃣  프로세스 상태"
PID=$(ps aux | grep "evaluate_ragas_full.py" | grep -v grep | awk '{print $2}' | head -1)
if [ ! -z "$PID" ]; then
    CPU_MEM=$(ps -p "$PID" -o %cpu,%mem,etime,stat --no-headers 2>/dev/null)
    echo "   ✅ 실행 중 (PID: $PID)"
    echo "   📊 상태: $CPU_MEM"
else
    echo "   ❌ 프로세스가 실행 중이지 않습니다."
fi

echo ""

# 2. 로그 분석
echo "2️⃣  로그 분석"
if [ -f "$LOG_FILE" ]; then
    TOTAL_LINES=$(wc -l < "$LOG_FILE" | tr -d ' ')
    echo "   📝 총 로그 라인: $TOTAL_LINES"
    
    # 배치 진행 상황
    BATCH_COUNT=$(grep -c "배치.*처리 중" "$LOG_FILE" 2>/dev/null || echo "0")
    echo "   📦 배치 처리 시작: $BATCH_COUNT개"
    
    # 에러 개수 (중요한 에러만)
    CRITICAL_ERRORS=$(grep -cE "(Traceback|Fatal|실패|Failed)" "$LOG_FILE" 2>/dev/null || echo "0")
    echo "   ⚠️  중요 에러: $CRITICAL_ERRORS개"
    
    # 최근 배치 진행 상황
    LAST_BATCH=$(grep "배치.*처리 중" "$LOG_FILE" | tail -1 | sed 's/^[[:space:]]*//')
    if [ ! -z "$LAST_BATCH" ]; then
        echo "   📌 최근 배치: ${LAST_BATCH:0:80}"
    fi
else
    echo "   ⚠️  로그 파일이 없습니다."
fi

echo ""

# 3. 결과 파일
echo "3️⃣  결과 파일 상태"
if [ -f "$OUTPUT_FILE" ]; then
    FILE_SIZE=$(ls -lh "$OUTPUT_FILE" | awk '{print $5}')
    ROW_COUNT=$(python3 << PYEOF
import pandas as pd
try:
    df = pd.read_csv("$OUTPUT_FILE")
    print(len(df))
except:
    print(0)
PYEOF
)
    echo "   ✅ 파일 존재 (크기: $FILE_SIZE)"
    echo "   📊 평가 완료: ${ROW_COUNT}개"
    
    if [ "$ROW_COUNT" -gt 0 ]; then
        # 평균 점수 계산
        python3 << PYEOF
import pandas as pd
try:
    df = pd.read_csv("$OUTPUT_FILE")
    metrics = ['context_recall', 'context_precision', 'faithfulness', 'answer_relevancy']
    print("\n   📈 현재 평균 점수:")
    for metric in metrics:
        if metric in df.columns:
            avg = df[metric].mean()
            print(f"      - {metric}: {avg:.4f}")
except Exception as e:
    pass
PYEOF
    fi
else
    echo "   ⚠️  결과 파일이 아직 생성되지 않았습니다."
fi

echo ""

# 4. 예상 완료 시간
echo "4️⃣  예상 완료 시간"
if [ -f "$OUTPUT_FILE" ] && [ "$ROW_COUNT" -gt 0 ]; then
    python3 << PYEOF
import pandas as pd
from pathlib import Path
import time
from datetime import datetime, timedelta

output_file = Path("$OUTPUT_FILE")
df = pd.read_csv(output_file)
completed = len(df)
total = 2223

if completed > 0:
    file_mtime = output_file.stat().st_mtime
    elapsed = time.time() - file_mtime
    
    time_per_q = elapsed / completed
    remaining = total - completed
    est_remaining = time_per_q * remaining
    
    progress = (completed / total) * 100
    print(f"   📊 진행률: {progress:.1f}% ({completed}/{total})")
    print(f"   ⏱️  경과: {elapsed/60:.1f}분")
    print(f"   ⚡ 속도: {time_per_q:.2f}초/질문")
    print(f"   ⏰ 예상 남은 시간: {est_remaining/60:.1f}분")
    
    completion = datetime.now() + timedelta(seconds=est_remaining)
    print(f"   🎯 예상 완료: {completion.strftime('%H:%M:%S')}")
else:
    print("   ⚠️  아직 데이터가 없습니다.")
PYEOF
else
    echo "   ⚠️  계산 불가 (결과 파일 없음)"
fi

echo ""
echo "=========================================="
echo "💡 모니터링 명령어:"
echo "   tail -f $LOG_FILE"
echo "   ./check_ragas_status.sh"
echo "=========================================="

