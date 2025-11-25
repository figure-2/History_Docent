#!/bin/bash
# 전체 Validation Set 테스트 상태 확인 스크립트

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/results/full_test.pid"
PROGRESS_FILE="$SCRIPT_DIR/results/full_test_progress.json"
LOG_FILE="$SCRIPT_DIR/results/full_test_progress.log"
RESULTS_FILE="$SCRIPT_DIR/results/llm_selected_model_full_test.csv"

echo "=========================================="
echo "📊 전체 Validation Set 테스트 상태 확인"
echo "=========================================="
echo ""

# 프로세스 상태 확인
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "✅ 프로세스 실행 중 (PID: $PID)"
        
        # CPU/메모리 사용량 확인
        if command -v ps > /dev/null; then
            CPU_MEM=$(ps -p "$PID" -o %cpu,%mem,etime --no-headers 2>/dev/null)
            if [ ! -z "$CPU_MEM" ]; then
                echo "   CPU/메모리: $CPU_MEM"
            fi
        fi
    else
        echo "❌ 프로세스가 실행 중이지 않습니다 (PID: $PID)"
        echo "   PID 파일 정리 중..."
        rm -f "$PID_FILE"
    fi
else
    echo "⚠️  PID 파일이 없습니다. 프로세스가 실행 중이지 않을 수 있습니다."
fi

echo ""

# 진행 상황 확인
if [ -f "$PROGRESS_FILE" ]; then
    echo "📈 진행 상황:"
    python3 << EOF
import json
from pathlib import Path

progress_file = Path("$PROGRESS_FILE")
if progress_file.exists():
    with open(progress_file, 'r', encoding='utf-8') as f:
        progress = json.load(f)
    
    completed = progress.get('total_completed', 0)
    last_checkpoint = progress.get('last_checkpoint', 'N/A')
    
    print(f"   완료된 질문: {completed}개")
    print(f"   마지막 체크포인트: {last_checkpoint}")
    
    # 전체 질문 수 확인
    import json
    from pathlib import Path
    data_file = Path("$SCRIPT_DIR/../03_Embedding/data/validation_set_20.json")
    if data_file.exists():
        with open(data_file, 'r', encoding='utf-8') as f:
            total = len(json.load(f))
        print(f"   전체 질문 수: {total}개")
        if total > 0:
            percentage = (completed / total) * 100
            print(f"   진행률: {percentage:.1f}%")
            remaining = total - completed
            print(f"   남은 질문: {remaining}개")
EOF
else
    echo "⚠️  진행 상황 파일이 없습니다."
fi

echo ""

# 결과 파일 확인
if [ -f "$RESULTS_FILE" ]; then
    echo "📁 결과 파일:"
    python3 << EOF
import pandas as pd
from pathlib import Path

results_file = Path("$RESULTS_FILE")
if results_file.exists():
    df = pd.read_csv(results_file)
    print(f"   파일: {results_file}")
    print(f"   총 응답 수: {len(df)}개")
    if 'latency' in df.columns:
        print(f"   평균 지연시간: {df['latency'].mean():.2f}초")
    if 'type' in df.columns:
        print(f"   질문 유형별 분포:")
        for q_type, count in df['type'].value_counts().items():
            print(f"     - {q_type}: {count}개")
EOF
else
    echo "⚠️  결과 파일이 아직 생성되지 않았습니다."
fi

echo ""

# 로그 파일 확인
if [ -f "$LOG_FILE" ]; then
    echo "📝 최근 로그 (마지막 10줄):"
    tail -n 10 "$LOG_FILE" 2>/dev/null || echo "   로그 파일 읽기 실패"
else
    echo "⚠️  로그 파일이 없습니다."
fi

echo ""
echo "=========================================="

