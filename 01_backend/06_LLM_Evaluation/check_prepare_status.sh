#!/bin/bash
# 전처리 작업 상태 확인 스크립트

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_FILE="$SCRIPT_DIR/results/llm_selected_model_full_test_with_contexts.csv"
INPUT_FILE="$SCRIPT_DIR/results/llm_selected_model_full_test.csv"
LOG_FILE="$SCRIPT_DIR/nohup_prepare_ragas.out"

echo "=========================================="
echo "📊 전처리 작업 상태 확인"
echo "=========================================="
echo ""

# 프로세스 확인
PID=$(ps aux | grep "prepare_ragas_data.py" | grep -v grep | awk '{print $2}')
if [ ! -z "$PID" ]; then
    echo "✅ 프로세스 실행 중 (PID: $PID)"
    
    # CPU/메모리 사용량 확인
    CPU_MEM=$(ps -p "$PID" -o %cpu,%mem,etime --no-headers 2>/dev/null)
    if [ ! -z "$CPU_MEM" ]; then
        echo "   CPU/메모리: $CPU_MEM"
    fi
else
    echo "❌ 프로세스가 실행 중이지 않습니다."
fi

echo ""

# 결과 파일 확인
if [ -f "$OUTPUT_FILE" ]; then
    echo "📁 결과 파일:"
    python3 << EOF
import pandas as pd
from pathlib import Path

output_file = Path("$OUTPUT_FILE")
input_file = Path("$INPUT_FILE")

if output_file.exists():
    df_output = pd.read_csv(output_file)
    df_input = pd.read_csv(input_file)
    
    total = len(df_input)
    completed = len(df_output) if 'contexts' in df_output.columns else 0
    
    print(f"   파일: {output_file}")
    print(f"   총 질문 수: {total}개")
    print(f"   처리 완료: {completed}개")
    
    if total > 0:
        percentage = (completed / total) * 100
        print(f"   진행률: {percentage:.1f}%")
        remaining = total - completed
        print(f"   남은 질문: {remaining}개")
        
        # contexts 컬럼 확인
        if 'contexts' in df_output.columns:
            non_empty = df_output['contexts'].notna().sum()
            print(f"   contexts 추가됨: {non_empty}개")
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

