#!/bin/bash
# QA 데이터셋 생성 진행 상황 확인 스크립트

SCRIPT_DIR="/home/pencilfoxs/00_new/History_Docent/06_LLM_Evaluation"
LOG_FILE="${SCRIPT_DIR}/generation_progress.log"
STATS_FILE="${SCRIPT_DIR}/generation_stats.json"
OUTPUT_FILE="${SCRIPT_DIR}/balanced_qa_dataset_full.jsonl"

echo "=========================================="
echo "📊 QA 데이터셋 생성 진행 상황"
echo "=========================================="
echo ""

# 1. 프로세스 실행 여부 확인
if pgrep -f "generate_balanced_qa_dataset_robust.py" > /dev/null; then
    echo "✅ 스크립트 실행 중"
    PID=$(pgrep -f "generate_balanced_qa_dataset_robust.py" | head -1)
    echo "   PID: $PID"
else
    echo "❌ 스크립트 실행 중이 아님"
fi
echo ""

# 2. 통계 파일 확인
if [ -f "$STATS_FILE" ]; then
    echo "📈 통계 정보:"
    python3 -c "
import json
import sys
from datetime import datetime

try:
    with open('$STATS_FILE', 'r', encoding='utf-8') as f:
        stats = json.load(f)
    
    total_chunks = stats.get('total_chunks', 0)
    processed = stats.get('processed_chunks', 0)
    remaining = stats.get('remaining_chunks', 0)
    total_saved = stats.get('total_saved', 0)
    by_type = stats.get('by_type', {})
    
    print(f'   전체 청크: {total_chunks}')
    print(f'   처리 완료: {processed}')
    print(f'   남은 청크: {remaining}')
    if total_chunks > 0:
        progress = (processed / total_chunks) * 100
        print(f'   진행률: {progress:.1f}%')
    print(f'   생성된 질문 수: {total_saved}')
    print(f'   - Keyword: {by_type.get(\"keyword\", 0)}')
    print(f'   - Context: {by_type.get(\"context\", 0)}')
    print(f'   - Abstract: {by_type.get(\"abstract\", 0)}')
    
    if 'started_at' in stats:
        print(f'   시작 시간: {stats[\"started_at\"]}')
    if 'completed_at' in stats:
        print(f'   완료 시간: {stats[\"completed_at\"]}')
except Exception as e:
    print(f'   오류: {e}')
" 2>/dev/null || echo "   통계 파일 읽기 실패"
else
    echo "⚠️  통계 파일 없음 (아직 시작하지 않았거나 오류 발생)"
fi
echo ""

# 3. 출력 파일 확인
if [ -f "$OUTPUT_FILE" ]; then
    LINE_COUNT=$(wc -l < "$OUTPUT_FILE" 2>/dev/null || echo "0")
    FILE_SIZE=$(du -h "$OUTPUT_FILE" 2>/dev/null | cut -f1)
    echo "📁 출력 파일:"
    echo "   파일: $(basename $OUTPUT_FILE)"
    echo "   질문 수: $LINE_COUNT"
    echo "   파일 크기: $FILE_SIZE"
else
    echo "⚠️  출력 파일 없음"
fi
echo ""

# 4. 최근 로그 (마지막 10줄)
if [ -f "$LOG_FILE" ]; then
    echo "📝 최근 로그 (마지막 10줄):"
    tail -n 10 "$LOG_FILE" | sed 's/^/   /'
else
    echo "⚠️  로그 파일 없음"
fi
echo ""

echo "=========================================="



