import json
import statistics
from pathlib import Path
from collections import Counter, defaultdict

# -----------------------------------------------------------------------------
# 설정 (Configuration)
# -----------------------------------------------------------------------------
BASE_DIR = Path("/home/pencilfoxs/00_new/History_Docent")
CHUNK_FILE = BASE_DIR / "02_Chunking/output/all_chunks.json"

# 임계값 설정 (Thresholds for Warnings)
MIN_LENGTH_WARNING = 20   # 20자 미만은 노이즈(페이지 번호 등)일 가능성 높음
MAX_LENGTH_WARNING = 1500 # 1500자 초과는 임베딩 시 잘릴(Truncation) 위험 있음

class ChunkValidator:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = []
        self.stats = {
            "total_count": 0,
            "by_source": Counter(),
            "by_type": Counter(),
            "empty_text": 0,
            "missing_meta": 0,
            "lengths": []
        }
        self.anomalies = {
            "too_short": [],
            "too_long": []
        }

    def load_data(self):
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            print(f"✅ 데이터 로드 성공: {len(self.data)}개의 청크")
        except Exception as e:
            print(f"❌ 데이터 로드 실패: {e}")
            exit(1)

    def run_checks(self):
        print("\n🔍 검증 시작...\n")
        
        for idx, chunk in enumerate(self.data):
            # 1. 필수 필드 검사
            if "text" not in chunk or "metadata" not in chunk:
                print(f"⚠️ [Index {idx}] 필수 필드 누락")
                continue
            
            text = chunk.get("text", "")
            meta = chunk.get("metadata", {})
            
            # 2. 텍스트 무결성
            if not text.strip():
                self.stats["empty_text"] += 1
            
            # 3. 메타데이터 검사
            source = meta.get("source", "UNKNOWN")
            c_type = meta.get("type", "unknown")
            
            if source == "UNKNOWN":
                self.stats["missing_meta"] += 1

            # 4. 통계 수집
            self.stats["by_source"][source] += 1
            self.stats["by_type"][c_type] += 1
            self.stats["lengths"].append(len(text))
            
            # 5. 이상치 탐지 (Anomalies)
            if len(text) < MIN_LENGTH_WARNING and c_type == 'text':
                self.anomalies["too_short"].append((chunk["chunk_id"], len(text), text[:20]))
            
            if len(text) > MAX_LENGTH_WARNING:
                self.anomalies["too_long"].append((chunk["chunk_id"], len(text), source))

    def print_report(self):
        print("="*60)
        print("📊 청킹 데이터 품질 리포트 (Chunking Quality Report)")
        print("="*60)
        
        # 1. 기본 통계
        print(f"1. 총 청크 수: {len(self.data):,}")
        print(f"2. 빈 텍스트(Empty) 수: {self.stats['empty_text']}개 (0이어야 함)")
        print(f"3. 소스(Source) 누락 수: {self.stats['missing_meta']}개")
        
        # 2. 파일별 분포 (Completeness Check)
        print("\n4. 파일별 청크 분포 (Source Distribution):")
        print(f"   - 총 {len(self.stats['by_source'])}개의 소스 파일 감지됨")
        for source, count in self.stats['by_source'].most_common():
            print(f"   - {source:<20}: {count:4,} chunks")
        
        # 3. 타입 분포
        print("\n5. 데이터 타입 분포:")
        for c_type, count in self.stats['by_type'].items():
            ratio = (count / len(self.data)) * 100
            print(f"   - {c_type:<10}: {count:4,} ({ratio:.1f}%)")

        # 4. 길이 분석 (Length Analysis)
        lengths = self.stats["lengths"]
        if lengths:
            avg_len = statistics.mean(lengths)
            max_len = max(lengths)
            min_len = min(lengths)
            print(f"\n6. 텍스트 길이 분석 (글자 수 기준):")
            print(f"   - 평균: {avg_len:.1f} 자")
            print(f"   - 최대: {max_len:,} 자")
            print(f"   - 최소: {min_len:,} 자")
            
            # 길이 분포 시각화 (Simple ASCII Histogram)
            print("\n   [길이 분포 히스토그램]")
            buckets = [0, 200, 500, 1000, 1500, 99999]
            bucket_counts = defaultdict(int)
            for l in lengths:
                for i in range(len(buckets)-1):
                    if buckets[i] <= l < buckets[i+1]:
                        bucket_counts[f"{buckets[i]}~{buckets[i+1]}"] += 1
                        break
            
            for k, v in bucket_counts.items():
                bar = "█" * int((v / len(lengths)) * 50)
                print(f"   {k:<10}: {v:4,} |{bar}")

        # 5. 경고 및 이상치 (Warnings)
        print("\n7. 경고 및 이상치 (Anomalies):")
        
        short_cnt = len(self.anomalies['too_short'])
        if short_cnt > 0:
            print(f"   ⚠️  너무 짧은 청크 (<{MIN_LENGTH_WARNING}자): {short_cnt}개 (노이즈 가능성)")
            print(f"      예시: {self.anomalies['too_short'][:3]} ...")
        else:
            print(f"   ✅ 너무 짧은 청크 없음")
            
        long_cnt = len(self.anomalies['too_long'])
        if long_cnt > 0:
            print(f"   ⚠️  너무 긴 청크 (>{MAX_LENGTH_WARNING}자): {long_cnt}개 (검색 정확도 저하 위험)")
            print(f"      예시: {self.anomalies['too_long'][:3]} ...")
        else:
            print(f"   ✅ 너무 긴 청크 없음")

        print("="*60)
        
        # 최종 판정
        if self.stats['empty_text'] == 0 and self.stats['missing_meta'] == 0:
            print("🎉 [PASS] 데이터 구조 무결성 검증 통과!")
        else:
            print("🔥 [FAIL] 데이터에 문제가 있습니다. 위 로그를 확인하세요.")

if __name__ == "__main__":
    validator = ChunkValidator(CHUNK_FILE)
    validator.load_data()
    validator.run_checks()
    validator.print_report()
