#!/usr/bin/env python3
"""
업스테이지 parser(2025 스펙) + 후처리 파이프라인을 한 번에 실행하는 CLI 스크립트.

기능 요약
---------
1. 지정한 PDF를 batch-size 단위로 분할
2. 각 분할본을 Upstage document-digitization API로 분석하여 JSON 생성
3. 기존 2025 파이프라인(`누락_데이터.py`) 로직을 이용해 텍스트/이미지/표를 정리하고
   Gemini를 활용한 요약과 Markdown을 생성
4. `document_analysis_results.*`, `text_summaries.json`, `image_summaries.json`,
   `table_markdowns.json` 등 산출물을 저장

환경 변수
---------
* `UPSTAGE_API_KEY` : Upstage Document API Key (필수)
* `GEMINI_API_KEY`  : Google Gemini API Key (멀티모달 요약 시 필수)
  - 기본적으로 `.env2` 파일을 로드하며, `--env-file` 인자로 다른 파일을 지정할 수 있음

사용 예시
---------
python History_Docent/01_Data_Preprocessing/UPSTAGE_parser_pipeline.py \\
    --base-dir History_Docent/1_Data_Preprocessing/조선편_2025 \\
    --pdf data/벌거벗은한국사-조선편.pdf
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Iterable, List

import pymupdf
import requests
from dotenv import load_dotenv
import re
import shutil
import unicodedata

# 한글 파일명을 그대로 사용하므로 번역 기능 제거


# ---------------------------------------------------------------------------
# 환경 변수 & 모듈 로드
# ---------------------------------------------------------------------------

def load_env_file(env_path: Path | None) -> None:
    """지정된 env 파일을 우선 로드하고, 없으면 기본 .env 탐색."""
    if env_path and env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()  # fallback (.env 등)


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise EnvironmentError(f"환경 변수 {name} 이(가) 설정되지 않았습니다.")
    return value


def load_parser_module(base_dir: Path):
    """누락_데이터.py 모듈을 importlib으로 불러옵니다."""
    target = base_dir / "누락_데이터.py"
    if not target.exists():
        raise FileNotFoundError(f"parser module not found: {target}")

    spec = importlib.util.spec_from_file_location("parser2025", target)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


# ---------------------------------------------------------------------------
# Upstage API 호출 & 데이터 준비
# ---------------------------------------------------------------------------

UPSTAGE_ENDPOINT = "https://api.upstage.ai/v1/document-digitization"


def split_pdf(pdf_path: Path, batch_size: int) -> list[Path]:
    """PDF를 batch-size 단위로 분할하고 경로 목록을 반환합니다."""
    data_dir = pdf_path.parent
    pattern = f"{pdf_path.stem}_*.pdf"
    split_files = sorted(data_dir.glob(pattern))
    if split_files:
        return split_files

    print("⚠️  분할 PDF가 없어 새로 생성합니다.")
    doc = pymupdf.open(pdf_path)
    try:
        num_pages = len(doc)
        for start in range(0, num_pages, batch_size):
            end = min(start + batch_size, num_pages) - 1
            out_path = data_dir / f"{pdf_path.stem}_{start:04d}_{end:04d}.pdf"
            with pymupdf.open() as out_doc:
                out_doc.insert_pdf(doc, from_page=start, to_page=end)
                out_doc.save(out_path)
            split_files.append(out_path)
            print("생성:", out_path)
    finally:
        doc.close()
    return split_files


def call_upstage(pdf_file: Path, api_key: str, timeout: int = 300) -> Path:
    """단일 분할 PDF를 Upstage API로 분석해 JSON을 생성합니다."""
    headers = {"Authorization": f"Bearer {api_key}"}
    data = {
        "model": "document-parse",
        "ocr": "force",
        "coordinates": True,
        "chart_recognition": True,
        "merge_multipage_tables": True,
        "output_formats": ["html", "markdown"],
        "base64_encoding": ["figure", "table"],
    }
    files = {"document": open(pdf_file, "rb")}
    try:
        response = requests.post(
            UPSTAGE_ENDPOINT,
            headers=headers,
            data=data,
            files=files,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Upstage 요청 실패: {exc}") from exc
    finally:
        files["document"].close()

    if response.status_code != 200:
        try:
            detail = response.json()
        except Exception:
            detail = response.text
        raise RuntimeError(
            f"Upstage 분석 실패 ({response.status_code}): {detail}"
        )

    output_file = pdf_file.with_suffix(".json")
    with open(output_file, "w", encoding="utf-8") as fout:
        json.dump(response.json(), fout, ensure_ascii=False)
    print(f"✅ Upstage 분석 완료: {output_file}")
    return output_file


def ensure_layout_json(
    split_files: Iterable[Path],
    api_key: str,
    force: bool = False,
) -> list[Path]:
    """분할 PDF 목록에 대해 JSON이 없으면 Upstage 분석을 수행합니다."""
    json_files: list[Path] = []
    for pdf in split_files:
        json_path = pdf.with_suffix(".json")
        if json_path.exists() and not force:
            print(f"기존 JSON 사용: {json_path}")
        else:
            json_path = call_upstage(pdf, api_key)
        json_files.append(json_path)
    return json_files


# ---------------------------------------------------------------------------
# 파이프라인 실행
# ---------------------------------------------------------------------------

def run_pipeline(
    parser_module,
    pdf_path: Path,
    split_files: list[Path],
    batch_size: int,
    skip_gemini: bool,
):
    """누락_데이터.py의 함수들을 이용해 후처리를 실행합니다."""
    GraphState = parser_module.GraphState  # type: ignore[attr-defined]

    with pymupdf.open(pdf_path) as doc:
        num_pages = len(doc)

    state = GraphState(
        filepath=str(pdf_path),
        batch_size=batch_size,
        split_filepaths=[str(p) for p in split_files],
        page_numbers=list(range(num_pages)),
    )

    print("📄 JSON 복원...")
    state.update(parser_module.restore_state_from_files(state))

    print("📊 페이지 메타데이터 추출...")
    state.update(parser_module.extract_page_metadata(state))

    print("🔍 페이지 요소 추출...")
    state.update(parser_module.extract_page_elements(state))

    print("📝 페이지 텍스트 재구성...")
    state.update(parser_module.extract_page_text(state))

    print("🧾 텍스트 요약 생성...")
    state.update(parser_module.create_text_summary(state))

    print("🖼️ 이미지/표 크롭...")
    state.update(parser_module.crop_image(state))
    state.update(parser_module.crop_table(state))

    print("📦 요약 배치 생성...")
    state.update(parser_module.create_image_summary_data_batches(state))
    state.update(parser_module.create_table_summary_data_batches(state))

    if skip_gemini:
        print("⚠️  --skip-gemini 옵션으로 멀티모달 요약을 건너뜁니다.")
    else:
        print("🤖 Gemini 이미지 요약...")
        state.update(parser_module.create_image_summary(state))

        print("🤖 Gemini 테이블 요약...")
        state.update(parser_module.create_table_summary(state))

        print("📝 Gemini 테이블 Markdown 변환...")
        state.update(parser_module.create_table_markdown(state))

    print("💾 결과 저장...")
    parser_module.save_results(state)
    print("✅ 완료! 결과가 저장되었습니다.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Upstage + Gemini 통합 파이프라인")
    parser.add_argument(
        "--template-dir",
        default="History_Docent/1_Data_Preprocessing/조선편_2025",
        help="파이프라인 템플릿 디렉토리 (누락_데이터.py 포함)",
    )
    parser.add_argument(
        "--pdf",
        default=None,
        help="단일 PDF 경로 (base-dir 기준 상대 경로 허용)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="PDF 분할 시 사용할 페이지 수",
    )
    parser.add_argument(
        "--env-file",
        default=".env2",
        help="환경 변수 로드를 위한 파일 경로 (기본값: .env2)",
    )
    parser.add_argument(
        "--force-layout",
        action="store_true",
        help="기존 JSON이 있어도 Upstage 분석을 다시 수행합니다.",
    )
    parser.add_argument(
        "--input-dir",
        default="History_Docent/PDF_history",
        help="복수 PDF 처리 시 사용할 디렉토리",
    )
    parser.add_argument(
        "--output-root",
        default="History_Docent/01_Data_Preprocessing",
        help="결과 저장 루트 폴더",
    )
    parser.add_argument(
        "--skip-gemini",
        action="store_true",
        help="Gemini 이미지/표 요약 단계를 건너뜁니다.",
    )
    return parser


def compute_target_paths(pdf_path: Path, output_root: Path, translator=None) -> Path:
    """원본 파일명을 그대로 폴더명으로 사용합니다 (한글 포함)."""
    # 확장자를 제거한 파일명을 그대로 사용
    folder_name = pdf_path.stem
    return output_root / folder_name


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)

    load_env_file(Path(args.env_file) if args.env_file else None)
    upstage_key = require_env("UPSTAGE_API_KEY")
    if not args.skip_gemini:
        require_env("GEMINI_API_KEY")

    pdfs: list[Path]
    if args.pdf:
        pdf_path = Path(args.pdf).resolve()
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        pdfs = [pdf_path]
    else:
        input_dir = Path(args.input_dir).resolve()
        if not input_dir.exists():
            raise FileNotFoundError(f"입력 디렉토리를 찾을 수 없습니다: {input_dir}")
        pdfs = sorted(input_dir.glob("*.pdf"))
        if not pdfs:
            raise FileNotFoundError("입력 디렉토리에 PDF가 없습니다.")

    template_dir = Path(args.template_dir).resolve()
    parser_module = load_parser_module(template_dir)

    for pdf_path in pdfs:
        target_base = compute_target_paths(
            pdf_path,
            Path(args.output_root).resolve(),
        )
        
        # 이미 완료된 PDF는 건너뛰기
        result_file = target_base / "document_analysis_results.json"
        if result_file.exists():
            print(f"⏭️  이미 완료된 PDF 건너뛰기: {pdf_path.name} (결과 파일 존재: {result_file})")
            continue
        
        data_dir = target_base / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        local_pdf = data_dir / pdf_path.name
        if not local_pdf.exists() or os.path.getmtime(pdf_path) > os.path.getmtime(local_pdf):
            shutil.copy2(pdf_path, local_pdf)

        original_cwd = Path.cwd()
        try:
            os.chdir(target_base)
            split_files = split_pdf(local_pdf, args.batch_size)
            ensure_layout_json(split_files, upstage_key, force=args.force_layout)
            run_pipeline(
                parser_module=parser_module,
                pdf_path=local_pdf,
                split_files=split_files,
                batch_size=args.batch_size,
                skip_gemini=args.skip_gemini,
            )
        finally:
            os.chdir(original_cwd)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("중단되었습니다.")

