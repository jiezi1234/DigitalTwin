#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export CHROMA_PERSIST_DIR="${PROJECT_ROOT}/chroma_db_mm"
export PDF_EXPORT_ROOT="${PROJECT_ROOT}/output/course_mm"
export TUTOR_MM_TEXT_COLLECTION="textbook_mm_text_embeddings"
export TUTOR_MM_IMAGE_COLLECTION="textbook_mm_image_embeddings"
export TUTOR_OCR_TEXT_COLLECTION="textbook_ocr_text_embeddings"
export TUTOR_VL_MODEL="qwen-vl-plus"

python -m src.run_server
