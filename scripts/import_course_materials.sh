#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

python -m src.skills.course_material_import.scripts.import_course_materials_cli \
  --persist-dir "${PROJECT_ROOT}/chroma_db_mm" \
  --output-root "${PROJECT_ROOT}/output/course_mm" \
  --notes-files \
    "${PROJECT_ROOT}/data/pdf/notes1_2022.pdf" \
    "${PROJECT_ROOT}/data/pdf/notes7_2022.pdf" \
  --textbook-file "${PROJECT_ROOT}/data/pdf/textbook.pdf" \
  --mm-text-collection "textbook_mm_text_embeddings" \
  --mm-image-collection "textbook_mm_image_embeddings" \
  --ocr-text-collection "textbook_ocr_text_embeddings" \
  --pdf-workers 1 \
  --mm-text-batch-size 8 \
  --mm-text-workers 4 \
  --mm-image-batch-size 1 \
  --mm-image-workers 2 
