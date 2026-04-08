# DigitalTwin API 与命令文档（当前实现）

## 服务入口

- 启动服务：`python -m src.run_server`
- 默认地址：`http://localhost:8080`
- 助教页面：`http://localhost:8080/tutor`

## HTTP 接口

### 1. 分身会话

- `POST /chat`
  - 请求：
    ```json
    {
      "message": "你好",
      "session_id": "session_123",
      "persona_id": "optional"
    }
    ```
  - 返回：`status/reply/session_id/debug`

- `POST /reset`
  - 请求：
    ```json
    {
      "session_id": "session_123"
    }
    ```

### 2. 分身管理

- `GET /api/personas`
- `DELETE /api/personas/<persona_id>`

### 3. 助教会话

- `POST /tutor/chat`
  - 请求：
    ```json
    {
      "message": "课程成绩构成是什么",
      "session_id": "tutor-123",
      "stream": true
    }
    ```
  - `stream=true` 时使用 SSE 推送：`token/sources/images/done`

- `POST /tutor/reset`
  - 请求：
    ```json
    {
      "session_id": "tutor-123"
    }
    ```

- `GET /tutor/stats`
  - 返回集合统计与 agent 状态：`text_records/image_records/ocr_records/agent_ready/...`

- `POST /tutor/import`
  - 后台触发 `data/pdf/*.pdf` 导入（服务内异步）

## 导入命令

### 1. 微信聊天导入（分身）

```bash
python -m src.skills.wechat_csv_import.scripts.import_wechat_csv_cli
```

### 2. 课程资料导入（助教）

默认增量导入：

```bash
python -m src.skills.course_material_import.scripts.import_course_materials_cli \
  --persist-dir ./chroma_db_mm \
  --notes-files data/pdf/notes1_2022.pdf data/pdf/notes7_2022.pdf \
  --textbook-file data/pdf/textbook.pdf
```

全量重建：

```bash
python -m src.skills.course_material_import.scripts.import_course_materials_cli \
  --persist-dir ./chroma_db_mm \
  --notes-files data/pdf/notes1_2022.pdf data/pdf/notes7_2022.pdf \
  --textbook-file data/pdf/textbook.pdf \
  --reset
```

仅导入 notes：

```bash
python -m src.skills.course_material_import.scripts.import_course_materials_cli \
  --persist-dir ./chroma_db_mm \
  --notes-files data/pdf/notes1_2022.pdf data/pdf/notes7_2022.pdf \
  --skip-textbook
```

## 关键环境变量

- `DASHSCOPE_API_KEY`
- `CHROMA_PERSIST_DIR`
- `EMBED_MODEL`
- `MM_EMBED_MODEL`
- `TUTOR_MM_TEXT_COLLECTION`
- `TUTOR_MM_IMAGE_COLLECTION`
- `TUTOR_OCR_TEXT_COLLECTION`

注意：`CHROMA_PERSIST_DIR` 必须与导入命令的 `--persist-dir` 一致。
