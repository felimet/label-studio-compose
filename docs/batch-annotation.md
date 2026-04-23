# 批次標注指南

使用 `scripts/batch_annotate.py`（或 port 8085 的 Web UI）對整個 Label Studio 專案執行
SAM3/SAM2.1 推論，無需逐一手動開啟任務。

---

## 前置需求

| 需求 | 確認方式 |
|---|---|
| ML 後端已啟動 | `curl http://localhost:9090/health` |
| Label Studio 已啟動 | `curl http://localhost:8080/health` |
| 已設定 `LABEL_STUDIO_API_KEY` | `echo $LABEL_STUDIO_API_KEY` |
| 專案標注設定含 `<BrushLabels>` | Label Studio UI → Settings → Labeling |

---

## 快速入門

### CLI

```bash
# 演習模式 — 列出任務但不執行標注
python scripts/batch_annotate.py --project-id 1 --backend sam3 --dry-run

# 標注所有任務
python scripts/batch_annotate.py --project-id 1 --backend sam3

# Makefile 捷徑
make batch-annotate PROJECT_ID=1
```

### Web UI（無需終端機）

batch-server 隨 `make up` 一起啟動；也可以單獨重建：

```bash
make batch-server
```

在瀏覽器開啟 `http://<your-server>:8085` 並填寫表單即可提交批次任務。

---

## CLI 參數說明

| 旗標 | 預設值 | 說明 |
|---|---|---|
| `--project-id INT` | （必填）| Label Studio 專案 ID |
| `--backend sam3\|sam21` | `sam3` | ML 後端類型 |
| `--backend-url URL` | `$ML_BACKEND_URL` 或 `http://localhost:9090` | ML 後端端點 |
| `--ls-url URL` | `$LABEL_STUDIO_URL` 或 `http://localhost:8080` | Label Studio 端點 |
| `--dry-run` | 關閉 | 列出任務但不呼叫 ML 後端 |
| `--force` | 關閉 | 對已有人工標注的任務也寫入預測 |
| `--confirm-force` | 關閉 | 使用 `--force` 時必須同時加上此旗標 |
| `--concurrency N` | `1` | 並行 HTTP 請求數 |
| `--max-tasks INT` | （全部）| 限制處理任務數 |
| `--resume` | 關閉 | 從上次成功的任務繼續 |
| `--resume-file PATH` | `scripts/.batch_resume.json` | 續接狀態檔路徑 |
| `--text-prompt STR` | （空）| SAM3 文字提示（描述要分割的物件，例：`cow, grass, fence`）|
| `--task-ids STR` | （空）| 針對特定 task ID 處理（例：`1, 3, 17`）|
| `--confidence FLOAT` | `0.5` | SAM3 信心閾值（僅 SAM3 有效）|
| `--sam21-mode grid` | 關閉 | 啟用 SAM2.1 格點模式（實驗性）|
| `--grid-n INT` | `3` | SAM2.1 格點邊長（N×N 個點）|
| `--basic-auth-user STR` | 空 | ML 後端 HTTP Basic Auth 帳號 |
| `--basic-auth-pass STR` | 空 | ML 後端 HTTP Basic Auth 密碼 |

---

## 認證

API 金鑰**僅**從 `LABEL_STUDIO_API_KEY` 環境變數讀取。金鑰不會被記錄，也不會作為
CLI 引數傳遞。

```bash
export LABEL_STUDIO_API_KEY=your-legacy-token-here
python scripts/batch_annotate.py --project-id 1
```

> **⚠ 必須使用 Legacy Token，不得使用 Personal Access Token（JWT Bearer）。**
> CLI 以 `Authorization: Token <key>` 呼叫 Label Studio API，僅 Legacy Token
> 符合此格式。PAT 格式不同，會導致 401。

Docker Compose 環境下，`LABEL_STUDIO_API_KEY` 從 `.env.ml` 自動注入 batch-server
容器（`env_file: .env.ml required: false`），無需在 `.env` 重複設定。

---

## SAM3 vs SAM2.1

| 特性 | SAM3（`--backend sam3`）| SAM2.1（`--backend sam21`）|
|---|---|---|
| 上下文類型 | 文字提示（label 名稱）| 格點幾何（grid keypoints）|
| 多物件 | 支援（每個 label 一個遮罩）| 不支援（每張影像一個遮罩，取最高分）|
| 準確度 | 較高 | 較低（幾何啟發式）|
| 啟用方式 | 預設 | 須加 `--sam21-mode grid` |

> **SAM2.1 為實驗性功能。** 每張影像只回傳**一個遮罩**（最高分點），
> 不適用於多物件或空景場景。

---

## 並行處理

```bash
# 4 個並行請求（需在 .env.ml 設定 SAM3_IMAGE_WORKERS=4）
python scripts/batch_annotate.py --project-id 1 --concurrency 4
```

後端 `WORKERS` 決定真正的並行數量。若 `WORKERS=1`（預設），
設定 `--concurrency > 1` 只會讓請求排隊，不會提升吞吐量。

確認 `.env.ml`：

```
SAM3_IMAGE_WORKERS=4
```

---

## 任務狀態說明

| 狀態 | 含義 |
|---|---|
| `success` | 預測已寫入 Label Studio |
| `skip_human` | 任務已有人工標注，略過（加 `--force` 可覆寫）|
| `skip_race` | 在列表與寫入之間出現標注（TOCTOU 視窗）|
| `zero_match` | ML 後端回傳零個預測（確認 label 名稱與信心閾值）|
| `fail` | HTTP 錯誤或網路異常 |

> **`--force` 安全說明**：`--force --confirm-force` 同時使用時，預測與現有人工標注**並存**，
> **不會刪除任何人工標注**。

---

## 結束代碼

| 代碼 | 含義 |
|---|---|
| `0` | 全部成功（`zero_match` 計為成功）|
| `1` | 部分失敗 |
| `2` | 全部失敗 |
| `3` | 設定錯誤（缺少引數、API 金鑰未設定）|
| `4` | 認證失敗（API 金鑰無效）|

---

## 續接中斷的批次

```bash
# 從上次成功處繼續
python scripts/batch_annotate.py --project-id 1 --resume
```

狀態儲存於 `scripts/.batch_resume.json`（已排除在 git 外，本地跨 session 保留）。

---

## Web UI（batch_server）

已整合至 `docker-compose.yml`，與主服務一起啟動。
瀏覽器表單位於 `http://<host>:8085`。

**API 端點：**

```
GET  /                      → HTML 表單
POST /batch                 → 啟動批次，回傳 {"job_id": "..."}
GET  /batch/{job_id}/status → 輪詢進度與日誌
POST /batch/{job_id}/stop  → 終止執行中的批次作業
GET  /health                → {"status": "ok"}
```

![batch-annotation-ui](../image/batch-annotation-ui.png)

**Web UI 表單欄位說明：**

| 欄位 | 說明 |
|---|---|
| Project ID | Label Studio 專案 ID（可從 URL `/projects/{id}` 取得，必填）|
| Backend | SAM3（建議，使用文字提示）或 SAM2.1（實驗性，使用格點幾何）|
| ML Backend URL | ML 後端 `/predict` 端點完整 URL（預設指向 Docker Compose 服務名稱）|
| Text Prompt | 自由文字描述欲分割的物件（例：`cow, grass, fence`）。SAM3 將此作為唯一輸入上下文。SAM3 必填；SAM2.1 忽略 |
| SAM2.1 Mode | 僅選擇 SAM2.1 時顯示。`grid` 對每張影像送出 3×3 格點 (不推薦) |
| Confidence | 分數閾值（0–1）。低於此值的預測會被捨棄。降低可取得更多遮罩，但可能包含雜訊 |
| Max tasks | 限制處理任務數（留空代表全部）|
| Task IDs | 指定僅處理特定的 Task ID（逗號分隔，例：`1, 3, 17`）。若填寫，過濾條件優先於 `Max tasks` |
| Dry run | 演習模式：不呼叫 ML 後端，不寫入預測。用於確認任務數量與設定 |
| Force overwrite | 對已有人工標注的任務也寫入預測（預測與現有標注**並存**，不會刪除任何人工標注）|
| Auth (Username / Password) | ML 後端 HTTP Basic Auth 帳密（選填）。對應 `.env.ml` 中的 `BASIC_AUTH_USER` / `BASIC_AUTH_PASS` |
| Stop 按鈕 | 終止執行中的作業（先送 SIGTERM，5 秒後 SIGKILL）。僅在作業執行中時顯示 |

---

## 常見問題排查

### 所有任務都回傳 `zero_match`

1. 確認 label 名稱與標注設定中 `<Label value="...">` 完全相符（大小寫敏感）
2. 降低 `--confidence`（例如 `--confidence 0.3`）
3. 確認 ML 後端正在運行：`curl http://localhost:9090/health`
4. 先用單任務測試：`--max-tasks 1 --dry-run`，再去掉 `--dry-run`

### 每個任務都是 `skip_race`

另一個行程或使用者正在對同一專案並行標注。**每個專案同時只應有一個批次 CLI 行程執行。**

### Connection refused

- `--ls-url` 或 `--backend-url` 指向錯誤的主機/埠號
- 確認兩個服務都在啟動批次前已正常運行

---

## 注意事項

- **不要對同一專案同時執行多個批次行程。** `delete_cli_predictions()` 使用
  `model_version` 範圍化，但並行刪除可能競爭並刪除對方剛寫入的預測。
- **若使用者正在主動標注，不要執行批次**（除非使用 `--force`）。從列出任務到寫入
  預測之間存在微小的 TOCTOU 視窗。草稿標注（尚未提交）不受保護。
