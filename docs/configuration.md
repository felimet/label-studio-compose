# 環境變數設定說明

所有變數定義於 `.env`（從 `.env.example` 複製後填入）。

> **官方參考**：[Label Studio 部署指南](https://github.com/HumanSignal/label-studio/tree/4b222c5d63acd150277cc43d8326269ef567b595/docs/source/guide) — 環境變數、儲存後端、部署選項的原始說明文件。

## PostgreSQL

以下為使用者填入 `.env` 的變數；compose 會將 `POSTGRES_*` 轉譯成 Label Studio 內部所需的 `POSTGRE_*` 格式（注意：LS 官方 env var 名稱無 `SDB` 中綴）。

| 變數 | 預設值 | 說明 |
|------|--------|------|
| `POSTGRES_USER` | `labelstudio` | 資料庫使用者名稱 |
| `POSTGRES_PASSWORD` | — | **必填。** 強隨機密碼 |
| `POSTGRES_DB` | `labelstudio` | 資料庫名稱 |

## Redis

| 變數 | 說明 |
|------|------|
| `REDIS_PASSWORD` | **必填。** Redis AUTH 密碼 |

## MinIO

| 變數 | 範例 | 說明 |
|------|------|------|
| `MINIO_ROOT_USER` | — | 管理員帳號 |
| `MINIO_ROOT_PASSWORD` | — | 管理員密碼（≥8 字元） |
| `MINIO_BUCKET` | `label-studio-bucket` | 儲存桶名稱；由 `make init-minio` 自動建立 |
| `MINIO_EXTERNAL_HOST` | `minio.example.com` | 對外公開網域；嵌入 Presigned URL |

> **CORS**：MinIO 開源版已移除 S3 `PutBucketCors` API。CORS 改由 docker-compose.yml 的 `MINIO_API_CORS_ALLOW_ORIGIN=*` 環境變數控制，無需在 `make init-minio` 中設定。

> **重要：** `MINIO_EXTERNAL_HOST` 必須可從瀏覽器端解析。MinIO 用此值產生 Presigned URL，Label Studio 內部請求仍走 `http://minio:9000`。

## Label Studio

| 變數 | 範例 | 說明 |
|------|------|------|
| `LABEL_STUDIO_HOST` | `https://label-studio.example.com` | 對外公開 URL；同時作為 `CSRF_TRUSTED_ORIGINS` 的值（需含 `https://`） |
| `LABEL_STUDIO_SECRET_KEY` | `openssl rand -hex 32` | Django Session 金鑰 |
| `LABEL_STUDIO_USERNAME` | `admin@example.com` | 初始管理員 Email |
| `LABEL_STUDIO_PASSWORD` | — | 初始管理員密碼 |
| `LABEL_STUDIO_DISABLE_SIGNUP_WITHOUT_LINK` | `true` | 關閉公開註冊（需邀請連結） |
| `LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT` | `/label-studio/data/file` | Local files storage 根目錄；須為 `./label-studio-data` 的子路徑（容器內 `/label-studio/data/file`） |
| `LABEL_STUDIO_USER_TOKEN` | `openssl rand -hex 32` | 首次啟動時寫入 admin 的 legacy API token；**僅首次生效**（見下方說明） |
| `DATA_UPLOAD_MAX_NUMBER_FILES` | `100000` | Django 上傳檔案數上限；預設 100 過低，修復 [issue #6777](https://github.com/HumanSignal/label-studio/issues/6777) |
| `JSON_LOG` | `1` | 輸出結構化 JSON log；LS 原始碼支援但官方文件未列 |

### `LABEL_STUDIO_USER_TOKEN` 注意事項

`LABEL_STUDIO_USER_TOKEN`、`LABEL_STUDIO_USERNAME`、`LABEL_STUDIO_PASSWORD` 均為 **first-boot only**：只在 Postgres 資料庫為空（首次初始化）時寫入。若 stack 已跑過一次，這些變數在後續啟動時被忽略。

- **全新部署**：在 `.env` 填好後直接 `docker compose up -d`，admin 帳號與 token 均自動寫入。
- **已跑過 stack**：透過 LS UI（Settings → Access Tokens）手動建立 token，並填入 `.env` 的 `LABEL_STUDIO_API_KEY`；或 `docker compose down -v` 重置資料庫（**⚠️ 會刪除所有標注資料**）。

### CSRF 設定說明

Label Studio 讀取的 env var 是 `CSRF_TRUSTED_ORIGINS`（**非** `DJANGO_CSRF_TRUSTED_ORIGINS`）。
值必須含 scheme，例如 `https://label-studio.example.com`，否則登入時會出現 **403 CSRF verification failed**。

此值由 docker-compose.yml 自動從 `LABEL_STUDIO_HOST` 繼承，無需額外設定。

### `LABEL_STUDIO_ENABLE_LEGACY_API_TOKEN`

已設為 `true`，允許使用靜態 bearer token（`Authorization: Token <value>`）。此選項為相容性設定——Label Studio 正逐步推行 JWT / Personal Access Token；若未來官方 deprecate，需改走 PAT 流程。

## Cloudflare Tunnel

| 變數 | 說明 |
|------|------|
| `CLOUDFLARE_TUNNEL_TOKEN` | Zero Trust 儀表板產生的 Tunnel Token |
| `MINIO_EXTERNAL_HOST` | MinIO 公開網域（同上，雙重用途） |

詳細設定步驟見 [cloudflare-tunnel.md](cloudflare-tunnel.md)。

## SAM3 ML 後端

<!-- AUTO-GENERATED from .env.example + docker-compose.ml.yml -->
### 必填

| 變數 | 說明 |
|------|------|
| `HF_TOKEN` | HuggingFace Token；下載 `facebook/sam3.1` 必填（需先接受 Meta 授權） |
| `LABEL_STUDIO_API_KEY` | 兩個 SAM3 後端共用的 LS API 金鑰。建議在 LS UI（Settings → Access Tokens）建立專用 token，與 `LABEL_STUDIO_USER_TOKEN` 分開管理 |

### 模型設定

| 變數 | 預設值 | 說明 |
|------|--------|------|
| `SAM3_IMAGE_MODEL_ID` | `facebook/sam3.1` | 影像後端 HuggingFace Hub 模型 ID |
| `SAM3_VIDEO_MODEL_ID` | `facebook/sam3.1` | 影片後端 HuggingFace Hub 模型 ID |
| `DEVICE` | `cuda` | `cuda`（GPU）或 `cpu`（備援，極慢） |
| `MAX_FRAMES_TO_TRACK` | `10` | 影片後端每次 predict 最多追蹤畫格數 |

### PCS（文字概念提示）設定

| 變數 | 預設值 | 說明 |
|------|--------|------|
| `SAM3_ENABLE_PCS` | `true` | 啟用自然語言文字提示（PCS）功能。`false` = 退化為 SAM2 幾何模式 |
| `SAM3_CONFIDENCE_THRESHOLD` | `0.5` | 文字提示偵測的最低置信分數（`0`–`1`；越低偵測越多但可能有假陽性） |
| `SAM3_RETURN_ALL_MASKS` | `false` | `true` = 回傳所有偵測實例；`false` = 只回傳得分最高的一個 |

### Flash Attention 3 設定（影片後端）

| 變數 | 預設值 | 說明 |
|------|--------|------|
| `SAM3_ENABLE_FA3` | `false` | 啟用 Flash Attention 3 推論加速（需在 build-time 先傳入 `--build-arg ENABLE_FA3=true`） |

### Gunicorn 並發設定

| 變數 | 預設值 | 說明 |
|------|--------|------|
| `ML_WORKERS` | `1` | Gunicorn workers 數（每個 worker preload 完整模型至 VRAM）。單 GPU 建議保持 1；雙 GPU 可設 2 |
| `ML_THREADS` | `8` | 每個 worker 的執行緒數（gthread 模式，共享模型權重，不額外佔用 VRAM）。8-core CPU 適用 8；16-core 可設 16 |
| `ML_BASIC_AUTH_USER` | — | ML backend API 的 Basic Auth 帳號（選填） |
| `ML_BASIC_AUTH_PASS` | — | ML backend API 的 Basic Auth 密碼（選填） |

> `TIMEOUT` 由 Dockerfile ENV 定義（image: 120 s、video: 300 s），可在 compose 的 environment 區塊中覆寫。
<!-- END AUTO-GENERATED -->

### 說明

- 兩個後端各自維護獨立的 `sam3-image-models` / `sam3-video-models` Volume 儲存下載的權重。
- 共用 `hf-cache` Volume（`/home/appuser/.cache/huggingface`）避免重複下載 HF 元資料。
- 首次啟動下載約 3.5 GB 權重；健康檢查設 `start_period: 300s` 留足緩衝。

## 產生強密碼

```bash
# Django Secret Key / LABEL_STUDIO_USER_TOKEN
openssl rand -hex 32

# 資料庫 / Redis / MinIO 密碼
openssl rand -base64 24
```

## 資料目錄說明

<!-- AUTO-GENERATED from docker-compose.yml volumes -->
所有資料均以 bind mount 方式存在專案根目錄，重裝系統或搬機時直接複製資料夾即可：

| 目錄 | 對應服務 | 內容 | 備份方式 |
|------|----------|------|----------|
| `./label-studio-data/` | label-studio | 標注資料、匯出檔、Local files | 直接 `tar` 壓縮 |
| `./postgres-data/` | db | 資料庫（PostgreSQL 內部格式）| **必須用 `pg_dump`**，不可直接複製 |
| `./minio-data/` | minio | 上傳的影像、影片等媒體檔案 | 直接 `tar` 壓縮 |
| `./redis-data/` | redis | 任務佇列暫存（重啟後自動恢復）| 通常不需備份 |
<!-- END AUTO-GENERATED -->

> `postgres-data/` 是 PostgreSQL 的二進位內部格式，**直接複製無法還原**。備份請用 `pg_dump`（見 [Runbook → Backup](RUNBOOK.md#backup)）。

## 從舊版 Named Volume 遷移到 Bind Mount

若先前已用 Docker named volume 跑過 stack，切換前需手動搬資料：

```bash
docker compose down

# Label Studio 媒體資料
docker compose cp label-studio:/label-studio/data/. ./label-studio-data/
mkdir -p label-studio-data/file

# PostgreSQL（直接複製二進位檔案）
docker run --rm \
  -v label-studio_postgres-data:/src \
  -v "$(pwd)/postgres-data:/dst" \
  alpine sh -c "cp -a /src/. /dst/"

# MinIO 媒體檔案
docker run --rm \
  -v label-studio_minio-data:/src \
  -v "$(pwd)/minio-data:/dst" \
  alpine sh -c "cp -a /src/. /dst/"

# redis-data 不需遷移（任務佇列重啟後自動恢復）

docker compose up -d
```

> Windows Docker Desktop 不需要 `chown`，可略過 Linux/WSL 的權限相關步驟。
