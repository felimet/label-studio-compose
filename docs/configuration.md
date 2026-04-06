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
| `LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT` | `/label-studio/data/file` | Local files storage 根目錄（**容器內路徑**）。host 目錄 `./label-studio-data/local_storage_file` 透過 volume mount 對應至此路徑，填入 UI 時也須用容器內路徑 |
| `LABEL_STUDIO_USER_TOKEN` | `openssl rand -hex 20` | 首次啟動時寫入 admin 的 legacy API token；**僅首次生效**（見下方說明）。**必須 ≤40 字元**：用 `hex 20`（40 chars），勿用 `hex 32`（64 chars 會靜默破壞首次啟動建立使用者） |
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

<!-- AUTO-GENERATED from .env.ml.example + docker-compose.ml.yml -->
> 所有 SAM3 變數定義於 `.env.ml`（從 `.env.ml.example` 複製後填入）。
> `docker-compose.ml.yml` 透過 `env_file: .env.ml` 載入；`environment:` 區塊僅含靜態值（URL、路徑、port）。

### 必填

| 變數 | 說明 |
|------|------|
| `LABEL_STUDIO_API_KEY` | 兩個 SAM3 後端共用的 LS API 金鑰。建議在 LS UI（Settings → Access Tokens）建立專用 token，與 `LABEL_STUDIO_USER_TOKEN` 分開管理 |
| `HF_TOKEN` | HuggingFace Token；下載 `facebook/sam3.1` 必填（需先接受 Meta 授權） |

### 模型設定

| 變數 | 預設值 | 說明 |
|------|--------|------|
| `SAM3_MODEL_ID` | `facebook/sam3.1` | HuggingFace Hub 模型 ID（影像與影片後端共用） |
| `SAM3_CHECKPOINT_FILENAME` | `sam3.1_multiplex.pt` | 模型 checkpoint 檔名（與 `SAM3_MODEL_ID` 配對，約 3.5 GB） |
| `DEVICE` | `cuda` | `cuda`（GPU）或 `cpu`（備援，極慢）。CUDA 需 NVIDIA driver ≥ 535.x |

### 影片後端設定

| 變數 | 預設值 | 說明 |
|------|--------|------|
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
| `SAM3_ENABLE_FA3` | `false` | 啟用 Flash Attention 3 推論加速。需兩步驟：1) `docker compose build --build-arg ENABLE_FA3=true sam3-video-backend`；2) 此處設 `true`。僅支援 Ampere（A100、RTX 3090）以上 GPU |

### Gunicorn 並發設定

| 變數 | 預設值 | 說明 |
|------|--------|------|
| `WORKERS` | `1` | Gunicorn worker 數（每個 worker preload 完整模型至 VRAM）。單 GPU 建議保持 1；雙 GPU 可設 2 |
| `THREADS` | `8` | 每個 worker 的執行緒數（gthread 模式，共享模型權重，不額外佔用 VRAM）。8-core CPU 適用 8；16-core 可設 16 |

### 日誌與驗證

| 變數 | 預設值 | 說明 |
|------|--------|------|
| `LOG_LEVEL` | `INFO` | 日誌層級（`DEBUG`、`INFO`、`WARNING`、`ERROR`） |
| `BASIC_AUTH_USER` | — | ML backend API 的 Basic Auth 帳號（選填；留空則停用） |
| `BASIC_AUTH_PASS` | — | ML backend API 的 Basic Auth 密碼（選填；留空則停用） |

> `TIMEOUT` 由 Dockerfile ENV 定義（image: 120 s、video: 300 s），可在 `docker-compose.ml.yml` 的 `environment:` 區塊中覆寫。
<!-- END AUTO-GENERATED -->

### 說明

- 兩個後端各自維護獨立的 `sam3-image-models` / `sam3-video-models` Volume 儲存下載的權重。
- 共用 `hf-cache` Volume（`/home/appuser/.cache/huggingface`）避免重複下載 HF 元資料。
- 首次啟動下載約 3.5 GB 權重；健康檢查設 `start_period: 300s` 留足緩衝。

## 產生強密碼

```bash
# Django Secret Key
openssl rand -hex 32

# LABEL_STUDIO_USER_TOKEN  ← 必須 ≤40 chars；Token field max_length=40
openssl rand -hex 20

# 資料庫 / Redis / MinIO 密碼
openssl rand -base64 24
```

## 資料目錄說明

<!-- AUTO-GENERATED from docker-compose.yml volumes -->
所有資料均以 bind mount 方式存在專案根目錄，重裝系統或搬機時直接複製資料夾即可：

| 目錄（host） | 容器內路徑 | 對應服務 | 內容 | 備份方式 |
|-------------|-----------|----------|------|----------|
| `./label-studio-data/` | `/label-studio/data/` | label-studio | 標注資料、匯出檔 | 直接 `tar` 壓縮 |
| `./label-studio-data/local_storage_file/` | `/label-studio/data/file/` | label-studio | **Local files storage 根目錄**（詳見下方說明） | 直接 `tar` 壓縮 |
| `./postgres-data/` | `/var/lib/postgresql/data/` | db | 資料庫（PostgreSQL 內部格式）| **必須用 `pg_dump`**，不可直接複製 |
| `./minio-data/` | `/data/` | minio | 上傳的影像、影片等媒體檔案 | 直接 `tar` 壓縮 |
| `./redis-data/` | `/data/` | redis | 任務佇列暫存（重啟後自動恢復）| 通常不需備份 |
<!-- END AUTO-GENERATED -->

> `postgres-data/` 是 PostgreSQL 的二進位內部格式，**直接複製無法還原**。備份請用 `pg_dump`（見 [Runbook → Backup](RUNBOOK.md#backup)）。

## 兩種媒體存放方式：MinIO vs Local Files

Label Studio 支援兩種方式存放要標注的媒體檔案，適用情境不同：

| | **MinIO** | **Local Files Storage** |
|---|---|---|
| 存放位置 | `./minio-data/`（物件儲存） | `./label-studio-data/local_storage_file/` |
| 如何放入 | 透過 Label Studio UI 上傳 | 直接複製到 host 資料夾 |
| 適合情境 | 需要遠端多人協作、檔案從網路上傳 | 本機已有大量資料、不想重複上傳 |
| 存取方式 | 透過 Presigned URL（可公開分享） | 僅限容器內路徑存取 |
| 容量限制 | 受磁碟空間限制，但可擴充 | 同上 |

**一般建議**：資料已在本機（如 `D:\datasets\`）→ 用 Local Files；需要從其他電腦上傳或多人共用 → 用 MinIO。

---

## Local Files Storage：直接掛載本機資料夾，不用上傳

只要把本機已有的資料夾掛進容器，Label Studio 就能直接讀取，**省去上傳步驟**。

### 運作方式

```
本機                              容器內
./label-studio-data/
  local_storage_file/     →      /label-studio/data/file/
    project1/                      project1/
      img001.jpg                     img001.jpg
      img002.jpg                     img002.jpg
```

`docker-compose.yml` 已設定：
```yaml
- ./label-studio-data/local_storage_file:/label-studio/data/file
```

### 使用步驟

1. 把要標注的資料放到 host 上的 `./label-studio-data/local_storage_file/` 子目錄，例如：
   ```
   label-studio-data/local_storage_file/project1/img001.jpg
   ```

2. 在 Label Studio UI → Project → Cloud Storage → Add Source Storage：
   - Storage type：**Local files**
   - Absolute local path：`/label-studio/data/file/project1`（容器內路徑）

3. 點 **Sync**，Label Studio 即可讀取該目錄下的所有檔案，無需上傳。

> **注意**：UI 中填的是**容器內路徑**（`/label-studio/data/file/...`），不是 host 路徑。

### 想掛載其他本機資料夾

若資料已存放在本機其他位置（例如 `D:\datasets\my-project`），可在 `docker-compose.yml` 額外新增一個 volume：

```yaml
label-studio:
  volumes:
    - ./label-studio-data/local_storage_file:/label-studio/data/file
    - D:\datasets\my-project:/label-studio/data/file/my-project  # 額外掛載
```

掛載後在 UI 中填 `/label-studio/data/file/my-project` 即可，資料完全不需要複製或上傳。

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
