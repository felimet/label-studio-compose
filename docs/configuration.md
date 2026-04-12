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
| `MINIO_ROOT_USER` | — | 管理員帳號（僅用於 minio-init 初始化及 Admin UI） |
| `MINIO_ROOT_PASSWORD` | — | 管理員密碼（≥8 字元） |
| `MINIO_BUCKET` | `default-bucket` | 儲存桶名稱（逗號分隔可列多個，例如 `default-bucket,test`）；所列 bucket 均由 `make init-minio` 自動建立，service account 亦同時取得所有 bucket 的存取權 |
| `MINIO_EXTERNAL_HOST` | `minio.example.com` | 對外公開網域；嵌入 Presigned URL |
| `MINIO_LS_ACCESS_ID` | `openssl rand -hex 10` | Label Studio 專用 access key（最小權限，限 `MINIO_BUCKET` 所列的所有 bucket）。由 `minio-init` 建立；**設定 LS Cloud Storage 時使用此帳號，不要用 root** |
| `MINIO_LS_SECRET_KEY` | `openssl rand -hex 20` | 對應 secret key（≥8 字元） |
| `MINIO_BUCKET_QUOTA_GB` | `200` | Bucket 容量上限（GiB）。**留空停用** |

> **CORS**：MinIO 開源版已移除 S3 `PutBucketCors` API。CORS 改由 docker-compose.yml 的 `MINIO_API_CORS_ALLOW_ORIGIN=*` 環境變數控制，無需在 `make init-minio` 中設定。

> **重要：** `MINIO_EXTERNAL_HOST` 必須可從瀏覽器端解析。MinIO 用此值產生 Presigned URL，Label Studio 內部請求仍走 `http://minio:9000`。

> **服務帳號安全**：`MINIO_ROOT_USER`/`MINIO_ROOT_PASSWORD` 應僅用於 Admin UI 管理，不應填入 Label Studio Cloud Storage 設定。Label Studio 連線 MinIO 請使用 `MINIO_LS_ACCESS_ID`/`MINIO_LS_SECRET_KEY`，此帳號只有 `MINIO_BUCKET` 的 Get/Put/Delete/List 權限，root 帳密不會因 LS 被入侵而外洩。

### 取得 Access Key 的兩種方式

**方式 A — `make init-minio`（自動化）**

在 `.env` 填好 `MINIO_LS_ACCESS_ID` / `MINIO_LS_SECRET_KEY` 後執行：

```bash
make init-minio
```

`minio-init` 會自動建立對應使用者並套用 bucket-scoped policy。完成後直接在 LS Cloud Storage 填入這組 key。

**方式 B（推薦） — Admin UI（手動）**

適合不想在 `.env` 硬寫 key、或需要多組 key 的情境：

1. 以 root 帳號登入 **Full Admin UI**（`http://localhost:19002`）
2. **Identity → Users → Create User**：建立一個非 root 使用者並指派 policy
3. 以**該使用者**登入 Admin UI（或 Console，`http://localhost:19001`）
4. 右上角頭像 → **Access Keys → Create Access Key**
5. 複製 **Access Key**（對應 LS 的 *Access Key ID*）與 **Secret Key**
6. 在 LS Cloud Storage 設定頁填入這兩個值

> **⚠️ 重要：首次部署或建立使用者後，務必修改密碼。**
> 以該使用者身分登入 Admin UI → 右上角頭像 → **Access Keys → Change Password**，將初始密碼換成強隨機密碼並妥善保存。
> `make init-minio` 自動建立的帳號密碼即 `.env` 的 `MINIO_LS_SECRET_KEY`，建議部署完成後透過此流程輪換，並更新 LS Cloud Storage 設定與 `.env`。

### MinIO Bucket Access Policy

在 MinIO Admin UI 或 Console 的 Buckets 頁面，每個 bucket 可設定 **Access Policy**：

| Policy | 說明 | 適用情境 |
|--------|------|----------|
| `private` | 所有操作皆須驗證（預設） | **Label Studio 標準配置** — LS 透過 Presigned URL 存取，不需公開讀 |
| `public` | 任何人皆可未經驗證讀取（HTTP GET） | 靜態資源公開分享；**⚠️ 不建議用於標注資料** |
| `custom` | 自訂 IAM JSON policy | 需要細粒度控制時（例如：指定 IP 範圍、特定 prefix 公開） |

**為何選 `private`**：Label Studio 以 Presigned URL 方式存取 MinIO 物件，URL 本身已內嵌時效性簽名；bucket 不需設為 public。`make init-minio` 建立的 `MINIO_LS_ACCESS_ID` service account 僅有 `MINIO_BUCKET` 的 Get/Put/Delete/List 權限，root 帳密不暴露於 LS。

**Access Policy 與 Bucket Encryption 無關**：Policy 控制「誰可以存取」，Encryption 控制「資料如何靜態加密」，兩者獨立設定。

### Bucket Encryption（SSE-S3 / SSE-KMS）

在 Admin UI → Buckets → 選擇 bucket → **Encryption** 頁籤可設定：

| 加密類型 | 說明 |
|----------|------|
| **Default（無加密）** | 不啟用靜態加密；資料以明文儲存於 `./minio-data/`（仍受 Linux 檔案系統權限保護）。適合本機/實驗環境 |
| **SSE-S3（MinIO managed）** | MinIO 用自身管理的 master key 在寫入時加密每個物件。需在 `.env` 設定 `MINIO_KMS_SECRET_KEY`（格式：`<key-name>:<base64-32-byte-key>`）。適合不需外部 KMS 的場景 |
| **SSE-KMS（外部 KMS）** | 透過外部 KMS（如 HashiCorp Vault）管理 master key；適合企業合規要求。本 stack 未內建 KES，需另行部署 |

**SSE-S3 注意事項**：

```bash
# 產生 MINIO_KMS_SECRET_KEY
printf 'minio-sse-key:'; openssl rand -base64 32
```

- 啟用後**已存在的未加密物件**會觸發 `PrefixAccessDenied`（MinIO 嘗試以 KMS 解密既有明文物件）。建議在空 bucket 或全新部署時才啟用 SSE-S3。
- 若已有資料需加密，先備份，再以加密設定重建 stack，最後重新上傳資料。

## Label Studio

| 變數 | 範例 | 說明 |
|------|------|------|
| `LABEL_STUDIO_HOST` | `https://label-studio.example.com` | 對外公開 URL；同時作為 `CSRF_TRUSTED_ORIGINS` 的值（需含 `https://`） |
| `LABEL_STUDIO_SECRET_KEY` | `openssl rand -hex 32` | Django Session 金鑰 |
| `LABEL_STUDIO_USERNAME` | `admin@example.com` | 初始管理員 Email |
| `LABEL_STUDIO_PASSWORD` | — | 初始管理員密碼 |
| `LABEL_STUDIO_DISABLE_SIGNUP_WITHOUT_LINK` | `true` | 關閉公開註冊（需邀請連結） |
| `LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT` | `/label-studio/data/file` | Local files storage 根目錄（**容器內路徑**）。host 目錄 `./ls-data/file` 透過 volume mount 對應至此路徑，填入 UI 時也須用容器內路徑 |
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
**Last Updated:** 2026-04-12 (GPU assignment vars; per-service WORKERS; MAX_FRAME_LONG_SIDE; TORCH_DTYPE; sm_75+ minimum enforced at startup)

> 所有 SAM3 變數定義於 `.env.ml`（從 `.env.ml.example` 複製後填入）。
> `docker-compose.ml.yml` 透過 `env_file: .env.ml` 載入；`environment:` 區塊僅含靜態值（URL、路徑、port）。

### 必填

| 變數 | 說明 |
|------|------|
| `LABEL_STUDIO_API_KEY` | 兩個 SAM3 後端共用的 LS API 金鑰。**必須使用 Legacy Token**（LS UI → Account & Settings → Legacy Token）。ML backend SDK 以 `Authorization: Token <key>` 格式驗證，Personal Access Token（JWT Bearer）會導致 401。此值與 `.env` 的 `LABEL_STUDIO_USER_TOKEN` 相同 |
| `HF_TOKEN` | HuggingFace Token；下載 SAM3 模型必填（需先接受 Meta 授權） |

### 模型設定

影像與影片後端使用**不同架構**的 checkpoint，請分別設定：

| 變數 | 預設值 | 說明 |
|------|--------|------|
| `SAM3_IMAGE_MODEL_ID` | `facebook/sam3` | 影像後端 HuggingFace Hub 模型 ID |
| `SAM3_IMAGE_CHECKPOINT_FILENAME` | `sam3.pt` | 影像後端 checkpoint（約 3.45 GB）。對應 `build_sam3_image_model`，**無 SAM3.1 影像版本** |
| `SAM3_VIDEO_MODEL_ID` | `facebook/sam3.1` | 影片後端 HuggingFace Hub 模型 ID |
| `SAM3_VIDEO_CHECKPOINT_FILENAME` | `sam3.1_multiplex.pt` | 影片後端 checkpoint（約 3.5 GB）。對應 `build_sam3_multiplex_video_predictor` |
| `DEVICE` | `cuda` | `cuda`（GPU）或 `cpu`（備援，極慢）。CUDA 需 NVIDIA driver ≥ 535.x 且 **Turing（sm_75+）或更新的 GPU**。Volta（sm_70/72）與 Pascal（GTX 10xx, sm_61）不支援——SAM3 的 `torch.ops.aten._addmm_activation` bfloat16 kernel 需要 sm_75+；啟動時拋出 `RuntimeError` |

> 若未設定 `SAM3_IMAGE_*` / `SAM3_VIDEO_*`，會 fallback 讀取共用的 `SAM3_MODEL_ID` / `SAM3_CHECKPOINT_FILENAME`（向下相容）。

### GPU 指定

兩個後端可獨立 pin 到不同實體 GPU。`start.sh` 在容器啟動時讀取 `SAM3_*_GPU_INDEX`（由 `env_file: .env.ml` 提供）並將其 export 為 `CUDA_VISIBLE_DEVICES`。若 INDEX 含多個 GPU，gunicorn `post_fork` hook 再對多 worker 做一對一分配（worker *i* → `gpus[i-1]`，每個 worker 內部的 `cuda:0` = 分配到的實體 GPU）：

| 變數 | 預設 | 說明 |
|------|------|------|
| `SAM3_IMAGE_GPU_INDEX` | `0` | 影像後端的 GPU 編號（`nvidia-smi` index）。多 GPU 以逗號分隔（如 `0,1`）；留空 = 暴露所有 GPU（單 GPU 環境適用） |
| `SAM3_VIDEO_GPU_INDEX` | `1` | 影片後端的 GPU 編號。多 GPU 以逗號分隔（如 `1,2,3`）；留空 = 暴露所有 GPU |

> **多 GPU 範例**：影像用 GPU 0、影片分散在 GPU 1+2+3：設 `SAM3_IMAGE_GPU_INDEX=0` / `SAM3_IMAGE_WORKERS=1`；`SAM3_VIDEO_GPU_INDEX=1,2,3` / `SAM3_VIDEO_WORKERS=3`。

### 影片後端設定

| 變數 | 預設值 | 說明 |
|------|--------|------|
| `MAX_FRAMES_TO_TRACK` | `10` | **雙重作用：** (1) 每次 predict 傳給 `propagate_in_video` 的 `max_frame_num_to_track`；(2) 記憶體預算 — 後端只抽取 `[start_frame, last_frame + MAX_FRAMES_TO_TRACK + 1)` 的畫格至暫存資料夾，再以 image folder 模式開啟 SAM3 session，避免長影片 OOM。數值越小追蹤越短但記憶體越低。 |
| `MAX_FRAME_LONG_SIDE` | `1024` | 抽取畫格的長邊上限（像素）。`0` = 不縮放。SAM3 ViT attention 複雜度正比於 spatial tokens 的平方，1080p 降至 1024 可節省約 3× VRAM；建議保持 `1024` |

### PCS（文字概念提示）設定

| 變數 | 預設值 | 說明 |
|------|--------|------|
| `SAM3_ENABLE_PCS` | `true` | 啟用自然語言文字提示（PCS）功能，支援純文字（text-only）與 text+geo mixed 兩種路徑。`false` = 幾何提示專用模式（影片後端退化為 SAM2 fallback 幾何模式） |
| `SAM3_CONFIDENCE_THRESHOLD` | `0.5` | 文字提示偵測的最低置信分數（`0`–`1`；越低偵測越多但可能有假陽性） |
| `SAM3_RETURN_ALL_MASKS` | `false` | `true` = 回傳所有偵測實例；`false` = 只回傳得分最高的一個 |

### Flash Attention 3 設定（影片後端）

| 變數 | 預設值 | 說明 |
|------|--------|------|
| `SAM3_ENABLE_FA3` | `false` | 啟用 Flash Attention 3 推論加速。需兩步驟：1) 建置時加 `--build-arg ENABLE_FA3=true`；2) 此處設 `true`。**僅支援 NVIDIA Hopper（H100/H800，sm_90+）**；`flash_attn_interface` 非 pip 套件，需自行 build，除非具備 Hopper 硬體與自訂 build 否則留 `false` |

### Gunicorn 並發設定

| 變數 | 預設值 | 說明 |
|------|--------|------|
| `SAM3_IMAGE_WORKERS` | `1` | 影像後端 gunicorn worker 數（每個 worker 獨立載入模型至 VRAM）。設為 `SAM3_IMAGE_GPU_INDEX` 中的 GPU 數量；單 GPU 保持 `1` |
| `SAM3_VIDEO_WORKERS` | `1` | 影片後端 gunicorn worker 數。設為 `SAM3_VIDEO_GPU_INDEX` 中的 GPU 數量；單 GPU 保持 `1` |
| `TORCH_DTYPE` | `（空）` | 覆寫自動精度偵測。留空 = 自動（Ampere sm_80+ → bfloat16 + TF32；Turing sm_75–79 → bfloat16；< sm_75 → 啟動中止）。可設 `fp16` 強制 float16 或 `bf16` 強制 bfloat16 |
| `THREADS` | `8` | 每個 worker 的執行緒數（gthread 模式，共享模型權重，不額外佔用 VRAM）。8-core CPU 適用 8；16-core 可設 16 |

### 日誌與驗證

| 變數 | 預設值 | 說明 |
|------|--------|------|
| `GPU_IDLE_TIMEOUT_SECS` | `3600` | 無推論請求後模型從 VRAM 自動卸載的等待秒數（預設 1 小時）。Watchdog 每 60 秒檢查一次，實際釋放最多延遲 60 秒 |
| `LOG_LEVEL` | `INFO` | 日誌層級（`DEBUG`、`INFO`、`WARNING`、`ERROR`） |
| `BASIC_AUTH_USER` | — | ML backend API 的 Basic Auth 帳號（選填；留空則停用） |
| `BASIC_AUTH_PASS` | — | ML backend API 的 Basic Auth 密碼（選填；留空則停用） |

> `TIMEOUT` 由 Dockerfile ENV 定義（image: 120 s、video: 300 s），可在 `docker-compose.ml.yml` 的 `environment:` 區塊中覆寫。
<!-- END AUTO-GENERATED -->

### 說明

- SAM3 兩個後端共用 `hf-cache` Volume（`/home/appuser/.cache/huggingface`）。Checkpoint 在**首次推論時**透過 `hf_hub_download` 下載至 HF cache，回傳 blob 路徑供模型載入。
- 首次啟動：影像後端下載 ~3.45 GB，影片後端下載 ~3.5 GB；健康檢查設 `start_period: 300s` 留足緩衝。

## SAM2.1 ML 後端

<!-- AUTO-GENERATED from .env.ml.example + docker-compose.ml.yml -->
**Last Updated:** 2026-04-13

> 所有 SAM2.1 變數定義於 `.env.ml`（從 `.env.ml.example` 複製後填入）。
> Checkpoint 在 **build time** 下載至 `model-cache` Volume（`/data/models`），runtime 直接讀取，不需網路。

### 模型選擇

| 變數 | 預設值 | 說明 |
|------|--------|------|
| `SAM21_DEFAULT_MODEL` | `sam2.1_hiera_large` | 後端啟動時的預設模型。可在標注介面的 Choices checkbox 切換；切換後系統記憶最後一次選擇。可選值：`sam2.1_hiera_tiny` / `sam2.1_hiera_small` / `sam2.1_hiera_base_plus` / `sam2.1_hiera_large` |

可選 checkpoint 規格：

| 模型 key | 參數量 | Checkpoint 大小 |
|----------|--------|----------------|
| `sam2.1_hiera_tiny` | ~38M | ~155 MB |
| `sam2.1_hiera_small` | ~46M | ~185 MB |
| `sam2.1_hiera_base_plus` | ~80M | ~325 MB |
| `sam2.1_hiera_large` | ~224M | ~900 MB |

> SAM2.1 repos 為公開存取，不需接受 Meta 授權即可下載。提供 `HF_TOKEN` 可提升 HuggingFace 速率限制。Build time 執行 `download_models.py`（primary）或 `download_ckpts.sh`（fallback，Meta CDN）。

### GPU 指定

| 變數 | 預設 | 說明 |
|------|------|------|
| `SAM21_IMAGE_GPU_INDEX` | `（空）` | 影像後端 GPU 編號（`nvidia-smi` index）。多 GPU 逗號分隔；留空 = 暴露所有 GPU |
| `SAM21_VIDEO_GPU_INDEX` | `（空）` | 影片後端 GPU 編號。留空 = 暴露所有 GPU |

### Gunicorn 並發設定

| 變數 | 預設值 | 說明 |
|------|--------|------|
| `SAM21_IMAGE_WORKERS` | `1` | 影像後端 gunicorn worker 數。設為 `SAM21_IMAGE_GPU_INDEX` 的 GPU 數量 |
| `SAM21_VIDEO_WORKERS` | `1` | 影片後端 gunicorn worker 數。設為 `SAM21_VIDEO_GPU_INDEX` 的 GPU 數量 |

> `DEVICE`、`THREADS`、`GPU_IDLE_TIMEOUT_SECS`、`LOG_LEVEL`、`BASIC_AUTH_USER`、`BASIC_AUTH_PASS` 由 SAM3 與 SAM2.1 共用，定義見上方 SAM3 節。

### Volume 說明

| Volume | 掛載路徑 | 服務 | 說明 |
|--------|---------|------|------|
| `hf-cache` | `/home/appuser/.cache/huggingface` | sam3-image, sam3-video | SAM3 runtime 下載快取 |
| `model-cache` | `/data/models` | sam21-image, sam21-video | SAM2.1 build-time 下載的 `.pt` 檔案（image 與 video 共用） |

<!-- END AUTO-GENERATED -->

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
| `./ls-data/` | `/label-studio/data/` | label-studio | 標注資料、匯出檔 | 直接 `tar` 壓縮 |
| `./ls-data/file/` | `/label-studio/data/file/` | label-studio | **Local files storage 根目錄**（詳見下方說明） | 直接 `tar` 壓縮 |
| `./postgres-data/` | `/var/lib/postgresql/data/` | db | 資料庫（PostgreSQL 內部格式） | **必須用 `pg_dump`**，不可直接複製 |
| `./minio-data/` | `/data/` | minio | 上傳的影像、影片等媒體檔案 | 直接 `tar` 壓縮 |
| `./redis-data/` | `/data/` | redis | 任務佇列暫存（重啟後自動恢復） | 通常不需備份 |
| `./scripts/` | `/scripts/` | minio-init（唯讀） | MinIO 初始化腳本 | 版本控制中，無需另行備份 |
<!-- END AUTO-GENERATED -->

> `postgres-data/` 是 PostgreSQL 的二進位內部格式，**直接複製無法還原**。備份請用 `pg_dump`（見 [Runbook → Backup](RUNBOOK.md#backup)）。

## `pg-db` vs `ls-data` — 資料分層說明

兩者存放的是完全不同層次的資料：

### `pg-db`（PostgreSQL，`./postgres-data/`）

存放 Label Studio 的**應用程式元資料**，純關聯式結構：

- 專案定義（Project）、標注模板（Labeling Config XML）
- 任務清單（Task）：每筆任務的 URL/路徑指標，**不含媒體檔案本體**
- 標注結果（Annotation）：標注者畫的框、分割 mask、標籤 JSON
- 使用者帳號、組織、權限
- ML backend 設定、預測結果（Prediction）
- 匯入/匯出歷史記錄

本質上是「誰、在哪個任務、標了什麼」的索引與結果。

### `ls-data`（Label Studio 應用層，`./ls-data/`）

存放 Label Studio **應用程式自身產生的檔案**：

- 匯出的標注檔（JSON/CSV/COCO/YOLO 等）
- Local files storage 的媒體檔案（放在 `./ls-data/file/`）
- LS 內部 cache、session 檔

### 對比

| | `pg-db` | `ls-data` |
|--|---------|-----------|
| 格式 | PostgreSQL 二進位（不可直接複製） | 普通檔案系統 |
| 內容 | 元資料、標注 JSON | 媒體檔、匯出檔 |
| 備份方式 | **必須** `pg_dump` | 直接 `tar` |
| 如果丟失 | 所有標注結果、任務清單消失 | 媒體檔與匯出結果消失，但標注 JSON 仍在 pg-db |

### 媒體檔案在哪裡？

取決於使用哪種 storage：

- **MinIO**：媒體本體在 `./minio-data/`，pg-db 只存 S3 URL 指標
- **Local Files**：媒體本體在 `./ls-data/file/`，pg-db 存容器內路徑

三個目錄缺一不可：pg-db 是索引，minio-data 與 ls-data/file 是實體，遺失任何一個都會導致標注工作流斷裂。

## 兩種媒體存放方式：MinIO vs Local Files

Label Studio 支援兩種方式存放要標注的媒體檔案，適用情境不同：

| | **MinIO** | **Local Files Storage** |
|---|---|---|
| 存放位置 | `./minio-data/`（物件儲存） | `./ls-data/file/` |
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
./ls-data/
  file/                   →      /label-studio/data/file/
    project1/                      project1/
      img001.jpg                     img001.jpg
      img002.jpg                     img002.jpg
```

`docker-compose.yml` 已設定：
```yaml
- ./ls-data/file:/label-studio/data/file:rw
```

### 使用步驟

1. 把要標注的資料放到 host 上的 `./ls-data/file/` 子目錄，例如：
   ```
   ls-data/file/project1/img001.jpg
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
    - ./ls-data/file:/label-studio/data/file:rw
    - D:\datasets\my-project:/label-studio/data/file/my-project:ro  # 額外掛載（唯讀）
```

掛載後在 UI 中填 `/label-studio/data/file/my-project` 即可，資料完全不需要複製或上傳。

## 從舊版 Named Volume 遷移到 Bind Mount

若先前已用 Docker named volume 或舊版 `./ls-data/` 跑過 stack，切換前需手動搬資料：

```bash
docker compose down

# Label Studio 媒體資料（從舊目錄重新命名）
mv ./ls-data ./ls-data
mkdir -p ls-data/file

# 若舊版已有 local_storage_file，確保結構正確
# 舊：./ls-data/local_storage_file/
# 新：./ls-data/file/
# 如無自動完成，手動遷移：
# mv ./ls-data/local_storage_file/* ./ls-data/file/ (if exists)

# 從 named volume 遷移（若先前使用 named volume）
docker run --rm \
  -v label-studio_ls-data:/src \
  -v "$(pwd)/ls-data:/dst" \
  alpine sh -c "cp -a /src/. /dst/"

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
