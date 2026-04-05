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

> **重要：** `MINIO_EXTERNAL_HOST` 必須可從瀏覽器端解析。MinIO 用此值產生 Presigned URL，Label Studio 內部請求仍走 `http://minio:9000`。

## Label Studio

| 變數 | 範例 | 說明 |
|------|------|------|
| `LABEL_STUDIO_HOST` | `https://label-studio.example.com` | 對外公開 URL；用於 CSRF 信任來源 |
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

### `LABEL_STUDIO_ENABLE_LEGACY_API_TOKEN`

已設為 `true`，允許使用靜態 bearer token（`Authorization: Token <value>`）。此選項為相容性設定——Label Studio 正逐步推行 JWT / Personal Access Token；若未來官方 deprecate，需改走 PAT 流程。

## Cloudflare Tunnel

| 變數 | 說明 |
|------|------|
| `CLOUDFLARE_TUNNEL_TOKEN` | Zero Trust 儀表板產生的 Tunnel Token |
| `MINIO_EXTERNAL_HOST` | MinIO 公開網域（同上，雙重用途） |

詳細設定步驟見 [cloudflare-tunnel.md](cloudflare-tunnel.md)。

## SAM3 ML 後端

| 變數 | 預設值 | 說明 |
|------|--------|------|
| `HF_TOKEN` | — | HuggingFace Token；下載 `facebook/sam3` 必填 |
| `LABEL_STUDIO_API_KEY` | — | sam3-ml-backend 使用的 LS API 金鑰。建議建立專用 token（Settings → Access Tokens），與 `LABEL_STUDIO_USER_TOKEN` 分開管理 |
| `SAM3_MODEL_ID` | `facebook/sam3` | HuggingFace Hub 模型 ID |
| `DEVICE` | `cuda` | `cuda`（GPU）或 `cpu`（備援） |
| `EMBED_CACHE_SIZE` | `80` | 記憶體中最大快取影像數 |
| `EMBED_CACHE_TTL` | `300` | 快取 TTL（秒） |

## 產生強密碼

```bash
# Django Secret Key / LABEL_STUDIO_USER_TOKEN
openssl rand -hex 32

# 資料庫 / Redis / MinIO 密碼
openssl rand -base64 24
```

## 從舊版 Named Volume 遷移到 Bind Mount

若先前已用預設 named volume 跑過 stack，切換到 `./label-studio-data` bind mount 前需手動搬資料：

```bash
# 1. 匯出舊 volume 內容
docker compose cp label-studio:/label-studio/data/. ./label-studio-data/

# Linux / WSL — 確保 appuser（UID 1001）可寫入
sudo chown -R 1001:1001 label-studio-data

# 2. 停止 stack 並移除舊 named volume
docker compose down
docker volume rm label-studio_label-studio-data   # 名稱視 project name 而定

# 3. 建立 Local files 子目錄（若不存在）
mkdir -p label-studio-data/file

# 4. 重啟
docker compose up -d
```

> **⚠️ 風險提示**：`docker volume rm` 前務必確認資料已完整複製到 `./label-studio-data`。Windows Docker Desktop 不受 chown 影響，可略過步驟 2。
