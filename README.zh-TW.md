# label-anything-sam

[Label Studio](https://labelstud.io) SAM3/3.1 完整部署方案：PostgreSQL · Redis · MinIO (S3) · Nginx · Cloudflare Tunnel · SAM3/3.1 互動式影像分割。

> 截至 2026 年 4 月 11 日 [Label Studio ml-backend](https://github.com/HumanSignal/label-studio-ml-backend) 尚未更新加入 [Meta (Facebook) SAM 3: Segment Anything with Concepts](https://github.com/facebookresearch/sam3) 後端分割模型，故在本 repo 中提供基於 SAM 3 的 Label Studio ML Backend 自訂實作，供有需求的使用者直接整合。並盡可能的符合 [Label Studio ml-backend](https://github.com/HumanSignal/label-studio-ml-backend) 資料夾結構，詳見 `./ml-backends`。

> **English documentation** → [README.md](README.md)

## 服務架構

| 服務 | 映像 | 用途 | 資料儲存說明連結  |
|------|------|------|----------|
| `label-studio` | `heartexlabs/label-studio:latest` | 標注 UI + API | [`./ls-data/`](docs/configuration.md#pg-db-vs-ls-data--資料分層說明) — 匯出檔、local files |
| `pg-db` | `postgres:17` | 資料庫 | [`./postgres-data/`](docs/configuration.md#pg-db-vs-ls-data--資料分層說明) — 任務、標注結果、使用者 |
| `redis` | `redis:8.6.2` | 任務佇列 / 快取 | [`./redis-data/`](docs/configuration.md#資料目錄說明) — 暫存佇列狀態 |
| `minio` | `firstfinger/minio:latest` | S3 相容物件儲存 + 完整 Admin UI（port 9002） | [`./minio-data/`](docs/configuration.md#minio) — 媒體檔案 |
| `minio-init` | `minio/mc:RELEASE.2025-08-13T08-35-41Z` | 一次性 bucket 初始化 + service account + quota | — |
| `nginx` | `nginx:1.28.3-alpine3.23` | 反向代理 | — |
| `cloudflared` | `cloudflare/cloudflared:2026.3.0` | Zero Trust Tunnel | — |
| `sam3-image-backend` | (自訂建置) | SAM3 影像分割 → BrushLabels *(需 GPU，可選)* | `hf-cache`（共用 volume）— 模型權重 |
| `sam3-video-backend` | (自訂建置) | SAM3 影片物件追蹤 → VideoRectangle *(需 GPU，可選)* | `hf-cache`（共用 volume）— 模型權重 |

> **MinIO CE 說明**：MinIO 於 2025-05-24 從社群版移除全部 Admin UI，並於 2025-09-07 後停止推送 CE Docker image。本 stack 改用 `firstfinger/minio`——每日從上游原始碼自動建置並恢復完整 Admin Console（port 9001 Console、port 9002 Full Admin UI）。參考：[Harsh-2002/MinIO](https://github.com/Harsh-2002/MinIO)

## 前置需求

- Docker Engine ≥ 26 + Docker Compose v2
- NVIDIA GPU + `nvidia-container-toolkit`（僅 SAM3 後端需要）
- Cloudflare 帳號，已開啟 Zero Trust
- HuggingFace 帳號，已同意 Meta `facebook/sam3.1` 使用條款

## 快速開始

```bash
git clone https://github.com/felimet/label-anything-sam
cd label-anything-sam

# 1. 核心服務
cp .env.example .env
$EDITOR .env           # 填入所有 <PLACEHOLDER> 值
                       # LABEL_STUDIO_USER_TOKEN: openssl rand -hex 20（必須 ≤40 字元）

make up                # 啟動核心服務（管理員帳號於首次啟動時自動建立）
make init-minio        # 建立 S3 儲存桶 + 存取政策

# 2. 取得 Label Studio API Token（SAM3 後端需要）
#    登入 → 右上角頭像 → Account & Settings → Legacy Token → 複製
#    ⚠ 必須使用 Legacy Token（不可用 Personal Access Token）——ML SDK 傳送
#      "Authorization: Token <key>"；PAT 使用 JWT Bearer → 401 Unauthorized。

# 3. SAM3 ML 後端（可選，需 NVIDIA GPU）
cp .env.ml.example .env.ml
$EDITOR .env.ml        # 填入 LABEL_STUDIO_API_KEY（步驟 2）及 HF_TOKEN

make ml-up
```

在 Label Studio 中連接 MinIO 儲存：
**專案 → Settings → Cloud Storage → Add Source Storage → S3**
（endpoint: `http://minio:9000`，使用 `MINIO_LS_ACCESS_ID` / `MINIO_LS_SECRET_KEY`——由 `make init-minio` 建立的最小權限 service account。**請勿**填入 root 帳密）

> **⚠️ 首次部署後：** 立即輪換 service account 密碼。
> Admin UI（`http://localhost:19002`）→ 右上角頭像 → **Access Keys → Change Password**
> 同步更新 `.env` 的 `MINIO_LS_SECRET_KEY` 與 LS Cloud Storage 設定。

## Makefile 指令

| 指令 | 說明 |
|------|------|
| `up / down / restart / logs / ps` | 核心服務生命週期管理 |
| `ml-up / ml-down` | SAM3 ML 疊加層（影像 + 影片） |
| `build-sam3-image / build-sam3-video` | 建置 ML 後端映像 |
| `test-sam3-image / test-sam3-video` | 在容器內執行 pytest |
| `init-minio` | 一次性儲存桶初始化 |
| `create-admin` | 建立管理員帳號 |
| `health` | 檢查所有服務狀態 |
| `push` | git add + commit + push |

## 文件

| 文件 | 內容 |
|------|------|
| [docs/configuration.md](docs/configuration.md) | `.env` 環境變數說明 · [MinIO 存取政策](docs/configuration.md#minio-bucket-access-policy) · [Bucket 加密](docs/configuration.md#bucket-encryptionsse-s3--sse-kms) · [pg-db vs ls-data](docs/configuration.md#pg-db-vs-ls-data--資料分層說明) |
| [docs/cloudflare-tunnel.md](docs/cloudflare-tunnel.md) | Zero Trust 設定 + WAF 規則 + 替代方案 |
| [docs/sam3-backend.md](docs/sam3-backend.md) | SAM3 模型設定 + 標注流程 |
| [docs/architecture.md](docs/architecture.md) | 服務拓撲、Volume、網路 |
| [docs/RUNBOOK.md](docs/RUNBOOK.md) | 營運指南（健康檢查、升級、排除故障） |

## 授權

Apache-2.0 © 2026 Jia-Ming Zhou
