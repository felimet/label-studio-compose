# label-anything-sam

適用於生產環境的 Label Studio 部署方案，內含可選用的 SAM3 與 SAM2.1 ML 後端。

English version: [README.md](README.md)

## 為何有這個專案

截至 2026-04，上游 [Label Studio ML backend](https://github.com/HumanSignal/label-studio-ml-backend) 尚未提供可直接用於生產部署的 SAM3 整合路徑。本專案提供可落地的完整堆疊：

- 核心服務：Label Studio + PostgreSQL + Redis + MinIO + Nginx + Cloudflare Tunnel
- 可選 GPU 疊加：SAM3 影像/影片後端與 SAM2.1 影像/影片後端
- 以安全為先的預設：S3 最小權限、Token 使用規範、對外暴露邊界

## 快速開始

```bash
git clone https://github.com/felimet/label-anything-sam
cd label-anything-sam

# 1) 核心服務
cp .env.example .env
# 填入所有 <PLACEHOLDER>
# LABEL_STUDIO_USER_TOKEN 必須 <= 40 字元（建議：openssl rand -hex 20）

make up
make init-minio

# 2) 可選 ML 後端（需 GPU）
cp .env.ml.example .env.ml
# 設定 LABEL_STUDIO_API_KEY（Legacy Token）與 HF_TOKEN

make ml-up

# 3) 可選 RedisInsight（Redis GUI）
cp .env.tools.example .env.tools
make tools-up

# 4) 可選 Supabase 管理疊加層（Studio + Meta API）
cp .env.supabase.example .env.supabase
# 啟動前先設定必填值：
#   POSTGRES_PASSWORD=<與 .env 相同密碼>
#   PG_META_CRYPTO_KEY=<至少 32 字元>
make supabase-up

# 5) 可選 Supabase S3 storage profile（進階）
# 需先執行 `make init-minio` 建立 MinIO service account，且需可用的 PostgREST 端點。
# 例如：
#   SUPABASE_STORAGE_POSTGREST_URL=http://<your-postgrest-host>:3000
make supabase-s3-up
```

可選的 Cloudflare Tunnel 管理面路由請直接在 Cloudflare UI 設定（不是填 env 變數），例如：

```text
supabase-studio.example.com -> http://supabase-studio:3000
supabase-meta.example.com   -> http://supabase-meta:8080
redisinsight.example.com    -> http://redisinsight:5540
```

若有調整 `SUPABASE_META_CONTAINER_PORT`，請同步更新 Cloudflare 中 `supabase-meta` 目標埠號。

完整對映與 CF Access 建議請見 [docs/cloudflare-tunnel.md](docs/cloudflare-tunnel.md)。

開啟：

- Label Studio：`http://localhost:18090`
- MinIO Console：`http://localhost:19001`
- MinIO Full Admin UI：`http://localhost:19002`

檢查服務健康：

```bash
make health
```

## 開始前請先注意

- ML 後端必須使用 **Legacy Token**，不可使用 Personal Access Token。
- Label Studio 連 S3 請用 `MINIO_LS_ACCESS_ID` / `MINIO_LS_SECRET_KEY`，不要使用 root 帳密。
- 首次部署完成後，請立即輪換 MinIO service account 密碼。
- 變更 `.env` 後請用 `down` + `up` 重建容器，不要只做 `restart`。

## 環境檔分層

為避免單一 env 檔過長，變數按範圍拆分：

- `.env.example` → `.env`：核心執行堆疊（必填）
- `.env.ml.example` → `.env.ml`：SAM3/SAM2.1 後端（選填）
- `.env.tools.example` → `.env.tools`：RedisInsight 等本機工具（選填）
- `.env.supabase.example` → `.env.supabase`：Supabase 管理疊加層（Studio + Meta API，選填）

`.env.example` 為唯一完整核心模板。

## 依角色閱讀

| 角色 | 起點 | Cookbook | 深入文件 |
|------|------|----------|----------|
| 使用者 / 專案管理者 | [docs/README.md](docs/README.md) | [docs/cookbook/user-cookbook.md](docs/cookbook/user-cookbook.md) | [docs/user-guide.md](docs/user-guide.md) |
| 開發者 | [docs/README.md](docs/README.md) | [docs/cookbook/developer-cookbook.md](docs/cookbook/developer-cookbook.md) | [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) |
| 維運 / SRE | [docs/README.md](docs/README.md) | [docs/cookbook/ops-cookbook.md](docs/cookbook/ops-cookbook.md) | [docs/RUNBOOK.md](docs/RUNBOOK.md) |

## 文件地圖

- [docs/README.md](docs/README.md)：文件入口與閱讀路線
- [docs/user-guide.md](docs/user-guide.md)：使用者流程與管理操作
- [docs/configuration.md](docs/configuration.md)：環境變數單一真相來源
- [docs/architecture.md](docs/architecture.md)：拓撲、資料流與安全設計
- [docs/cloudflare-tunnel.md](docs/cloudflare-tunnel.md)：對外暴露、Tunnel、WAF
- [docs/sam3-backend.md](docs/sam3-backend.md)：SAM3 後端行為與限制
- [docs/sam21-backend.md](docs/sam21-backend.md)：SAM2.1 後端行為與限制
- [docs/RUNBOOK.md](docs/RUNBOOK.md)：維運、事故排除、備份與還原
- [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md)：開發流程與貢獻規範
- [docs/releases/v1.1.0-plan.md](docs/releases/v1.1.0-plan.md)：次版本節奏與遷移里程碑

## 常用 Make 指令（精簡）

- `make up / down / restart / logs / ps`：核心服務生命週期
- `make ml-up / ml-down`：核心服務 + ML 疊加層
- `make tools-up / tools-down / tools-logs`：RedisInsight 本機 GUI 疊加層
- `make supabase-up / supabase-down / supabase-logs`：Supabase 管理疊加層（Studio + Meta API）
- `make supabase-s3-up / supabase-s3-down / supabase-s3-logs`：Supabase 可選 S3 storage profile（storage-api + imgproxy）
- `make build-sam3-image / build-sam3-video / build-sam21-image / build-sam21-video`：建置 ML 映像
- `make test-sam3-image / test-sam3-video / test-sam21-image / test-sam21-video`：執行 ML 後端測試
- `make init-minio`：首次建立 bucket 與 service account
- `make health`：全棧健康檢查

## 授權

Apache-2.0 © 2026 Jia-Ming Zhou
