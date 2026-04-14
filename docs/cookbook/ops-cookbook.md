# 維運 Cookbook

本文件面向維運與 SRE，提供任務導向的操作劇本。
若需要完整背景、邊界條件與長版排障，請搭配 [../RUNBOOK.md](../RUNBOOK.md)。

## 任務 1：每日健康巡檢

### 目標
在 3 分鐘內確認核心服務與 ML 服務是否健康。

### 步驟
```bash
make ps
make health
```

### 驗證
- 核心服務狀態為 `Up` 或 `healthy`
- `minio-init` 一次性任務為 `Exited (0)` 視為正常

### 異常處理
- 先看即時日誌：`make logs`
- 單服務深入：`docker compose logs -f <service-name>`

---

## 任務 2：版本升級（低風險流程）

### 目標
在不破壞資料的前提下升級服務映像。

### 步驟
1. 先備份（至少 PostgreSQL dump + MinIO 檔案備份）
2. 拉取新映像
```bash
docker compose pull
```
3. 逐步重啟關鍵服務
```bash
docker compose up -d --no-deps label-studio
docker compose up -d --no-deps nginx
docker compose up -d --no-deps cloudflared
```
4. 驗證
```bash
make health
```

### 回復策略
- 若升級後異常，先回退版本標籤再重啟服務
- 若涉及 DB migration，必要時從備份還原

---

## 任務 3：快速事故排查（高頻）

### 症狀 A：ML 後端 401 Unauthorized
- 檢查 `.env.ml` 的 `LABEL_STUDIO_API_KEY` 是否為 Legacy Token
- 確認沒有誤用 Personal Access Token

### 症狀 B：S3 資料無法載入
- 檢查 Storage endpoint 是否 `http://minio:9000`
- 檢查 service account 對 bucket 權限
- 檢查 `MINIO_EXTERNAL_HOST` 是否可從瀏覽器解析（Presigned URL 模式）

### 症狀 C：GPU OOM
- 降低 workers（`SAM3_*_WORKERS` / `SAM21_*_WORKERS`）
- 調低追蹤長度或畫面尺寸（影片任務）
- 必要時切換到 CPU（效能會顯著下降）

### 症狀 D：CSRF 403
- 檢查 `LABEL_STUDIO_HOST` 是否為正確 `https://` URL
- 檢查反向代理是否正確帶入 `X-Forwarded-Proto`

---

## 任務 4：備份與還原演練

### 目標
建立可驗證的資料保全流程。

### 建議備份
```bash
# PostgreSQL（建議每日）
docker compose exec db pg_dump -U labelstudio labelstudio > backup-$(date +%Y%m%d).sql

# Label Studio 檔案資料
tar -czf ls-data-$(date +%Y%m%d).tar.gz ./ls-data/

# MinIO 媒體資料
tar -czf minio-data-$(date +%Y%m%d).tar.gz ./minio-data/
```

### 還原重點
- PostgreSQL 要用 SQL dump 還原，不要直接複製 `postgres-data` 目錄
- 還原後先跑 `make health`，再做抽樣任務驗證

---

## 任務 5：服務回滾

### 目標
當升級造成中斷時，快速回到可用版本。

### 步驟
1. 將版本標籤回退至前一版
2. 逐服務重啟（先 app 再 proxy）
3. 若 schema 不相容，執行 DB 還原

### 驗證
- 登入與標註流程可用
- 既有任務與媒體可讀
- ML 後端 Validate 成功

### 延伸閱讀
- [../RUNBOOK.md](../RUNBOOK.md)
- [../configuration.md](../configuration.md)
- [../architecture.md](../architecture.md)
- [../cloudflare-tunnel.md](../cloudflare-tunnel.md)
