# 使用者 Cookbook

本文件面向「想快速完成任務」的使用者與專案管理者。
若你需要完整背景與參數定義，請回到 [../README.md](../README.md) 與 [../configuration.md](../configuration.md)。

## 任務 1：核心服務首次啟動

### 目標
在本機啟動 Label Studio + Supabase standalone（含 PostgreSQL）+ Redis + MinIO + Nginx，並通過健康檢查。

### 前置條件
- 已安裝 Docker Engine 與 Docker Compose v2
- 已建立 `.env`（從 `.env.example` 複製）
- 已建立 `.env.supabase`（從 `.env.supabase.example` 複製）
- `.env` 與 `.env.supabase` 的 `POSTGRES_PASSWORD` 必須一致
- `LABEL_STUDIO_USER_TOKEN` 長度不超過 40 字元（建議 `openssl rand -hex 20`）

### 步驟
```bash
cp .env.example .env
cp .env.supabase.example .env.supabase
# 編輯 .env，填入所有 <PLACEHOLDER>
# 編輯 .env.supabase，填入所有 <PLACEHOLDER>

make supabase-up SUPABASE_STANDALONE_ENV=.env.supabase

make up
make init-minio
make health
```

### 驗證
- `make health` 核心服務為 healthy 或正常回應
- 可開啟 `http://localhost:18090`
- `make supabase-logs SUPABASE_STANDALONE_ENV=.env.supabase` 無明顯錯誤

### 失敗回復
- 查看即時日誌：`make logs`
- 核心服務重啟：`make down && make up`

---

## 任務 2：連接 MinIO S3 資料來源

### 目標
在 Label Studio 專案中新增 MinIO S3 Source Storage，能成功同步資料。

### 步驟
1. 進入 Project → Settings → Cloud Storage → Add Source Storage → S3。
2. 填入：
   - Endpoint: `http://minio:9000`
   - Access Key: `MINIO_LS_ACCESS_ID`
   - Secret Key: `MINIO_LS_SECRET_KEY`
   - Bucket: `MINIO_BUCKET`
3. 按 Save 並執行 Sync。

### 驗證
- Sync 成功，任務列表可看到資料
- 標註介面可正常載入影像/影片

### 常見錯誤
- 使用 root 帳號而非 service account
- Bucket 名稱不在 `.env` 的 `MINIO_BUCKET` 清單內

---

## 任務 3：使用 Local Files（不經上傳）

### 目標
直接使用本機資料夾（`./ls-data/file`）建立任務。

### 步驟
1. 將檔案放到 `./ls-data/file/<your-folder>/`。
2. 在 Label Studio 新增 Local Files Source Storage。
3. Absolute local path 填：`/label-studio/data/file/<your-folder>`。
4. Save 後執行 Sync。

### 驗證
- 任務可被匯入且可開啟
- 不需要額外手動上傳檔案

---

## 任務 4：啟用 SAM 後端預標註

### 目標
為專案加入 SAM3 或 SAM2.1 後端，啟用 Auto-Annotation。

### 前置條件
- 若使用 SAM3：需 GPU + HF_TOKEN + 已同意模型授權
- `.env.ml` 已設定 `LABEL_STUDIO_API_KEY`（Legacy Token）

### 步驟
```bash
cp .env.ml.example .env.ml
# 編輯 .env.ml

make ml-up
make health
```

在 Label Studio：
1. Project → Settings → Machine Learning → Add Model
2. 填入後端 URL（擇一或多個）
   - `http://sam3-image-backend:9090`
   - `http://sam3-video-backend:9090`
   - `http://sam21-image-backend:9090`
   - `http://sam21-video-backend:9090`
3. Validate and Save，開啟 Auto-Annotation

### 驗證
- Validate 成功
- 在標註畫面觸發 smart prompt 可收到預測結果

---

## 任務 5：匯出標註成果

### 目標
將標註結果匯出供訓練或下游流程使用。

### 步驟
1. 進入專案 Export 功能。
2. 選擇格式（JSON/COCO/YOLO 等）。
3. 匯出檔案會存放於 `./ls-data/export/`（依專案與時間戳命名）。

### 驗證
- 匯出檔存在於 `ls-data/export`
- 檔案內容可被下游流程讀取

---

## 任務 6：重設管理員密碼與基本帳號管理

### 目標
安全地維護管理員帳號與一般使用者。

### 步驟
- 重設管理員密碼：
```bash
make reset-password
```
- 建立管理員（冪等）：
```bash
make create-admin
```

### 驗證
- 可用新密碼登入
- 使用者列表與權限符合預期

### 延伸閱讀
- [../user-guide.md](../user-guide.md)
- [../configuration.md](../configuration.md)
- [../RUNBOOK.md](../RUNBOOK.md)
