# Runbook

Operations reference for the Label Studio production stack.

## Health Checks

```bash
make health          # full stack check (PostgreSQL · Redis · MinIO · LS · Nginx · SAM3)
```

Individual service health endpoints:

| Service | Endpoint | Expected |
|---------|----------|---------- |
| Label Studio | `http://localhost:18086/health` *(dev port)* | HTTP 200 |
| nginx | `http://localhost:18090/health` *(dev port)* | `OK` |
| sam3-image-backend | `http://sam3-image-backend:9090/health` *(internal)* | HTTP 200 |
| sam3-video-backend | `http://sam3-video-backend:9090/health` *(internal)* | HTTP 200 |
| MinIO | `mc ready local` *(via docker exec)* | `The cluster is ready` |

## Deployment

### First-time deployment

```bash
cp .env.example .env
# Fill in all <PLACEHOLDER> values — see docs/configuration.md
# Generate secrets:
openssl rand -hex 32    # LABEL_STUDIO_SECRET_KEY
openssl rand -hex 20    # LABEL_STUDIO_USER_TOKEN  ← must be ≤40 chars; hex 32 (64 chars) breaks first-boot
openssl rand -base64 24 # POSTGRES_PASSWORD, REDIS_PASSWORD, MINIO_ROOT_PASSWORD

docker compose up -d
make init-minio         # create bucket + CORS (one-time)
make health
```

### Upgrade services

```bash
# 1. Update version pins in .env (LABEL_STUDIO_VERSION, POSTGRES_VERSION, etc.)
# 2. Pull new images
docker compose pull

# 3. Rolling restart (LS → nginx → cloudflared)
docker compose up -d --no-deps label-studio
docker compose up -d --no-deps nginx
docker compose up -d --no-deps cloudflared

# 4. Verify
make health
```

### Start ML backends (GPU)

```bash
make ml-up             # core stack + sam3-image-backend + sam3-video-backend
make health            # includes SAM3 checks
```

After startup, register each backend in Label Studio:

1. **Project → Settings → Machine Learning → Add Model**
2. Image backend URL: `http://sam3-image-backend:9090`
3. Video backend URL: `http://sam3-video-backend:9090`
4. Click **Validate and Save**, enable **Auto-Annotation**

## Rollback

### Application rollback

```bash
# Revert LABEL_STUDIO_VERSION in .env to previous value, then:
docker compose up -d --no-deps label-studio
```

> PostgreSQL schema migrations are not automatically reversed. If the new version ran `migrate`, rolling back may require a database restore.

### Database restore from backup

```bash
docker compose stop label-studio
docker compose exec db psql -U labelstudio -c "DROP DATABASE labelstudio;"
docker compose exec db psql -U labelstudio -c "CREATE DATABASE labelstudio;"
docker compose exec -T db psql -U labelstudio labelstudio < backup.sql
docker compose start label-studio
```

## Common Issues

### Label Studio won't start — "database does not exist"

```bash
# Check postgres health
docker compose exec db pg_isready -U labelstudio

# Inspect init log
docker compose logs db | grep -i error
```

If the database was never created, re-run init:
```bash
docker compose down
docker volume rm label-studio_postgres-data   # ⚠️ deletes all data
docker compose up -d db
docker compose logs -f db  # wait for "database system is ready"
```

### Label Studio 登入 403 CSRF verification failed

原因：nginx 的 `X-Forwarded-Proto` 送出 `http`（Cloudflare Tunnel 以 HTTP 連到 nginx），Django CSRF 與瀏覽器的 HTTPS Origin 不符。

確認 nginx 設定是否已套用：
```bash
docker compose exec nginx grep -r "X-Forwarded-Proto" /etc/nginx/conf.d/
# 應顯示: proxy_set_header   X-Forwarded-Proto https;
```

若未套用（`$scheme` 而非 `https`）：
```bash
docker compose up -d --no-deps nginx
```

同時確認 `LABEL_STUDIO_HOST` 含 `https://` 前綴（影響 `CSRF_TRUSTED_ORIGINS`）。

### 首次登入帳號不存在 / 登入後 500 "No memberships found"

`LABEL_STUDIO_USERNAME`/`LABEL_STUDIO_PASSWORD` 只在 Postgres DB **首次初始化**時生效。若 DB 已存在，需手動建立：

```bash
docker compose exec label-studio bash -c '
cd /label-studio
python label_studio/manage.py shell -c "
from django.contrib.auth import get_user_model
from organizations.models import Organization
U = get_user_model()
user = U.objects.create_superuser(email=\"admin@example.com\", password=\"your_password\")
Organization.create_organization(created_by=user, title=\"Default\")
print(\"Done\")
"'
```

> 若 DB 已存在但帳號不存在（如重建容器未 `down -v`），執行上述指令即可；Organization 建立後才能正常登入。

### MinIO bucket not found

```bash
make init-minio
# Or manually:
docker compose run --rm minio-init
```

### SAM3 backend — model download fails

Common causes:
- `HF_TOKEN` not set or expired
- Meta license not accepted at https://huggingface.co/facebook/sam3.1

```bash
docker compose -f docker-compose.yml -f docker-compose.ml.yml \
  logs sam3-image-backend | grep -i error
```

Fix: update `HF_TOKEN` in `.env.ml`, then:
```bash
docker compose -f docker-compose.yml -f docker-compose.ml.yml \
  restart sam3-image-backend sam3-video-backend
```

### SAM3 backend — Cannot re-initialize CUDA in forked subprocess

若看到 `RuntimeError: Cannot re-initialize CUDA in forked subprocess`，表示 CUDA 在 gunicorn master 程序中被初始化，fork 後 worker 無法重新初始化。

**根因**：`start.sh` 中使用了 `--preload`，或模組級程式碼觸發了 CUDA 初始化（如 `torch.cuda.get_device_properties()`）。

**修復**：
1. 確認 `start.sh` **沒有** `--preload` 旗標
2. 確認 `model.py` 使用 `_ensure_loaded()` 延遲載入模式（不在模組層級建模型）
3. 確認 `gunicorn.conf.py` 存在且包含 `post_fork` hook 重置 CUDA 狀態

```bash
# 檢查是否誤用 --preload
docker compose -f docker-compose.yml -f docker-compose.override.yml -f docker-compose.ml.yml \
  exec sam3-image-backend cat /app/start.sh | grep preload
# 應該沒有輸出；若有 --preload 須移除
```

### SAM3 video backend — GPU 架構太舊 (sm_61 / Pascal)，Triton kernel 無法編譯

**症狀**：video backend 啟動後第一次呼叫 predict 即失敗，log 出現大量：

```
Feature '.acq_rel' requires .target sm_70 or higher
```

加上 traceback 最後指向：

```
sam3/sam3/perflib/triton/connected_components.py::connected_components_triton
ptxas fatal: Unsupported .target sm_61
```

**根因**：SAM3 video predictor 的 connected components 後處理是用 Triton 寫的自訂 kernel，kernel 內使用 `.acq_rel` 記憶體語意（需要 Volta 以上 `sm_70+`）。Pascal 世代（GTX 1080Ti / P40 等 `sm_61`）的 ptxas 直接拒絕編譯，與任何參數設定無關，純粹是硬體世代不符。

> 為什麼 **image backend 正常**：image 推論路徑只用標準 PyTorch kernel，不碰 Triton 的 connected components，因此舊卡可以跑。

**解法選項**（按推薦優先）：

| 方案 | 說明 | 工程量 | 風險 |
|------|------|--------|------|
| 換 GPU（`sm_70+`，V100 / RTX 20xx 以後） | 官方支援路徑；換卡後照正常流程跑 | 中（硬體 + driver 更新） | 低 |
| 改 SAM3 — 把 Triton kernel 換成 PyTorch 備援 | 偵測 `torch.cuda.get_device_capability() < (7, 0)` 時走 `scipy.ndimage.label` 或純 PyTorch 版 | 高（需理解 perflib 邏輯） | 中 |
| 只用 image 路徑，自行實作 tracking | 用 SAM3 image 做逐 frame segmentation，tracking 改用 IoU matching / Kalman filter | 中 | 低 |
| 改用不依賴 Triton 的 video segmentation 模型 | DeAOT / XMem / Mask2Former 等純 PyTorch 實作，對舊卡友好 | 中 | 低 |

**不建議**：嘗試欺騙 ptxas 把 target 改成 `sm_70`——即使能編過，Pascal 硬體實際上不支援這些 memory ordering feature，runtime 行為不可預測。

**確認你的 GPU compute capability**：

```bash
docker compose -f docker-compose.yml -f docker-compose.ml.yml \
  exec sam3-video-backend python -c \
  "import torch; print(torch.cuda.get_device_properties(0))"
# sm_61 = Pascal (GTX 1080Ti / Titan Xp 等) → 無法跑 SAM3 video
# sm_70+ = Volta 以後 → OK
```

### SAM3 backend — CUDA out of memory

Reduce concurrent workers (already set to 1 in Dockerfile CMD). If OOM still occurs:

1. Close other GPU workloads
2. Use `DEVICE=cpu` in `.env.ml` (very slow, no GPU required)
3. Try a smaller model: set `SAM3_MODEL_ID=facebook/sam3` and `SAM3_CHECKPOINT_FILENAME=sam3.pt` in `.env.ml`

### Cloud Storage 概念速查

**Source Cloud Storage**（來源儲存）：存放「原始資料」或「任務定義」。

- 可放影像、音訊、影片等媒體檔，或 JSON / JSONL / Parquet 格式的任務定義檔。
- 專案按 **Sync** 時，Label Studio 掃描 bucket，將物件轉成任務（tasks），但實際檔案不搬入資料庫——以 URL 或 presigned URL 方式直接讀取。
- **Import Method** 決定掃描行為：
  - `Tasks`（預設）：把每個檔案視為 JSON/JSONL/Parquet 任務定義來解析。
  - `Files`：把每個檔案視為媒體來源物件，自動包裝成帶 URL 的任務。

**Target Cloud Storage**（目標儲存）：存放「標注結果」。

- Annotator 按 Submit / Update 後，Label Studio 同時寫入自身 DB 與目標 bucket。
- 下游 ML pipeline 直接從此 bucket 撈標注 JSON 做訓練或備份遷移。

---

### Local File Storage — UnsupportedFileFormatError（.mp4 / 媒體檔）

**症狀**：

```
io_storages.exceptions.UnsupportedFileFormatError: File "...BC500-001.mp4" is not a
JSON/JSONL/Parquet file. Only .json, .jsonl, and .parquet files can be processed.
```

**根因**：Local Storage 的 Import Method 設成 `Tasks`（只接受 JSON），遇到 `.mp4` 就拒絕。

**修法 1 — UI**：

1. Project → Settings → Cloud Storage → 編輯該 Local Storage
2. Step 2 **Import Settings & Preview** → **Import Method** 改選 `Files - Treat each file as a source object`
3. **File Name Filter** 改為 `.*\.(mp4|avi|mov|webm)$`（或清空以接受所有格式）
4. Load Preview → Next → **Save & Sync**

**修法 2 — API**：

```bash
# 修改 storage（換成實際 storage id）
curl -X PATCH \
  "${LABEL_STUDIO_HOST}/api/storages/localfiles/<STORAGE_ID>" \
  -H "Authorization: Token ${LABEL_STUDIO_USER_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{"use_blob_urls": true}'

# 重新同步
curl -X POST \
  "${LABEL_STUDIO_HOST}/api/storages/localfiles/<STORAGE_ID>/sync" \
  -H "Authorization: Token ${LABEL_STUDIO_USER_TOKEN}"
```

修正後，每個 `.mp4` 會自動產生如下任務：

```json
{"data": {"video": "/data/local-files/?d=test-video/BC500-001.mp4"}}
```

> **前提**：`LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true` 與 `LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT` 已正確設定（見 [configuration.md](configuration.md)）。

---

### Redis connection refused

```bash
docker compose exec redis redis-cli -a "$REDIS_PASSWORD" ping
# Expected: PONG
docker compose logs redis | tail -20
```

### nginx 502 Bad Gateway

```bash
docker compose ps label-studio          # check healthy status
docker compose logs label-studio --tail=50
```

Label Studio typically takes 60–90 s to start on first boot.

## Log Access

```bash
make logs                                     # all services, last 100 lines
docker compose logs -f label-studio           # LS only
docker compose logs -f db redis               # infra only

# ML backends
docker compose -f docker-compose.yml -f docker-compose.ml.yml \
  logs -f sam3-image-backend sam3-video-backend
```

Structured JSON logs are enabled (`JSON_LOG=1`). Use `jq` to filter:

```bash
docker compose logs label-studio 2>&1 | jq 'select(.level=="ERROR")'
```

## Backup

### Data volumes

所有資料目錄已使用 bind mount，直接存在專案根目錄，可用一般檔案工具備份。

```bash
# Label Studio 標注資料 + Local files
tar -czf ls-data-$(date +%Y%m%d).tar.gz ./label-studio-data/

# PostgreSQL — 必須用 pg_dump（postgres-data/ 是內部格式，直接複製無法還原）
docker compose exec db pg_dump -U labelstudio labelstudio \
  > backup-$(date +%Y%m%d).sql

# MinIO 媒體檔案
tar -czf minio-data-$(date +%Y%m%d).tar.gz ./minio-data/
```

### Exclude from backup

- `redis-data/` — 任務佇列暫存，重啟後自動恢復，不需備份
- `hf-cache`, `sam3-image-models`, `sam3-video-models` — 可從 HuggingFace 重新下載

## Monitoring

No bundled monitoring stack. Recommended additions:

| Tool | Purpose |
|------|---------|
| Prometheus + cAdvisor | Container resource metrics |
| Loki + Promtail | Log aggregation (forward docker json logs) |
| Grafana | Dashboard for above |
| Uptime Kuma | External endpoint health checks |

Cloudflare provides basic WAF analytics and request metrics for public endpoints.
