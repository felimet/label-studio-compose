# Cloudflare Tunnel 設定

本專案透過單一 `cloudflared` 容器對外暴露 7 條路由（核心 3 條 + 管理面選用 4 條）。

> 讀者對象：維運、網路管理者
>
> 本文件涵蓋：Tunnel 建立、Public Hostname、WAF 規則、替代方案
>
> 本文件不涵蓋：一般部署與應用排障（請見 [RUNBOOK.md](RUNBOOK.md)）
>
> 任務導覽入口： [README.md](README.md)

## 路由規劃

| 公開網域 | 內部目標 | 存取控制 |
|----------|----------|----------|
| `label-studio.example.com` | `http://nginx:80` | 可選加 CF Access（SSO） |
| `minio-api.example.com` | `http://minio:9000` | WAF 規則：僅允許 GET/HEAD |
| `minio-console.example.com` | `http://minio:9001` | 建議加 CF Access（SSO） |
| `minio-admin.example.com` | `http://minio:9002` | **必須加 CF Access**（高風險管理面） |
| `supabase-studio.example.com` | `http://supabase-studio:3000` | **必須加 CF Access**（Studio 為管理 UI） |
| `supabase-meta.example.com` | `http://supabase-meta:8080` | **必須加 CF Access**（postgres-meta 無內建驗證） |
| `redisinsight.example.com` | `http://redisinsight:5540` | **必須加 CF Access** |

> **不對 MinIO API（`minio-api`）使用 CF Access 的原因：** Presigned URL 已內嵌 HMAC-SHA256 驗證且有時效限制，若再套 CF Access 登入牆，瀏覽器將無法直接存取 Presigned 下載 / 上傳連結。WAF 規則已提供足夠防護。

## 步驟一：建立 Tunnel

1. 登入 [Cloudflare Zero Trust](https://one.dash.cloudflare.com) → **Networks → Tunnels → Create a tunnel**
2. 選擇 **Cloudflared** connector 類型
3. 為 Tunnel 命名（例如 `label-studio`）
4. 複製 Tunnel Token → 貼入 `.env` 的 `CLOUDFLARE_TUNNEL_TOKEN`
5. 儲存後不需要在本機安裝 cloudflared（容器會自動連線）

## 步驟二：設定 Public Hostnames

在 Tunnel 的 **Public Hostnames** 頁籤新增：

| Subdomain | Domain | Service |
|-----------|--------|---------|
| `label-studio` | `example.com` | `http://nginx:80` |
| `minio-api` | `example.com` | `http://minio:9000` |
| `minio-console` | `example.com` | `http://minio:9001` |
| `minio-admin` | `example.com` | `http://minio:9002` |
| `supabase-studio` | `example.com` | `http://supabase-studio:3000` |
| `supabase-meta` | `example.com` | `http://supabase-meta:8080` |
| `redisinsight` | `example.com` | `http://redisinsight:5540` |

> **先決條件（管理面路由）**：`minio-admin`、`supabase-studio`、`supabase-meta`、`redisinsight` 必須先完成 CF Access Policy 再建立 Public Hostname，避免管理面短暫裸露。

> 若使用 `supabase-studio` / `supabase-meta` 與 `redisinsight` 路由，請先啟用對應疊加層：`make supabase-up`、`make tools-up`。

> `supabase-meta` 為 PostgreSQL 管理 REST API（可開 Swagger 文件），不是 Studio 介面。實際管理 UI 服務是 `supabase-studio`。

> 若在 `.env.supabase` 調整 `SUPABASE_META_CONTAINER_PORT`，Cloudflare 中 `supabase-meta` 的 service target 埠號也要同步調整。

> cloudflared 容器以**主動出站**方式連線至 Cloudflare Edge，主機防火牆無需開放入站埠號。

## 步驟三：設定 MinIO WAF 規則

前往 **Security → WAF → Custom Rules**，針對 `minio-api.example.com` 新增以下規則：

### 規則 1：封鎖非 GET/HEAD 請求

```
(http.host eq "minio-api.example.com") and not (http.request.method in {"GET" "HEAD"})
```

動作：**Block**

### 規則 2：封鎖儲存桶列舉

```
(http.host eq "minio-api.example.com") and (
  (http.request.uri.query contains "list-type") or
  (http.request.uri.query contains "list-objects") or
  (http.request.uri.path eq "/") or
  (http.request.uri.path matches "^/[^/]+/?$" and not http.request.uri.query contains "X-Amz-Signature")
)
```

動作：**Block**

## 步驟四（可選）：以 CF Access 保護 Label Studio

如需在 Label Studio 前加 SSO 登入牆：

1. **Access → Applications → Add an Application → Self-hosted**
2. Domain：`label-studio.example.com`
3. 設定 Policy（對應你的 IdP）
4. 新增 **Service Auth Bypass** 規則（允許 API Token 呼叫，避免 ML 後端被擋）：
   - Rule type：`Service Token`
   - 建立一組 Service Token 供 `sam3-ml-backend` 使用，或改用 IP CIDR 放行內部容器網段

## 步驟五（必要）：以 CF Access 保護 MinIO Admin、Supabase Studio / Meta 與 RedisInsight

`minio-admin`、`supabase-studio`、`supabase-meta` 與 `redisinsight` 都是管理面，不應裸露到公開網路。

1. **Access → Applications → Add an Application → Self-hosted**
2. 分別建立：
  - `minio-admin.example.com`
  - `supabase-studio.example.com`
  - `supabase-meta.example.com`
  - `redisinsight.example.com`
3. Policy 要求：
  - 僅允許特定 IdP 群組（維運、DBA）
  - 關閉匿名存取
  - 可選加上 Device posture / mTLS 條件

## 無 Cloudflare 帳號的替代方案

若無 Cloudflare 帳號或不想使用 Zero Trust，有兩個替代選項：

### 選項 1：使用 ngrok

[ngrok](https://ngrok.com/) 將本機埠對應至公開 HTTPS URL，無需設定 DNS 或防火牆。

```bash
# 安裝 ngrok（從 https://ngrok.com/download）
ngrok http 18090   # 暴露 nginx 埠（見 docker-compose.override.yml）
```

ngrok 會輸出公開 URL，例如 `https://abc123.ngrok.io`，複製此 URL 至 `.env`：

```bash
LABEL_STUDIO_HOST=https://abc123.ngrok.io
```

重啟 Label Studio：
```bash
docker compose up -d --no-deps label-studio
```

**優點**：無需 DNS 設定、無需靜態公開 IP、立即可用  
**缺點**：ngrok 免費方案有帶寬限制、URL 重啟後變更（可升級到付費方案固定 URL）

### 選項 2：純本地存取（localhost）

若只需在本機或內部網路存取，無需對外公開：

```bash
# .env
LABEL_STUDIO_HOST=http://localhost:18090
```

**訪問方式**：
- 本機瀏覽器：`http://localhost:18090`
- 同一網路其他電腦：`http://<your-machine-ip>:18090`（例如 `http://192.168.1.100:18090`）
- 跨越網際網路：不支援

**ML 後端連接**：
- ML backends（sam3-image, sam3-video）使用內部 Docker network 自動連接，不受 `LABEL_STUDIO_HOST` 影響
- 後端透過 `http://label-studio:8080`（容器內 DNS）連接應用，與外部公開 URL 無關

**優點**：無需外部服務、最簡單、最安全  
**缺點**：無法遠端存取、無法公開分享註釋結果

---

## 故障排除

| 症狀 | 可能原因 | 解決方式 |
|------|----------|----------|
| Label Studio 回傳 CSRF 403 | `LABEL_STUDIO_HOST` 未正確設定 | 確認 `.env` 中 `LABEL_STUDIO_HOST` 與實際網域一致 |
| Presigned URL 無法在瀏覽器存取 | `MINIO_EXTERNAL_HOST` 設定錯誤 | 確認與 CF Tunnel Public Hostname 相符 |
| cloudflared 容器持續重啟 | Token 無效或 Tunnel 已刪除 | 至 CF Zero Trust 重新產生 Token |
| `supabase-studio` / `supabase-meta` / `redisinsight` 回 502 | 對應 overlay 未啟動 | 執行 `make supabase-up` / `make tools-up` |
| MinIO Presigned URL 422 | CF WAF 規則過於嚴格 | 檢查規則是否誤擋帶有 `X-Amz-Signature` 的 GET 請求 |
