# label-anything-sam

適用於生產環境的 Label Studio 部署方案，內含可選用的 SAM3 與 SAM2.1 ML 後端。

English version: [README.md](README.md)

## 為何有這個專案

截至 2026-04，上游 [Label Studio ML backend](https://github.com/HumanSignal/label-studio-ml-backend) 尚未提供可直接用於生產部署的 SAM3 整合路徑。本專案提供可落地的完整堆疊：

- 核心服務：Label Studio + PostgreSQL + Redis + MinIO + Nginx + Cloudflare Tunnel
- 可選 GPU 疊加：SAM3 影像/影片後端與 SAM2.1 影像/影片後端
- 以安全為先的預設：S3 最小權限、Token 使用規範、對外暴露邊界

> [!NOTE]
> 版本使用建議：
>
> - `v1.0.4` 是原生 PostgreSQL 線（無需 Supabase）的最新穩定版本，
>   包含所有 SAM3 修正與強化：影像 + 影片原生 point embedding、
>   遮罩選擇模式（`adaptive`/`top1`/`topk`/`threshold`/`all`）、
>   執行期門檻與選擇模式 UI 覆蓋、雙向影片追蹤、多物件 track 合併、
>   純文字 / 混合用途雙提示欄位，以及可選的 SAM3 Agent（LLM 輔助遮罩選擇）。
> - 若需要 Supabase 整合版本，請使用 `main` 或 `v1.1.4`。
>
> ```bash
> git fetch --tags
> git checkout tags/v1.0.4 -b local-v1-native-pg
> ```

## 5 分鐘快速開始

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

# 3) 可選 SAM3 Agent（LLM 輔助遮罩選擇，SAM3 後端專用）
cp .env.sam3_agent.example .env.sam3_agent
# 設定 SAM3_AGENT_ENABLED=true 並填入 LLM 端點（URL / KEY / MODEL）
# 各支援平台範例（vLLM、Ollama、OpenAI、Gemini、Groq 等）詳見 .env.sam3_agent.example
# LLM 必須具備視覺（multimodal）能力

make up-sam3-agent
# 若只變更 LLM 設定，無需重建映像：make restart-sam3-agent
```

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

## 常用 Make 指令（精簡）

- `make up / down / restart / logs / ps`：核心服務生命週期
- `make ml-up / ml-down`：核心服務 + ML 疊加層
- `make build-sam3-image / build-sam3-video / build-sam21-image / build-sam21-video`：建置 ML 映像
- `make test-sam3-image / test-sam3-video / test-sam21-image / test-sam21-video`：執行 ML 後端測試
- `make init-minio`：首次建立 bucket 與 service account
- `make health`：全棧健康檢查

## 授權

Apache-2.0 © 2026 Jia-Ming Zhou
