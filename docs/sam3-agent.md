# SAM3 Agent

> 讀者對象：ML 開發者、進階標注流程設計者
>
> 本文件涵蓋：SAM3 Agent 架構、LLM 設定、推論流程、支援的 LLM 端點
>
> 本文件不涵蓋：SAM3 基礎後端操作（請見 [sam3-backend.md](sam3-backend.md)）

SAM3 Agent 是 SAM3 影像後端的可選推論模式。啟用後，以 LLM 作為推理引擎，將 SAM3 回傳的候選遮罩可視化影像（base64）送給 LLM，由 LLM 根據 text prompt 決定最終保留哪些遮罩。

> **視覺能力必要條件**：Agent 傳送遮罩可視化影像給 LLM，LLM **必須具備 multimodal（視覺）能力**。使用純文字模型時，預測呼叫靜默失敗並 fallback 回標準 SAM3 文字路徑，不中斷服務。

## 架構

SAM3 Agent **不是獨立服務**，而是 `sam3-image-backend` 中的可選推論路徑，由 `SAM3_AGENT_ENABLED` 控制。

**觸發條件（三者同時滿足）：**
1. `SAM3_AGENT_ENABLED=true`
2. 有 text prompt
3. 無手動幾何提示（KeyPoint / Rectangle）

任一條件不滿足，或 Agent 呼叫失敗，均 fallback 回標準 SAM3 路徑，標注流程不中斷。

**推論流程：**

```
predict()
    ├── AGENT_ENABLED=true AND text_prompt AND no geometry
    │       └── _predict_sam3_agent()
    │               ├── agent_call_sam_service()     ← 本地呼叫 SAM3（不走 HTTP）
    │               │       └── set_image() → set_text_prompt() → 候選遮罩
    │               ├── agent_inference()             ← LLM agentic loop
    │               │       └── send_generate_request()  ← 可視化影像 + prompt → LLM
    │               └── _select_mask_indices()        ← 依 selection_mode 過濾
    │                       → BrushLabels RLE
    └── fallback → _predict_sam3()（標準路徑）
```

## 前置需求

- `sam3-image-backend` 已建置並運行（見 [sam3-backend.md](sam3-backend.md)）
- 可存取的 LLM 端點，**必須支援視覺（multimodal）能力**
- `.env.sam3_agent` 設定檔（從 `.env.sam3_agent.example` 複製）

## 設定

### 1. 建立設定檔

```bash
cp .env.sam3_agent.example .env.sam3_agent
```

編輯 `.env.sam3_agent`：

| 變數 | 預設值 | 說明 |
|------|--------|------|
| `SAM3_AGENT_ENABLED` | `false` | `true` 啟用 Agent 路徑 |
| `SAM3_AGENT_LLM_URL` | `http://localhost:8000/v1` | OpenAI SDK `base_url`（**不含** `/chat/completions`，SDK 自動附加） |
| `SAM3_AGENT_LLM_KEY` | `sk-xxxx` | API Key（本地 vLLM / Ollama 可填任意非空字串） |
| `SAM3_AGENT_MODEL_NAME` | `meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8` | 模型名稱（需與端點一致） |

> **注意**：`SAM3_AGENT_LLM_URL` 只填到版本前綴為止（如 `/v1`），OpenAI SDK 會自動補上 `/chat/completions`。若誤填完整路徑，請求 URL 會變成 `.../chat/completions/chat/completions` 並回傳 404。

完整的 LLM 端點設定範例（vLLM、Ollama 本地 / 雲端、OpenAI、Gemini、Groq、Claude 等）見 [`.env.sam3_agent.example`](../.env.sam3_agent.example)。

### 2. 啟動 / 重啟 image backend

`docker-compose.ml.yml` 在 `.env.sam3_agent` 存在時自動載入（`required: false`），不存在時靜默略過。

```bash
# 首次（或更新 requirements.txt 後）：須重建映像
make build-sam3-image
make up-sam3-agent       # 確認 .env.sam3_agent 存在後啟動 sam3-image-backend

# 僅變更 Agent 設定（SAM3_AGENT_ENABLED / LLM URL / KEY / MODEL）：重啟即可
make restart-sam3-agent
```

> `SAM3_AGENT_ENABLED` 在 `false` 與 `true` 之間切換，或修改任何 LLM 設定時，執行 `make restart-sam3-agent` 即可（內部使用 `up --force-recreate`，確保容器重讀 env_file）。
> 若修改了 `requirements.txt`，必須先執行 `make build-sam3-image` 再重啟，否則新套件不會被安裝。

## Makefile 指令

| 指令 | 說明 |
|------|------|
| `make up-sam3-agent` | 確認 `.env.sam3_agent` 存在，啟動（或重建）sam3-image-backend |
| `make restart-sam3-agent` | 重啟 sam3-image-backend，套用最新 Agent 設定 |
| `make build-sam3-image` | 重建 image backend 映像（更新程式碼後使用） |
| `make test-sam3-image` | 在容器內執行 pytest（mock 測試，不需 GPU） |

## 支援的 LLM

SAM3 Agent 使用 OpenAI SDK（`from openai import OpenAI`）搭配 `base_url` 參數，任何 OpenAI-compatible endpoint 均可對接。

| LLM | 視覺能力 | 備註 |
|-----|:--------:|------|
| vLLM（自建） | 依模型 | 推薦搭配 Llama-4-Maverick 等 multimodal 模型 |
| Ollama（本地）| 需 vision 模型 | `llama3.2-vision`、`llava`、`minicpm-v` 等 |
| Ollama Cloud（直連）| 需確認模型 | `SAM3_AGENT_LLM_URL=https://ollama.com/v1` |
| Ollama Cloud（via daemon）| 需確認模型 | `ollama signin` 後，model name 加 `-cloud` 後綴 |
| OpenAI GPT-4o | ✓ | |
| Google Gemini | ✓ | 使用 Google 提供的 OpenAI-compatible 端點 |
| Groq | 需 vision 模型 | `llama-3.2-90b-vision-preview` 等 |
| DeepSeek VL2 | ✓ | |
| Mistral Pixtral | ✓ | `pixtral-large-latest` |
| Anthropic Claude | ✓ | **無原生相容端點**，需 LiteLLM proxy（見下） |

**Claude via LiteLLM proxy：**

```bash
pip install litellm
litellm --model anthropic/claude-sonnet-4-6 --port 4000
# 然後設：
# SAM3_AGENT_LLM_URL=http://localhost:4000/v1
# SAM3_AGENT_LLM_KEY=<your-anthropic-api-key>
# SAM3_AGENT_MODEL_NAME=anthropic/claude-sonnet-4-6
```

## Log 確認

啟用 Agent 後，每次觸發會在 `sam3-image-backend` 容器日誌中輸出以下訊息，可用來確認實際走了 agent 路徑（而非 fallback）：

```
# 推論開始
[SAM3-Agent] Triggered: prompt='<text>'  endpoint=<LLM_URL>  model=<MODEL_NAME>

# 每個保留遮罩（rank 從 0 起）
  [SAM3-Agent] mask #0  score=0.9123  box(xywh_norm)=[0.123,0.456,0.321,0.234]
  [SAM3-Agent] mask #1  score=0.8701

# 推論完成摘要
[SAM3-Agent] Completed: found=5  kept=2  best=0.9123  prompt='<text>'
```

| 欄位 | 說明 |
|------|------|
| `found` | SAM3 回傳的候選遮罩總數 |
| `kept` | LLM 選擇後保留的遮罩數 |
| `best` | 保留遮罩中最高的 SAM3 信心分數 |
| `box(xywh_norm)` | 正規化 (0–1) 的 x, y, width, height 座標（若有） |

Agent 路徑同時會在 Label Studio 標注結果的 **Inference Scores** TextArea 顯示摘要：

```
mode=agent  endpoint=<URL>  model=<MODEL>  selection_mode=<mode>  threshold=0.3000
#0  score=0.9123  box(xywh_norm)=[0.123,0.456,0.321,0.234]
#1  score=0.8701
(+3 filtered candidates)
```

若容器日誌中完全沒有 `[SAM3-Agent]` 前綴，代表 fallback 回標準 SAM3 路徑（檢查 `SAM3_AGENT_ENABLED`、是否有 text prompt、是否無幾何提示）。

## 已知限制

| 限制 | 說明 |
|------|------|
| 僅支援純文字觸發 | 有幾何提示（KeyPoint / Rectangle）時 Agent 不觸發，走標準混合路徑 |
| 視覺能力必要 | 純文字 LLM 靜默失敗並 fallback，不回傳錯誤訊息給 UI |
| Claude 需 LiteLLM | Anthropic API 無 OpenAI-compatible 端點 |
| Agent 呼叫延遲 | 需額外一次 LLM API 往返；本地 vLLM 延遲較低，雲端 API 視網路而定 |
| 暫存目錄 | Agent 於 `TemporaryDirectory` 下寫入遮罩可視化暫存，容器重啟後自動清除 |

## 參考資料

- [Meta SAM3 Agent 官方 Notebook](https://github.com/facebookresearch/sam3/blob/main/examples/sam3_agent.ipynb)
- [SAM3 後端文件](sam3-backend.md)
- [`.env.sam3_agent.example`](../.env.sam3_agent.example)
- [設定參考](configuration.md)
