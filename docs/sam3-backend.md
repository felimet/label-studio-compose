# SAM3 ML 後端

自訂 Label Studio ML 後端，封裝 Meta [SAM3](https://huggingface.co/facebook/sam3)（Segment Anything Model 3），透過 HuggingFace `transformers ≥ 4.57` 載入。

## 前置需求

1. **同意 Meta 使用條款**：前往 [facebook/sam3](https://huggingface.co/facebook/sam3) → *Agree and access repository*
2. 產生 HuggingFace **Read Token** → 填入 `.env` 的 `HF_TOKEN`
3. NVIDIA GPU，VRAM ≥ 6 GB（bfloat16 推論）
4. 主機已安裝 `nvidia-container-toolkit`

## 啟動後端

```bash
make gpu             # 建置映像 + 以 GPU Compose overlay 啟動
make build-sam3      # 僅建置映像（不啟動）
make test-sam3       # 在執行中容器內執行 pytest
```

首次啟動時，容器會從 HuggingFace 下載 `facebook/sam3` 權重（約 3.45 GB，0.9B 參數 F32）至 `hf-cache` Docker Volume。後續啟動直接使用快取。

健康檢查設定 `start_period: 300s`，模型下載期間不會觸發重啟。

## 連接至 Label Studio

1. 確認 `make gpu` 已啟動且健康
2. Label Studio → 專案 → **Settings → Machine Learning → Add Model**
3. URL 填入：`http://sam3-ml-backend:9090`
4. 點選 **Validate and Save**
5. 在標注頁面開啟 **Auto-Annotation** 切換

> **API Key 設定**：`LABEL_STUDIO_USER_TOKEN` 為首次啟動時自動寫入的 admin bootstrap token；`LABEL_STUDIO_API_KEY` 為 sam3-ml-backend 專用的存取金鑰，建議在 LS UI（Settings → Access Tokens）建立獨立 token 後填入 `.env`。詳見 [docs/configuration.md](configuration.md#label-studio-user_token-注意事項)。

## 標注配置

將 [sam3-ml-backend/labeling_config.xml](../sam3-ml-backend/labeling_config.xml) 匯入專案：

```
Settings → Labeling Interface → Code → 貼上 XML
```

| 控制項 | 類型 | 用途 |
|--------|------|------|
| `<KeyPointLabels smart="true">` | 點擊提示 | 正向（Object）或負向（Background）點 |
| `<RectangleLabels smart="true">` | 框選提示 | SAM3 邊界框約束 |
| `<BrushLabels>` | 輸出 | SAM3 遮罩輸出（RLE 格式） |

## 標注流程

1. 工具列開啟 **Auto-Annotation**
2. 選擇 **Object** 標籤 → 點擊目標物件 → SAM3 回傳遮罩
3. 選擇 **Background** 標籤 → 點擊不需要的區域 → 遮罩精修
4. 框選矩形 → 提供邊界框約束，適合定位明確的物件
5. 遮罩會逐步累積；每次新的提示都會精修當前選取

## 推論流程

```
Label Studio（點擊事件）
    │  POST /predict  { task, context: {keypointlabels | rectanglelabels} }
    ▼
SAM3Backend.predict()
    ├── _fetch_image()     ← PIL 影像，附 LRU + TTL 快取
    ├── _parse_context()   ← 百分比座標 → 像素座標；正向/負向標籤解析
    ├── Sam3TrackerProcessor + Sam3TrackerModel (bfloat16)
    └── mask2rle()         ← Label Studio RLE（label_studio_converter.brush）
    │  ModelResponse { brushlabels, rle }
    ▼
Label Studio（渲染遮罩覆蓋層）
```

## 執行測試

```bash
# 不需要 GPU 或真實模型——使用 mock 進行 CPU 測試
cd sam3-ml-backend
DEVICE=cpu python -m pytest tests/ --tb=short -v
```

測試涵蓋：座標轉換、標注配置解析、mocked 模型的完整 predict() 路徑、RLE 編解碼往返驗證。

## 已知限制

| 限制 | 說明 |
|------|------|
| 無 encoder 層級快取 | 每次點擊都重跑 SAM3 Image Encoder（GPU 約 500–1500 ms）。`image_embeddings=` 快取已標記為 TODO |
| SAM3.1 未支援 | `facebook/sam3.1` 尚未整合進 HuggingFace `transformers`；待上游整合後更新 `SAM3_MODEL_ID` |
| 單物件輸出 | 每次 predict 呼叫回傳一個遮罩；未實作多物件批次分割 |

## 環境變數

詳細說明見 [docs/configuration.md](configuration.md#sam3-ml-後端)。
