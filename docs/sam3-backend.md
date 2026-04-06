# SAM3 ML 後端

SAM3（Segment Anything Model 3）是 Meta 於 2025 年 11 月釋出的下一代分割模型，支援影像分割（Image）與影片物件追蹤（Video），並新增 **PCS（Promptable Concept Segmentation）**——純文字自然語言驅動的分割能力。本專案以兩個獨立 ML 後端服務的形式整合進 Label Studio：

| 服務 | 路徑 | 監聽埠 | 功能 |
|------|------|--------|------|
| `sam3-image-backend` | `ml-backends/sam3-image/` | `:9090` | 靜態影像分割 → BrushLabels RLE 遮罩 |
| `sam3-video-backend` | `ml-backends/sam3-video/` | `:9090` | 影片物件追蹤 → VideoRectangle 序列 |

## 前置需求

1. **同意 Meta 使用條款**：前往 [facebook/sam3.1](https://huggingface.co/facebook/sam3.1) → *Agree and access repository*
2. 產生 HuggingFace **Read Token** → 填入 `.env.ml` 的 `HF_TOKEN`
3. NVIDIA GPU，VRAM ≥ 8 GB（bfloat16 推論，torch 2.7 + CUDA 12.6）
4. 主機已安裝 `nvidia-container-toolkit`

> **SAM3 與 SAM2 的差異**：SAM3 使用 `facebookresearch/sam3`（源碼安裝，非 HuggingFace transformers）。
> 影像端：`build_sam3_image_model()` + `Sam3Processor`（state-dict API，`set_image → set_text_prompt / add_geometric_prompt`）
> 影片端：`Sam3VideoPredictorMultiGPU`（session-based API，`handle_request / handle_stream_request`）
> 不需手動 cv2 畫格分割——SAM3 影片端以 `video_loader_type="cv2"` 在內部處理。

## 啟動

```bash
make ml-up              # 建置映像 + 以 ML Compose overlay 啟動（含核心服務）
make ml-down            # 停止所有服務（含核心）

make build-sam3-image   # 僅建置影像後端映像
make build-sam3-video   # 僅建置影片後端映像

make test-sam3-image    # 在容器內執行影像後端 pytest
make test-sam3-video    # 在容器內執行影片後端 pytest
```

首次啟動時，容器從 HuggingFace Hub 下載 `facebook/sam3.1` 權重（約 3.5 GB）至 `sam3-image-models` / `sam3-video-models` Docker Volume。健康檢查 `start_period: 300s`，下載期間不觸發重啟。

## 連接至 Label Studio

**影像後端**

1. Label Studio → 專案 → **Settings → Machine Learning → Add Model**
2. URL：`http://sam3-image-backend:9090`
3. 點選 **Validate and Save**，開啟 **Auto-Annotation**

**影片後端**

1. 同上，URL：`http://sam3-video-backend:9090`
2. 需要含有 `<Video>` 與 `<VideoRectangle>` 標籤的標注配置

> 兩個後端共用同一個 `LABEL_STUDIO_API_KEY`。建議在 LS UI（Settings → Access Tokens）建立獨立的 token 後填入 `.env`。

## 影像後端

### 標注配置（image）

將 [ml-backends/sam3-image/labeling_config.xml](../ml-backends/sam3-image/labeling_config.xml) 匯入專案：

```
Settings → Labeling Interface → Code → 貼上 XML
```

| 控制項 | 類型 | 用途 |
|--------|------|------|
| `<TextArea name="text_prompt">` | 文字提示 | PCS 自然語言提示（純 SAM3 功能） |
| `<KeyPointLabels smart="true">` | 點擊提示 | 正向（foreground）或負向（background）點 |
| `<RectangleLabels smart="true">` | 框選提示 | SAM3 邊界框約束 |
| `<BrushLabels>` | 輸出 | SAM3 遮罩（Label Studio RLE 格式） |

### Predict 路徑

<!-- AUTO-GENERATED from ml-backends/sam3-image/model.py -->
**Last Updated:** 2026-04-06

三條路徑依序判斷，並可任意組合：

| 路徑 | 觸發條件 | SAM3 呼叫 | 說明 |
|------|----------|-----------|------|
| **混合（優先）** | TextArea + 幾何提示（KeyPoint / Rectangle） | `set_text_prompt()` → `add_geometric_prompt()` | 文字概念約束 + 幾何定位，最精確 |
| **純文字（PCS）** | 只有 TextArea，無幾何提示 | `set_text_prompt()` | 回傳 N 個概念實例遮罩；影像尺寸從 PIL 推導 |
| **純幾何** | 只有 KeyPoint / Rectangle，無 TextArea | `add_geometric_prompt()` 一或多次 | 模型自動補 `'visual'` 虛擬文字；等同 SAM2 行為 |
| **SAM2 fallback** | `sam3` package 未安裝（import 失敗） | `SAM2ImagePredictor.predict()` | 文字提示記 WARNING 並被忽略；幾何提示仍可用 |

> **Points 的限制**：`Sam3Processor.add_geometric_prompt()` 僅接受 box，不接受點。點提示以邊長 ±0.5% 影像尺寸的微小 box 代替（`eps_x = eps_y = 0.005`），正向/負向透過 `label=True/False` 傳遞。
<!-- END AUTO-GENERATED -->


### 推論流程（image）

```
Label Studio（點擊事件）
    │  POST /predict  { task, context: {keypointlabels | rectanglelabels | textarea} }
    ▼
NewModel.predict()
    ├── get_local_path()            ← 透過 LS SDK 下載影像（支援 local / S3 / MinIO）
    ├── context 解析               ← textarea → text_prompt
    │                                keypointlabels → (pixel x, y, is_positive)
    │                                rectanglelabels → pixel xyxy box
    ├── _predict_sam3()
    │    ├── torch.autocast(bfloat16) context 包裹
    │    ├── processor.set_image(PIL)
    │    ├── processor.set_text_prompt(prompt, state)          [有文字時]
    │    └── processor.add_geometric_prompt(box, label, state) [有幾何時，可多次]
    │         state["masks"] [N,1,H,W] bool → mask2rle() → BrushLabels
    │  ModelResponse { brushlabels[], rle }
    ▼
Label Studio（渲染遮罩覆蓋層）
```

## 影片後端

### 標注配置（video）

將 [ml-backends/sam3-video/labeling_config.xml](../ml-backends/sam3-video/labeling_config.xml) 匯入專案：

```
Settings → Labeling Interface → Code → 貼上 XML
```

| 控制項 | 類型 | 用途 |
|--------|------|------|
| `<TextArea name="text_prompt">` | 文字提示 | PCS 自然語言提示（純 SAM3 功能） |
| `<Labels name="videoLabels">` | 追蹤標籤 | 為追蹤物件指派語意標籤 |
| `<VideoRectangle name="box" smart="true">` | 框選提示 | 在目標畫格拉框，SAM3 向前追蹤 |
| `<KeyPointLabels name="kp" smart="true">` | 點擊提示 | 正向（Object）或負向（background）點 |

### 推論流程（video）

```
Label Studio（使用者在影格 N 畫框或點擊，可選填文字）
    │  POST /predict  { task, context: {videorectangle?, keypointlabels?, textarea?} }
    ▼
NewModel.predict()
    ├── get_local_path()                      ← 下載影片
    ├── _get_geo_prompts()                    ← VideoRectangle sequence → [{type:"box", frame_idx, x_pct…}]
    │                                            KeyPointLabels → [{type:"point", frame_idx, x_pct, is_positive…}]
    ├── _get_text_prompt()                    ← TextArea → str（可選）
    ├── _predict_sam3()
    │    ├── torch.autocast(bfloat16) context 包裹
    │    ├── cv2.VideoCapture → 取得 vid_w/vid_h（百分比 → 像素換算）
    │    ├── handle_request({ type: "start_session", resource_path: video_path })
    │    ├── handle_request({ type: "add_prompt",              ← 依畫格分組，每 obj_id 一次
    │    │        frame_index, obj_id,
    │    │        bounding_boxes (pixel xywh),                ← box 提示
    │    │        points / point_labels,                      ← point 提示
    │    │        text? })                                    ← 文字提示（SAM3 only）
    │    ├── handle_stream_request({ type: "propagate_in_video",
    │    │        start_frame_index: last_frame,
    │    │        max_frame_num_to_track: MAX_FRAMES_TO_TRACK })
    │    │        → yields {frame_index, outputs: {out_binary_masks}}
    │    └── finally: handle_request({ type: "close_session" })    ← 必定執行
    │    mask → _mask_to_bbox_pct() → VideoRectangle sequence（與 context sequence 合併）
    │  ModelResponse { videorectangle: {sequence: [{frame, x, y, width, height, time}…]} }
    ▼
Label Studio（渲染多畫格追蹤框）
```

### 已知限制

| 限制 | 說明 |
|------|------|
| 追蹤長度上限 | 最多 `MAX_FRAMES_TO_TRACK` 畫格（預設 10） |
| Flash Attention 3 | 需 build-time `--build-arg ENABLE_FA3=true` 且設 `SAM3_ENABLE_FA3=true` |
| SAM2 fallback 文字提示 | SAM2 不支援 PCS，文字提示記 WARNING 並被忽略 |

## Flash Attention 3（選用加速）

FA3 可大幅提升影片推論速度（主要對 Transformer attention 有效），適用於 Ampere 及以後的 GPU（A100、RTX 3090 等）。

**注意**：`Sam3VideoPredictorMultiGPU` 目前不透過建構子參數控制 FA3，而是依賴 GPU 能力偵測（`model_misc.py`）。`.env.ml` 的 `SAM3_ENABLE_FA3=true` 設定需與 build-time 的 `--build-arg ENABLE_FA3=true` 共同使用，確保 `flash-attn-3` 套件已安裝於映像中。

```bash
# 1. 建置時啟用 FA3（安裝 flash-attn-3）
docker compose -f docker-compose.yml -f docker-compose.override.yml -f docker-compose.ml.yml \
  build --build-arg ENABLE_FA3=true sam3-video-backend

# 2. .env.ml 中同時設定
SAM3_ENABLE_FA3=true

# 3. 重啟影片後端
make ml-up
```

> Flash Attention 3 目前僅對影片後端有效；影像後端（`build_sam3_image_model()`）不暴露此參數。

## 執行測試

```bash
# 不需要 GPU 或真實模型——全部使用 mock 進行 CPU 測試
cd ml-backends/sam3-image
DEVICE=cpu python -m pytest tests/ --tb=short -v

cd ml-backends/sam3-video
DEVICE=cpu python -m pytest tests/ --tb=short -v
```

**影像後端測試涵蓋**：`Sam3Processor` mock、純文字 / 純幾何 / 混合三條路徑、SAM2 fallback（文字忽略）、box 正規化（normalized cxcywh）、`set_image` 呼叫、RLE roundtrip。

**影片後端測試涵蓋**：`_get_geo_prompts` / `_get_text_prompt` 解析、`handle_request` / `handle_stream_request` mock、session 生命週期（`close_session` 必定呼叫）、`_mask_to_bbox_pct`、SAM2 fallback propagation。

## 環境變數

詳細說明見 [docs/configuration.md](configuration.md#sam3-ml-後端)。
