# SAM2.1 ML 後端

> 讀者對象：ML 開發者、進階標註流程設計者
>
> 本文件涵蓋：SAM2.1 架構、推論流程、模型選擇、限制與測試
>
> 本文件不涵蓋：核心服務部署與一般操作（請見 [user-guide.md](user-guide.md) 與 [RUNBOOK.md](RUNBOOK.md)）
>
> 快速任務路徑： [cookbook/user-cookbook.md](cookbook/user-cookbook.md#任務-4啟用-sam-後端預標註) / [cookbook/developer-cookbook.md](cookbook/developer-cookbook.md)

SAM2.1（Segment Anything Model 2.1）是 Meta 釋出的分割模型，支援靜態影像分割（Image）與影片物件追蹤（Video）。本專案以兩個獨立 ML 後端服務的形式整合進 Label Studio：

| 服務 | 路徑 | 監聽埠 | 功能 |
|------|------|--------|------|
| `sam21-image-backend` | `ml-backends/sam21-image/` | `:9090` | 靜態影像分割 → BrushLabels RLE 遮罩 |
| `sam21-video-backend` | `ml-backends/sam21-video/` | `:9090` | 影片物件追蹤 → VideoRectangle 序列 |

> **與 SAM3 的差異**：SAM2.1 不支援 PCS 文字提示；僅接受幾何提示（KeyPoint / Rectangle / VideoRectangle）。checkpoint 在 **build time** 下載（`download_models.py`），無需 HuggingFace Token 即可推論（下載時需要）。

## 前置需求

1. NVIDIA GPU，VRAM ≥ 4 GB（Tiny/Small），建議 ≥ 8 GB（Base+/Large）
2. 主機已安裝 `nvidia-container-toolkit`
3. 首次 build 需要能連線至 HuggingFace Hub（選擇性：設定 `HF_TOKEN` build-arg 可提高速率上限）

## 架構：Build-time 下載 + Lazy Loading

與 SAM3（runtime lazy download）不同，SAM2.1 在 **Docker build 階段**執行 `download_models.py` 將所有 checkpoint 寫入 `/data/models`（`model-cache` volume）。容器啟動後 checkpoint 已就緒，不依賴外部網路。

模型本體在 **首次 `predict()` 呼叫**時才於 worker 程序內載入（`_ensure_model()`）。關鍵約束：

- `start.sh` **禁止** 使用 `--preload`（觸發 CUDA 初始化，fork 後 worker 失敗）
- `gunicorn.conf.py` 的 `post_fork` hook 重置 PyTorch CUDA 狀態
- 模組層級 **禁止** 呼叫 CUDA 初始化函式

## 模型選擇

> **`SAM21_DEFAULT_MODEL` 已移除** — 請改用 `MODEL_CONFIG` + `MODEL_CHECKPOINT`（見下）。

### 推薦方式：MODEL_CONFIG + MODEL_CHECKPOINT

`.env.ml` 中設定一組變數即可指定任意 SAM2.1 模型，無需修改程式碼：

```bash
# 使用內建 large 模型（預設值）
MODEL_CONFIG=configs/sam2.1/sam2.1_hiera_l.yaml
MODEL_CHECKPOINT=sam2.1_hiera_large.pt

# 使用自訂模型（checkpoint 需放在 /data/models 或使用絕對路徑）
MODEL_CONFIG=/data/models/my_custom.yaml
MODEL_CHECKPOINT=/data/models/my_custom.pt
```

- `MODEL_CONFIG` + `MODEL_CHECKPOINT` 兩者皆設定時優先於內建模型字典
- 變更後重啟服務（`make restart-sam21-image restart-sam21-video`）即可生效，**無需重建 Docker image**
- 支援絕對路徑或 `$MODEL_DIR` 下的檔名

### 內建模型（已棄用，不建議新使用）

`SAM21_DEFAULT_MODEL` 已完全移除，請僅使用 `MODEL_CONFIG` + `MODEL_CHECKPOINT`。

| 模型 key | checkpoint 檔 | 參數量 | 備註 |
|----------|--------------|--------|------|
| `sam2.1_hiera_tiny` | `sam2.1_hiera_tiny.pt` | ~38M | 最快，VRAM 最少 |
| `sam2.1_hiera_small` | `sam2.1_hiera_small.pt` | ~46M | 快 |
| `sam2.1_hiera_base_plus` | `sam2.1_hiera_base_plus.pt` | ~80M | 速度與精度平衡 |
| `sam2.1_hiera_large` | `sam2.1_hiera_large.pt` | ~224M | 最精準 |

## 啟動

```bash
make ml-up                # 建置映像 + 以 ML Compose overlay 啟動（含核心服務）
make ml-down              # 停止所有服務

make build-sam21-image    # 僅建置影像後端映像（Dockerfile/requirements 變更時）
make build-sam21-video    # 僅建置影片後端映像

make up-sam21-image       # 單獨啟動影像後端（假設 label-studio 已運行，--no-deps）
make up-sam21-video       # 單獨啟動影片後端

make restart-sam21-image  # 重啟影像後端（model.py 或 config/env 變更）
make restart-sam21-video  # 重啟影片後端（model.py 或 config/env 變更）

make test-sam21-image     # 在容器內執行影像後端 pytest
make test-sam21-video     # 在容器內執行影片後端 pytest
```

> **注意**：`docker-compose.ml.yml` 已將 `./ml-backends/sam21-*/model.py` 掛載到容器 `/app/model.py`。只改 `model.py` 時通常只需 `restart-sam21-*`；僅在 Dockerfile、requirements 或其他映像層變更時才需要 `build-sam21-*`。

首次 build 下載所有 4 個 checkpoint（合計約 700 MB）。健康檢查 `start_period: 300s`，下載期間不觸發重啟。

## 連接至 Label Studio

**影像後端**

1. Label Studio → 專案 → **Settings → Machine Learning → Add Model**
2. URL：`http://sam21-image-backend:9090`
3. 點選 **Validate and Save**，開啟 **Auto-Annotation**

**影片後端**

1. 同上，URL：`http://sam21-video-backend:9090`
2. 需要含有 `<Video>` 與 `<VideoRectangle>` 標籤的標注配置

> 兩個後端共用同一個 `LABEL_STUDIO_API_KEY`。**必須使用 Legacy Token**（LS UI → Account & Settings → Legacy Token），與 `.env` 的 `LABEL_STUDIO_USER_TOKEN` 為同一組值。

## 影像後端

### 標注配置（image）

將 [ml-backends/sam21-image/labeling_config.xml](../ml-backends/sam21-image/labeling_config.xml) 匯入專案：

```
Settings → Labeling Interface → Code → 貼上 XML
```

| 控制項 | 類型 | 用途 |
|--------|------|------|
| `<KeyPointLabels smart="true">` | 點擊提示 | 任意標籤（非 Exclude）為正向點；Exclude 為負向點 |
| `<RectangleLabels smart="true">` | 框選提示 | 任意標籤（非 Exclude）為正向框；Exclude 為負向框（轉為背景中心點） |
| `<BrushLabels smart="true">` | 輸出 | SAM2.1 分割遮罩（RLE 格式）；`smart="true"` 為必要條件 |
| `<TextArea name="scores">` | 推論資訊 | 由 ML backend 自動填入：模型名稱 + 每個候選 mask 的分數 |

> `brushlabels` 回傳值會動態解析：優先採用 context 中與目標 `to_name` 對應的合法標籤（排除 `Exclude` / `background`），若無可用值則回退到 `<BrushLabels>` 第一個 label。

### 提示解析規則

| 提示 | 轉換 | SAM2 參數 |
|------|------|-----------|
| KeyPoint 非 Exclude | pixel (x, y) | `point_coords`, `point_labels=1` |
| KeyPoint Exclude | pixel (x, y) | `point_coords`, `point_labels=0` |
| RectangleLabels 非 Exclude | pixel [x0,y0,x1,y1] | `box`（只取第一個 FG box） |
| RectangleLabels Exclude | BG box 中心點 | `point_coords`, `point_labels=0` |

> SAM2 每次呼叫只接受一個 box prompt；多個 FG box 以第一個為準。BG box 自動轉為負向中心點。

### 推論流程（image）

```
Label Studio（點擊事件）
    │  POST /predict  { task, context: {keypointlabels | rectanglelabels} }
    ▼
NewModel.predict()
    ├── _resolve_model_key()          ← 從 MODEL_CHECKPOINT 推導邏輯模型名稱
    ├── _ensure_model()               ← lazy load / swap model
    ├── _load_image()                 ← 本機路徑直接讀取；s3:// → LS resolve endpoint；http → Token auth GET
    ├── _resolve_brush_output()       ← 動態解析 from_name/to_name 與輸出標籤
    ├── _parse_prompts()              ← context → point_coords, point_labels, box
    └── _run_inference()
         ├── torch.autocast(bfloat16 / float16 by GPU arch)
         ├── predictor.set_image(PIL)
         ├── predictor.predict(point_coords, point_labels, box, multimask_output=True)
         │    → masks [N,H,W] bool, scores [N] float
         ├── 取最高分 mask（argmax）
         ├── _mask_to_rle()          ← mask * 255 → brush.mask2rle → list[int]
         └── scores TextArea         ← "model: <key>\n#1  score=0.xxxx\n#2 … ✓ 最高分"
    │  ModelResponse { brushlabels（單一最佳 mask，label 與 from/to_name 皆動態）+ scores textarea }
    ▼
Label Studio（渲染遮罩覆蓋層）
```

## 影片後端

### 標注配置（video）

將 [ml-backends/sam21-video/labeling_config.xml](../ml-backends/sam21-video/labeling_config.xml) 匯入專案：

```
Settings → Labeling Interface → Code → 貼上 XML
```

| 控制項 | 類型 | 用途 |
|--------|------|------|
| `<VideoRectangle name="box" smart="true">` | 框選提示 | 在目標畫格拉框選取物件（不會自動觸發推論，需配合 Submit 按鈕） |
| `<Labels name="videoLabels">` | 追蹤標籤 | 任意標籤（非 Exclude）為正向；Exclude 為負向 |
| `<TextArea name="run_trigger" showSubmitButton="true">` | 推論觸發器 | 框選後點擊 **Submit** 觸發 SAM2.1 追蹤（仿 sam3-video 作法） |
| `<TextArea name="scores">` | 追蹤資訊 | 由 ML backend 自動填入：模型名稱 + 逐畫格 obj/mask info |

> 影片幾何提示若缺少 label，後端會先從 labeling config 動態選預設 label（優先非 `Exclude`）；僅在未顯式傳入預設值的舊呼叫路徑，才維持 `Object` 相容 fallback。
>
> `KeyPointLabels` 已從影片後端移除：Label Studio 的 `KeyPointLabels` 僅支援 `Image` tag，不支援 `Video`；影片模式的點提示目前無對應 tag。

### 推論流程（video）

```
Label Studio（使用者在畫格 N 畫框 → 點擊 run_trigger Submit 按鈕）
    │  POST /predict  { task, context: {videorectangle, run_trigger textarea} }
    ▼
NewModel.predict()
    ├── context 無結果 → 立即回傳空（不再有 batch fallback）
    ├── _get_geo_prompts()            ← VideoRectangle → [{type:"box", frame_idx, x_pct…}]
    ├── _resolve_model_key() / _ensure_model()
    ├── 下載影片（S3/MinIO/本機，含 token auth；無副檔名時建立 symlink）
    └── _predict_sam2()
         ├── torch.autocast(bfloat16)
         ├── _extract_frames()        ← cv2 抽取 [start, last+MAX_FRAMES_TO_TRACK) 畫格
         │                               長邊 > MAX_FRAME_LONG_SIDE 時下縮（減少 OOM）
         ├── predictor.init_state(frame_dir)
         ├── predictor.add_new_points_or_box() × 每畫格每物件
         ├── predictor.propagate_in_video(start_frame_idx=last, max=MAX_FRAMES_TO_TRACK)
         │    → (rel_idx, obj_ids, masks [N,1,H,W] float32 logits)
         │    → binary = masks > 0 → _mask_to_bbox_pct() → VideoRectangle sequence
         └── predictor.reset_state()  ← 必定執行（finally）
    │  ModelResponse { videorectangle sequence（含原始 context + 追蹤結果）
    │                  + scores textarea }
    ▼
Label Studio（渲染多畫格追蹤框）
```

### 影片後端已知限制

| 限制 | 說明 |
|------|------|
| 追蹤長度 | 最多 `MAX_FRAMES_TO_TRACK` 畫格（預設 10），從最後一個提示畫格起算 |
| 解析度縮放 | 長邊超過 `MAX_FRAME_LONG_SIDE`（預設 1024）自動縮小，避免 OOM |
| 文字提示 | SAM2.1 不支援 PCS，TextArea 配置已移除 text_prompt |
| VideoRectangle 自動觸發 | `smart="true"` 上傳框選 BBox；需點擊 `run_trigger` Submit 按鈕（仿 sam3-video 作法） |
| Batch mode | 已移除 `_get_latest_annotation_result` 批次回退路徑；僅支援互動模式（context 必須含結果） |
| gunicorn `--preload` | **禁止使用** |

## GPU 精度

兩個後端均按 GPU 計算能力自動選擇 autocast dtype：

| GPU 世代 | Compute Capability | 影像後端 dtype | 影片後端 dtype |
|---------|------------------|--------------|--------------|
| Ampere（RTX 30xx, A100）| sm_80+ | bfloat16 + TF32 | bfloat16 + TF32 |
| Turing（RTX 20xx, T4）| sm_75–79 | float16 | bfloat16 |
| Pascal（GTX 10xx）| sm_61 以下 | 無 autocast | bfloat16（軟體模擬） |
| CPU | — | 無 autocast | 無 autocast |

## GPU 空閒釋放

模型在 `GPU_IDLE_TIMEOUT_SECS` 秒無推論請求後自動從 VRAM 卸載（`del predictor` + `cuda.empty_cache()`）。

- 預設值：3600 秒（1 小時）
- 調整：`.env.ml` 設定 `GPU_IDLE_TIMEOUT_SECS=1800`（或更短）

## 已知問題與修正記錄

### `ValueError: Value must be a string`（setup 階段）

新版 `label_studio_ml` 的 `CACHE.__setitem__` 要求 value 必須為字串，但舊版程式碼呼叫 `set_extra_params({"predict_only": True})`（傳入 dict）導致 `ValidationError`，後端無法通過 `/health` 檢查。

**修正**：移除 `setup()` 內的 `set_extra_params` 呼叫。SAM2.1 未實作 `fit()`，`predict_only` hint 無功能作用。若只改 `model.py`，重啟對應後端即可套用；若涉及映像層檔案，才需 `build-sam21-*` + `up-sam21-*`。

### Brush mask 不顯示（RLE 值域問題）

`label_studio_converter.brush.mask2rle` 期望輸入值域為 `{0, 255}`。SAM2 回傳 bool mask，`astype(uint8)` 後值為 `{0, 1}`；部分版本的 converter 將值為 `1` 的像素判定為背景，導致 RLE 全空，前端無任何 mask 顯示。

**修正**：`_mask_to_rle` 改為 `mask.astype(np.uint8) * 255` 再傳入 `mask2rle`，與 sam3-image 慣例一致。若只改 `model.py`，重啟 `sam21-image-backend` 即可套用；若有映像層變更才需 rebuild。

### 影像後端僅回傳最高分 mask

SAM2 `multimask_output=True` 固定產生 3 個候選 mask。影像後端改為只回傳 `argmax` 最高分的單一 mask，避免 UI 同時出現多個重疊區域造成混淆。Inference Scores TextArea 仍顯示全部 3 個分數（最高分標記 `✓`）供參考。

## 環境變數

詳細說明見 [docs/configuration.md](configuration.md#sam21-ml-後端)。
