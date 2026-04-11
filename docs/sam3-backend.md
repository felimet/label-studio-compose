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

> **SAM3 安裝方式**：`facebookresearch/sam3`（源碼安裝，非 HuggingFace transformers）。
> 影像端：`build_sam3_image_model()` + `Sam3Processor`（state-dict API，`set_image → set_text_prompt / add_geometric_prompt`）
> 影片端（優先）：`build_sam3_multiplex_video_predictor()`（sam3.1 branch，`sam3.1_multiplex.pt`）
> 影片端（回退）：`build_sam3_video_predictor()`（sam3 main branch，`sam3.pt`）— 自動偵測，兩者均使用 `handle_request / handle_stream_request` API 並支援 PCS 文字提示。

## 架構：Lazy Model Loading

兩個後端均採用 **延遲載入** 模式：checkpoint 在模組載入時下載，但模型本體在首次 `predict()` 呼叫時才於 worker 程序內載入（`_ensure_loaded()`）。這避免了 gunicorn master 程序初始化 CUDA，導致 fork 後所有 worker 出現 `RuntimeError: Cannot re-initialize CUDA in forked subprocess`。

關鍵約束：
- `start.sh` **禁止** 使用 `--preload`（會在 master 載入 app，觸發 CUDA 初始化）
- `gunicorn.conf.py` 的 `post_fork` hook 會重置 PyTorch CUDA 狀態作為額外保險
- 模組層級 **禁止** 呼叫 `torch.cuda.get_device_properties()` 等 CUDA 初始化函式

## 啟動

```bash
make ml-up              # 建置映像 + 以 ML Compose overlay 啟動（含核心服務）
make ml-down            # 停止所有服務（含核心）

make build-sam3-image   # 僅建置影像後端映像
make build-sam3-video   # 僅建置影片後端映像

make test-sam3-image    # 在容器內執行影像後端 pytest
make test-sam3-video    # 在容器內執行影片後端 pytest
```

首次啟動時，容器從 HuggingFace Hub 下載 `facebook/sam3.1` 權重（約 3.5 GB）至共用的 `hf-cache` Docker Volume（`/home/appuser/.cache/huggingface`）。健康檢查 `start_period: 300s`，下載期間不觸發重啟。

## 連接至 Label Studio

**影像後端**

1. Label Studio → 專案 → **Settings → Machine Learning → Add Model**
2. URL：`http://sam3-image-backend:9090`
3. 點選 **Validate and Save**，開啟 **Auto-Annotation**

**影片後端**

1. 同上，URL：`http://sam3-video-backend:9090`
2. 需要含有 `<Video>` 與 `<VideoRectangle>` 標籤的標注配置

> 兩個後端共用同一個 `LABEL_STUDIO_API_KEY`。**必須使用 Legacy Token**（LS UI → Account & Settings → Legacy Token），與 `.env` 的 `LABEL_STUDIO_USER_TOKEN` 為同一組值。ML backend SDK 以 `Authorization: Token <key>` 驗證；Personal Access Token（JWT Bearer）會導致 401 Unauthorized。

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
| `<BrushLabels smart="true">` | 輸出 | SAM3 遮罩（Label Studio RLE 格式）；**必須** `smart="true"` 否則前端不會觸發 predict |

### Predict 路徑

<!-- AUTO-GENERATED from ml-backends/sam3-image/model.py -->
**Last Updated:** 2026-04-07

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
| Pascal / Volta GPU | sm_61 (GTX 1080) 在 image backend（sam3 main branch）實測可用；video backend（sam3.1）若推論時觸發 `addmm_act` kernel 缺失則失敗，改用 `sam3.pt` (sam3 main) 可解 |
| Flash Attention 3 | 需 build-time `--build-arg ENABLE_FA3=true` 且設 `SAM3_ENABLE_FA3=true`；需 sm_90+（Hopper H100/H800） |
| gunicorn `--preload` | **禁止使用**。`--preload` 在 master 程序載入 app，觸發 CUDA 初始化，fork 後所有 worker 失敗 |

## Flash Attention 3（選用加速）

FA3 可大幅提升影片推論速度（主要對 Transformer attention 有效），**僅支援 NVIDIA Hopper（H100/H800，sm_90+）**；`flash_attn_interface` 非 pip 套件，需自行 build。

### FA3 停用時的保護機制（預設行為）

SAM3 的 `sam3/model/model_misc.py::get_sdpa_settings()` 在模組匯入時執行，在 Ampere+ GPU（sm_80+）上會將模組層級變數 `USE_FLASH_ATTN` 設為 `True`。attention 區塊在推論時讀取此旗標，無條件執行：

```python
from sam3.perflib.fa3 import flash_attn_func   # → from flash_attn_interface import ...
```

若 `flash-attn-3` 未安裝（預設），`propagate_in_video` 會拋出 `ImportError: No module named 'flash_attn_interface'`，即使 `use_fa3=False` 已傳入 builder 也無法阻止（builder 參數不會覆蓋模組層級旗標）。

**修正**（`ml-backends/sam3-video/model.py::_ensure_loaded()`）：

1. **模組層級 patch**：SAM3 匯入後、`build_sam3_multiplex_video_predictor` 呼叫前，強制覆寫已載入模組物件的旗標：
   ```python
   import sam3.model.model_misc as _sam3_misc
   _sam3_misc.USE_FLASH_ATTN = False
   ```
2. **實例層級 patch**：predictor 建置後，掃描所有子模組並清除 `use_fa3` / `use_flash_attn` 屬性（防止 `__init__` 快取舊值）。

預設 `SAM3_ENABLE_FA3=false` 在 Ampere+ GPU 上現已安全，不再需要安裝 `flash-attn-3`。

### 啟用 FA3（選用）

需同時滿足兩個條件：

```bash
# 1. 建置時安裝 flash-attn-3
docker compose -f docker-compose.yml -f docker-compose.override.yml -f docker-compose.ml.yml \
  build --build-arg ENABLE_FA3=true sam3-video-backend

# or 修改 docker-compose.ml.yml 中 SAM3_ENABLE_FA3: true
sam3-video-backend:
    build:
      context: ./ml-backends/sam3-video
      dockerfile: Dockerfile
      args:
        ENABLE_FA3: true

# 2. .env.ml 中啟用
SAM3_ENABLE_FA3=true

# 3. 重啟影片後端
make ml-up
```

> Flash Attention 3 僅對影片後端有效；影像後端（`build_sam3_image_model()`）不暴露此參數。

## GPU 精度與多 GPU 支援

SAM3 後端自動偵測 GPU 計算能力並選擇最佳推論精度，無需手動配置：

| GPU 世代 | Compute Capability | 推論精度 | 備註 |
|---------|------------------|--------|------|
| Ampere（RTX 30xx, A100 等） | sm_80+ | **bfloat16 autocast** + TF32 | 原生 BF16 Tensor Core；TF32 加速剩餘 fp32 運算 |
| Turing（RTX 20xx, T4 等） | sm_75–79 | **bfloat16 autocast** | 無 BF16 硬體，軟體模擬；**最低支援世代** |
| Volta（TITAN V, V100 等） | sm_70–72 | **不支援** | SAM3 的 `torch.ops.aten._addmm_activation` bfloat16 kernel 需要 sm_75+；`_check_gpu_compatibility()` 啟動時攔截 |
| Pascal（GTX 10xx 等） | sm_61 以下 | **不支援** | 同上 |

> **為何全部使用 bfloat16 autocast**：SAM3 的 `sam3/perflib/fused.py::addmm_act` 函數在 MLP fc1 中無條件將所有張量 `.to(torch.bfloat16)` 後再呼叫 fused kernel，fc1 輸出必然為 bf16。若下游的 fc2 weight 仍是 fp32，PyTorch 會拋出 `RuntimeError: mat1 and mat2 must have the same dtype, but got BFloat16 and Float`。`torch.autocast(dtype=bfloat16)` 確保 fc2 執行期自動將 fp32 weight cast 為 bf16，解決不一致。

**多 GPU 支援**（兩個後端均支援）：
- 透過 `.env.ml` 的 `SAM3_IMAGE_GPU_INDEX` / `SAM3_VIDEO_GPU_INDEX` 獨立指定每個後端的 GPU 編號（逗號分隔可指定多個）
- `start.sh` 在容器啟動時讀取對應的 `SAM3_*_GPU_INDEX` 並 export 為 `CUDA_VISIBLE_DEVICES`（Docker Compose `${VAR}` 替換只讀 HOST 的 `.env`，無法直接從 `env_file:` 取值，故在 `start.sh` 處理）；gunicorn `post_fork` hook 再依 `worker.age` 做一對一分配：worker *i* → `gpus[i-1]`，每個 worker 內部的 `cuda:0` = 分配到的實體 GPU
- `SAM3_IMAGE_WORKERS` / `SAM3_VIDEO_WORKERS` 應設為對應 index 中的 GPU 數量，確保 worker 與 GPU 一一對應

**GPU 空閒釋放**：
- 模型在 `GPU_IDLE_TIMEOUT_SECS` 秒無推論請求後自動從 VRAM 卸載
- 預設值：3600 秒（1 小時）
- 調整範例（`.env.ml`）：`GPU_IDLE_TIMEOUT_SECS=1800`（30 分鐘）或 `300`（5 分鐘，更激進）

## 參考資料

**官方儲存庫與模型**：
- [Meta SAM3 官方源碼](https://github.com/facebookresearch/sam3)
- [SAM2 後備方案](https://github.com/facebookresearch/sam2)（若 SAM3 不可用）
- [HuggingFace 模型卡](https://huggingface.co/facebook/sam3.1)

**Label Studio ML 後端範例**：
- [影像後端範例](https://github.com/HumanSignal/label-studio-ml-backend/tree/master/label_studio_ml/examples/segment_anything_2_image)
- [影片後端範例](https://github.com/HumanSignal/label-studio-ml-backend/tree/master/label_studio_ml/examples/segment_anything_2_video)

> SAM3/SAM3.1 的完整架構、checkpoint 詳情與進階設定，詳見上述官方儲存庫與 HuggingFace 模型卡。GPU 精度配置會根據硬體自動適應，無需手動干預。

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
