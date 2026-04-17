# SAM3 ML 後端

> 讀者對象：ML 開發者、進階標註流程設計者
>
> 本文件涵蓋：SAM3 架構、推論流程、GPU 相容性、限制與測試
>
> 本文件不涵蓋：核心服務部署與一般操作（請見 [user-guide.md](user-guide.md) 與 [RUNBOOK.md](RUNBOOK.md)）
>
> 快速任務路徑： [cookbook/user-cookbook.md](cookbook/user-cookbook.md#任務-4啟用-sam-後端預標註) / [cookbook/developer-cookbook.md](cookbook/developer-cookbook.md)

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

make up-sam3-image      # 單獨啟動影像後端（假設 label-studio 已運行，--no-deps）
make up-sam3-video      # 單獨啟動影片後端

make restart-sam3-image # 重啟影像後端
make restart-sam3-video # 重啟影片後端

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
| `<TextArea name="text_prompt">` | 純文字提示 | 純文字路徑：無幾何提示時走全圖 PCS；有幾何提示且 mixed 欄位為空時作為回退 |
| `<TextArea name="text_prompt_mixed">` | 混合文字提示 | 混合模式優先欄位：Text+Box / Text+Point 時優先取此值；空則回退到 text_prompt |
| `<KeyPointLabels smart="true">` | 點擊提示 | 任意標籤（非 Exclude）為正向點；Exclude/background 為負向點 |
| `<RectangleLabels smart="true">` | 框選提示 | 任意標籤（非 Exclude）為正向框；Exclude 為負向框 |
| `<BrushLabels smart="true">` | 輸出 | SAM3 遮罩（Label Studio RLE 格式）；**必須** `smart="true"` 否則前端不會觸發 predict |
| `<TextArea name="confidence_threshold">` | 門檻覆蓋（選用） | 執行期覆蓋 `SAM3_CONFIDENCE_THRESHOLD`（0~1）；超出範圍自動夾回 |
| `<Choices name="apply_threshold_globally">` | 門檻套用範圍（選用） | 勾選→所有 selection mode 套用門檻；未勾選→僅 threshold mode 套用 |
| `<Choices name="selection_mode">` | 選擇模式（選用） | 覆蓋 `SAM3_MASK_SELECTION_MODE`：`adaptive`/`top1`/`topk`/`threshold`/`all` |
| `<TextArea name="selection_topk_k">` | Top-K 數量（選用） | `topk`/`adaptive` 模式的候選數量；覆蓋 `SAM3_MAX_RETURNED_MASKS` |

> 影像後端 `brushlabels` 會依 context 動態回填：優先採用使用者當前幾何提示標籤；若無幾何標籤則回退到 `<BrushLabels>` 第一個 label（再不行才 fallback `Object` 以維持相容）。

### Predict 路徑

<!-- AUTO-GENERATED from ml-backends/sam3-image/model.py -->
**Last Updated:** 2026-04-17 (v1.1.2)

推論路徑由 context 中提示類型動態決定，優先順序由上而下：

| 路徑 | 觸發條件 | 使用的文字提示 | SAM3 呼叫 | 說明 |
|------|----------|--------------|-----------|------|
| **混合（優先）** | text_prompt_mixed 或 text_prompt + 幾何提示（KeyPoint / Rectangle） | `mixed_text_prompt`（優先）或回退 `pure_text_prompt` | `set_text_prompt()` →（Rectangle）`add_geometric_prompt()` /（KeyPoint）`append_points()` | 文字概念約束 + 幾何定位，最精確 |
| **純文字（PCS）** | 只有 text_prompt，無幾何提示 | `pure_text_prompt` | `set_text_prompt()` | 回傳 N 個概念實例遮罩；影像尺寸從 PIL 推導 |
| **純幾何** | 只有 KeyPoint / Rectangle，無任何 TextArea | — | （Rectangle）`add_geometric_prompt()` /（KeyPoint）`append_points()` | 模型自動補 `'visual'` 虛擬文字；等同 SAM2 行為 |
| **SAM2 fallback** | `sam3` package 未安裝（import 失敗） | 忽略 | `SAM2ImagePredictor.predict()` | 文字提示記 WARNING 並被忽略；幾何提示仍可用 |

> **Point Prompt 行為**：影像後端優先使用 SAM3 原生 point embedding（`geometric_prompt.append_points()`）；僅在執行環境的 sam3 缺少該能力時退回 tiny box 近似（半邊長由 `SAM3_POINT_FALLBACK_HALF_SIZE` 控制，預設 `0.005`）。
>
> **Mask Selection 行為**：推論結果由 `SAM3_MASK_SELECTION_MODE` 控制（`adaptive`/`top1`/`topk`/`threshold`/`all`），可在 UI 的 `selection_mode` Choices 控制項執行期覆蓋。`SAM3_CONFIDENCE_THRESHOLD` 門檻套用範圍由 `SAM3_APPLY_THRESHOLD_GLOBALLY` 控制；同樣可透過 `confidence_threshold` TextArea 執行期覆蓋。
>
> **Smart-trigger fallback**：若 Label Studio smart-trigger 在 `context.result` 中省略非幾何控制項，後端會從當前 annotation 結果中補回 TextArea / Choices 值（context 值優先，annotation 值僅補空缺）。
<!-- END AUTO-GENERATED -->


### 推論流程（image）

```
Label Studio（點擊事件）
    │  POST /predict  { task, context: {keypointlabels | rectanglelabels | textarea | choices} }
    ▼
NewModel.predict()
    ├── get_local_path()            ← 透過 LS SDK 下載影像（支援 local / S3 / MinIO）
    ├── context 解析
    │    ├── textarea[text_prompt]          → pure_text_prompt
    │    ├── textarea[text_prompt_mixed]    → mixed_text_prompt（Text+Geo 優先欄位）
    │    ├── textarea[confidence_threshold] → confidence_threshold override（0~1）
    │    ├── choices[selection_mode]        → selection_mode override
    │    ├── choices[apply_threshold_globally] → apply_threshold_globally override
    │    ├── textarea[selection_topk_k]    → max_returned_masks override
    │    ├── keypointlabels                 → (pixel x, y, is_positive)
    │    └── rectanglelabels               → pixel xyxy box
    │    （若 smart-trigger 省略非幾何欄位，從 annotation 結果補回）
    ├── _predict_sam3()
    │    ├── torch.autocast(bfloat16) context 包裹
    │    ├── processor.set_image(PIL)
    │    ├── processor.set_text_prompt(prompt, state)          [有文字時]
    │    ├── rectangle → processor.add_geometric_prompt(box, label, state)
    │    ├── keypoint  → geometric_prompt.append_points(...)；若不支援則 fallback tiny box
    │    └── state["masks"] [N,1,H,W] + state["scores"] → mask selection（依 selection_mode）
    │         → mask2rle() → BrushLabels
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
| `<TextArea name="text_prompt">` | 純文字提示 | 純文字路徑：無幾何提示時走全圖 PCS；mixed 欄位為空時作為回退 |
| `<TextArea name="text_prompt_mixed">` | 混合文字提示 | Text+Box 混合模式優先欄位；空則回退到 text_prompt |
| `<Labels name="videoLabels">` | 追蹤標籤 | 為追蹤物件指派語意標籤 |
| `<VideoRectangle name="box" smart="true">` | 框選提示 | 在目標畫格拉框，SAM3 雙向追蹤 |
| `<KeyPointLabels name="kp" smart="true">` | 點擊提示 | 任意標籤（非 Exclude）正向點；Exclude/background 負向點；原生 point_entries 傳入 |

> 影片後端對缺少 label 的幾何提示，會從 labeling config 取預設標籤，避免結果與專案自訂標籤脫鉤。

### 推論流程（video）

```
Label Studio（使用者在影格 N 畫框或點擊，可選填文字）
    │  POST /predict  { task, context: {videorectangle?, keypointlabels?, textarea?} }
    ▼
NewModel.predict()
    ├── get_local_path()                      ← 下載影片
    ├── _get_geo_prompts()                    ← VideoRectangle sequence → [{type:"box", frame_idx, x_pct…}]
    │                                            KeyPointLabels → [{type:"point", frame_idx, x_pct, is_positive…}]
    ├── _get_text_prompt()                    ← TextArea → (mixed_text_prompt, pure_text_prompt)
    │                                            推論模式判定：geo_only / text_only / text_geo / mixed_text_geo / none
    ├── _predict_sam3()
    │    ├── torch.autocast(bfloat16) context 包裹
    │    ├── 提取雙向追蹤視窗                  ← [first_prompt_frame − MAX_FRAMES, last_prompt_frame + MAX_FRAMES]
    │    │                                      MAX_FRAMES_TO_TRACK=0 → 提取整段影片
    │    ├── 幾何提示正規化                    ← 百分比轉 normalized xywh/xy，sanitize/clamp 到 [0,1]
    │    │                                      非有限值（NaN/Inf）或完全離框提示會被略過
    │    │                                      Box → bounding_boxes；Point → point_entries（原生 point 支援）
    │    ├── handle_request({ type: "start_session", resource_path: frame_dir })
    │    ├── handle_request({ type: "add_prompt",              ← 依畫格分組，每 obj_id 一次
    │    │        frame_index, obj_id,
    │    │        bounding_boxes (normalized xywh),           ← box 提示
    │    │        point_coords / point_labels,                ← point 提示（原生）
    │    │        text? })                                    ← 文字提示（SAM3 only）
    │    ├── handle_stream_request({ type: "propagate_in_video", ... })
    │    │        → yields {frame_index, obj_id, out_binary_mask}
    │    └── finally: handle_request({ type: "close_session" })    ← 必定執行
    │    mask → _mask_to_bbox_pct() → per-obj_id VideoRectangle sequence
    │    已有 context sequence 與新 sequence 依 frame 排序合併（多物件各自維護）
    │  ModelResponse { videorectangle[]: {id, labels, sequence: [{frame, x, y, w, h, time}…]} }
    ▼
Label Studio（渲染多畫格多物件追蹤框）
```

### 已知限制

| 限制 | 說明 |
|------|------|
| 追蹤視窗 | 預設向前後各 `MAX_FRAMES_TO_TRACK` 畫格（預設 10）；設為 0 追蹤整段影片（OOM 風險） |
| 雙向追蹤 | `SAM3_ENABLE_BIDIRECTIONAL_TRACKING=true`（預設）；設為 false 退回純前向追蹤 |
| 多物件合併 | 新追蹤結果依 `obj_id` 與已有 context sequence 合併，同畫格依 frame 排序去重 |
| KeyPoint（video） | 原生 point_entries 傳入 predictor；SAM3 multiplex video predictor 支援 native points（不再 tiny-box 轉換） |
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

**影像後端測試涵蓋**：`Sam3Processor` mock、純文字 / 純幾何 / 混合三條路徑、SAM2 fallback（文字忽略）、box 正規化（normalized cxcywh）、`set_image` 呼叫、RLE roundtrip；runtime 控制項解析（`confidence_threshold`、`selection_mode`、`apply_threshold_globally`、`selection_topk_k`）；`pure_text_prompt` vs `mixed_text_prompt` 路徑選擇；smart-trigger annotation fallback 補回邏輯。

**影片後端測試涵蓋**：`_get_geo_prompts` / `_get_text_prompt` 解析（含 `mixed_text_prompt` 回傳）、`handle_request` / `handle_stream_request` mock、session 生命週期（`close_session` 必定呼叫）、`_mask_to_bbox_pct`、SAM2 fallback propagation；雙向追蹤視窗計算（`extract_start` / `extract_end`）；多物件 `tracked_sequences` dict 結構；原生 point_entries 傳遞；inference_mode 分類標記。

另含 prompt 正規化回歸測試：
- `xywh` sanitize（越界 clamp、退化框最小尺寸、非有限值/完全離框丟棄）
- `add_prompt` 的 `bounding_boxes`/`bounding_box_labels` 對齊（含 mixed 正負提示）
- 極端離框 keypoint 不會被靜默轉成邊界負樣本

## 環境變數

詳細說明見 [docs/configuration.md](configuration.md#sam3-ml-後端)。
