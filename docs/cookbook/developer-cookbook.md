# 開發者 Cookbook

本文件面向開發者，目標是用最短路徑完成「改程式、驗證、提交」。
規範與完整背景請搭配 [../CONTRIBUTING.md](../CONTRIBUTING.md)。

## 任務 1：啟動可開發環境

### 目標
啟動核心服務與（可選）ML 後端，讓本機可直接開發與測試。

### 步驟
```bash
cp .env.example .env
# 編輯 .env

make up
make init-minio

# 可選：需要 ML 後端
cp .env.ml.example .env.ml
# 編輯 .env.ml
make ml-up
```

### 驗證
```bash
make ps
make health
```

---

## 任務 2：修改 ML 後端後正確重建與驗證

### 目標
修改 `ml-backends/*` 後，用正確成本套用變更（能不 rebuild 就不 rebuild），並完成驗證。

### 步驟
1. 修改程式碼（例如 `ml-backends/sam3-image/`）。
2. 若只改 `model.py`，直接重啟對應服務：
```bash
make restart-sam3-image
```
3. 若改到映像層內容（例如 `Dockerfile`、`requirements*.txt`、`start.sh`、base image 或 build args），再重建映像：
```bash
make build-sam3-image
```
4. 啟動或重啟對應服務：
```bash
make up-sam3-image
# 或
make restart-sam3-image
```
5. 執行測試：
```bash
make test-sam3-image
```

### 驗證
- 服務健康檢查通過
- 測試全數通過
- Label Studio Validate model 成功

### 注意
- SAM2.1 / SAM3 後端彼此獨立，請只操作受影響服務。
- 目前四個 ML 後端都已將 `model.py` 以 volume 掛載到容器 `/app/model.py`，因此 `model.py` 變更通常只需 restart，不需 rebuild。
- 只有映像層內容變更時才需要 `build-sam*` / `build-sam21*`。

---

## 任務 3：新增或調整環境變數（不讓文件失真）

### 目標
調整 env 變數時，維持程式、範例檔、文件一致。

### 變更清單
1. 更新 `.env.example` 或 `.env.ml.example`
2. 更新 [../configuration.md](../configuration.md) 對應段落
3. 若操作流程受影響，更新對應 cookbook
4. 若健康檢查或啟動路徑受影響，更新 [../RUNBOOK.md](../RUNBOOK.md)

### 驗證
- 新增變數在全新環境可成功啟動
- 文件範例可直接複製執行

---

## 任務 4：快速測試

### 目標
在提交前最小成本確認主要路徑沒有壞掉。

### 指令
```bash
make health
make test-sam3-image
make test-sam3-video
make test-sam21-image
make test-sam21-video
```

### 失敗時優先排查
- Token 類問題（Legacy Token 與 PAT 混用）
- GPU/driver 相容性問題
- Compose 疊檔 (`docker-compose.yml` + `docker-compose.override.yml` + `docker-compose.ml.yml`) 未正確使用

---

## 任務 5：提交前檢查與 PR 品質

### 目標
確保 PR 可審查、可重現、可回滾。

### 提交前清單
- [ ] 指令可重現（至少在本機跑過一次）
- [ ] 文件同步（README / docs / cookbook）
- [ ] 無敏感資訊進版控（`.env` 不可提交）
- [ ] 測試與健康檢查通過

### 建議提交格式
`<type>(<scope>): <subject>`

例如：
- `docs(cookbook): add role-based quick recipes`
- `fix(sam3-video): sanitize out-of-range boxes`

### 延伸閱讀
- [../CONTRIBUTING.md](../CONTRIBUTING.md)
- [../RUNBOOK.md](../RUNBOOK.md)
- [../sam3-backend.md](../sam3-backend.md)
- [../sam21-backend.md](../sam21-backend.md)
