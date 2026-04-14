# 文件入口（Documentation Hub）

本頁是整份文件的導覽中樞。
如果你是第一次接觸本專案，請先從這裡選擇角色與任務，再進入對應文件。

## 1. 依角色閱讀

| 角色 | 先讀 | 任務型 Cookbook | 深入文件 |
|------|------|-----------------|----------|
| 使用者 / 專案管理者 | [user-guide.md](user-guide.md) | [cookbook/user-cookbook.md](cookbook/user-cookbook.md) | [configuration.md](configuration.md) |
| 開發者 | [CONTRIBUTING.md](CONTRIBUTING.md) | [cookbook/developer-cookbook.md](cookbook/developer-cookbook.md) | [sam3-backend.md](sam3-backend.md), [sam21-backend.md](sam21-backend.md) |
| 維運 / SRE | [RUNBOOK.md](RUNBOOK.md) | [cookbook/ops-cookbook.md](cookbook/ops-cookbook.md) | [architecture.md](architecture.md), [cloudflare-tunnel.md](cloudflare-tunnel.md) |

## 2. 依任務閱讀

| 我想完成的任務 | 建議路徑 |
|----------------|----------|
| 5 分鐘啟動核心服務 | [../README.md](../README.md) → [cookbook/user-cookbook.md](cookbook/user-cookbook.md#任務-1核心服務首次啟動) |
| 連接 MinIO S3 | [cookbook/user-cookbook.md](cookbook/user-cookbook.md#任務-2連接-minio-s3-資料來源) |
| 連接 SAM 後端做預標註 | [cookbook/user-cookbook.md](cookbook/user-cookbook.md#任務-4啟用-sam-後端預標註) → [sam3-backend.md](sam3-backend.md) / [sam21-backend.md](sam21-backend.md) |
| 修改後端程式並重建容器 | [cookbook/developer-cookbook.md](cookbook/developer-cookbook.md#任務-2修改-ml-後端後正確重建與驗證) |
| 線上故障排查與回滾 | [cookbook/ops-cookbook.md](cookbook/ops-cookbook.md) → [RUNBOOK.md](RUNBOOK.md) |

## 3. 深入參考文件

- [architecture.md](architecture.md)：服務拓撲、資料流、安全設計
- [configuration.md](configuration.md)：環境變數單一真相來源（SSOT）
- [cloudflare-tunnel.md](cloudflare-tunnel.md)：對外暴露、WAF、Tunnel 設定
- [sam3-backend.md](sam3-backend.md)：SAM3 後端機制、限制、效能議題
- [sam21-backend.md](sam21-backend.md)：SAM2.1 後端機制、限制、效能議題
- [RUNBOOK.md](RUNBOOK.md)：維運、事故、備份、還原
- [CONTRIBUTING.md](CONTRIBUTING.md)：開發流程與提交規範

## 4. 文件責任邊界（避免重複）

- 環境變數定義只放在 [configuration.md](configuration.md)。
- 架構與資料流只放在 [architecture.md](architecture.md)。
- 事故處理步驟只放在 [RUNBOOK.md](RUNBOOK.md)。
- Cookbook 只講「完成任務的最短路徑」，不重貼完整背景理論。

## 5. 維護規範

當你修改以下內容時，請同步更新文件：

- `.env.example` / `.env.ml.example` 變更：同步更新 [configuration.md](configuration.md) 與對應 cookbook。
- Makefile 指令變更：同步更新三份 cookbook 與 [CONTRIBUTING.md](CONTRIBUTING.md)。
- 新增服務或網路邊界：同步更新 [architecture.md](architecture.md) 與 [RUNBOOK.md](RUNBOOK.md)。
