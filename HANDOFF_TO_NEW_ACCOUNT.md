# 交接文件：BB-144 訓練搬遷到新帳號後怎麼續跑

> 寫給新帳號上的 Claude。作者：舊帳號（leo07010@NCHC）上的 Claude，2026-07-08。
> 你看不到舊帳號的對話，這份文件自含全部背景，照著做即可。有不確定的地方問 Leo，不要猜。

---

## 1. 背景

- 這個 repo 是 **Cascade Neural Decoder**（arXiv:2604.08358 重現）的訓練專案，位於 `QEC/cascade/`。
- 正在進行的 run：**BB-144 code、v6 min-weight logical basis（TAG `v6_bb144_mw`）**，總共 40000 steps。
- 原本在舊帳號 nano4 上以 SLURM job chain 跑（job 165337 → 165338 → 165339 pending），每段上限 48h，靠 checkpoint 自動接力。
- Leo 把整個 `QEC/` 資料夾搬到你這個帳號，之後工作都在這邊。你的任務：**從 checkpoint 續跑這個訓練，不是從頭跑**。

## 2. 目前進度（以舊帳號 2026-07-08 為準）

| 項目 | 值 |
|------|-----|
| Checkpoint 目錄 | `checkpoints/bb_144_12_12_v6_bb144_mw/` |
| `last.pt`（完整 resume 快照：模型+optimizer+step） | step **24000** / 40000（同步時間不同可能更新，以實際 resume log 為準） |
| `best.pt` | best p_block 對應的 EMA 權重 |
| 檔案大小 | 兩個都 ~50 MB |
| 原本的 GPU | NVIDIA H200（141 GB），1 張，~0.4 steps/s |
| 剩餘量估計 | ~16000 steps ≈ 11 小時以上 |

**Resume 機制**（`scripts/14_train_bb_v3.py:212`）：啟動時只要 `checkpoints/<code>_<tag>/last.pt` 存在就**自動載入接續**，不用加參數；log 會印 `[resume] loaded ... at step N`。快照每 1000 steps（每次 eval）原子寫入一次。

## 3. 開跑前 Checklist（照順序，全部打勾才 sbatch）

### 3.1 確認 checkpoint 是最終版
- [ ] `ls -la checkpoints/bb_144_12_12_v6_bb144_mw/` — `last.pt` 與 `best.pt` 都在、各 ~50 MB。
- [ ] `md5sum` 驗證與舊帳號最終版一致（2026-07-08 舊帳號側實測值）：
  ```
  189135dd5f8c676a474bed612a580dcc  last.pt
  2ba53eeaf411bf85939fe16c80c2d255  best.pt
  ```
- [x] **已確認（2026-07-08 12:52，舊帳號側 Claude 補記）**：舊帳號 job 全部停止——165338 FAILED（12:45，存檔路徑 bug，見 §6）、165339 接力後 fresh start 已 scancel（未寫任何檔）。最終 checkpoint＝`last.pt` **step 24000**（mtime 07-08 10:08）；舊帳號另留一份備份在 `checkpoints/bb_144_12_12_v6_bb144_mw.backup-20260708/`。在此時間點之後同步過來的即為最終版。

### 3.2 重建 venv（強制——複製來的 `.venv/` 不能用）
複製過來的 `.venv/` 裡所有 shebang 和路徑都寫死指向舊帳號的 `/work/leo07010/...`，直接 source 會壞掉或靜默用錯 Python。

- [ ] 刪掉或改名舊的：`mv .venv .venv.old-account`
- [ ] 用 Python ≥ 3.10 重建（原環境是 3.13，來自 cluster 的 miniconda module；新機找對應 module）：
  ```bash
  python3 -m venv .venv && source .venv/bin/activate
  pip install -e .        # 依賴都在 pyproject.toml
  ```
- [ ] 注意 `muon-optimizer` 是從 GitHub 裝的（`git+https://github.com/KellerJordan/Muon.git`），login node 要能連外。
- [ ] `torch>=2.5` 要裝到對的 CUDA 版本；裝完驗證：
  ```bash
  python -c "import torch, stim, sinter, einops, lion_pytorch; print(torch.__version__)"
  python -m py_compile scripts/14_train_bb_v3.py
  ```

### 3.3 改 SLURM 腳本 `slurm/train_bb_v6_mw.sh`（三個地方）
- [ ] **39 行 `WORKDIR`**：寫死了 `/work/leo07010/Ray/QEC/cascade`，改成新帳號的實際路徑。
- [ ] **26 行 `--account`**：原為 `GOV114009`，改成新帳號的 project account。
- [ ] **27 行 `--partition`**：原為 `8gpus`（nano4/nano5），改成新 cluster 的對應 partition。資源需求：1 node、1 GPU、8 CPU、48h。
- [ ] `mkdir -p logs`（腳本會做，但先確認 WORKDIR 下寫得進去）。
- [ ] **不要改 TAG（`v6_bb144_mw`）**——out_dir 是 `<code>_<tag>` 組出來的，TAG 變了就找不到 checkpoint，會從頭跑。
- [ ] **不要加 `--no-resume`。**
- [ ] **確認拿到的是 2026-07-08 修復後的版本**：`scripts/14_train_bb_v3.py` 的 `--out` 預設值原本寫死舊帳號的 `/home/leo07010/...`（165338/165339 就是因此 resume 失敗），已改為 repo-relative，且本腳本的 python 呼叫已明確傳 `--out "${WORKDIR}/checkpoints"`。檢查：`grep -- '--out' slurm/train_bb_v6_mw.sh` 要有結果、`grep '/home/leo07010' scripts/14_train_bb_v3.py` 要**沒有**結果。若拿到修復前版本，手動在呼叫加 `--out`。

### 3.4 GPU 記憶體備案
原本跑在 H200 141GB。若新機 GPU 較小而 OOM：降 micro-batch、等比例升 accum，維持有效 batch ≈ 3312（例如 `BATCH=24; ACCUM_STEPS=138`）。改之前先問 Leo。

## 4. 提交與驗證（最關鍵的一步）

```bash
sbatch slurm/train_bb_v6_mw.sh 144
```

Job 開跑後**幾分鐘內**就要檢查 log（`logs/cascade_bb_<jobid>.out`）：

- ✅ 必須看到：`[resume] loaded .../bb_144_12_12_v6_bb144_mw/last.pt at step 24000`（step ≥ 24000）
- ❌ 如果**沒有這行**、直接從 step 0 開始 → **立刻 `scancel`**。原因：fresh run 會在 step 1000 的第一次 eval 就**覆寫同一個目錄的 `last.pt`/`best.pt`，把搬來的 checkpoint 蓋掉**。先 scancel 保住檔案，再查路徑/TAG 哪裡不對。

確認 resume 成功後回報 Leo。**從這一刻起，絕對不要再從舊帳號同步 checkpoint 過來**（會把新進度蓋掉）。

## 5. 之後的接力與收尾

- 剩 ~16000 steps 若一段 48h 跑不完（不太可能，但保險），用 dependency chain 接力：
  ```bash
  J=$(sbatch --parsable slurm/train_bb_v6_mw.sh 144)
  sbatch --dependency=afterany:$J slurm/train_bb_v6_mw.sh 144
  ```
  接力段啟動時一樣會自動 resume。
- 跑完 40000 steps 後會自動做 final eval（200000 shots，p ∈ {0.001…0.005}），結果在 log 和 checkpoint 目錄。
- **公平比較注意**（來自腳本開頭註解）：之後跑 BP+OSD baseline 必須用**同一份 min-weight circuit**（`src/cascade/codes/bb.py` 現在回傳 min-weight logical representatives），否則 decoder 和 baseline 面對的 observables 不同，比較無效。

## 6. 舊帳號側資訊（追查歷史用）

- 舊帳號：`leo07010`，cluster nano4，account `GOV114009`，partition `8gpus`。
- Job chain（已全部結束，2026-07-08）：165337（步 0→18000）→ 165338（18000→24000+；07-08 12:45 FAILED——存 checkpoint 時 `--out` 指向的 `/home/leo07010/...` 路徑已不存在，RuntimeError；log 顯示訓練有跑過 step 25000，但 24000 之後的進度未能存檔，遺失 ~1-2h）→ 165339（afterany 接力自動起跑，因同一 bug 找不到 last.pt 而 fresh start，12:52 scancel，未寫任何 checkpoint）。
- 舊 log 在 `logs/cascade_bb_1653*.out`（若有同步過來），訓練歷史曲線可以從裡面 grep `step|eval`。
