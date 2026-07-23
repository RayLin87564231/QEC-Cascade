# 通用（code-agnostic）NN QEC decoder ＋ FPGA 即時部署：文獻調查與可行性評估

> **本報告的定位**：回答使用者讀完 Cascade 論文後的兩個問題——(1) 「訓練**一個通用 NN** 適用全部（或一族）QEC code，再**燒進 FPGA** 做即時解碼」，有沒有人做過？(2) 這個方向還有什麼可以嘗試的？
>
> **證據基礎與檢索日期**：本報告所有文獻宣稱皆出自**兩輪** deep-research 流程（universal-decoder 主輪 ＋ neural-BP 補核驗輪，見 JSON 內 `neuralBpRound`）的逐篇對抗式核驗結果（`reports/universal_decoder_survey_evidence.json`，檢索截至 **2026-07**），未在該結果中的論文/數字一律不進本報告。此領域月更速度快，**「交集為空」是對已核驗文獻集合的陳述**，非絕對不存在；引用前請對照下方 §5 的品質保證說明。
>
> **preprint 警示**：四篇關鍵文獻尚未同行評審——SAGU（2025-10）、NTU foundation decoder（2026-06，僅一個月新）、SUSTech FPGA 實機 demo（2026-05）、utility-scale latency 分析（2025-11）；其性能數字皆為**自報、無獨立重現**，跟教授談時請明確標注 preprint 身分。（補核驗輪的 Astra（arXiv:2408.07038）、Relay-BP（arXiv:2506.01779）、scaled-min-sum（arXiv:2605.10433）亦為 preprint。）

---

## TL;DR（直接回答兩個問題）

1. **有沒有人做過完整路徑？** ——「通用 NN decoder ＋ 燒進 FPGA」這條**整條路徑，在本次逐篇核驗的文獻中無人佔據**（交集為空＝機會）；但路徑的**兩端各自都已有人做**。
2. **通用端（線1）**：SAGU（arXiv:2510.06257, 2025）已證 multi-code 訓練的單一 NN 可解碼 **BB code 家族內、訓練集外**的 code 且無精度代價——但限 **code-capacity 噪聲、高 p（0.06–0.10）、同家族**；detector-graph GNN（Lange et al., PRResearch 2025）提供 **code-agnostic 的「圖」輸入表示法**；NTU（2026）、Transformer-QEC（2023）都只做**跨 distance**（非跨家族），且 Transformer-QEC 的「without retraining」實為 fine-tune。
3. **硬體端（線2）**：NN decoder 上 FPGA 已**實機**證實可行（SUSTech 2026：d=3、閉環 **550 ns**、NN 推論僅 **124 ns**；QUNET 2025：**4-bit QAT** + early-exit），但**每一個都 surface-code 專用**；非 NN 對照（Riverlane LCD/Collision Clustering、Yale Helios）把 FPGA/ASIC 基準壓到 **~0.01–1 µs/round、d 高達 17–23**，是任何 NN 硬體提案必須對打的標竿。
4. **與我們的關係**：BB-144 iteration 7 已在**噪聲軸**證明「泛化可由訓練分布設計學出」（mixed-p 五點全勝 BP+OSD）；SAGU 是它在 **code 軸**的初步對應物，但只到 code-capacity/高 p——**把它推到 circuit-level ＋ 低 p 正是空缺，也正是我們 repo 有一手能力去補的地方**。
5. **還能做什麼？** 兩層：**(a) repo 內下個 iteration 就能動手**的 E1–E4 跨 code 實驗梯（zero-shot 移植 → frozen-backbone 重訓 readout → mixed-code co-train → unseen-polynomial 驗證）；**(b) 更廣路線圖**——code-agnostic 的圖 message-passing 架構 ＋ FPGA 落地（先做一頁**即時性 roofline go/no-go 算術**，再 PTQ/QAT → HLS → 板上 latency 量測）。
6. **neural-BP fallback 的判定（補核驗輪，`neuralBpRound`）**：neural-BP／learned message passing **不接 OSD 在 BB code 可勝 BP+OSD**（Astra、Relay-BP，皆 3-0）——是**低風險 fallback、非準確率死路**；但必須走 **GNN／Relay-BP 式**（天真 neural min-sum 在 bicycle 家族已證訓練失敗），且其 **FPGA/ASIC 硬體代價數字本輪仍缺**。完整判定見 **§2.5**。

---

## 1. 已有的工作地圖：誰做過什麼、做到哪

四條檢索線的候選論文彙整如下。每條宣稱都可對回 `reports/universal_decoder_survey_evidence.json` 的對應 finding（見 §5）。「confidence」欄同時標注同行評審/preprint 身分。

### 1.1 線1 — 通用／跨 code 的 NN decoder

| 論文（arXiv/DOI） | 年份 | 一句話貢獻 | 與「一個通用 NN 適用所有 code」的關係 | confidence |
|---|---|---|---|---|
| **SAGU**（arXiv:2510.06257，建於 QuBA GNN 之上） | 2025 | multi-code 訓練框架：多個 BB code 聯合訓練的**單一 NN** 可解碼訓練集外（unseen）的 code | **最直接先例**。訓練於 [[72,12,6]]/[[90,8,10]]/[[144,12,12]]/[[288,12,18]]，held-out [[756,16,≤34]] 上表現僅差在小數第四位，且「comparable to or even outperforming」單 code 專訓 | high（**preprint**，OpenReview 審稿中） |
| **Lange et al.** detector-graph GNN（arXiv:2307.01241；Phys. Rev. Research 7, 023181） | 2025 | 把解碼形式化為 **detector graph 上的 graph classification**——輸入是「圖」而非綁 code 的 grid | **最重要的 code-agnostic 輸入表示法候選**；僅用模擬數據即在 surface circuit-level 噪聲下勝過拿到完整 error model 的 MWPM。**但每個 code/dataset 仍各訓一網，論文未示範單一模型跨 code**；另兩條次要限定：d=9 時尚未穩定勝過 belief-matching、訓練成本重（~1e9 樣本） | high（**同行評審**） |
| **Transformer-QEC**（arXiv:2311.16082，ICCAD） | 2023 | self-attention 取得對全部 syndrome 的 **global receptive field** | 其 transferability **僅跨 code distance**（surface 家族內，變長輸入）且**實需 fine-tuning**（約省 10× 訓練成本，非 zero-shot）——**反證通用 decoder 的空缺真實存在** | high |
| **NTU foundation decoder**（arXiv:2606.27119） | 2026 | neural transfer unification：用可擴展 code 家族共享的**代數結構**對齊不同 distance 的解碼任務，小 code 加速大 code 訓練 | 跨 **distance** transfer（surface d7→19、BB [[72,12,6]]→[[144,12,12]] 各自遞進），**非單一 code-agnostic 模型**；[[72,12,6]] 低 p 區自報勝 Relay-BP；**FPGA/quantization/real-time 全列為 future work**，自稱「first open-source foundation decoder」 | **medium**（preprint，1 個月新、無獨立重現） |

**線1 小結**：三類跨 code 泛化嘗試都存在，但**所有已核驗工作的泛化都限「同一 code 家族內」**（跨 distance 或跨 BB 參數），**尚無跨結構相異家族（surface↔BB↔color）的單一模型**。SAGU 是唯一真正「一個權重解多個 code」的先例，但被限定在 BB 家族、code-capacity 噪聲、高 p。

### 1.2 線2a — NN decoder 上 FPGA／即時硬體

| 論文（arXiv/DOI） | 年份 | 一句話貢獻 | 與想法的關係 | confidence |
|---|---|---|---|---|
| **SUSTech**（arXiv:2605.04892） | 2026 | 超導處理器上**實機**示範 real-time surface-code（**d=3**）QEC：確定性閉環 **550 ns**（含 NN 推論 **124 ns**），在 1.25 µs QEC cycle 內回饋 | **證明 NN decoder 可達 FPGA 即時預算**；但 d=3、per-stabilizer **LSTM**（每個 XX/ZZ stabilizer 一個 32-hidden LSTM）、輸入綁 surface 結構、**abstract 對 cross-code 零宣稱**；scalability 只談 distance 擴展 | high（preprint，~2.5 個月新、未評審；d=3 logical 僅「comparable to offline MWPM」） |
| **QUNET**（DOI 10.1109/DSD67783.2025，Euromicro DSD） | 2025 | quantized modular **UNet** decoder：QAT 把權重/activation 壓到 **as low as 4 bits**、**early-exit** 讓簡單 syndrome 走淺層平均省 **34%** 解碼時間、省最多 **50%** FPGA 資源 | **NN-on-FPGA 壓縮技術範本**；但 UNet grid 輸入、**surface 專用**，天然非 code-agnostic；未給絕對 latency（34% 為相對節省），「4 bits」可能混合精度 | high（**同行評審**） |
| **Overwater et al.**（arXiv:2202.05741，IEEE TQE） | 2022 | 小 d surface **fully-connected feed-forward** NN decoder 的設計空間探索，報告 post-place-and-route 時序：**FPGA <90 ns、ASIC <30 ns**，遠低於固態 qubit ~440 ns 預算 | 「NN 塞得進 FPGA latency 預算」**最早的量化證據**；但**是 Vivado post-implementation 估計非實機**、僅 d=3/5 塞進 Artix-7、噪聲為無測量錯誤的 depolarizing（**code-capacity**）、架構 code-specific | high（同行評審） |

### 1.3 線2b — 非 NN 的 FPGA/ASIC 對照基準（任何「NN 燒進 FPGA」提案必須對打）

| 論文（arXiv/DOI） | 年份 | 一句話貢獻 | 作為基準的意義 | confidence |
|---|---|---|---|---|
| **Riverlane Local Clustering Decoder**（arXiv:2411.10343；Nat. Commun. 16:11048） | 2025 | adaptive、distributed 的 Union-Find 式 clustering **純古典**演算法，Xilinx VU19P/ZCU111 FPGA 上 rotated planar surface code **到 d=17 均 <1 µs/round**（d=17 平均 0.622/0.676 µs） | 定義 NN 方案必須追平的**速度基準**；LCD 本身 surface 專用。限定：sub-µs 是 10M shots **平均**非 worst-case，時脈由 400 MHz(d=5) 降至 285 MHz(d=17) | high（同行評審） |
| **Riverlane Collision Clustering**（arXiv:2309.05558；Nat. Electron.） | 2025 | 古典 clustering 同做 FPGA 與 ASIC：FPGA(XCVU3P) d=21 平均 **810 ns/round**；ASIC(12nm) d=23 平均 **240 ns/round、0.06 mm²、8 mW** | **面積/功耗/速度資源效率標竿**。限定：ASIC 為 sign-off「ready to tape out」**未流片**（設計估計非實測矽；FPGA 數字為實機） | high（同行評審） |
| **Yale Helios distributed Union-Find**（arXiv:2406.08491；IEEE TQE） | 2024 | O(d³) 平行資源達成對 d 的 **sublinear 平均時間**：d=21 平均 **11.5 ns/round**（比 1 µs 低約兩個數量級）、d=17 circuit-level 23.7 ns、省資源模式 d=51 為 544 ns | 非 NN 演算法 decoder 的**速度上限參照**。限定：sublinear 為特定噪聲下經驗平均、非 worst-case 閉環延遲；d>21 後時脈劣化 | high（同行評審） |

### 1.4 latency 預算的定錨文獻

| 論文（arXiv） | 年份 | 一句話貢獻 | 對本方向的意義 | confidence |
|---|---|---|---|---|
| **1QBit/Quantum Machines utility-scale 分析**（arXiv:2511.10633） | 2025 | 即使 decoder 已達 sub-µs/round，在 10⁶–10¹¹ T gates、200–2000 logical qubits、Λ=9.3、**2.86 MHz** stabilization 下仍引入巨大開銷（magic state factory +100k–250k qubits、core processor 因 d→d+4 +300k–1.75M qubits、runtime ~100×） | **把即時目標收緊到比慣用 1 µs/round 更嚴**：2.86 MHz ≈ **~350 ns/round**。通用 NN 若因泛化而變大，量化/蒸餾要守的目標更緊 | **medium**（preprint；已於 APS 2026 發表、被同行評審後續引用） |

> **一條被否決的宣稱（防線有作用的證據）**：同源 arXiv:2511.10633 曾有一句「Riverlane FPGA UF 對 d=30 memory 的 1.0 µs/round 是 SOTA」被 3-vote **0-3 否決**、已剔除（詳見 §5）。引用 2511.10633 時避開該句。

### 1.5 線3 — 兩者交集

**核心結論（high confidence，綜合四篇 3-0 核驗結果）**：在本次逐篇對抗式核驗的全部文獻中，「**code-agnostic NN decoder ＋ FPGA/即時硬體部署**」的**交集為空**：

- **每一個已上 FPGA 的 NN decoder**（SUSTech d=3 LSTM、QUNET UNet、Overwater FC-FF）**都是 surface-code 專用**；
- **每一個跨 code 泛化的 NN decoder**（SAGU、detector-graph GNN、NTU）**都無任何硬體實作**——NTU 甚至明說 real-time/quantization/distillation 是 future work，並自稱「first open-source foundation decoder」，可作 real-time 優化的起點。

「機會空缺」假說成立：**把 multi-code 訓練出的通用 NN 量化後燒進 FPGA，目前無人佔據。**

---

## 2. 誠實的可行性評估（以 BB-144 iteration 7 為錨）

### 2.1 我們手上有什麼一手證據（數字對齊 `reports/iteration_7_status.md`）

BB-[[144,12,12]] iteration 7 的 mixed-p 重訓（log-uniform p∈[0.001, 0.0055]，16 點 log grid，TAG `v7_bb144_mixp`，40k steps）把 iter-6「單一 p=0.0055 訓練→低 p 泛化失敗（p_block 地板 ~0.40）」修成**五點全勝 BP+OSD（min-weight circuit baseline）**：

| p | Cascade p_block | 勝 BP+OSD（P_L/cycle 比） | 勝 BP+OSD（p_block 比） |
|---|---|---|---|
| 0.0055 | 0.350 | 1.9× | 1.6× |
| 0.005 | 0.206 | 2.2× | 1.9× |
| 0.004 | 0.0422 | 4.2× | 3.9× |
| 0.003 | 0.00577 | 6.4× | 6.3× |
| 0.002 | **0.00038** | **8.8×** | **8.8×** |

- **p=0.002 勝約 8.8×**（兩種比值定義在此點一致；區間約 **[4×, 18×]**，因 BP+OSD 在 p=0.002 僅 8 個失誤、CI 寬）。兩種比值定義在高 p 點差 15–20%（P_L/cycle vs p_block），跟教授引用時**須註明用哪一種**。
- **外推到訓練範圍外**：p=0.0005（低於訓練 p_min=0.001）在 zero-syndrome probe 中 **p_block=0.00000（0/64 shots）**，v6 在此點為 0.42。（註：64 shots 樣本少，是 probe 非正式 eval，作趨勢證據而非精確測量。）
- **健康度**：12 個 per-logical std 全程 5.9–7.5；zero-syndrome probe **0/12 heads fire**。

### 2.2 這對「通用 NN 跨 code」是支持還是困難？

**支持面（兩條軸都開始有證據）**：

- **噪聲軸（我們一手）**：iter-7 直接證明「**泛化可以透過訓練分布設計學出來**」——mixed-p 訓練分布讓模型在**沒單獨訓練過**的低 p 點全勝。
- **code 軸（文獻初步對應）**：SAGU 證明「multi-code 訓練的單一 NN 泛化到訓練集外的 code、無精度代價」。這正是我們噪聲軸經驗在 code 軸的**理論對應物**，且用的是 GNN（QuBA）——與 §4 主推的圖 message-passing 方向一致。**兩條軸現在都有正向證據，central bet 不是空想。**

**困難面（三道具體的牆，這是「為什麼難」）**：

1. **輸入表示法綁 code（第一道牆，我們一手經驗）**：現行模型輸入是綁 code 結構的 grid（BB 的 `(T, ℓ, m, C)`），兩個 model class（`CascadeModel` Conv3d vs `BBCascadeModel` torus conv）**零權重共享**，per-logical head 綁各 code 自己的 min-weight logical basis。跨 code 通用化**首先是輸入表示法/架構問題，其次才是訓練分布問題**。文獻中 Lange 的「detector graph」輸入是這道牆的解法方向，但**沒人示範單一模型跨家族**。
2. **兩個未驗證軸疊加（最大技術風險，openQuestion 之一）**：SAGU 的跨 code 泛化只在 **code-capacity 噪聲 ＋ 高 p（0.06–0.10）**；而我們的 BB-144 場景與 FPGA 部署場景恰恰是 **circuit-level ＋ 低 p**。**兩個未驗證軸疊加後泛化是否仍成立，是此方向最大的技術風險**——反過來說，**這也正是我們 repo 有一手實驗能力去補的空缺**（見 §4 E3）。
3. **as-is 模型對 FPGA 即時性可能太大（硬體端最可能的否證點）**：BB-144 v7 實測 hidden=256、12 blocks、**~4.16M 參數（16.67 MB fp32）**，粗估 **~3.3 GMAC/decode**（12-round 窗）。對照 surface code ~1 µs/round（甚至 utility-scale 指向的 ~350 ns/round），12 µs 窗內跑 3.3 GMAC 需 **~275 TMAC/s（int8）**，很可能**超過一般 FPGA 單卡算力**。文獻裡所有已上 FPGA 的 NN decoder 都是 **d=3/5 的小 surface code**——**沒有人把 BB-144 這種尺寸的 NN 做上 FPGA 即時**。這一步很可能**先否證「這個 as-is model class 放得進 FPGA 做串流即時」**，而非資源容量問題。

### 2.3 一句話可行性判斷

**方向真實且無人佔據，central bet 兩軸都有初步證據；但落地要跨三道牆——輸入表示法、circuit-level+低 p 的跨 code 泛化、以及 as-is 模型的即時性 roofline。** 前兩道 repo 內可用 E1–E4 直接測；第三道要靠先算後做的量化/蒸餾（甚至換 neural-BP），不是調 HLS 能救的。

### 2.4 核心論證：「parity check 從權重搬到輸入」

**問題的骨架**：不同 code 有不同的 parity check（Tanner graph），這是「一個通用 NN 適用所有 code」表面上最大的障礙。**破題方式**：把 parity check 從「烙進權重」改成「餵進輸入」——讓權重**跨 code 共享、只吃跟 code 無關的局部結構（邊型別、node degree）**，而 code-specific 的 Tanner 連結（鄰接、shift、圖）當成**輸入或編譯期常數**。這正是現有 `BBTorusConv` 已在做的事（權重 code-agnostic、只有 6 個 offset 的 shift 綁 code），只是被硬編成單一 code。四項**已核驗**證據構成這個論證的骨幹（引用皆對回 `reports/universal_decoder_survey_evidence.json`）：

- **最直接正例 — SAGU（arXiv:2510.06257，第一輪 finding，high／preprint）**：multi-code 訓練的**單一 NN** 可解碼訓練集外的 BB code、且無精度代價 → 「parity 進輸入」時「一個權重解多 code」在原則上成立（限 BB 家族、code-capacity、高 p）。
- **表示法 — Lange et al. detector-graph GNN（arXiv:2307.01241，第一輪 finding，high／同行評審）**：把解碼形式化為 **detector graph 上的分類**，輸入是「圖」而非綁 code 的 grid → 提供 code-agnostic 的輸入表示法（但每個 code/dataset 仍各訓一網，未示範單一模型跨 code）。
- **反例 — Transformer-QEC（arXiv:2311.16082，第一輪 finding，high）**：權重綁 code，transferability **僅跨 distance 且需 fine-tune** → 當 parity 留在權重、通用性就受限，**反證通用 decoder 的空缺真實存在**。
- **本輪 neural-BP 的機制層佐證（`neuralBpRound.findings`）**：Astra（arXiv:2408.07038）能把低 distance 訓練的 decoder 外推到 **surface code d=25、BB code d=34**，正因「**Tanner graph 節點 degree 不隨 distance 改變**」（`nbp-b-transfer-within-family`，3-0）——**可轉移的是 message-passing 函數**（綁在正規化局部結構／邊型別上的權重），而 code-specific 的 parity 結構經 degree／圖進入 = 輸入；反向，scaled-min-sum（arXiv:2605.10433，`nbp-b-degree-binding`，2-1）顯示**把 parity 留在固定權重**（固定 scaling 綁 CN degree）會隨 code 改變 penalty 遞增。

**小結**：文獻在**原理層一致支持**「parity check 搬到輸入」是通用化的正確軸；此論證與 §4.2.1 方案 A（edge-type 權重共享的 graph message passing）是同一邏輯。**尚未被任何工作跨越**的仍是兩點（併入 §3）：跨**結構相異家族**（surface↔BB↔color）的單一權重、以及 **circuit-level ＋ 低 p** 的疊加。

### 2.5 neural-BP 路線判定（補核驗輪 `neuralBpRound`）：低風險 fallback 還是準確率死路？

本輪補一輪 neural-BP 逐篇對抗式核驗（wf_7b6a70f5，19 sources、94 claims → 25 核驗 → **21 confirmed / 3 refuted / 1 unverified**，皆 3-vote；已併入 `reports/universal_decoder_survey_evidence.json` 的 `neuralBpRound`，含被隔離的 refuted/unverified）。

> **判定（一句話）：neural-BP／learned message passing 是「低風險 fallback」，不是「準確率死路」——但有條件（見下 (a) 反面與含意）。** confidence：核心「非死路」= **high**（兩個獨立來源交叉佐證）；操作性限定 = medium。

四個子問題的實際數字（全部對回 `neuralBpRound`）：

- **(a) 不接 OSD 的準確率上限、對 BP+OSD 的差距**
  - **正面（high，兩獨立工作）**：**Astra**（learned GNN message passing，**不接任何 post-processing**）在 surface（訓練至 d=11）與 BB（訓練至 d=18）threshold 與 LER **皆勝 BP+OSD**，正文報 surface threshold 約 **17%（Astra）對 14%（BP+OSD）**（**code-capacity**，2408.07038，3-0；⚠ 單點 LER 數值未能自 abstract/正文核實，該 framing 已被 0-3 否決並隔離，**不得引用單點 LER 對打我方 8.8×**）。**Relay-BP**（純 message passing、不接 OSD）於 BB **circuit-level** 顯著勝 **BP+OSD+CS-10**，[[144,12,12]] 於 **p=3×10⁻³ 約低 1–2 個數量級**（2506.01779，3-0；惟 Relay-BP **非** neural／learned）。小型 GB code 上，不接 OSD 的 overcomplete BP（OBP4）亦勝 min-sum+OSD reference（2308.08208／2212.10245，3-0），但機制是 **overcomplete matrix 非 learned 權重**。
  - **反面／限制**：經典 neural BP（Liu & Poulin，1811.07835／PRL 122.200501）只相對 **untrained BP** 提升最多 **3 個數量級**、**完全無 BP+OSD 比較、無絕對 LER**（3-0）；**天真的可訓練權重 neural min-sum 在 bicycle（GB）家族訓練失敗、作者略去結果**（2308.08208，3-0）；BP+GNN（2310.17758）僅勝三種**非 OSD** post-processing、不勝 OSD（2-0）；純 NN transformer 在 **BB-144 反而比 BP+OSD 差**（q-2026-06-30-2149，3-0）。
- **(b) 通用性**：learned message-passing 權重限**同家族內**跨尺寸／距離轉移（Liu & Poulin size；Astra distance，皆 3-0），固定 min-sum 權重綁 **CN degree**（2605.10433，2-1）；**無跨家族的單一權重組**——與主線 §3 第 1 項同一缺口。
- **(c) 硬體**：本輪**未產出任何 neural-BP 專屬的 FPGA/ASIC latency/資源數字**（缺口）；min-sum 定點化先例僅由 2605.10433 的 CN-degree scaling 間接觸及。硬體實測仍以 §1.2／§1.3 的非 NN 與 surface-NN 為準。
- **(d) 最新（2024–2026）**：Relay-BP 是 BB code 熱潮後**最重要的純 BP 改良**，但**不是** learned／neural（disordered memory 由 gradient-free 超參優化 + 每 leg 隨機取樣決定）。

**含意（對用戶想法）**：

1. **準確率不是死路**——no-OSD message passing 在 BB code 已被 **Astra ＋ Relay-BP 兩個獨立工作**證明可勝 BP+OSD。作為「若 as-is Cascade 上不了 FPGA、退而用更輕的 message-passing decoder」的 fallback，**準確率保險成立**。
2. **但 fallback 必須是 GNN 式／learned message passing（SAGU／Astra／QuBA）或 Relay-BP 式 disordered memory，不能是天真 neural min-sum**（在 bicycle 家族已證訓練失敗）。
3. **殘餘風險與主線相同**，落在通用性軸（權重綁 Tanner-graph degree、per-distance 單模型、無跨家族單一權重）與 circuit-level+低 p 的疊加。
4. **口徑警告**：上述勝差**多為 code-capacity**；且各 baseline 口徑不一（BP+OSD+CS-10／min-sum+OSD-10／我方 min-weight circuit BP+OSD），**不可與我方 8.8× 直接並列**。
5. **有利對照**：文獻裡**純 NN**（transformer）在 BB-144 反而**輸** BP+OSD，而我方 Cascade v7 在 BB-144 **全點勝** BP+OSD（§2.1）——凸顯 Cascade 結果的非平凡性（惟架構／訓練不同，非同類重現）。

---

## 3. 空缺（gap）：哪些沒人做／做得淺

1. **跨「家族」的單一模型**：無任何論文示範單一權重在 **surface↔BB↔color** 間 zero-shot 或 fine-tune 轉移。SAGU 限 BB/coprime-BB 家族內；Transformer-QEC、NTU 均為跨 distance 非跨家族。
2. **跨 code 泛化 × circuit-level × 低 p 的三重疊加**：現有跨 code 證據（SAGU）只在 code-capacity、高 p；FPGA 部署場景恰是 circuit-level、低 p。這個交集**完全沒有文獻數據**。
3. **通用 NN ＋ 硬體部署的整條路徑**（線3 核心）：無人把跨 code 泛化的 NN 量化後上 FPGA。NTU 明列為 future work。
4. **GNN/attention 的 message-passing 塞進 FPGA latency 預算**：已示範上板的 NN 全是 **LSTM/FC/UNet**；**GNN 的不規則稀疏運算**在 ~350 ns–1 µs/round 預算下量化後的資源/時序**完全沒有文獻數據**（openQuestion 之一）。
5. **真正的 latency 目標未定案**：1 µs/round 是慣用值，utility-scale 分析指向 ~350 ns（2.86 MHz）且 backlog 代價巨大；**通用 NN 若因泛化而變大，量化/蒸餾能守住哪個目標，無人回答**。
6. **BB/qLDPC code 沒有非 NN 的 FPGA 基準**：LCD/CC/Helios 全是 surface 專用。這是雙面刃——**BB-144 NN-on-FPGA 沒有現成硬體對手可比（新），但也意味著沒有現成的「贏過既有硬體」故事可套用**。
7. **neural-BP 在 BB/qLDPC 的準確率上限與硬體代價**（設計稿②的 fallback 路線）：**準確率子問題已由補核驗輪填補**（見 §2.5、`reports/universal_decoder_survey_evidence.json` 的 `neuralBpRound`）——不接 OSD 的 learned／message-passing decoder 在 BB code 上**可勝 BP+OSD**（Astra 2408.07038 code-capacity、Relay-BP 2506.01779 circuit-level，皆 3-0），故 neural-BP 是**低風險 fallback、非準確率死路**，惟須走 GNN／Relay-BP 式（天真 neural min-sum 在 bicycle 家族訓練失敗，2308.08208），且通用性殘餘風險（權重綁 CN degree，2605.10433）仍在。**唯一仍空的子問題是硬體**：本輪**未產出任何 neural-BP 專屬的 FPGA/ASIC latency/資源數字**（min-sum 定點化僅由 2605.10433 的 CN-degree scaling 間接觸及）——neural-BP 的硬體代價仍需再補一輪硬體文獻。

---

## 4. 可嘗試方向（分兩層：repo 內下個 iteration ＋ 更廣路線圖）

> 素材來源：第一層取自 opus 設計稿①（最小改動跨 code 實驗梯 E1–E4），第二層取自 opus 設計稿②（code-agnostic 架構 + FPGA 路徑）。**凡設計稿與文獻衝突或被文獻超車者，以文獻為準並註明**（見 4.1 開頭與 4.2 的文獻校準框）。

### 4.1 第一層 — repo 內下一個 iteration 就能動手：E1–E4 跨 code 實驗梯

**與文獻的關係（重要，先講）**：設計稿①寫作時**尚不知 SAGU 存在**（設計稿②的 §4 還把「是否有單一權重跨多 code family 的先例」列為待核驗）。**SAGU 已在原則上先行/部分超車了 E3 的中心假說**（multi-code 訓練的單一 backbone 可跨 code、無精度代價）。**但 SAGU 限 code-capacity、高 p、QuBA GNN 架構**；本 repo 的 E3 是 **circuit-level ＋ 低 p ＋ Cascade torus-conv 架構**——正好落在 §3 第 2 項那個「完全沒有文獻數據」的三重疊加上。**因此 E1–E4 不是重造 SAGU，而是把 SAGU 的結論推進到它明確未涵蓋、且對 FPGA 部署最關鍵的兩個軸。**

**梯子的可行性前提（設計稿①對 checkpoint 核實）**：BB-72 與 BB-144 **共用多項式** `a=(3,1,2), b=(3,1,2)`、同 `k=12`、同 `hidden=256`，故每個 learnable tensor 形狀相同，code-specific 結構全在 `persistent=False` buffer（構造時重建）。唯一要對齊的結構旋鈕是 block 數（BB-72=8、BB-144=12），且 `rounds/T` 因全卷積而可自由。**含意**：`BBCascadeModel(code=144, num_blocks=8).load_state_dict(bb72_state_dict, strict=True)` 可乾淨載入並跑出 BB-144 decoder——**transfer 是語意問題不是接線問題**，這正是梯子要隔離的變數。

> **內建 caveat（校準每級結論的強度）**：BB-72↔BB-144 是**最容易的 transfer pair**（只差 torus 尺寸 ℓ、多項式相同）。E1–E3 回答「**size** 是否 transfer」；「**operator** 是否跨家族 transfer」要靠不同多項式的 code（E4，仍維度相容，只有 `torch.roll` offset 變）。

| 級 | 科學問題 | 最小實作 | 資源估計 | 驗收（PASS） | 失敗含意 |
|---|---|---|---|---|---|
| **E1** zero-shot 權重移植（BB-72→BB-144）backbone 診斷 | BB-72 backbone 在 BB-144 上產生的是**非垃圾**特徵，還是 code-specialized？（接線相容已證，這裡量表示轉移） | NEW `scripts/34_zeroshot_bb_transfer.py`（clone `evaluate_bb`，建 `BBCascadeModel(144, num_blocks=8)` 載 BB-72 `best.pt`，dump `p_block`/per-logical `std`/`bce`；並跑反向 144→72）；NEW `slurm/eval_zeroshot_bb.sh`（`dev`）。**無 `src/` 改動** | **< 1 GPU-h**；~0.5 天（人力為主，dev 分區） | 12 個 head 的 per-logical **std > 0.1** 且 per-logical BCE 有結構（非齊平 ln2）→ backbone transfer，放行 E2 | 特徵塌到 chance/齊平 → backbone code-specialized；**跳過 E2 直達架構邊界**（此負結果本身值得報告） |
| **E2** frozen-backbone 重訓 readout（BB-72 backbone→BB-144 readout） | backbone 是否為**通用 BB error-propagation engine**、code 知識全在便宜的線性 readout？ | 給 `scripts/14_train_bb_v3.py` 加兩個**向後相容** flag：`--init-from PATH`（`load_state_dict(strict=False)`）、`--freeze-backbone`（`embed/blocks` 設 `requires_grad=False`）；**optimizer 不用改**（`optimizers.py:43` `split_params` 已跳過 frozen 參數）；用**與 v7 相同的 mixed-p 配方**，讓 E2 vs v7 是單變數比較；NEW `slurm/train_bb_v8_xfer.sh`，TAG `v8_bb144_xfer72` | **~25–30 GPU-h**；~1–1.5 天；單一 48h segment（無 chain） | 恢復瀑布、`p_block(0.002)` 落在 v7 的 0.00038 **~2× 內**、≥3 個掃描點勝/平 BP+OSD | readout-only 無法恢復瀑布 → code 知識分散在 backbone 而非 readout → 需 E3 或換架構（兩種結果都可發表、都精確定位 code 知識所在） |
| **E3** mixed-code co-training（BB-72+BB-144，shared backbone + per-code readout） | **中心假說**：iter-7 證噪聲軸可由分布設計泛化，**同一配方轉到 code 軸**能否讓一個 shared backbone 同時匹配兩個 code 的專訓模型？ | 去重兩個近重複 bottleneck（`cascade_bb.py:55` 重用 `blocks.py:17`，純 refactor）；不建新 model class，為兩 code 各建 `BBCascadeModel` 後把 `embed/blocks` 別名為同一組 module（梯度共同更新一個 backbone），各留自己的 `final_conv/heads/buffers`；NEW `scripts/36_cotrain_bb.py`（交錯兩 code batch、累積 loss、單步）；NEW `slurm/train_bb_v8_cotrain.sh`，TAG `v8_bbmix_cotrain` | **~170–190 GPU-h**；~7–8 天；4-segment 48h `afterany` chain（同 v7 resume 模式） | 兩 code 同時落在各自專訓的 **~1.5× 內** → 「一個 backbone 解多 code」**可由 code 軸分布設計學出**（mixed-p 勝利的直接對應） | joint 訓練被迫 Pareto 妥協（一/兩個 code 大幅退步）→ code 軸**不像**噪聲軸那樣泛化 → 這個負結果終結最小改動路線、**授權轉向 code-agnostic 架構** |
| **E4**（選配驗證）unseen-polynomial BB code | backbone 跨**家族**（不同多項式→不同 Tanner offset）還是只跨**尺寸**？（真正的通用性測試） | 加一個不同 `(a,b)`（另一個 Bravyi et al. `k=12` code）的 `BBCode` factory，對 E3 backbone 跑 E1 式 zero-shot 與/或 E2 式 readout-only。維度相容（只 roll offset 變） | zero-shot ~E1（**<1 GPU-h**）；readout-only ~E2（**~25 GPU-h**） | 成功＝真家族級通用；失敗＝只是 size-transfer 非 operator-transfer——**校準 E1–E3 宣稱的強度** | 定 E1–E3 主張的適用邊界 |

**最小改動路線的天花板（硬界，設計稿①）**：**E3 之後到頂**。最多交付「一個 shared backbone + 小的 per-code readout（`final_conv` + k heads，per-code 重訓/co-train）」。**無法**交付「readout 也 code-agnostic」的 decoder（`final_conv`、k heads、min-weight-basis pooling 全硬綁固定 k 與 code-specific logical basis）。若 **E2 失敗，或 E3 顯示硬 Pareto 妥協，就停**——答案是「最小改動已窮盡、換架構」，交棒給第二層。

**E3 執行風險（設計稿①已標，開工前先看）**：shared backbone 的 `BatchNorm3d` running stats 會被兩 code activation 混合驅動，若不穩，fallback 為 GroupNorm 或 per-code BN affine——列為 E3 smoke 第一個要盯的點；別名須嚴格限於 `embed_*`+`blocks`，per-code readout 要看到自己的 `scatter_matrix`/`per_logical_mask`（以 smoke 對齊獨立建的單 code 模型驗證）。**慣例**：每級走 SLURM（smoke 在 `dev`）、新 TAG、新 checkpoint dir、不動 v6/v7 checkpoints 或 `results/`。

### 4.2 第二層 — 更廣路線圖：code-agnostic 架構 ＋ FPGA 落地

#### 4.2.1 Code-agnostic 表示法：三方案（設計稿②，已用文獻校準）

**核心原則**：跨 code 通用必須落在**架構與權重**上，code 結構（Tanner/detector graph）當成**輸入或編譯期常數**——這正是現有 `BBTorusConv` 已在做的（權重跟 code 無關、只有 6 個 offset 的 shift 綁 code），只是被硬編成單一 code。

- **方案 A（主推）— edge-type 權重共享的 detector/Tanner-graph message passing**：把 decoder 定義在圖 `G(code)` 上，節點=detector/check＋data-qubit，邊=Tanner 關聯附邊型別（X/Z、A/B side、時間±1），每輪 message passing 用**一組 code-agnostic、按邊型別共享**的權重。**換 code 只換圖不換權重**。通用性上限最高，且與現有 Cascade 距離最小（現有 `BBTorusConv` 就是此方案在 circulant 圖上的特例，6 offset = 6 種 A/B 邊型別）。**文獻校準**：Lange 的 detector-graph 分類**驗證了「圖輸入」是可行的 code-agnostic 表示法**；SAGU 用 GNN（QuBA）做 multi-code 訓練**驗證了此類架構確實能承載跨 code 泛化**——**方案 A 的方向已被文獻背書，缺口在（i）跨結構相異家族、（ii）circuit-level+低 p、（iii）上 FPGA 三者，皆無人做**。
- **方案 C（並行低風險 fallback + 蒸餾/初始化來源）— neural-BP（unrolled BP as NN）／learned message passing**：Tanner-graph native 天生通用、硬體成熟度最高（min-sum/查表/int 定點，FPGA/ASIC 主流）、樣本效率最高。**準確率風險——已由補核驗輪回答（見 §2.5、`neuralBpRound`）**：vanilla BP 在 BB codes 會失敗需 OSD（本 repo baseline 就是 BP+**OSD**），**但改良的 message-passing 變體不接 OSD 即可勝 BP+OSD**——Astra（learned GNN MP、no post-processing）在 surface+BB threshold/LER 勝 BP+OSD（code-capacity，2408.07038，3-0）、Relay-BP（純 MP、no OSD）於 BB circuit-level 勝 BP+OSD+CS-10 且 [[144,12,12]] 於 p=3×10⁻³ 約低 **1–2 個數量級**（2506.01779，3-0）。**故方案 C 是低風險 fallback、非準確率死路**，但兩點必守：**(1) 不能用天真 neural min-sum**——其在 bicycle（GB）家族訓練已證失敗（2308.08208，3-0），要走 GNN／Relay-BP 式；**(2) 通用性殘餘風險**（權重綁 CN degree，2605.10433，2-1）與主線相同。**唯一仍空**：neural-BP 專屬的 FPGA/ASIC 資源/latency 數字本輪未涵蓋（§3 第 7 項、§5）。引用皆對回 `reports/universal_decoder_survey_evidence.json` 的 `neuralBpRound`。
- **方案 B（不推）— syndrome 序列 + code 描述子的 conditional 架構**：通用性上限高但不確定（要模型「讀懂」描述子，泛化到沒見過的家族風險大），硬體故事最弱（hypernetwork 讓權重動態化、破壞固定 dataflow），樣本效率最低。

> **統一實作技巧**：A 與 C 都是「圖上 edge-type 權重共享的 message passing」，差別只在層數/聚合子/是否 log-domain。可設計**同一份 graph-MP kernel**，A/C 只是超參配置——讓後續 FPGA kernel 只需寫一套。

#### 4.2.2 FPGA 路徑（以 BB-144 v7 為第一標的，可動手階梯）

**目標刻意務實**：先把**已訓練好的 `BBCascadeModel`-144 端到端搬上板**（不換架構），量真實 latency/資源，再決定是否換架構。使用者有 FPGA 板 + HDL/HLS 經驗，故以下具體到可動手。

> **repo 數字缺口（已核實更正）**：plan Context（Explore 盤點）記 BB-144 為「hidden=128×6 blocks、best.pt 純 EMA ~33.6MB」——此為過時舊值。經 `best.pt` config/state_dict 與訓練 log（`Hidden 256, Blocks 12, params=4,158,220`）雙重核對：**hidden=256、12 blocks、~4.16M 參數（16.67 MB fp32）**；`best.pt` 實測 **≈50.4 MB**（含 model+EMA+optimizer 等 state，非僅權重）。注意訓練腳本 `14_train_bb_v3.py` 的 `--hidden` **預設值是 128**，256 是 v3/v7 run 時的 CLI 覆蓋值——grep 腳本預設會誤導。兩份設計稿均用 256。**FPGA 資源估算以 checkpoint-verified 的 256/12/4.16M 為準**（檔案大小與部署無關，部署看的是權重參數量）。

- **Step 0 — 基準與可證偽點先行（先算再做，紙上 1 天）**
  - 用 `scripts/25_inference_latency.py` 建 GPU 參考點（batch 掃描，凸顯單 shot vs 批次：archive 顯示 batch=1 forward ~9.6 ms、batch=64 才降到 ~33 µs/cycle——**GPU throughput 導向、單 shot latency 極差**，正是 FPGA 確定性低延遲的切入理由）。
  - **粗算 ~3.3 GMAC/decode**（12 blocks ~1.8 GMAC + `final_conv` ~1.47 GMAC；heads pooling 後可忽略）。
  - **可證偽點 #1（最重要，go/no-go gate）**：**文獻已定錨即時預算**——surface code ~1 µs/round（LCD 到 d=17 <1 µs、Helios 11.5–23.7 ns、CC FPGA 810 ns 是已核驗的非 NN 實測基準），utility-scale 分析更指向 **~350 ns/round（2.86 MHz）**。BB-144 一個 12-round 窗（~12 µs）跑 3.3 GMAC → 需 **~275 TMAC/s（int8）**，很可能超過一般 FPGA 單卡算力（**目標器件實際 int8 TOPS/DSP 數待 datasheet 核對**）。**動任何 HLS 前先用這個算術做 go/no-go**；被否證就導向「必須大幅縮模型/換 neural-BP」而非「調 HLS」。

- **Step 1 — 量化（PTQ → QAT）**：先 **PTQ int8**（activation 是 SiLU 後有界值、輸入二值 detection events，動態範圍溫和），重點放參數大戶 **`final_conv`（786K）＋ 12 heads（1.57M）** 做 per-channel。**BN 折疊**（`BatchNorm3d` fold 進前一個 1×1，上板前必做）。掉點就落 **QAT**（沿用 iter-7 mixed-p 框架 `trainer_v2.py` 加 fake-quant）。**文獻範本**：QUNET 已示範 **surface decoder QAT 到 4 bits + early-exit**——**位寬選擇與 QAT 配方可直接借用**（但 QUNET 是 UNet/surface，本標的是 torus-conv/BB，需自行驗證）。**可證偽點 #2**：若 int8（甚至 int4 for 1×1）在低 p 外推區（如已知全對的 p=0.0005）明顯掉點，代表 head 的 XOR-parity readout 對量化敏感。交付物：fp32→int8 的 `p_block` 對照（用 `decoder_compare.py` 同管線）。

- **Step 2 — 算子盤點 → HLS 對應**（關鍵洞察：`roll` 與稀疏 scatter 的「不規則感」全來自**編譯期已知的 circulant/Tanner 結構**，HLS 可**完全靜態展開**成規則 dataflow，無 run-time 動態記憶體存取——這是本 model class 對 FPGA 友善的根本原因，無 attention 無遞迴）：

  | PyTorch 算子 | HLS 對應 | 難度 |
  |---|---|---|
  | Conv3d k=1（embed/proj/`final_conv`/`self_check`） | 純 GEMM / systolic，`final_conv` 是最大單一 GEMM | 低（最成熟） |
  | `torch.roll`×6 + einsum（`BBTorusConv`） | circulant shift = 靜態位址重排 + 6-tap 加權和，編譯期展開成固定 line-buffer | 中 |
  | 稀疏 Tanner scatter（`cascade_bb.py:273`） | 固定稀疏矩陣（每 data-qubit 度數小）→ 固定 MAC 樹/小 SpMV | 中 |
  | time Conv1d k=3 | 3-tap FIR，line-buffer | 低 |
  | masked mean-pool + per-logical MLP head | reduction + 小 GEMM | 低 |
  | SiLU | 分段線性 LUT 近似 | 低 |

  **建議**：先 **Vitis HLS**（C++ 迴圈 + `#pragma HLS pipeline/array_partition/dataflow`）做完整功能與資源探索，把 GEMM/roll 寫成可重用 template；只有 timing 關鍵路徑（systolic PE）才下沉手寫 HDL。**文獻校準**：SUSTech 的 d=3 LSTM 做到 **124 ns NN 推論**、Overwater FC-FF post-PnR **<90 ns**——證明 surface 小 code 的 NN 上板可行，但**沒有 BB-144 尺寸的先例**，本標的的 latency/資源要靠自己的 HLS report 校準。

- **Step 3 — 資源估算（roofline）**：**DSP** 以並行 MAC 估（int8 一顆 DSP48 packing 2 MAC；目標 latency 反推並行度）。**BRAM**：int8 權重 ~4.2 MB，**可能超過典型 FPGA 片上 BRAM/URAM（單卡 ~數 MB–數十 MB，待 datasheet）**，需權重流式（HBM/DDR）或剪枝/低秩化 `final_conv`＋head 或分時重用；activation 峰值 1872×256×int8 ≈ 0.48 MB 可片上。**做一張 roofline（模型算力/記憶體 vs 器件上限）**，一眼看出 compute-bound 或 memory-bound（本模型幾乎確定雙 bound）。

- **Step 4 — 板上量測協定**：**latency**（batch=1 從 DMA-in 到 logit 出的**確定性**時間，用板上 counter/ILA 量非 host 計時，報 µs/decode 與 µs/round）與 **throughput**（pipeline 滿載 shots/s）**分開報**；對照 surface ~1 µs/round 與 BB-144 12-round 窗，明確標「即時串流」vs「離線批次」。**正確性回歸**：板上 int8 forward 的 logits 與 GPU int8 reference bit-exact（或 p_block 一致），用 `decoder_compare.py` 在 p∈{0.001…0.0055} 掃描比對。**可證偽點 #3**：若板上 latency 達標但**必須靠大 batch** 才有吞吐（如 GPU），則喪失 FPGA 低延遲賣點＝否證「用 FPGA 換即時性」的動機。

  **最可能否證「這個 model class 放得進 FPGA」的排序**：**#1 即時性 roofline（Step 0，紙上就能否證）> BRAM 容納 4 MB int8 權重（Step 3）> QAT 掉點（Step 1）**。**先做 Step 0 算術 gate 再投入 HLS。**

#### 4.2.3 通用性 vs 硬體的張力，與化解

- **本質衝突**：通用＝圖結構 run-time 可變（動態鄰接/定址/層數）；FPGA 高效＝固定 dataflow、靜態定址、pipeline 深度編譯期定死。真正動態圖與固定電路互斥。
- **化解：「通用權重、per-code 編譯期特化 dataflow」**（本設計核心，與現有碼天然契合）：權重**跨 code 共享、離線訓練一次**；每個目標 code 在**編譯期**把它的 Tanner/detector 圖展開成**該 code 專屬的靜態 HLS dataflow**（roll→固定 line-buffer，scatter→固定 MAC 樹）。**換 code = 換 bitstream / partial-reconfigure，不換權重、不重訓**——保留「一個通用 NN」的科學主張，同時給硬體固定電路。
- **折衷光譜**：全靜態（每 code 一 bitstream，效能最好彈性最差)→ partial reconfiguration（秒級重配)→ **參數化 overlay**（shift/adjacency 做成可載入 table，dataflow 固定但支援同家族不同尺寸）。**對 circulant BB 家族特別可行**（只 offset 數值變、拓撲不變），是「通用電路」的甜蜜點，建議作第二階段目標。
- **對 GNN「不規則存取」硬傷的正式評估**：對**任意未知圖**成立；但對 **surface/BB 這類 Cayley/circulant 圖**，鄰接是 affine 位址、編譯期可全靜態化 →**硬傷在本專案目標 code 上基本消失**（換到真正不規則的 qLDPC 才回來）。這是方案 A 的底氣，也是要對外講清楚的邊界條件。**⚠ openQuestion**：GNN message-passing 量化後塞進 350 ns–1 µs 預算的資源/時序，**目前零文獻數據**——這是路線圖上最需要自己補的一格。

#### 4.2.4 建議執行順序（收斂用）

1. **§4.2.2 Step 0 即時性 roofline 算術 gate**（紙上，1 天）→ 決定是否值得搬 as-is 模型。
2. 平行兩線：**線 A** = `BBCascadeModel`-144 做 BN-fold + PTQ int8，量準確率；**線 C** = 實作 graph-MP kernel（A/C 共用），先在 GPU 驗證方案 A 一般化 = 現有 BB 特例（bit-level 對得上）。
3. HLS bring-up 先做最大單一算子 `final_conv` GEMM 與一個 `BBTorusConv` block，量 DSP/BRAM/latency 校準估算。
4. 依文獻/量測結果決定：搬 as-is（若即時預算寬鬆）或轉 neural-BP 蒸餾（若被 #1 否證）——**其準確率文獻已補齊（§2.5：低風險 fallback、走 GNN／Relay-BP 式、非天真 neural min-sum），但轉前仍須補 neural-BP 的硬體代價文獻**（§3 第 7 項）。

---

## 5. 方法與品質保證

**為什麼可以信這份報告的引文**：本報告底層是**兩輪** deep-research 式的多線 fan-out + 逐篇對抗式核驗流程。**第一輪（universal-decoder 主輪）**數字如下（`reports/universal_decoder_survey_evidence.json` 的 `stats`）：

- **5 條檢索角度**、**17 個來源抓取**；
- **抽出 84 條原始宣稱 → 核驗 25 條 → 24 confirmed / 1 killed**（`claimsExtracted=84, claimsVerified=25, confirmed=24, killed=1`）；核驗後合併同源宣稱，**綜合成 12 條 findings**；
- **每條宣稱經 3-vote 對抗式投票**（獨立 agent 對實際 abstract/內文核對「我們宣稱它做了什麼」，逐字比對）：全部 confirmed 宣稱均為 **3-0** 通過；
- **共 99 次 agent 呼叫**、9 個 URL 去重、4 條因預算未及核驗（budgetDropped，**未進本報告**）。

**第二輪（neural-BP 補核驗，`neuralBpRound.stats`，wf_7b6a70f5，檢索 2026-07）**：

- **5 條檢索角度**、**19 個來源抓取**；
- **抽出 94 條原始宣稱 → 核驗 25 條 → 21 confirmed / 3 refuted / 1 unverified**（`claims=94, verified=25, confirmed=21, killed=3, unverified=1`）；核驗後合併同源宣稱，**綜合成 10 條 findings（含 4 子問題分組與戰略判定）**；
- **每條宣稱經 3-vote 對抗式投票**；**101 次 agent 呼叫**（本輪 agent 計數取自流程回報）；
- **被隔離的 3 條 refuted + 1 條 unverified**（larger GB codes 上 OBP4 不勝 MS+OSD〔0-3〕、SAGMS 為非學習式〔1-2〕、Astra 在 BB「數個數量級」但無單點 LER〔0-3〕、Gong test-code 細節〔三票 errored〕）見 `neuralBpRound.refuted/unverified/isolationNote`，**不得當支持證據**。

**被 killed 的那一條（防線確實有作用）**：一條原始宣稱——「Riverlane 的 FPGA Union-Find 對 d=30 quantum memory 的 **1.0 µs/round** 是 surface code decoder 的 **state-of-the-art**，只在 QPU stabilization round ≥ 1.0 µs 時才夠用」——在 3-vote 中被 **0-3 全數否決**、剔除。**剔除理由**：這是一句**過度概括的「SOTA」宣稱**，來源（arXiv:2511.10633）在對抗式核對下**不支持該具體數字與「最先進」定位**（該文是 utility-scale 開銷分析，非 decoder benchmark；把它讀成「1.0 µs/round 是 SOTA」是幻覺式引申）。因此本報告引用 2511.10633 時**只取其已核實的 overhead 數字（~350 ns 目標、+qubit/runtime 開銷），避開被否決的那句**。這條 killed 記錄是「報告只引用經核驗材料」這道防線真的攔下了東西的直接證據。

**引文可追溯性**：本報告每一個文獻宣稱都可對回 `reports/universal_decoder_survey_evidence.json` 的某條 finding/evidence（arXiv/DOI 已於正文標注）；**未在該 JSON 出現的論文或數字，一律未寫入本報告**。BB-144 一手數字全部對齊 `reports/iteration_7_status.md`。

**已知材料缺口（誠實交代）**：

1. **neural-BP 準確率——已補核驗（本項原為缺口、現已填補）**：已補一輪 neural-BP 逐篇對抗式核驗（`neuralBpRound`，見下方方法段與 §2.5）。結論：不接 OSD 的 learned／message-passing decoder 在 BB code 可勝 BP+OSD（Astra、Relay-BP，3-0），故方案 C 是**低風險 fallback、非準確率死路**（惟須走 GNN／Relay-BP 式、非天真 neural min-sum）。**仍未填補的子缺口**：neural-BP 專屬的 FPGA/ASIC latency/資源數字（本輪硬體軸為空，min-sum 定點化僅間接觸及）；若要把方案 C 從「候選」升為「硬體決策」，需再補一輪 neural-BP 硬體/learned-OSD 文獻。
2. **目標 FPGA 器件的實際上限（int8 TOPS / DSP 數 / BRAM / HBM 頻寬）**未在文獻證據內，§4.2.2 的 roofline 數字待實際 datasheet 填入。
3. **NTU（arXiv:2606.27119）信心為 medium 且不宜與 BB-144 直接對打**：它的 code 是 [[72,12,6]] 非 [[144,12,12]]、baseline 是 Relay-BP 非 BP+OSD、1 個月新且自報無重現——引用時只作「跨 distance transfer 存在、real-time 是 future work」的證據，**不要拿它的 LER 數字跟我們的 8.8× 並列比較**。
4. **preprint 比例高**（§開頭警示）：SAGU/NTU/SUSTech/utility-scale 四篇未評審，性能皆自報。
5. **「交集為空」是集合內陳述**：檢索截至 2026-07，無法排除未檢得的新工作；此領域月更快，跟教授談時請標檢索日期。
