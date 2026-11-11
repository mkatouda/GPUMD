# 第 5 章 現状、未完了の作業、限界、再現手順

## 5.1 状態

DR-005 の traceability は **`partial`**（`implemented` ではありません）。

| 項目 | 状態 |
|---|---|
| モデル・カーネル（`gpumd`）、学習側（`nep`） | 実装済み・検証済み（合成データ）。04 章 |
| `gpumd_replica`（モード 2） | コード済み。**GPU 検証は未実施**（04 章 4.7 節） |
| 文書（英語 `doc/`、本マニュアル） | 完了（公式の Sphinx ビルドは、変更したページで警告なし。既存の無関係な警告 3 件が別に存在） |
| 全回帰（`run_regression.py --suite full`） | **未実施**。人間の合図があるまで、また再学習の終了まで開始しない |
| 再学習（段階 5） | 依頼元（GPUMD-polymer-apps）側で実施中（5.3 節） |
| 周期系の P そのものの出力（DES-0005 3b） | **未実装** |

## 5.2 未完了の作業（順序）

1. 再学習の終了後：段階 3b の GPU 検証（`pytest tests_pytest/test_replica_charge_transfer.py -q`、`python3 tests_regression/replica/test_replica.py`、`run_regression.py` の replica ケース）。合格後に、`doc/gpumd-replica/input_parameters/potential.rst`（「モード 2 は対応、モード 3 は非対応」）と VER-0034 の README の記述を更新。
2. 全回帰：現ビルド（候補）を、修正前の HEAD（DR-006 前の `d7806f1b` を基にしたもの）を基準に比較。AGENTS.md の既知失敗 37 件のリストの再測定も含む。人間の合図で開始。
3. `nep` 対 `gpumd` の差の切り分け（5.4 節、`gpumd` 同士の差の床の測定）。
4. 直接の P の出力（`compute_dpdt` の追加列など。既定は無効）：未実装。DES-0005 3b で承認済みだが、インターフェースは未確定。
5. 段階 5b の結果（held-out BEC の RMSE と ASR など）の受領後、C の対角 BEC のずれ（5.3 節）の原因の見立て。
6. upstream への PR を作る場合：テストが `specs/` の生成器に依存するため、PR の段階で再配置する（試験を `specs/` から import して、無ければ skip する形。決定済みだが延期）。`specs/` と `.claude/` は PR に含めない。
7. HIP ビルド（この機械に hipcc がなく、未検証）。

## 5.3 再学習（段階 5）の途中経過（依頼元の報告、2026-11-11）

出典：GPUMD-polymer-apps の `docs/work-requests/reply-to-gpumd-bec-rev0-checks.md`。本節の数値は**依頼元が測定したもので、GPUMD 側では再現していません**。依頼元のデータは Tier P（試作）で、Orb-ft2 の E/F ラベルのライセンスが未確認のため、**内部の開発・検証用に限り、再配布・商用モデルの学習には使いません**。

- **5a（表現力の短い確認）**：N50 データ、`charge_transfer 6`、generation 2000、seed 1。held-out BEC RMSE（全成分）0.159 e（旧モデル、同じ予算・seed 1 で 0.112 e、比 1.4、ゲート ≤ 2× を**満たす**）。まだ収束しておらず（世代 500/1000/1500/2000 で 0.238 → 0.192 → 0.173 → 0.164）。対角成分の C に系統的なずれ −0.28 e（旧 +0.003）、F に +0.13 e。ASR の残差 1.7×10⁻⁵ e。力 RMSE（test）0.330 対 0.409 eV/Å、エネルギー 0.045 対 0.030 eV/原子（並記のみ、ゲートなし）。限界：seed 1 のみ、2000 世代、held-out 20 構造。
- **5b**（generation 10000、seed 1〜3、同一データ）は 2026-11-11 18:24 に開始済み。結果は出次第共有される予定です。
- C の対角 BEC の系統的なずれ（−0.28 e）が、学習が途中だからか、`charge_transfer` の表現力の制約かは、**未確定**です。原因の見立ては 5b の結果の後で行います。中間の `save_potential` の `nep.txt` を使った比較と、旧モデル対新モデルの `dump_xyz charge` の比較が、依頼元から提案されています。
- 依頼元の合否の目安（DR-005 A6）：held-out BEC RMSE は、旧の現行値 0.073 ± 0.012 e の 2 倍（約 0.15 e）を目安とし、超える場合は人間が判断します。E/F は BEC を優先し、数値のゲートはありません（並記のみ）。依頼元の再試験（T1〜T6：連鎖律と BEC・勾配の一致、P の有界性、AC 電場の疑似散逸、NPT の安定性、空隙、外挿）は、再学習の後に依頼元が行います。

## 5.4 依頼元の確認で見つかった事項（GPUMD 側の見立て、未検証）

出典：同上（`bec_rev0` の受け取り確認）と、それへの返信 3（`specs/verification/VER-0035-nep-charge-transfer-trainer/REPLY_TO_REQUESTER_3_ja.md`）。

1. **旧モデルの新旧バイナリの「ビット同一」の主張は、小さな箱の静的計算に限られる**：検証した水 63 原子と BaTiO₃ 40 原子ではビット同一でしたが、依頼元の 248 原子（アモルファス C80F168、15.9 Å の立方セル）では、**旧バイナリ同士**でも ΔE 3.5×10⁻⁵ eV、Δq 9×10⁻⁷、ΔF 9×10⁻⁶ eV/Å の差が出ました（旧対新も同程度）。つまり新旧の差は**実行ごとの雑音の範囲**です。原因の仮説：セルリストの `atomicAdd` の順序（`force/neighbor.cu`）、PPPM の `atomicAdd`。**未検証**です。`bec_rev0` の `NOTES_2026-11-11.md` で記述を訂正済みです。
2. **`nep` 対 `gpumd` の差が、合成データの A4 の数値を超える**：実データの 14 構造（12〜768 原子、三斜晶 6 本）で、BEC 最大 4.6×10⁻⁶〜1.4×10⁻⁵ e、力 最大 1.2×10⁻⁵〜5.5×10⁻⁵ eV/Å、電荷 8.2×10⁻⁷〜4.4×10⁻⁶ e、エネルギー 2.6×10⁻⁹〜2.6×10⁻⁷ eV/原子。原子数が多いほど大きい傾向（N12：1.7×10⁻⁵、N768：5.5×10⁻⁵ eV/Å）。合成データの A4（BEC ≤ 5×10⁻⁶ e、力 ≤ 1×10⁻⁵ eV/Å）より数倍大きいですが、依頼元の許容（BEC の再学習後 RMSE ≲ 0.15 e）より 4 桁小さく、支障はないと判断されています。見立て：float32 の加算順序の雑音が N と |F| に比例して増える（未検証）。切り分けの提案：`gpumd` 同士の差の床を同じ構造で測り、それと比べる。トレーナーの長距離が Ewald、`kspace ewald` が対応する選択、`dump_xyz` の BEC は `scale_bec` 適用後（`nep` 側は未確認と依頼元が記載）、`kspace pppm` での比較は未実施。
3. 依頼元の `charge_transfer` 学習は、60 原子超・三斜晶の構造（最大 768 原子）で動作に問題はありませんでした（合成データの学習側の試験は最大 59 原子だった）。

## 5.5 限界のまとめ

- 実データでの**表現力と学習の収束は未確認**（5a は途中、5b は実行中）。物理的な精度の主張はしません。
- 検証はすべて合成データ。float と double の比較は未実施（カーネルは float）。大きな N（> 100 原子）の `nep` の合成試験なし（依頼元の実データでは動作）。
- `gpumd` での化学種の圧縮（`active`）の `charge_transfer` の `nep.txt` は、φ の部分選択を実装済みだが、実行は未確認。
- 多 GPU、`gpumd_replica` の GPU 検証、HIP は未検証。`charge_mode 3` は replica 非対応。
- 周期系の P そのものの出力は未実装（`thermo` のエネルギーに −E·P は含まれない）。
- 電荷の目標値での学習、非ゼロの総電荷は不可。可変 ZBL と `charge_transfer` の併用は replica で拒否。`vdw`/`charge_vdw`/spin/`fine_tune` とは併用不可。
- 疑似散逸が消えるかどうかは、DR-005 の範囲外（BEC が勾配になっても、散逸の原因が他にある可能性がある。依頼元が T3 で確認）。

## 5.6 再現手順

`PY=/home/katouda/miniforge3/envs/py312-gpumd/bin/python`（`ase calorine numpy pytest hypothesis`）。生成物（`*.out`、ダンプ、`nep.txt`、実行ディレクトリ）は**リポジトリの外のスクラッチ**に作り、`examples/` の中では `gpumd` を実行しません。

```bash
cd src && make CUDA_ARCH="-arch=sm_86" -j4 2>&1 | tail -30

# 参照実装（CPU のみ、約 25 秒）
$PY specs/verification/VER-0029-qnep-charge-transfer-reference/scripts/ct_reference.py

# 生成器（合成の nep.txt と 5 構造）
S=specs/verification/VER-0032-qnep-charge-transfer-gpu/scripts
$PY $S/make_ct_model.py OUTDIR [--seed N] [--rc-radial 4] [--rc-phi 4] [--no-ct]

# BEC（FD との比較）
$PY $S/ct_bec_fd.py src/gpumd /scratch/fd --structure thick_n45_L14 [--max-atoms 12] [--delta 0.02] [--rc-radial 5.5 --rc-phi 4.0]

# 力・ビリアル（VER-0034）、学習側 A4・学習の動作確認（VER-0035）
$PY specs/verification/VER-0035-nep-charge-transfer-trainer/scripts/ct_a4_compare.py src/nep src/gpumd /scratch/a4 [--charge-mode 3] [--rc-phi 3] [--nt 3]
$PY specs/verification/VER-0035-nep-charge-transfer-trainer/scripts/ct_train_smoke.py src/nep src/gpumd /scratch/smoke --generations 100

# pytest（GPU 必要）
cd tests_pytest
$PY -m pytest -q test_qnep_charge_transfer_parse.py test_qnep_charge_transfer_bec.py \
   test_qnep_charge_transfer_forces.py test_qnep_charge_transfer_field.py test_nep_charge_transfer_train.py
# 3b（未実行。GPU 検証は再学習の終了後）
$PY -m pytest -q test_replica_charge_transfer.py
```

実行前に `nvidia-smi` で GPU の使用状況を確認してください（GPU は共有です）。

## 5.7 関連ファイル

| 内容 | 場所 |
|---|---|
| 公式マニュアルページ（英語） | `doc/nep/input_parameters/charge_transfer.rst`、`doc/theory/qnep_charge_transfer.rst`、`doc/nep/input_parameters/index.rst`、`doc/gpumd/input_parameters/{add_efield,compute_dpdt,dump_xyz}.rst`（BEC の注記） |
| 要求仕様・設計・決定 | `specs/requirements/DR-005-qnep-bec-gradient-consistency.md`、`specs/design/DES-0005-qnep-charge-conserving-bec.md`、`specs/decisions/ADR-0006-qnep-charge-transfer.md` |
| 検証記録 | `specs/verification/VER-0028`、`VER-0029`、`VER-0032`〜`VER-0036` |
| 依頼元への報告書 | `specs/verification/VER-0028-qnep-bec-gradient/REPORT_TO_REQUESTER_ja.md`、`REPLY_TO_REQUESTER_2_ja.md`、`VER-0035-.../REPLY_TO_REQUESTER_3_ja.md` |
| 配布バイナリ | `/home/katouda/bin/GPUMD/bec_rev0/`（`README.md`、`NOTES_2026-11-11.md`、`SHA256SUMS`。`93d203d4` から。変更しない。修正版は `bec_rev1` へ） |
| 追跡 | `specs/traceability.yaml`（DR-005：`partial`） |
| 関連する別機能 | `docs_ja/dftd/`（D2/D3 分散力補正の日本語マニュアル） |
