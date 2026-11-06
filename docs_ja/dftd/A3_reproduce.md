# 付録 A3 検証・測定の再現手順、用語集、パラメータの出典、変更履歴

本付録は、他の章で引用した検証・測定を自分で再現する手順と、用語、パラメータの出典、変更履歴をまとめます。パスはリポジトリのルート（`GPUMD/`）からの相対パスです。**本付録の手順は、記録（VER-xxxx）のヘッダや README、スクリプトの冒頭コメントを読んで整理したもので、作成者が本付録のために実行して確認したものではありません**（ただし、A1, A2 の数値検証スクリプト、`check_d3zero_derivation.py` と `check_d2_qe_plain.py` を含め、`docs_ja/dftd/scripts/` の CPU スクリプトだけは実行して結果を確認しています）。ソースファイルのパスは `develop` `aba6b28b` 時点（upstream の再構成でサブフォルダに移動する前）です。

## A3.1 環境とビルド

| 項目 | 内容 |
|---|---|
| ビルド | `cd src && make CUDA_ARCH="-arch=sm_86" -j4`（既定の sm_60 は CUDA 13 で失敗するので、GPU に合わせて指定。引数やフラグを変えたら `make clean`）[AGENTS.md] |
| 生成物 | `src/gpumd`, `src/nep` ほか。`.gitignore` 対象 |
| Python 環境 | `ase calorine numpy pytest hypothesis` が必要。開発機では conda 環境 `/home/katouda/miniforge3/envs/py312-gpumd` を使い、`/home/katouda/miniforge3/envs/py312-gpumd/bin/python -m pytest ...` で実行 [AGENTS.md] |
| GPU | 検証記録の環境は RTX A6000（sm_86）、CUDA 12.3。GPU を使うビルドやテストは一度に一つだけ動かす |
| NEP モデル | D3/D2 の試験は `potentials/nep/nep89_20250409/nep89_20250409.txt`（NEP89）を使う [`test_dftd3_reference.py` の前提] |

実行結果の出力（`*.out`、ダンプ、`nep.txt`、回帰テストの作業ディレクトリ）は、ソースツリーの外のスクラッチディレクトリに置いてください。`examples/` の中で直接 `gpumd` を走らせると、追跡されているファイルが書き換わります（01 章 1.7 節）。

## A3.2 pytest（`tests_pytest/`）

`tests_pytest/` で実行します。`pytest -m "not slow" -q` が既定の短いループです。D3/D2 に関するファイルは次のとおりです [コード読解、`tests_pytest/` の各ファイル冒頭]。

| ファイル | 内容 | 備考 |
|---|---|---|
| `test_dftd3_reference.py` | D3 の GPU 結果を simple-dftd3 の参照値（`fixtures/dftd3/reference.json`、7 ケース）と比較。`num_sub`（スレッド数）の全値での一致と再現性、近傍リストの正確性など | マーカー `fast`。NEP89 モデルが必要。許容値の規則は 06 章 6.2 節 |
| `test_dftd2_reference.py` | 内蔵表と dftd3-lib の一致、NumPy 参照実装と dftd3-lib の比較（打ち切りなしの極限）、有限差分による自己無矛盾性、不変性、カットオフでの連続性、GPU と NumPy 参照の比較、入力エラー、ビット単位の再現性 | 参照実装は `tests_pytest/dftd2_reference.py`（ASE 近傍リスト）。表の検査の一部は dftd3-lib のソースを必要とし、`prg/extras.f90` のパスを環境変数 `DFTD3_LIB_EXTRAS_F90` で指定する（`test_dftd2_reference.py:44-45`。既定値は開発機の `dftd3-lib/prg/extras.f90`） |
| `test_dftd2_consistency.py` | GPU の力と $-\partial E/\partial x$（5 点中心差分、刻み 0.01 Å）、ビリアルとひずみ微分の比較 | 環境変数 `DFTD2_TEST_GPUMD` で `gpumd` 実行ファイルを差し替える |
| `test_dftd2_src_table.py` | `src/force/dftd2para.cuh` が生成スクリプトの出力とバイト単位で同一であること | |
| `test_dftd3_zero.py` | D3(0)（DR-003）。(1) simple-dftd3（zero damping、二体項）の参照値との比較（`fixtures/dftd3_zero/reference.json`、PBE の全構造 + 重元素のセル、revPBE・BLYP・B3LYP・TPSS は 3 構造）。(2) 解析とエラー（ADR-0003、DR-003 D1・D4・D5）。(3) BJ の互換性：`dftd3 pbe 12 6` と `... bj` が互いに、および変更前のバイナリの凍結出力（`fixtures/dftd3_zero/bj_frozen.json`）とビット同一（全ゼロの NEP を使用）。有限差分、小箱／大箱の一致、微小な r など | 85 件（単独で 649 s、VER-0024 2 節）。`DFTD3_ZERO_TEST_GPUMD` で `gpumd` を差し替え可能 |
| `test_dftd2_qe.py` | `dftd2 … table qe` / `trunc plain`（DR-004、ADR-0004）。依頼元の参照（`fixtures/dftd2_qe/`）、`dftd2_qe_oracle.py`（QE 7.2 の表から作る NumPy 参照）との比較、既定の形（`grimme2006/sf`）の凍結出力、解析とエラー、有限差分、複製、R3 の凍結出力との許容値比較（`test_default_form_matches_pre_r3_frozen_output_within_ac1_caps`） | 139 件（VER-0026 5 節） |
| `test_dftd2_qe_extended.py` | DR-004 のレビュー後の追加試験：3741 個の C6・r0ab と CSV・QE 7.2 のソース（別方式の再解析）との照合、混合元素（H C N O Si Cl Rn）、境界（ちょうど rc のペアは除外）、最悪ケースのペア（Rn-Rn、H-H） | 14 件。QE 7.2 のソースは環境変数 `DFTD2_QE72_SRC`（`pars.f90` と `core.f90` のあるディレクトリ）か、`/home/katouda/build/q-e` のタグ `qe-7.2` から読む。読めない環境ではその部分は実行されない |

実行例（開発機の環境）：

```
cd tests_pytest
/home/katouda/miniforge3/envs/py312-gpumd/bin/python -m pytest test_dftd3_reference.py -q
/home/katouda/miniforge3/envs/py312-gpumd/bin/python -m pytest test_dftd2_reference.py test_dftd2_consistency.py test_dftd2_src_table.py -q
/home/katouda/miniforge3/envs/py312-gpumd/bin/python -m pytest test_dftd3_zero.py test_dftd2_qe.py test_dftd2_qe_extended.py -q   # DR-003, DR-004
```

**新しい参照データ（`tests_pytest/fixtures/`）**

| ディレクトリ | 内容と出典 |
|---|---|
| `dftd3_zero/` | `reference.json`（simple-dftd3 の `ZeroDampingParam`、二体項。生成は `generate_reference.py`）、`bj_frozen.json`（変更前のバイナリ `86899ba2` の凍結出力、`generate_bj_frozen.py`）、`heavy_cluster.xyz`、`tiny24.xyz`、全ゼロの NEP（`make_zero_nep.py`、`zero_nep_H_C_O_Cu.txt`）、`README.md`（出典） |
| `dftd2_qe/` | 依頼元の `d2_reference.py`（未改変のコピー。sha256 は README に記録され、テストが確認）、PTFE の 4 構造と 16 個の参照 `ref_cryst157_*_rc12.xyz`（依頼元が作成。**再生成しない**）、`summary_rc12.csv`、VER-0025 の CSV（`qe72_c6_first_point_Eh_bohr6_Z86.csv`、`qe72_r0ab_angstrom_Z86.csv`）、`default_frozen.json`（R3 バイナリの実行ごとのビット比較の基準）と `default_frozen_pre_r3.json`（R3 前のバイナリ `af269060` の出力。許容値比較の基準）、`generate_default_frozen.py`、`README.md`（出典） |

参照値は、テストを通すために再生成しません。`--update-golden` は人の承認と理由がある場合のみです。R3 の凍結出力の再生成は ADR-0005 項目 5 に基づく**一度きりの承認**で、前例にはなりません（06 章 6.11.4 節）[AGENTS.md]。

テスト専用の環境変数（`GPUMD_DFTD3_*`、`GPUMD_DFTD2_TARGET_THREADS`）はテストが自動で設定します。手で設定する必要はありません（01 章 1.4 節）。参照データを作り直す `--update-golden` は、人の承認がある場合のみ使います。`fixtures/dftd3/reference.json` は `fixtures/dftd3/generate_reference.py`（simple-dftd3 1.7.0 の `dftd3-python`、PBE、BJ、二体項のみ、倍精度）で、`fixtures/dftd2/` は同ディレクトリの `generate_params.py` と `generate_oracle_reference.py` で生成されています。

## A3.3 回帰テスト（`tests_regression/`）

D3/D2 に関するケースは `tests_regression/manifest.json` にあります。`dftd3_single_late`、`dftd3_kspace_one_each`、`dftd3_duplicate`、`dftd2_basic`、`dftd2_dftd3_exclusive` など [コード読解]。**新しいキーワード形式（`zero`、`table`、`trunc`）のケースはマニフェストにありません**（ランナーが基準側と候補側に同じ期待値を適用するため、人間の判断待ち。VER-0024 6 節、VER-0026 6 節）。

```
python3 tests_regression/run_regression.py --repo-root <repo> --baseline <master ビルドの gpumd> \
        --candidate <開発版の gpumd> --suite full --case 'dftd3_*'
```

- 受入基準は `--suite full` です。`--case` で絞った実行は診断用です。
- ベースラインは upstream `master`（`3b2c8db8`）をビルドしたバイナリです [VER-0004, VER-0010]。
- `--keep-all` を付けると `tests_regression/.work/` と `tests_regression/reports/` が残るので、終わったら削除します（いずれも git 管理外）。
- 同一ソースの `--suite full` でも、GPU 上の非決定性により約 37 件が失敗することが測定されています（`nvt_bdp` など、AGENTS.md の既知リスト）。リスト外のケースの失敗、またはリスト内のケースの数値差が小さくない失敗だけが疑わしいものです [AGENTS.md]。
- DR-003・DR-004 後の `--suite full`（基準 `master` `3b2c8db8`）は 216 passed、41 failed、関係テスト 25/25 でした。41 件は、既知の 37 件（上のリスト）+ `dftd2_basic`・`dftd2_dftd3_exclusive`（**`master` に `dftd2` が無いので基準側が失敗する**。`master` 基準の `dftd2` ケースは評価できない）+ `dftd3_single_late`・`dftd3_kspace_one_each`（develop と master の差。変更前のバイナリ `86899ba2` を候補にしても同じ値で失敗。原因未調査）です [VER-0024 6 節、VER-0026 2・5 節]。したがって `dftd2` の回帰は、同じバイナリを基準と候補にした実行と、凍結出力のテスト（A3.2 節）で確認します。
- 回帰の一致は物理的な正しさを意味しません（06 章 6.1 節）。

## A3.4 検証記録のスクリプト（`specs/verification/VER-xxxx/scripts/`）

各記録の README に目的、条件、結果が書かれています。多くのスクリプトは、作業用ディレクトリのパスやバイナリのパスが記録の環境に合わせてあるので、**使う前に冒頭のコメントと引数を確認し、パスを調整してください**（VER-0004 の README にも同趣旨の注記）。

| 記録 | スクリプト | 用途（README 冒頭より） | 呼び出しの形（冒頭の usage より） |
|---|---|---|---|
| VER-0001 | `accuracy.py`, `build_systems.py`, `slope.py`, `timing.py` | D3 の精度（simple-dftd3 比）、構造の作成、スロープ法のタイミング | 記録参照 |
| VER-0003 | `d2ref.py`, `pbc_cmp.py`, `oracle_cellgradient_check.py` | D2 の NumPy 参照実装と dftd3-lib の D2 モード（オラクル）の比較 | 記録参照（dftd3-lib をスクラッチで `make dftd3`） |
| VER-0004 | `bench.py`, `alt.sh`, `run.sh`, `harness.cu` | D3 の最適化の性能比較（nsys のカーネル時間）、近傍リストのハーネス | `run.sh tag binary xyz...` |
| VER-0005 | `make_structures.py`, `cutoff_static.py`, `analyze.py`, `cost.py`, `nve.py` | D2 のカットオフ収束、コスト、NVE | 記録参照 |
| VER-0006, VER-0010 | `fd_check.py`, `nve.py`, `timing.py` | D2 の有限差分整合性（力・ビリアル）、NVE、スロープ法 | `fd_check.py GPUMD_BIN NEP_TXT FIXTURE_DIR WORKDIR [s6] [rc]`、`timing.py GPUMD_BIN NEP_TXT WATER27_XYZ WORKDIR` |
| VER-0008, VER-0009 | `timing.py`, `fit.py`, `harness.cu` ほか | D2 の最適化（スレッド数・セル数の選択） | 記録参照 |
| VER-0011 | `nve.py`, `summarize_nve.py`, `pooled_stats.py`, `analyze_ac3.py`, `timing.py`, `cn_scan.py` | D3 の NVE エネルギー保存、コスト測定、極端な配位数の走査 | 記録参照 |
| VER-0014, VER-0015 | `accuracy.py`, `compare_ac2.py`, `run_s2.py` ほか | D3 の最適化後の精度・回帰比較 | 記録参照 |
| VER-0020 | `nve.py`, `stats.py`, `summarize_nve.py`, `pooled_stats.py` | D3 の最終 NVE 反復 | `nve.py CAND_BIN BASE_BIN NEP_TXT WATER27_XYZ WORKDIR SYSTEM NSTEPS_NVE [SEEDS...]`。環境変数 `D3=0`（D3 なしの対照）、`OUTTAG`（出力名の接尾辞） |
| VER-0023 | `gen_d3zero_tables.py`, `crosscheck_d3zero.py`, `probe.f90`（`data/` に生成した表と結果） | D3(0) の r0ab と汎関数パラメータを dftd3-lib の `core.f90` から生成し、dftd3-lib の実行時の値と simple-dftd3 で照合 | `gen_d3zero_tables.py <dftd3-lib dir> <out dir>`、`crosscheck_d3zero.py <probe binary> <data dir> [results]`（simple-dftd3 の Python バインディングが必要。dftd3-lib をスクラッチへコピーして `make`、`lib/api.f90` の 59 行目の `private` を外した**スクラッチのコピー**で `probe` をビルド。VER-0023 3 節） |
| VER-0024 | （リポジトリに保存されていない） | D3(0) の Phase 4（精度、性能、NVE、サニタイザ） | 「スクリプトと生の出力はエージェントのスクラッチに置かれ、リポジトリには残っていない」（VER-0024 末尾）。条件は記録の本文から再構成する |
| VER-0025 | `extract_qe_d2_params.py`, `compare_qe_gpumd.py`, `stage2_pairs_constants.py`, `compare_xyz_reference.py`, `stage3_pw_vs_numpy.py`, `d2_reference.py`（依頼元のコピー） | QE 7.2 の `pars.f90`・`core.f90` から C6（最初の点）と r0ab を抽出し、GPUMD の表と照合（Stage 1）、ペアの値と定数（Stage 2）、pw.x との比較（Stage 3） | `extract_qe_d2_params.py <dft-d3 source dir> <outdir> [label]`（ソースは `git show qe-7.2:dft-d3/<file>`）、`compare_qe_gpumd.py <csv dir> <repo root> <out dir>`、`stage2_pairs_constants.py <csv dir> [out report]`、`compare_xyz_reference.py A.xyz B.xyz`。`stage3_pw_vs_numpy.py` は QE 7.6-dev の pw.x と MPI 環境が必要（README に手順）。**QE 7.2 のバイナリは使っていない** |
| VER-0026 | （スクリプトは保存されていない。結果は記録とテストに） | `table qe`/`trunc plain` の精度・NVE、R3 の時間測定と精度 | 条件は記録の本文（slope 法、nsys など）から再構成する |


テーブルの生成：`tools/Miscellaneous/dftd2/generate_dftd2qe_table.py <QE dft-d3 source dir> <output .cuh>`（`pars.f90` と `core.f90` を置いたディレクトリを指定し、`src/force/dftd2qe_table.cuh` を生成）。生成物は VER-0025 の CSV と、テスト `test_dftd2_qe_extended.py` により照合されます。

スロープ法とは、100 ステップと 500 ステップの実行時間の差から 1 ステップあたりの時間を求める方法で、D3 の項の追加時間は「補正あり − 補正なし」の差で見ます [VER-0001, VER-0011]。ベンチマークの結果と読み方は 03 章を参照してください。

## A3.5 本マニュアルの数値検証スクリプト（`docs_ja/dftd/scripts/`）

CPU のみ（Python、NumPy）、数秒で終わります。いずれも `python -I <script>` で実行でき（開発機では `/home/katouda/miniforge3/envs/py312-gpumd/bin/python`）、GPU もビルドも不要です。

| スクリプト | 内容 | 本文での参照 |
|---|---|---|
| `check_d3_derivation.py` | D3 の力・ビリアル（直接項、間接項）、$\partial C_6/\partial\mathrm{CN}$、重みの分解、$k_{12}$ を有限差分と比較。**参照 C6 は合成した値** | A1.7 節 |
| `check_d2_derivation.py` | D2 のシフト付き力の力・ビリアルを有限差分と比較。C, H, O の実際のパラメータを使用 | A2.7 節 |
| `check_d2_tail.py` | シフト付き力の連続体近似 $7\pi\rho c/(3r_c^3)$ を数値積分で確認（SymPy は未インストールのため NumPy） | A2.6 節 |
| `check_d3zero_derivation.py` | D3(0) の $g_n$ の微分、力・ビリアル（直接項 + 配位数を通る項）を有限差分と比較。float32 の桁あふれ対策の確認。**参照 C6 と R0 は合成した値**。`check_d3_derivation.py` を読み込む。終了コードで PASS/FAIL | A1.9.5 節 |
| `check_d2_qe_plain.py` | `src/force/dftd2qe_table.cuh` から C-C、C-F、F-F の QE 型パラメータを読み、単純打ち切りの段差（7.4×10⁻⁶ eV、3.7×10⁻⁶ eV/Å）、$\phi'$ の有限差分、補償付き float 累算のエミュレーションを確認 | A2.9.4 節 |

## A3.6 用語集

| 用語 | 意味 |
|---|---|
| D2, D3 | Grimme の分散力補正。D2 は元素固定の C6、D3 は配位数に依存する C6 |
| BJ ダンピング（rational damping） | $r^n + (a_1R_0 + a_2)^n$ を分母にした、短距離で有限になる減衰。`dftd3` の既定のダンピング（`zero` も選べる） |
| CN（配位数、coordination number） | 原子の周りの近傍数を、距離の滑らかな関数で数えたもの（2.3 節） |
| C6, C8 | ペアの分散係数（距離の $-6$ 乗、$-8$ 乗）。D3 では C8 は C6 と `r2r4` から決まる |
| `rc_pot`, `rc_cn` | `dftd3` の第 2, 第 3 引数。ポテンシャルと配位数のカットオフ（コード内部の `rc_radial`, `rc_angular`） |
| 小箱経路 / 大箱経路 | D3 の二つの実装経路（セルの厚さとカットオフの関係で切り替え。04・05 章） |
| `extra`（カーネル） | $C_6$ の配位数依存性から生じる力（間接項）を加えるカーネル（2.5 節） |
| shifted-force（シフト付き力） | エネルギーと力の両方を打ち切り位置で 0 にそろえる補正（D2、2.6 節） |
| ゼロダンピング（D3(0)、`zero`） | $f_n(r) = 1/(1 + 6\,(r/(s_{r,n}R_0))^{-\alpha_n})$ を $1/r^n$ に掛ける減衰。短距離（$r \to 0$）で減衰関数が 0 に向かう（"zero"）。`dftd3 ... zero`（2.2 節、A1.9 節） |
| `r0ab` | D3 のペアごとの基準半径 $R_0^{ij}$（Å の表、Z ≤ 94 まで定義、実装は Z ≤ 86）。ゼロダンピングの $R_0$ と、`table qe` の $R_0 = 1.1\,r_0$ に使う |
| `table qe` / `trunc plain` | `dftd2` のオプション。QE 7.2 の `dftd3_version = 2` と同じパラメータ（D2 の式に D3 の表）、シフトなしのハードカットオフ（A2.8 節） |
| TwoSum（補償付き和） | 浮動小数点の和の丸め誤差を、誤差なしに別の変数へ取り出す演算。誤差項を足し上げて、和の精度を上げる（A2.9 節、`CompensatedSum`） |
| フラッシュ（flush） | float の部分和（補償項を含む）を double の累算へ足し込み、部分和をゼロに戻すこと。D2 では 4 近傍セルごと（`FLUSH_CELLS = 4`） |
| ADR | Architecture Decision Record（設計判断の記録、`specs/decisions/`）。ADR-0003（`dftd3` のダンピング指定）、ADR-0004（`dftd2` の `table`/`trunc`）、ADR-0005（`dftd2` の高速化の数値方針） |
| `num_sub` | 1 原子を何スレッドで処理するか（並列化の細かさ。05 章） |
| 検証（verification）/ 回帰（regression）/ 妥当性確認（validation） | 00 章 0.6 節、06 章 6.1 節 |
| `master` ベースライン | upstream の `master`（`3b2c8db8`）をビルドしたバイナリ |
| AC-n | 要求仕様 DR-001〜DR-004 の受入基準（Acceptance Criterion）n |
| UPSTREAM-xxxx | upstream コードで見つかった課題の記録（`specs/upstream-issues/`） |

## A3.7 パラメータの出典

| パラメータ | 出典 | 状態 |
|---|---|---|
| D2 の C6, R0（H〜Rn、$Z\le86$） | dftd3-lib（コミット `3c99ee9`、Grimme の DFTD3 V3.1 Rev 1 の再配布）の `prg/extras.f90` の `loadoldpar`。`tests_pytest/fixtures/dftd2/generate_params.py` が `dftd2_params.csv` と `src/force/dftd2para.cuh` を生成 [VER-0003 1 節] | H〜Xe の 54 元素は Turbomole マニュアルの表 6.3（Grimme 2006 の再掲）と**不一致 0 件**で照合済み（C6 は完全一致、R0 は 6e-4 Å 以内）。Cs〜Rn（$Z=55$〜86）は dftd3-lib のみで、グループ値（例：Ce〜Lu で C6 = 140.68、$r_0$ = 1.594）の穴埋めを含み、**他の出典との照合は無い** [VER-0003 2 節] |
| D3(0) の `r0ab` と汎関数パラメータ（$s_6, s_8, s_{r,6}, s_{r,8}$、def2-QZVP 版） | dftd3-lib（コミット `3c99ee981266548a37588abe3a6f5cb56092d51a`、DFTD3 V3.1 Rev 1 の再配布）の `lib/core.f90`（`setr0ab` と `setfuncpar` の `version.eq.3`）。`specs/verification/VER-0023-dftd3-zero-parameters/scripts/gen_d3zero_tables.py` が生成し（手入力なし）、`src/force/dftd3zero_table.cuh` に反映 [VER-0023 1・3 節] | Z ≤ 86 と全 48 汎関数で、独立なオラクル simple-dftd3 1.7.0（Python バインディング）と**不一致 0**。論文（Grimme et al. 2010）は**読んでいない**。Z = 87〜94 は dftd3-lib と simple-dftd3 で不一致（最大 29 %）のため未採用（エラー）[VER-0023 4・5 節] |
| `table qe` の C6（最初の参照点）と `r0ab` | Quantum ESPRESSO 7.2（タグ `qe-7.2`、コミット `b4c5e8deeb7d1863e7553a87b45f8d22c21926a5`）の `dft-d3/pars.f90` と `dft-d3/core.f90`。`tools/Miscellaneous/dftd2/generate_dftd2qe_table.py` が `src/force/dftd2qe_table.cuh`（倍精度、3741 ペア、Z ≤ 86）を生成。定数は QE 7.2 の CODATA 2018（`constants.f90`）[VER-0025, ADR-0004 項目 3] | GPUMD の既存の表（`c6_ref_cpu`、`d3zero_r0ab`）と 3741 ペアで不一致 0。QE 7.2 と開発版 HEAD（7.6-515）で、C6 は同一、`r0ab` の 10 進値も同一（`real(4)` と `_wp` の違いのみ）。QE 7.2 のバイナリは**実行していない**（7.6-dev で CF4 を実行し相対 4.6×10⁻⁸ で一致）[VER-0025 4・7 節] |
| D2 の $s_6$ | ユーザーが指定。汎関数ごとの値は文献を参照（PBE は 0.75） | コードに表は無い |
| D3 の汎関数パラメータ（$s_6, s_8, a_1, a_2$）と表（`c6_ref_cpu`, `cn_ref`, `num_cn`, `r2r4`, `covalent_radius`） | `src/force/dftd3/dftd3.cu` の `set_para` と `dftd3para.cuh` に値のみ。**コード中に出典の記載が無い** | 出典は不明。`dftd3.rst` は Bonn 大学のページの汎関数リストを参照しているが、コード中は 54 の名前で、リスト（未照合）と一致するかは確認していない。PBE の値は simple-dftd3 の参照値と 6.2 節の精度で一致 [VER-0001] |
| D3 の定数 `HartreeBohr6` など | コード中に出典なし（2.7 節、UPSTREAM-0002 B） | CODATA 2018 から求めた値に対し相対 +8.7e-7 |
| 参照実装 | simple-dftd3 1.7.0（conda-forge の `dftd3-python`、BJ と zero の精度の参照）、dftd3-lib（D2 のオラクル、D3(0) の表の出典）、依頼元の `d2_reference.py` と `dftd2_qe_oracle.py`（`table qe`） | BJ の参照は同一のパラメータを使うので、パラメータ表の独立な検証にはならない（D2 は VER-0003 の表照合で別途確認）。D3(0) の表は simple-dftd3 と独立に照合（VER-0023）。`table qe` は QE ソースからの生成と照合（VER-0025） |
| 理論文献 | D3: Grimme et al., J. Chem. Phys. 132, 154104 (2010)、BJ: Grimme, Ehrlich, Goerigk, J. Comput. Chem. 32, 1456 (2011)（`dftd3.rst` の [Grimme2010], [Grimme2011]、`dftd3.cu` のヘッダ）。D2: Grimme, J. Comput. Chem. 27, 1787 (2006)（`dftd2.cu` のヘッダ）。D3(0) の式：Grimme et al., J. Chem. Phys. 132, 154104 (2010)（DR-003 の記述）。QE の D2 形式（`dftd3_version = 2`）は文献ではなく QE 7.2 のソースが出典 | 論文は DR-003・VER-0023 で読まれていない |

## A3.8 変更履歴

### 本マニュアルに関する版情報

| 日付 | 内容 |
|---|---|
| 2026-11-04 | 初版（README の版情報を参照） |
| 2026-11-07 | DR-003（D3(0)）と DR-004（`dftd2` の QE 互換テーブル、単純打ち切り、高速化）を 00・01・02・03・06 章と付録 A1〜A3 に反映（`develop` `aba6b28b` 時点）。04・05 章は upstream の再構成後の構成に更新済み。01・02・04・05・A1・A2 の行番号引用は `7c15e99c` で再導出済み |

### 実装の履歴（`git log` の抜粋、`develop` ブランチ）

| 日付 | コミット | 内容 |
|---|---|---|
| 2024-07〜2026-02 | upstream の履歴 | D3 の追加、汎関数の追加、`dftd3 faster`（2026-02-08、`204a6cde`）など |
| 2026-11-03 | `84e5fa83` | D3 の高速化：大箱のスレッド並列化、小箱の近傍リスト（DR-001 Phase 3, 4、VER-0004） |
| 2026-11-03 | `cc5129d2` ほか | D2 の実装（DR-002、Phase 3-4）。パラメータ表と NumPy 参照は `2d0f40f4`（Phase 0-1） |
| 2026-11-03 | `27ca8d72` | D2 の既定カットオフを 30 Å に（VER-0005） |
| 2026-11-03 | `1beb905b` | D2 のレビュー修正（決定的なセル順序、`float` カットオフ、200 Å 超の警告） |
| 2026-11-03 | `1476643a`, `a012b76c` | D2 の最適化（VER-0008, VER-0009） |
| 2026-11-03 | `22467397` | D2 の最終検証（VER-0010） |
| 2026-11-03/04 | `ebf85206`, `e0a6918d` | D3 の NVE エネルギー保存の比較（VER-0011）、AC-3 は区別不能として終了 |
| 2026-11-04 | `db2bf407` | 小箱の 1 原子あたり複数スレッド化（VER-0013） |
| 2026-11-04 | `0ddcd1f7` | C6 の重みの因数分解（VER-0014、本書 A1.5 節） |
| 2026-11-04 | `7e37f67b` | `extra` カーネルと CN 近傍走査の共有（VER-0015） |
| 2026-11-04 | `0ac42758` | `float` のプリフィルタは棄却（VER-0016） |
| 2026-11-04 | `b5301c0c`, `bafbb6b9`, `0c353281`, `251271b0` | UPSTREAM-0001 の記録、VER-0017〜0020。D3 コードの最終コミットは `0c353281` |
| 2026-11-04 | `8fe8ab99` | UPSTREAM-0002 の記録 |

### DR-003・DR-004 の履歴（`git log 8fe8ab99..develop` の抜粋、`develop` ブランチ、日付はコミット日）

| 日付 | コミット | 内容 |
|---|---|---|
| 2026-11-06 | `e4f396e9` | DR-003：仕様（DR-003、DES-0003、ADR-0003）、VER-0023（パラメータの出典） |
| 2026-11-06 | `337a73c9` | DR-003：ゼロダンピングの実装とテスト（途中。PBE のビリアル 2 ケースが暫定許容値超過） |
| 2026-11-06 | `c200f20c` | DR-003：`zero` のとき小箱の配位数を double で計算（BJ の経路は不変） |
| 2026-11-06 | `26ef3c5c` | DR-003：VER-0024（Phase 4 の検証記録） |
| 2026-11-06 | `5efef9cb` | DR-003：レビュー修正（マニュアル、zero の例、解析メッセージ、r = 0 のガード） |
| 2026-11-06 | `38ac91f0`, `c909ff6b` | DR-003：BJ の出力を従来どおりにし（ADR-0003 項目 4 の改訂）、AC-5 の限界を記録 |
| 2026-11-06 | `6931004e` | `wt-d3-zero` を `develop` にマージ（DR-003） |
| 2026-11-06 | `cd10a599` | DR-003：traceability を「実装済み」（AC-5 は承認済みの例外）に |
| 2026-11-06 | `1296a145`, `c758bf56`, `830c08c8` | DR-004：仕様（DR-004、DES-0004）、ADR-0004 の起草と承認 |
| 2026-11-06 | `af269060` | DR-004：VER-0025（QE 7.2 の D2 テーブルの出典）、ADR-0004 の定数とテーブルの決定。**R1/R2 のビット一致の基準バイナリ** |
| 2026-11-06 | `83e020dd` | DR-004：テスト先行（参照、既定のビット一致、解析、有限差分、複製） |
| 2026-11-06 | `3704128d`, `040dc8ff`, `c3257df2` | DR-004：ADR-0004 項目 2・7 の改訂、`table qe` / `trunc plain` の実装、レビュー修正（整形、`static_assert`、生成スクリプトを `tools/Miscellaneous/dftd2/` へ移動、拡張テスト、VER-0025 の訂正） |
| 2026-11-06 | `6e5d0fc2`, `44ae8461` | DR-004：ADR-0005、VER-0026、`wt-d2-qe` のマージ（R1/R2） |
| 2026-11-07 | `e1036a4e` | DR-004 R3：`dftd2` のペアカーネルを補償付き float 累算（4 セルごとに double へフラッシュ）に。凍結出力は ADR-0005 に基づき一度だけ再生成 |
| 2026-11-07 | `9068fa61`, `ee6e24af` | docs_ja の D2 累算の記述更新、VER-0026 の R3 の記録、traceability を「実装済み」に |
| 2026-11-07 | `aba6b28b` | `wt-d2-qe` のマージ（R3）。本改訂の対象 HEAD |

上の履歴は `git log -- src/force/dftd3/dftd3.cu src/force/dftd2.cu src/force/dftd2para.cuh specs/verification` の件名から抜粋したもので、コミットの内容は未再確認です。詳細は各 VER 記録と `specs/traceability.yaml` を参照してください。
