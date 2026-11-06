# 付録 A3 検証・測定の再現手順、用語集、パラメータの出典、変更履歴

本付録は、他の章で引用した検証・測定を自分で再現する手順と、用語、パラメータの出典、変更履歴をまとめます。パスはリポジトリのルート（`GPUMD/`）からの相対パスです。**本付録の手順は、記録（VER-xxxx）のヘッダや README、スクリプトの冒頭コメントを読んで整理したもので、作成者が本付録のために実行して確認したものではありません**（ただし、A1, A2 の数値検証スクリプトだけは実行して結果を確認しています）。

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

実行例（開発機の環境）：

```
cd tests_pytest
/home/katouda/miniforge3/envs/py312-gpumd/bin/python -m pytest test_dftd3_reference.py -q
/home/katouda/miniforge3/envs/py312-gpumd/bin/python -m pytest test_dftd2_reference.py test_dftd2_consistency.py test_dftd2_src_table.py -q
```

テスト専用の環境変数（`GPUMD_DFTD3_*`、`GPUMD_DFTD2_TARGET_THREADS`）はテストが自動で設定します。手で設定する必要はありません（01 章 1.4 節）。参照データを作り直す `--update-golden` は、人の承認がある場合のみ使います。`fixtures/dftd3/reference.json` は `fixtures/dftd3/generate_reference.py`（simple-dftd3 1.7.0 の `dftd3-python`、PBE、BJ、二体項のみ、倍精度）で、`fixtures/dftd2/` は同ディレクトリの `generate_params.py` と `generate_oracle_reference.py` で生成されています。

## A3.3 回帰テスト（`tests_regression/`）

D3/D2 に関するケースは `tests_regression/manifest.json` にあります。`dftd3_single_late`、`dftd3_kspace_one_each`、`dftd3_duplicate`、`dftd2_basic`、`dftd2_dftd3_exclusive` など [コード読解]。

```
python3 tests_regression/run_regression.py --repo-root <repo> --baseline <master ビルドの gpumd> \
        --candidate <開発版の gpumd> --suite full --case 'dftd3_*'
```

- 受入基準は `--suite full` です。`--case` で絞った実行は診断用です。
- ベースラインは upstream `master`（`3b2c8db8`）をビルドしたバイナリです [VER-0004, VER-0010]。
- `--keep-all` を付けると `tests_regression/.work/` と `tests_regression/reports/` が残るので、終わったら削除します（いずれも git 管理外）。
- 同一ソースの `--suite full` でも、GPU 上の非決定性により約 37 件が失敗することが測定されています（`nvt_bdp` など、AGENTS.md の既知リスト）。リスト外のケースの失敗、またはリスト内のケースの数値差が小さくない失敗だけが疑わしいものです [AGENTS.md]。
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

スロープ法とは、100 ステップと 500 ステップの実行時間の差から 1 ステップあたりの時間を求める方法で、D3 の項の追加時間は「補正あり − 補正なし」の差で見ます [VER-0001, VER-0011]。ベンチマークの結果と読み方は 03 章を参照してください。

## A3.5 本マニュアルの数値検証スクリプト（`docs_ja/dftd/scripts/`）

CPU のみ（Python、NumPy）、数秒で終わります。いずれも `python <script>` で実行でき（開発機では `/home/katouda/miniforge3/envs/py312-gpumd/bin/python`）、GPU もビルドも不要です。

| スクリプト | 内容 | 本文での参照 |
|---|---|---|
| `check_d3_derivation.py` | D3 の力・ビリアル（直接項、間接項）、$\partial C_6/\partial\mathrm{CN}$、重みの分解、$k_{12}$ を有限差分と比較。**参照 C6 は合成した値** | A1.7 節 |
| `check_d2_derivation.py` | D2 のシフト付き力の力・ビリアルを有限差分と比較。C, H, O の実際のパラメータを使用 | A2.7 節 |
| `check_d2_tail.py` | シフト付き力の連続体近似 $7\pi\rho c/(3r_c^3)$ を数値積分で確認（SymPy は未インストールのため NumPy） | A2.6 節 |

## A3.6 用語集

| 用語 | 意味 |
|---|---|
| D2, D3 | Grimme の分散力補正。D2 は元素固定の C6、D3 は配位数に依存する C6 |
| BJ ダンピング（rational damping） | $r^n + (a_1R_0 + a_2)^n$ を分母にした、短距離で有限になる減衰。D3 の `dftd3` が実装する唯一のダンピング |
| CN（配位数、coordination number） | 原子の周りの近傍数を、距離の滑らかな関数で数えたもの（2.3 節） |
| C6, C8 | ペアの分散係数（距離の $-6$ 乗、$-8$ 乗）。D3 では C8 は C6 と `r2r4` から決まる |
| `rc_pot`, `rc_cn` | `dftd3` の第 2, 第 3 引数。ポテンシャルと配位数のカットオフ（コード内部の `rc_radial`, `rc_angular`） |
| 小箱経路 / 大箱経路 | D3 の二つの実装経路（セルの厚さとカットオフの関係で切り替え。04・05 章） |
| `extra`（カーネル） | $C_6$ の配位数依存性から生じる力（間接項）を加えるカーネル（2.5 節） |
| shifted-force（シフト付き力） | エネルギーと力の両方を打ち切り位置で 0 にそろえる補正（D2、2.6 節） |
| `num_sub` | 1 原子を何スレッドで処理するか（並列化の細かさ。05 章） |
| 検証（verification）/ 回帰（regression）/ 妥当性確認（validation） | 00 章 0.6 節、06 章 6.1 節 |
| `master` ベースライン | upstream の `master`（`3b2c8db8`）をビルドしたバイナリ |
| AC-n | 要求仕様 DR-001, DR-002 の受入基準（Acceptance Criterion）n |
| UPSTREAM-xxxx | upstream コードで見つかった課題の記録（`specs/upstream-issues/`） |

## A3.7 パラメータの出典

| パラメータ | 出典 | 状態 |
|---|---|---|
| D2 の C6, R0（H〜Rn、$Z\le86$） | dftd3-lib（コミット `3c99ee9`、Grimme の DFTD3 V3.1 Rev 1 の再配布）の `prg/extras.f90` の `loadoldpar`。`tests_pytest/fixtures/dftd2/generate_params.py` が `dftd2_params.csv` と `src/force/dftd2para.cuh` を生成 [VER-0003 1 節] | H〜Xe の 54 元素は Turbomole マニュアルの表 6.3（Grimme 2006 の再掲）と**不一致 0 件**で照合済み（C6 は完全一致、R0 は 6e-4 Å 以内）。Cs〜Rn（$Z=55$〜86）は dftd3-lib のみで、グループ値（例：Ce〜Lu で C6 = 140.68、$r_0$ = 1.594）の穴埋めを含み、**他の出典との照合は無い** [VER-0003 2 節] |
| D2 の $s_6$ | ユーザーが指定。汎関数ごとの値は文献を参照（PBE は 0.75） | コードに表は無い |
| D3 の汎関数パラメータ（$s_6, s_8, a_1, a_2$）と表（`c6_ref_cpu`, `cn_ref`, `num_cn`, `r2r4`, `covalent_radius`） | `src/force/dftd3/dftd3.cu` の `set_para` と `dftd3para.cuh` に値のみ。**コード中に出典の記載が無い** | 出典は不明。`dftd3.rst` は Bonn 大学のページの汎関数リストを参照しているが、コード中は 54 の名前で、リスト（未照合）と一致するかは確認していない。PBE の値は simple-dftd3 の参照値と 6.2 節の精度で一致 [VER-0001] |
| D3 の定数 `HartreeBohr6` など | コード中に出典なし（2.7 節、UPSTREAM-0002 B） | CODATA 2018 から求めた値に対し相対 +8.7e-7 |
| 参照実装 | simple-dftd3 1.7.0（conda-forge の `dftd3-python`）、dftd3-lib（D2 のオラクル） | 同一のパラメータを使うので、パラメータ表の独立な検証にはならない（D2 は VER-0003 の表照合で別途確認） |
| 理論文献 | D3: Grimme et al., J. Chem. Phys. 132, 154104 (2010)、BJ: Grimme, Ehrlich, Goerigk, J. Comput. Chem. 32, 1456 (2011)（`dftd3.rst` の [Grimme2010], [Grimme2011]、`dftd3.cu` のヘッダ）。D2: Grimme, J. Comput. Chem. 27, 1787 (2006)（`dftd2.cu` のヘッダ） | |

## A3.8 変更履歴

### 本マニュアルに関する版情報

| 日付 | 内容 |
|---|---|
| 2026-11-04 | 初版（README の版情報を参照） |

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

上の履歴は `git log -- src/force/dftd3/dftd3.cu src/force/dftd2.cu src/force/dftd2para.cuh specs/verification` の件名から抜粋したもので、コミットの内容は未再確認です。詳細は各 VER 記録と `specs/traceability.yaml` を参照してください。
