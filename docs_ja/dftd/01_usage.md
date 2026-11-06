# 第1章 使い方

本章では `run.in` の `dftd3` と `dftd2` キーワードの書き方、全引数、制限、メッセージ、入力例、注意事項を説明します。出典の記号：`D3` = `src/force/dftd3/dftd3.cu`、`D2` = `src/force/dftd2.cu`、`run.cu` = `src/main_gpumd/run.cu`（行番号は `develop` ブランチ `8fe8ab99` 時点。コード読解で確認したもので、メッセージの文言は grep で確認済み）。

## 1.1 構文

```
dftd3 <functional> <potential_cutoff> <coordination_number_cutoff> [zero|bj]
dftd2 <s6> [<cutoff>] [table grimme2006|qe] [trunc sf|plain]
```

どちらのキーワードも `run.in` の**任意の位置**に書けます（`run` の後ろでも読まれます）。ポテンシャルの生成時に `run.in` の全行が走査されるためです [コード: `nep.cu` のコンストラクタ、`DFTD2::find_in_run_input`。回帰テスト `dftd3_single_late` が `run 1` の後ろに `dftd3 pbe 4 3` を置く場合を確認]。

### `dftd3` の引数

| 引数 | 意味 | 単位 | 既定値 | 備考 |
|---|---|---|---|---|
| `functional` | 汎関数名（BJ ダンピングのパラメータ選択） | なし | なし（必須） | 大文字小文字は区別しない（内部で小文字化）。対応名は 1.2 節 |
| `potential_cutoff`（`rc_pot`） | D3 ポテンシャルのカットオフ半径 | Å | なし（必須） | コード内部では `rc_radial` |
| `coordination_number_cutoff`（`rc_cn`） | 配位数を数えるカットオフ半径 | Å | なし（必須） | コード内部では `rc_angular` |
| `damping`（任意） | ダンピング関数：`bj`（Becke-Johnson）または `zero`（ゼロダンピング、D3(0)） | なし | `bj` | 小文字のみ（大文字小文字を区別する）。それ以外の値、または 6 トークン以上は入力エラーで、メッセージに有効値を示す（DR-003、ADR-0003） |

- トークン数は 4（BJ、既定）または 5（5 番目が `zero` か `bj`）でなければなりません（`dftd3` を含む）。4 未満は `dftd3 must have 3 parameters ...`、5 番目が不正または 6 以上はエラーです [`DFTD3::parse_damping`、`nep.cu`、`nep_charge.cu` から呼ばれる]。起動時の出力に `damping = bj` または `damping = zero (D3(0))` の行が加わります（既存の行の文言と順序は変わりません）。
- 二つのカットオフの値域は**コードで検査されません**（負値、0、`rc_cn` > `rc_pot` も黙って受理されます。1.10 節の注意 3）。
- `functional` は、NEP の訓練データを作った汎関数に合わせます [dftd3.rst]。

### `dftd2` の引数

| 引数 | 意味 | 単位 | 既定値 | 検査 |
|---|---|---|---|---|
| `s6` | 大域的スケーリング係数 | なし | なし（必須） | 正の有限値でなければエラー |
| `cutoff` | カットオフ半径 | Å | 30 Å（`DFTD2::DEFAULT_CUTOFF`）[`dftd2.cuh:30`] | 正の有限値でなければエラー。200 Å を超えると警告のみ |

- 第 2 引数が数値ならカットオフとして読みます（カットオフは `s6` の直後にだけ置けます）。
- 続けて任意のオプション `table`（`grimme2006` または `qe`）と `trunc`（`sf` または `plain`）を、キーワードと値の組として任意の順序で、それぞれ高々 1 回書けます。値は小文字だけです（大文字はエラー）。
- 既定は `table grimme2006`、`trunc sf` で、オプションなしの入力の結果と画面出力は従来と同一です（ビット一致をテストしています）。
- `table qe` または `trunc plain` を使うときは、カットオフを**明示**しなければエラーです（ラベル生成時のカットオフと一致させるため。既定の 30 Å は Grimme 表とシフト付き力の収束調べのための値です）。
- `table qe` は Quantum ESPRESSO 7.2（`dftd3_version = 2`）のパラメータです。QE の `INPUT_PW` は「Original Grimme-D2 parametrization」と説明していますが、`dftd3_version = 2` についてこれは不正確です。エネルギーは **D2 の式に D3 の表を使うもの**（C6 は両原子の最初の D3 参照点の C6、R0 = 1.1 × D3 の `r0ab`）です。C6 は固定のペア値で、H（配位数 0.9118）と Ce（配位数 2.7991）では自由原子の C6 ではありません。Z ≤ 86 のみ対応し、構造に Z ≥ 87 の元素があると、その元素名を示してエラーになります（最初の力評価時）。
- `table qe` が再現するのは `dftd3_version = 2` のエネルギーだけです。QE の本来の D2（`vdw_corr = 'grimme-d2'`、独自の表、`london_rcut` = 200 bohr）ではありません。`table grimme2006 trunc plain` を QE の grimme-d2 に対して検証してはいません。`dsd-blyp` は減衰の傾き 60 を使うため（`table qe` は 20 固定）、再現できません。QE 7.2 では `s6` の 0.75 以外の値も単精度リテラルです。
- QE 7.2 は `r0ab` を単精度リテラルとして持ちます（10 進値との相対差は最大 5.9×10⁻⁸、減衰指数では最大 1.2×10⁻⁶）。また `r0ab` を autoang = 0.52917720859 で割ります（相対 4.4×10⁻⁹、意図的に無視）。したがって QE 7.2 との一致は相対 1×10⁻⁶ 程度までで、「一致」ではなく「整合」と表現します。検証は、依頼元の QE 7.2 データと QE 7.6-dev に対して行いました。QE 7.2 のバイナリは実行していません（検証であり、妥当性評価ではありません）。
- `trunc plain` はシフトなしの打ち切り（QE と同じ）で、ペアがカットオフを横切るたびにエネルギーが |φ(rc)|（C-C、12 Å、`table qe`、s6 = 0.75 で約 7×10⁻⁶ eV）だけ、力が約 6|φ(rc)|/rc（約 3.7×10⁻⁶ eV/Å）だけ跳びます。NVE では全エネルギーが小さな段差を持ちます。平衡では横切りが相殺するため系統的なドリフトは想定しませんが、短い NVE 測定ではドリフトは見えませんでした：180 原子の PTFE 結晶、300 K、NVE 20 ps、NEP89 + `dftd2 0.75 12 table qe`、0.5 fs、各 3 シードでは、全エネルギーの傾きは `trunc plain` で |傾き| ≤ 2×10⁻⁴ meV/atom/ps、`trunc sf` で ≤ 1×10⁻⁴ meV/atom/ps、標準偏差は plain で 5.1–5.7×10⁻⁴ eV、sf で 4.9–5.1×10⁻⁴ eV でした。7×10⁻⁶ eV の個々の段差は積分の刻みごとのノイズ（中央値 9×10⁻⁵ eV）より小さく、分離できません。ペアの規約は r < rc で、ちょうど rc のペアは除外されます（QE は r > rc のみ除外。差は無視できます）。
- ラベル生成（例えば QE の計算）と MD で、テーブルと打ち切りの設定を記録し、一致させることを推奨します。
- 汎関数表はありません。`s6` はユーザーが直接指定します。PBE 用の Grimme の値は 0.75 ですが、他の値の選択はユーザーの責任です [dftd2.rst Limitations]。
- 単位：エネルギーは eV、力は eV/Å、ビリアルは eV です [dftd2.rst Model]。

## 1.2 対応する汎関数（`dftd3`）

以下は BJ ダンピング（既定）の名前です。ゼロダンピングの対応汎関数は本節の末尾に示します。コード（`D3:1818-1871`、`set_para`）に登録されている 54 個の名前を次に示します（`lcwpbe` と `lc-wpbe08` は同一パラメータなので、パラメータセットとしては 53 種類）。綴りは**そのまま**入力してください。ハイフンやアンダースコアの正規化はありません（`hse-hjs`、`lc-wpbe08`、`dsd-pbep86` にはハイフンが含まれ、`b97d`、`camb3lyp`、`mn12sx` には含まれません）。

| | | | | | |
|---|---|---|---|---|---|
| b1b95 | b2gpplyp | b2plyp | b3lyp | b3pw91 | b97d |
| bhlyp | blyp | bmk | bop | bp86 | bpbe |
| camb3lyp | dsdblyp | hcth120 | hf | hse-hjs | lc-wpbe08 |
| lcwpbe | m11 | mn12l | mn12sx | mpw1b95 | mpwb1k |
| mpwlyp | n12sx | olyp | opbe | otpss | pbe |
| pbe0 | pbe38 | pbesol | ptpss | pw6b95 | pwb6k |
| pwpb95 | revpbe | revpbe0 | revpbe38 | revssb | rpbe |
| rpw86pbe | scan | sogga11x | ssb | tpss | tpss0 |
| tpssh | b2kplyp | dsd-pbep86 | b97m | wb97x | wb97m |

- 各名前の（s6, a1, s8, a2）の値はコードの `set_para` 呼び出しにあります。起動時に、選ばれた値が標準出力に表示されます（1.5 節）。例えば `pbe` は s6 = 1、a1 = 0.4289、s8 = 0.7875、a2 = 4.4407 Bohr です [`D3:1818-1871`。この値は VER-0001 でも使われている]。
- **注意**：パラメータ表と C6 参照表（`dftd3para.cuh`）の出典（どの版の表か）は、コードにもフォーク内の仕様書にも記載がありません [UPSTREAM-0002 G]。`dftd3.rst` は Bonn 大学の BJ 汎関数一覧へのリンクのみを示し、名前を列挙していません。コードの 54 個の名前の一部がその一覧に無い可能性がありますが、確認していません。
- 上記以外の名前を指定すると、次のメッセージで停止します（1.6 節）。

### ゼロダンピング D3(0)（`dftd3 <f> <rc> <rc_cn> zero`）

エネルギーは E = -Σ[ s6 C6 f6(r)/r^6 + s8 C8 f8(r)/r^8 ]、f_n(r) = 1/(1 + 6 (r/(sr_n R0))^(-alpha_n))（alpha6 = 14、alpha8 = 16）です（Grimme ら、J. Chem. Phys. 132, 154104 (2010)）。パラメータ（def2-QZVP 版）は dftd3-lib（コミット `3c99ee9`）から生成したもので、ユーザーは変更できません。sr6 は汎関数ごとに異なり、sr8 = 1 は `slaterdiracexchange`（sr6 = 0.999、sr8 = 0.697、s8 = -1.957）以外のすべてで成り立ちます。s6 = 1 でないのは二重ハイブリッドの `b2plyp`（0.64）、`pwpb95`（0.82）、`b2gpplyp`（0.56）、`ptpss`（0.75）だけです（全表は `src/force/dftd3/dftd3zero_table.cuh`）。汎関数名は小文字・ハイフンなしで、D3(0) 表の名前です（BJ 表の名前ではありません）：`b1b95 b2gpplyp b2plyp b3lyp b3pw91 b97d bhlyp blyp bmk bop bp86 bpbe camb3lyp hcth120 hf hse06 lcwpbe m05 m052x m06 m06hf m06l m062x mpw1b95 mpwb1k mpwlyp olyp opbe otpss pbe pbe0 pbe38 pbesol ptpss pw6b95 pwb6k pwpb95 revpbe revpbe0 revpbe38 revssb rpbe rpw86pbe ssb tpss tpss0 tpssh slaterdiracexchange`（48 個）。選んだダンピングに表が無い汎関数は、ダンピング名と汎関数名を示すエラーになり、もう一方のダンピングへの暗黙の切り替えはありません。名前の多くは BJ 表にも同じ綴りで存在しますが、`hse06`、`m05`、`m052x`、`m06`、`m06l`、`m062x`、`m06hf`、`slaterdiracexchange` はゼロダンピングにのみあります。逆に BJ 表の `hse-hjs` にはゼロダンピング側の対応がなく、`hse06` は別の汎関数定義です（別名ではありません）。

- 元素は H-Rn（Z <= 86）に限ります（DR-003 D5）。構造に Z >= 87 の原子種があると、最初の力の評価で元素名を示してエラー終了します（この検査は最初の力評価時に 1 回だけで、実行中に原子種が変わっても再実行されません）。モデルファイルに載っているだけの元素は対象外です。`bj` には適用されません。
- 二体項のみです。三体項（Axilrod-Teller-Muto）は実装していません。Quantum ESPRESSO の既定（`dftd3_version = 3`、`dftd3_threebody = .true.`）とは三体項の分だけ異なります。
- 長さは Å で扱い、R0 は Å で表になっており、換算せずそのまま使います（r/R0 の形でのみ現れるため）。dftd3-lib は bohr 値を 0.52917726 で換算するので、simple-dftd3・dftd3-lib とはダンピング領域で約 1.5 ppm の差があります（ADR-0003 項目 9）。

## 1.3 制限

### 回数・排他

| 制限 | 内容 | 出典 |
|---|---|---|
| 回数 | `dftd3` は 1 回だけ、`dftd2` も 1 回だけ | `run.cu:373-387` |
| 排他 | `dftd3` と `dftd2` は同時に使えない（どちらの順でもエラー） | `run.cu:377-387`, `D2:610-612` |

### 対応ポテンシャル

| `potential` キーワード | `dftd3` | `dftd2` |
|---|---|---|
| `nep4`、`nep4_zbl`、`nep4_temperature`、`nep4_zbl_temperature`（GPU 1 枚） | 可 | 可 |
| `nep4_charge1/2/3`、`nep4_zbl_charge1/2/3` | 可 | 可 |
| `nep_vdw`、`qnep3_vdw` | エラー | 可（二重計上はユーザーの責任） |
| NEP を GPU 複数枚で使う場合（`NEP_MULTIGPU`） | エラー | エラー |
| 上記以外（`tersoff`、`eam`、`lj`、`nep_ilp` など） | **黙って無視される**（エラーなし） | エラー |

[出典: `force.cu:117-135`、`nep.cu`、`nep_charge.cu`、`nep_charge_vdw.cu`、`nep_multigpu.cu`、`run.cu:510-516`]

`dftd3` が NEP 以外で黙って無視される点は、`dftd3.rst` にも「他のポテンシャルでは効果がない」と記載されています。`dftd2` はエラーになる点と非対称です（1.10 節の注意 1）。

### 元素

| | D3 | D2 |
|---|---|---|
| 対応元素 | H から Pu（94 元素）。元素記号は**最初の `potential` ファイルの 1 行目**から取る | H から Rn（Z ≤ 86）。組込み表 [`dftd2para.cuh`] |
| 範囲外の扱い | 1.6 節の注意（未知記号のエラーは確実には出ない） | 構造中に**実在する**原子型に Z > 86 の元素があるとエラー。最初の力評価時に一度だけ検査 |

D2 の C6, R0 表は dftd3-lib（`prg/extras.f90`、`loadoldpar`、コミット `3c99ee9`）から生成したものです。Z = 55〜86 の値と Sc-Zn、Y-Cd の群値は単一の出典（dftd3-lib 表）に由来し、Ce-Lu のように群で同じ値を共有するものがあります。独立には検証されていません [dftd2.rst、DR-002 D5]。NEP89 の場合、89 元素のうち 83 元素が D2 の表に含まれ、Ac、Th、Pa、U、Np、Pu（Z = 89〜94）は含まれません [VER-0003]。これらの元素が**構造に含まれていなければ** NEP89 でも `dftd2` を使えます。

### 境界条件、複数ポテンシャル

- 周期境界と非周期境界のどちらも扱えます。薄い板（スラブ）では、周期方向の鏡像をカットオフ内まで足し合わせるため、真空層がカットオフより薄いと、スラブが自分自身の鏡像と相互作用します [dftd2.rst Limitations]。
- 複数の NEP `potential` 行があると、各 NEP オブジェクトがそれぞれ補正を評価します。D2 のコストはポテンシャルの数だけ増えます [dftd2.rst]。D3 は元素リストを**最初の `potential` ファイルのみ**から取ります（異なる NEP ファイルを複数使った場合の挙動は未検証）。
- 原子数は 2 個以上が必要です（`gpumd` 一般の要件）[dftd2.rst]。
- 同一位置の原子（距離 0）は D2 で非有限値になります（NEP も同様）。検査はありません [dftd2.rst]。

## 1.4 テスト専用の環境変数

次の環境変数は、スレッド配分などを強制してテストするためのものです。コード中で「ユーザー向けには意図的に非公開」と記されており、**通常の計算では設定しないでください**（挙動の保証はありません）。

`GPUMD_DFTD3_LARGE_BOX_NUM_SUB`、`GPUMD_DFTD3_SMALL_BOX_NUM_SUB`、`GPUMD_DFTD3_LARGE_BOX_LIST_CAPACITY`、`GPUMD_DFTD2_TARGET_THREADS`（pytest 専用で `gpumd` が読まないもの：`DFTD2_TEST_GPUMD`）[`D3:463-481, 818-846, 866-891`、`D2:633-646`]。設定すると起動時や各ステップにデバッグ行が出力されます。

## 1.5 起動時に表示されるメッセージ

標準出力に次が出ます（`potential` 行を読んでポテンシャルを構築する時点で表示されます。コード読解のみ、表示順は未確認）。

D3（`D3:1880-1888`）：

```
    Add DFT-D3:
        potential cutoff = <rc_pot> Angstrom
        coordination number cutoff = <rc_cn> Angstrom
        functional = <小文字の汎関数名>
            s6 = <値>
            s8 = <値>
            a1 = <値>
            a2 = <値> Angstrom
```

`a2` はコード内の表では Bohr 単位ですが、表示は Å に変換した値です。値はデフォルト書式の浮動小数（有効数字 6 桁）です。

D2（`D2:674`）：

```
Use the DFT-D2 correction (s6 = <値>, shifted-force cutoff = <値> A).
```

`trunc plain` のときは `shifted-force cutoff` が `cutoff` になります。既定値でないオプションがあるときだけ、続けて `    table = qe (...)` と `    trunc = plain (...)` の行が出ます（既定値の入力の出力は従来と同一）。

カットオフ値は float に丸めた後の値です。`dftd2` のカットオフが 200 Å を超えると、次の警告も出ます。

```
Warning: the dftd2 cutoff (<値> A) is larger than 200 A; the cost grows as the cube of the cutoff and the run may be very slow or exhaust memory.
```

## 1.6 エラーメッセージ一覧

`PRINT_INPUT_ERROR(text)` は標準エラー出力に `Input Error:` に続けてファイル名・行番号・`Error text: <text>` を出力して終了します（終了コード 1）[`src/utilities/error.cuh`]。`std::cout` 出力＋`exit(1)` のものは標準出力に出ます（表の「出力先」）。

| メッセージ（原文） | 出力先 | 原因 | 対処 |
|---|---|---|---|
| `dftd3 must have 3 parameters (...) and an optional damping (zero or bj)` | 標準出力 | `dftd3` 行のトークン数が 4 未満 | `dftd3 <functional> <rc_pot> <rc_cn> [zero\|bj]` にする |
| `dftd3 has too many parameters ...` / `dftd3 damping must be zero or bj (lower case), not <token>` | 標準出力 | 6 トークン以上、または 5 番目が `zero`・`bj` 以外 | 構文を確認する |
| `The <functional> functional is not supported for DFT-D3 with zero damping.` | 標準出力 | `zero` で表に無い汎関数 | 本節の名前を使う（`bj` にしたいなら `bj` を明示） |
| `dftd3 with zero damping has no parameters for element(s): <元素 (Z=n)> ...` | 標準エラー | `zero` で構造に Z >= 87 の原子種がある（最初の力評価時） | 該当元素を含まない構造にするか `bj` を使う |
| `The <functional> functional is not supported for DFT-D3 with BJ damping.`（汎関数名は小文字で表示） | 標準出力 | 未対応の汎関数名 | 1.2 節の名前を使う。綴り（ハイフンの有無）を確認 |
| `DFTD3 only supports NEP models.` | 標準エラー | 最初の `potential` ファイルの 1 行目が `nep` で始まらない | NEP のファイルを最初の `potential` に指定する |
| `Cannot open potential file.` | 標準エラー | D3 側でポテンシャルファイルを開けない | パスを確認する |
| `The first line of the NEP model file should have <n> atom symbols.` | 標準出力 | NEP ファイル 1 行目のトークン数が不正 | モデルファイルが壊れていないか確認する |
| `DFTD3 only supports elements from H to Pu.` | 標準出力 | 原子番号が範囲外 | **注意**：未知の元素記号では確実には出ない（1.10 節の注意 4） |
| `Error:` `    The box has` `        a thickness < 2.5 radial cutoffs in a periodic direction.` `        and a thickness > 10 radial cutoffs in another direction.` `    Please increase the periodic direction(s).`（実際は複数行） | 標準出力 | 小箱経路（周期方向のどれかの厚さが 2.5 `rc_pot` 以下）で、かつ三つの厚さのうちどれかが 10 `rc_pot` を超える | 薄い周期方向を広げる、または `rc_pot` を小さくする。コードは非周期方向も含めて判定する（`dftd3.rst` の記述より少し広い）[`D3:196-230`] |
| `Invalid number of threads per atom in DFT-D3.` | 標準エラー | 内部エラー（大箱経路のスレッド配分が不正）。環境変数の誤設定は無視される設計だが、通常は出ない | 通常の入力では出ないはずです。出たら報告してください |
| `dftd3 can only appear once.` | 標準エラー | `dftd3` が 2 回以上 | 1 行にする |
| `dftd3 and dftd2 cannot be used together; use only one of them.` | 標準エラー | `dftd3` と `dftd2` を併用（`dftd3` が `dftd2` の後） | どちらか一つにする |
| `dftd2 can only appear once.` | 標準エラー | `dftd2` が 2 回以上 | 1 行にする |
| `dftd2 and dftd3 cannot be used together; use only one of them.` | 標準エラー | `dftd2` と `dftd3` を併用（`dftd2` が `dftd3` の後、またはどこかに `dftd3` がある） | どちらか一つにする |
| `dftd2 requires at least 1 parameter: dftd2 <s6> [<cutoff in Angstrom>] [table grimme2006\|qe] [trunc sf\|plain].` | 標準エラー | `s6` が無い | `dftd2 <s6> ...` にする |
| `dftd2: the option '<table/trunc>' can only be given once.` / `... needs a value (...)` | 標準エラー | オプションの重複、値の欠落 | 1 回だけ、値付きで書く |
| `dftd2: unknown table '<値>'; ...` / `dftd2: unknown truncation '<値>'; ...` | 標準エラー | `table` は `grimme2006\|qe`、`trunc` は `sf\|plain`（小文字）以外 | 有効な値にする |
| `dftd2: unexpected number '<値>'; the cutoff, if given, must come directly after s6 ...` / `dftd2: unknown token '<値>'; ...` | 標準エラー | オプションの後ろの数値、未知のトークン | 構文に従う |
| `dftd2: the cutoff must be given explicitly with 'table qe' or 'trunc plain' ...` | 標準エラー | `table qe` / `trunc plain` でカットオフ無し | カットオフを明示する |
| `dftd2: s6 must be a positive number.` | 標準エラー | `s6` ≤ 0、NaN、無限大 | 正の値を指定する |
| `dftd2: cutoff must be a positive number (Angstrom).` | 標準エラー | カットオフ ≤ 0、NaN、無限大 | 正の値を Å で指定する |
| `dftd2 is only supported for NEP potentials (nep4*, nep4_charge*, nep_vdw, qnep3_vdw); the potential used here is not supported.` | 標準エラー | NEP 系以外のポテンシャル、または `potential` が無い | NEP 系のポテンシャルを使う。`dftd2` 行を外す |
| `dftd2 has not been implemented for multi-GPU version.` | 標準エラー | 複数 GPU 版の NEP と `dftd2` | GPU 1 枚で実行する |
| `dftd3 has not been implemented for multi-GPU version.` | 標準エラー | 複数 GPU 版の NEP と `dftd3` | 同上 |
| `The dftd3 command is not supported for vdW NEP models.` | 標準エラー | `nep_vdw`/`qnep3_vdw` と `dftd3` | `dftd3` を外す（`dftd2` は使える。二重計上に注意） |
| `dftd2: atom type out of range.` | 標準エラー | 原子型がモデルの型数の範囲外（最初の評価時） | `model.xyz` の型とポテンシャルの元素リストを確認する |
| `dftd2 has no parameters for element(s): <元素名 (Z=n)> ... . The built-in DFT-D2 table covers Z <= 86 (H-Rn) only.`（`table qe` のときは末尾が `The 'table qe' parameters cover Z <= 86 (H-Rn) only.`。元素が認識不能の場合は `(unrecognized element)` のみ） | 標準エラー | 構造に実在する原子の Z が 1 未満または 86 超（最初の力評価時に一度だけ検査） | その元素を含まない構造にするか、`dftd3` を使う |

補足：

- `dftd2` で `potential` 行が全く無いときは、`dftd2` 側のメッセージより先に、`gpumd` の別の箇所（`compact_nep.cu`、`read_xyz.cu`）が `potential` 無しを報告する可能性があります。どちらが先に出るかは未実行です [コード読解のみ]。
- D2 の元素エラーは最初の力評価時に出ます。このため、起動メッセージが出た後にエラーになります。
- 引用した文言はコードからの転記です。テスト側の正規表現（`test_dftd2_reference.py`）の文言とは一致しない場合があります。

## 1.7 入力例

### D3：`examples/gpumd_dftd3`

```
potential   ../../potentials/nep/nep89_20250409/nep89_20250409.txt
dftd3       pbe 12 6

velocity    300 seed 1
ensemble    nve
time_step   0.5
dump_thermo 10
dump_xyz    100 dump.xyz force precision double
run         100
```

27 個の水分子（81 原子、立方体 9.3 Å）の NVE 100 ステップ（0.5 fs）です。`dftd3 pbe 12 6` は、PBE のパラメータ、ポテンシャルカットオフ 12 Å、配位数カットオフ 6 Å を意味します。この汎関数とカットオフは説明用であり、推奨値ではありません [examples の README]。

### D2：`examples/gpumd_dftd2`

```
potential   ../../potentials/nep/nep89_20250409/nep89_20250409.txt
dftd2       0.75 15
(以下は D3 の例と同じ)
```

`s6` = 0.75、カットオフ 15 Å（既定値は 30 Å）です。値は説明用です [examples の README]。

QE `dftd3_version = 2` と同じ D2（D3 の表、ハードカットオフ）の例は `examples/gpumd_dftd2/run_qe.in` です。

```
dftd2       0.75 12 table qe trunc plain
```

### 作業用ディレクトリで実行する

`gpumd` を `examples/<名前>/` の中で直接実行すると、`thermo.out`、`dump.xyz` などがリポジトリ内に出力されます。**必ず作業用ディレクトリにコピーして**実行してください。例：

```bash
cp -r examples/gpumd_dftd3 /tmp/gpumd_dftd3 && cd /tmp/gpumd_dftd3
/path/to/gpumd
```

`run.in` 内のポテンシャルのパス（`../../potentials/...`）は相対パスなので、コピー先に合わせて書き換えてください（コピー先が `/tmp/` 直下だと `../../potentials` は存在しません）。

### 補正だけを取り出す

`dftd3`（または `dftd2`）の行を入れた計算と外した計算を 1 回ずつ実行し、エネルギーと力の差を取ります。1 点計算には `time_step 0` と `run 1` を使います [examples の README]。補正は力・エネルギー・ビリアルに加算されるので、差が補正分です。

## 1.8 出力の見方

- 補正のエネルギー・力・ビリアルは、NEP の値に**直接加算**されます。`thermo.out` のポテンシャルエネルギー、`dump_xyz` の力、ビリアルから求める圧力に反映されます。補正を単独で出力する機能はありません。
- 原子ごとのエネルギー（`potential_per_atom`）には、各原子対のエネルギーの半分が両原子に配分されます [コード: `D3:598`、`D2:451`]。
- 例題の README によれば、81 原子の水の例では、構造が緩和されていないため 100 ステップで温度が約 600〜700 K に上がり、全エネルギー（運動＋ポテンシャル、`thermo.out` の 2 列目と 3 列目）は約 0.06 eV の範囲で一定です [examples/gpumd_dftd3/README.md]。これは入力の動作確認用であり、平衡化した計算ではありません。
- D3 の起動時表示（1.5 節）で、実際に使われた s6、s8、a1、a2 を必ず確認してください。

## 1.9 カットオフの選び方

- **コストは概ねカットオフの 3 乗**に比例します。近傍原子数が `rc^3` に比例するためです。カットオフを 2 倍にすると原子対の数は約 8 倍になります [dftd3.rst Cost、dftd2.rst Cost]。ただし、小さな系や小さなカットオフでは GPU が埋まらず固定コストが支配的なので、時間の増え方はもっと緩やかです。逆に、一部の条件（水 5,184 原子で 16 Å から 20 Å）では 3 乗より急に増える場合があり、原因は調べられていません [dftd3.rst]。実測値と系の大きさへの依存は **03 章**を参照してください。
- D3 の配位数カットオフ（`rc_cn`）は、通常 6 Å 程度で、コストへの寄与は小さいとされています [dftd3.rst]。`rc_cn` は `rc_pot` 以下にしてください（`rc_cn` > `rc_pot` のときの挙動は未検証。1.10 節の注意 3）。
- D3 は二つの経路を自動で切り替えます。周期方向のどれかの箱の厚さが `rc_pot` の 2.5 倍以下なら**小箱経路**（近傍リストを作る。1 原子あたり 10,000 近傍までの上限があり、超過は検査されない）、それ以外は**大箱経路**（セルリスト）です [dftd3.rst Cost]。小箱経路の上限は、水でおおよそ `rc_pot` < 28 Å の目安です [dftd3.rst]。
- D3 のカットオフの打ち切り誤差：VER-0001 の試験系（12/6 Å）では、simple-dftd3 の既定カットオフとの差が 0.03〜0.05 eV 程度、25/15 Å で約 1×10⁻³ eV でした [VER-0001]。これらは試験系（ベンゼン二量体、グラファイト、水）での値で、一般の系に外挿できるとは限りません。
- D2 の既定 30 Å は、軽元素（水、グラファイト、MgO）の収束テストに基づく値で、**重元素・高密度系では収束していません**。`ΔE ∝ rc^-3` の関係があり、密度と C6 に比例して誤差が大きくなります。そのような系ではカットオフ収束を必ず確認してください [dftd2.rst Truncation effects、VER-0005, VER-0007]。数値は 06 章を参照してください。
- `s6` やカットオフの値を変えるとエネルギーが変わります。NPT では、打ち切りによる圧力の系統誤差（正の方向）で体積がわずかに大きくなり得ます [dftd2.rst]。

## 1.10 既知の制限と注意事項

以下は利用者向けの注意です。各項目に、確認の度合いを付記します。

### 実測で確認されたもの

1. **極端な配位数での D3 の C6 誤差（UPSTREAM-0001）**：C6 補間の重みを float で計算しているため、配位数がどの参照配位数からも約 4.2 以上離れると `exp(-4 diff^2)` が float でアンダーフローし、`W < 1e-30f` で「最後の参照インデックス」の C6 が使われ、C6 の配位数微分は 0 になります。simple-dftd3 は double で保持し、参照値の最大配位数のものを使います。この挙動は upstream の `master` と同一で、このフォークの最適化で入ったものではありません。例：密な H70 クラスタで simple-dftd3 とのエネルギー差 8.1 eV、力 4.4×10⁻² eV/Å、ビリアル 1.1 eV。Cu35H35 密クラスタで 7.3 eV。圧縮 Cu fcc（a = 2.2 Å）、膨張結晶、短い原子対では許容誤差内でした。水・結晶・有機物など NEP89 の典型的な系は影響を受けないとされています。ただし、H や Cu の高密度クラスタなど、極端に配位数の大きい系では D3 の結果を疑ってください [UPSTREAM-0001, VER-0017]。原因はコード読解によるもので、double 重みでの確認実験はまだ行っていません。
2. **D2 のカットオフ収束**：重元素・高密度系（fcc Au で 30 Å のとき 10.1 meV/原子、圧力 96 MPa のずれ）[dftd2.rst, VER-0007]。
3. **大箱経路は実行ごとのビット一致が保証されません**：セルリストの構築に整数の `atomicAdd` を使うためです（upstream でも同様）。小箱経路の D3 は実行ごとにビット一致するようになり [VER-0013, VER-0014, VER-0015]、D2 も同一 GPU・同一バイナリでビット一致します [VER-0007, VER-0009]。大箱経路の非再現性は VER-0004 に記録があります。

### コード読解のみ（未再現）

以下は UPSTREAM-0002 に記録したコード読解の結果で、**実行して再現していません**。このフォークでは「既知の制限として文書化し、コードは変更しない」と決定しています [UPSTREAM-0002 ヘッダ]。

| 番号 | 内容 | 利用者への注意 |
|---|---|---|
| 注意 1 | `dftd3` は NEP 以外のポテンシャルでは黙って無視される（`dftd2` はエラー）[UPSTREAM-0002 G、`run.cu:373-380`] | NEP 以外で `dftd3` を書いても補正は入らない。起動メッセージ（1.5 節）が出ているかを確認する |
| 注意 2 | 小箱経路のメモリは 1 原子あたり約 320 kB（`N × 10000 × 32 B`）。`N × 10000` の int 演算は N が約 35,800 を超えるとオーバーフローし得る [UPSTREAM-0002 C] | 薄い箱に多数の原子を置く設定は避ける。そもそもこの大きさでは小箱経路は実用的でない |
| 注意 3 | `rc_pot`、`rc_cn` の範囲検査が無く、負値・0・`rc_cn` > `rc_pot` が受理される。小箱判定とセル幅は `rc_pot` だけで決まるため、`rc_cn` > `rc_pot` では配位数の数え落としが起こり得る（推論）[UPSTREAM-0002 E] | `0 < rc_cn ≤ rc_pot` を守る |
| 注意 4 | 元素記号が一致しないとき `atomic_number` が未初期化のまま使われ、`DFTD3 only supports elements from H to Pu.` が確実には出ない [UPSTREAM-0002 A] | NEP ファイル 1 行目の元素記号の綴りを確認する |
| 注意 5 | 近傍数が `MN = 10000` を超えても検査がなく、範囲外書き込みが起こり得る（再現はしていない）。`dftd3.rst` にも「検査しない」と記載 [UPSTREAM-0002 D、dftd3.rst] | 水で `rc_pot` を 28 Å 以上にしない（目安。密度が高い系ではより小さくする） |
| 注意 6 | NPT で箱が `2.5 rc_pot` のしきい値をまたいで小箱経路から大箱経路に変わる場合、セルリスト配列の確保が行われない可能性がある（偽陽性かもしれない）[UPSTREAM-0002 F] | しきい値付近で箱が大きく変化する NPT 計算は、`dftd3` 付きでは慎重に扱う（未検証） |
| 注意 7 | 配位数項による力の、原子ごとのビリアルへの配分が非対称（相手の D_j のみ）。全ビリアルは正しい [コード、UPSTREAM-0002 G] | 原子ごとのビリアルを使う解析では、この配分に注意する |
| 注意 8 | 定数 `HartreeBohr6` が CODATA 2018 から求めた値より相対 +8.7×10⁻⁷ 大きい。出典がコードに明記されていない [UPSTREAM-0002 B] | D3 のエネルギーが約 1×10⁻⁶ の相対オーダーで系統的にずれ得る（本フォークの精度許容値 2×10⁻⁶ の約 44%） |

### その他

- 二体項のみ（ATM 三体項なし）、汎関数は PBE との比較のみ検証、D3 の参照試験に使った元素は H、C、O、Cu のみです [VER-0001]。
- D2 は中性原子の固定 C6 で、イオン性・金属系は過剰結合になり得ます。熱輸送計算（`compute_hac`、`compute_hnemd`、`compute_shc`）は D2 の既定カットオフで検証していません [dftd2.rst]。
- 複数 GPU 版 NEP、HIP、`gpumd_replica` は 1.3 節の通り非対応または未検証です。
- DFT・実験に対する妥当性確認は行っていません（06 章）。

## 1.11 関連ファイル

| 内容 | 場所 |
|---|---|
| 公式マニュアルページ（英語） | `doc/gpumd/input_parameters/dftd3.rst`、`dftd2.rst` |
| 入力例 | `examples/gpumd_dftd3/`、`examples/gpumd_dftd2/` |
| 要求仕様 | `specs/requirements/DR-001`（D3 高速化）、`DR-002`（D2） |
| 既知課題 | `specs/upstream-issues/UPSTREAM-0001`、`UPSTREAM-0002` |
