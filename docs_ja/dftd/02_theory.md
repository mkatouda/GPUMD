# 第2章 理論（実装されている式）

本章は、`dftd3` と `dftd2` が**実際にコードで計算している式**をまとめます。結論（式）を先に示し、導出は付録 [A1](A1_derivation_d3.md)（D3）と [A2](A2_derivation_d2.md)（D2）に回します。各式にはソースの行番号を付けます。出典の記号：`D3` = `src/force/dftd3/dftd3.cu`、`D2` = `src/force/dftd2.cu`、`P3` = `src/force/dftd3/dftd3para.cuh`（行番号は `develop` ブランチ `cd0242e0` の内容を読んだもの）。

要点は次のとおりです。

- どちらも**二体項のみ**の補正で、エネルギー・力・ビリアルの三つを同じ式の微分として与えます。
- D3 の力には、直接の二体項に加えて、**C6 が配位数に依存することから生じる項**（コードでは `extra` カーネル）があります。この項は、単純に「ペアのエネルギーを距離で微分する」だけでは出てきません。
- D2 は**エネルギーと力の両方をカットオフ位置で 0 にそろえる**シフト付き力（shifted-force）打ち切りを使います。D3（BJ、ゼロとも）には打ち切りの平滑化はありません。D2 の `trunc plain` も同様に平滑化なしです。
- 式の代数（解析的な力・ビリアルが、エネルギーの有限差分と一致すること）は、付録のスクリプトで倍精度の NumPy により確認しました（A1, A2 の各節に最大相対誤差を記載）。**GPU コードそのものの精度は本章では扱いません**（06 章）。

## 2.1 記号と単位

| 記号 | 意味 | 単位 |
|---|---|---|
| $i, j$ | 原子（周期イメージを含む。$i \ne j$ または別イメージ） | |
| $\mathbf{r}_{ij} = \mathbf{r}_j - \mathbf{r}_i$, $r = \lvert\mathbf{r}_{ij}\rvert$ | 原子間ベクトルと距離（コードの `r12`, `d12` は $\mathbf{r}_j - \mathbf{r}_i$） | Å |
| $E$ | 補正エネルギー | eV |
| $\mathbf{F}_i = -\partial E/\partial \mathbf{r}_i$ | 原子 $i$ にかかる力 | eV/Å |
| $W_{ab} = -\partial E/\partial \varepsilon_{ab}$ | ビリアル（変形 $\mathbf{x} \to (1+\varepsilon)\mathbf{x}$ に対する）。GPUMD の規約で、$\sum_i x_{ia} F_{ib}$ に等しい | eV |
| $C_6$ | 分散係数 | eV Å$^6$（コード内の D3 の表は Hartree bohr$^6$、読み出し後に変換） |

**ビリアルの規約**：中心力のペアでは、ペアのビリアルは $-\mathbf{r}_{ij} \otimes \mathbf{F}_i$（$\mathbf{F}_i$ は $j$ から $i$ への力）です。コードは、このペアのビリアルを両原子に半分ずつ割り当てます [D3:605-613, D2:459-467]。全原子の和は、ペアごとの $-\mathbf{r}\otimes\mathbf{F}$ の和に等しくなります。出力順は xx, yy, zz, xy, xz, yz, yx, zx, zy です [D3:658-666, D2:483-491]。

## 2.2 DFT-D3（BJ ダンピング、二体項のみ）

### エネルギー

$$
E_{\mathrm{D3}} = -\sum_{i<j}\left[\, s_6 \frac{C_6^{ij}}{r_{ij}^{6} + f_{ij}^{6}} + s_8 \frac{C_8^{ij}}{r_{ij}^{8} + f_{ij}^{8}} \right],
\qquad f_{ij} = a_1 R_0^{ij} + a_2,
\qquad R_0^{ij} = \sqrt{\frac{C_8^{ij}}{C_6^{ij}}}
$$

$$
C_8^{ij} = 3\, C_6^{ij}\, \sqrt{Q_i Q_j}, \qquad \sqrt{Q_i} \equiv \texttt{r2r4}[Z_i]
\quad\Longrightarrow\quad R_0^{ij} = \sqrt{3\,\texttt{r2r4}[Z_i]\,\texttt{r2r4}[Z_j]}
$$

- $s_6, s_8, a_1, a_2$ は汎関数ごとの BJ パラメータです。`dftd3 pbe` では $s_6 = 1.000$, $a_1 = 0.4289$, $s_8 = 0.7875$, $a_2 = 4.4407$ bohr です [D3:1847]。表の $a_2$ は bohr 単位で、`set_para` が `a2 * Bohr` で Å に変換して保持します [D3:1400]。
- $C_6^{ij}$ は配位数に依存します（2.4 節）。
- 総和は、各原子対とその周期イメージのうち距離が `rc_pot`（`dftd3` の第2引数）未満のものすべてにわたります。打ち切りは鋭く、エネルギーも力も打ち切り位置で連続にはなりません（平滑化の処理がコードに無い。打ち切り誤差の大きさは記録に無い）。
- 3 体項（ATM）は含みません [VER-0001]。

**コードとの対応**：コードは**順序付きペア** $(i, j)$（全近傍リストに各ペアが 2 回現れる）ごとに計算し、原子 $i$ のエネルギーに半分を加えます。

| 式 | コード |
|---|---|
| $C_6^{ij}$ [eV Å$^6$] $= \texttt{HartreeBohr6}\times C_6[E_h\,a_0^6]$ | `c6 *= HartreeBohr6` [D3:588-589, 1124-1125] |
| $R_0^2 = C_8/C_6 = 3\,\texttt{r2r4}_i \texttt{r2r4}_j \cdot \texttt{Bohr2}$ [Å$^2$] | `c8_over_c6` [D3:591, 1127] |
| $C_8 = C_6 \cdot (C_8/C_6)$ | `c8` [D3:590, 1128] |
| $f = a_1\sqrt{C_8/C_6} + a_2$ | `damp` [D3:593, 1129] |
| $1/(r^6 + f^6)$, $1/(r^8 + f^8)$ | `damp_6`, `damp_8` [D3:593-597, 1129-1133] |
| 原子エネルギー $-\tfrac12 (s_6 C_6 d_6 + s_8 C_8 d_8)$ | `s_pe -= (...) * 0.5f` [D3:598, 1134] |

### ゼロダンピング D3(0)（`dftd3 ... zero`、DR-003）

$$
E_{\mathrm{D3(0)}} = -\sum_{i<j}\Big[\, s_6\,C_6^{ij}\,\frac{f_6(r_{ij})}{r_{ij}^{6}} + s_8\,C_8^{ij}\,\frac{f_8(r_{ij})}{r_{ij}^{8}} \Big],\qquad
f_n(r) = \frac{1}{1 + 6\,\big(r/(s_{r,n}R_0^{ij})\big)^{-\alpha_n}},\quad \alpha_6 = 14,\ \alpha_8 = 16,\ s_{r,8} = 1\ (\text{例外あり})
$$

- $C_6^{ij}$、$C_8^{ij}$、配位数、C6 補間は BJ と**共通**（2.3〜2.4 節）です。違いは減衰関数だけです。
- $R_0^{ij}$ は D3 の `r0ab`（Å、表から読む。コード上は `d3zero_r0ab`）で、BJ の $R_0 = \sqrt{C_8/C_6}$ とは別の量です。$r/R_0$ の形でしか現れないので、Å のまま使います。
- $s_6, s_8, s_{r,6}, s_{r,8}$ は汎関数ごとの表（def2-QZVP 版、dftd3-lib `3c99ee9` から生成、VER-0023）です。PBE は $s_6 = 1$、$s_8 = 0.722$、$s_{r,6} = 1.217$ で、dftd3-lib と simple-dftd3 の両方と一致を確認しました [VER-0023 4 節]。$\alpha_8 = \alpha_6 + 2$ は `prg/main.f90` と `lib/api.f90` に基づきます [VER-0023 2 節]。論文 [Grimme et al., J. Chem. Phys. 132, 154104 (2010)] 自体は**読んでいません**（パラメータは二つのソフトウェア実装に対して検証したもの）[VER-0023 1 節]。
- 力は、$g_n(r) = f_n(r)/r^n$ とおいて $\mathrm{d}g_n/\mathrm{d}r = -(g_n/r)\,\big[n - \alpha_n(1 - f_n)\big]$ から得ます（導出は A1.9 節）。C6 の配位数依存の項（`extra`）は BJ と同じ機構で、$D_i$ の中の $d_n$ を $g_n$ に置き換えるだけです。
- 数値の扱い：$t = r/(s_{r,n}R_0) < 1$ と $t \ge 1$ で式の形を切り替え、どの $r$ でも桁あふれしません（A1.9 節）。小箱の配位数だけは double で計算します（`zero` のときのみ。BJ は float のまま、ビット単位で不変）[VER-0024 1 節]。
- 長さの定数は BJ と同じ `Bohr` で、simple-dftd3（CODATA 2018）との差により、ダンピング領域で相対 1.5×10⁻⁶ 程度の差が残ります [ADR-0003 項目 9, VER-0023 2 節]。
- 三体項（ATM）は含みません。

コードの対応（`src/force/dftd3/dftd3.cu`、`develop` `90bff065`）：$g_n$ と $h_n = n - \alpha_n(1-f_n)$ は `d3_zero_term<n>`（172-205 行）、BJ とゼロの切り替えは `d3_damped_pair`（212-253 行）。表は `src/force/dftd3/dftd3zero_table.cuh`（生成物）です。

### r2r4 と共有半径（コードに無い定義）

`r2r4` の表 [P3:76-] は値だけで、定義はコードにありません。Grimme らの文献では $\texttt{r2r4}_i = \sqrt{\tfrac12 \sqrt{Z_i}\,\langle r^4\rangle_i/\langle r^2\rangle_i}$ ですが、これは**文献上の定義**で、コードや本リポジトリの記録からは確認していません（H: 2.00734898 は $\langle r^4\rangle/\langle r^2\rangle = 8.0589$ から、He: 1.56637132 は 3.4698 から、この式で再現されますが、これらの文献値は作成者の記憶に基づき、本リポジトリで照合したものではありません）。

同様に、共有結合半径の表 `covalent_radius` [P3:90-] の値（H: 0.80628308 bohr）は、Pyykkö の半径に倍率 $k_2 = 4/3$ を掛けたもの（H の 0.32 Å $\times 4/3 / 0.529177 = 0.80628$）と一致しますが、倍率 4/3 はコード中に明記されていません（推定）。D3 のパラメータ表（`c6_ref_cpu`, `cn_ref`, `num_cn`, `r2r4`, `covalent_radius`）と汎関数パラメータの**出典はコードにありません**（ヘッダはライセンス表示のみ）。

## 2.3 配位数（D3）

$$
\mathrm{CN}_i = \sum_{\substack{j\ (\text{images included})\\ r_{ij} < r_{c,\mathrm{cn}}}} \frac{1}{1 + \exp\!\left[-k_1\left(\dfrac{R_i + R_j}{r_{ij}} - 1\right)\right]},
\qquad k_1 = 16,\quad R_i = \texttt{Bohr}\times\texttt{covalent\_radius}[Z_i]\ [\text{Å}]
$$

- $k_1 = 16$ は定数としてコードに埋め込まれています [D3:189, 978]。
- 総和は `rc_cn`（`dftd3` の第3引数）未満のペアだけです。打ち切りは鋭く、平滑化はありません。
- 配位数は `float` で累積されます [D3:181, 934]。
- この関数の微分 $\partial\,\mathrm{count}/\partial r = -k_{12}\,r$ には $k_{12} = 16(R_i+R_j)\,e/[(1+e)^2 r^3]$ が現れます。$e = \exp[-16((R_i+R_j)/r - 1)]$ です（2.5 節、A1.3 節）。

## 2.4 C6 の補間（D3）

各元素 $Z$ には最大 5 個の参照配位数 $\mathrm{CN}^{\mathrm{ref}}_{Z,a}$ があり（`num_cn[Z]` 個を使用）、元素対ごとに参照 C6 の行列 $C_{6,ab}^{\mathrm{ref}}(Z_i, Z_j)$（最大 5×5）が表にあります。ガウス重み（$k_3 = 4$）を

$$
w_i[a] = \exp\!\left[-k_3\left(\mathrm{CN}_i - \mathrm{CN}^{\mathrm{ref}}_{Z_i,a}\right)^2\right],\qquad k_3 = 4
$$

とすると、C6 は次の重み付き平均です。

$$
C_6^{ij} = \frac{\sum_{a,b} C^{\mathrm{ref}}_{6,ab}\; w_i[a]\; w_j[b]}{\left(\sum_a w_i[a]\right)\left(\sum_b w_j[b]\right)}
= \frac{\mathcal{Z}}{\mathcal{W}}
$$

- $k_3 = 4$ も定数です [D3:69]。重みはその原子の配位数が確定した後に**原子ごとに一度**計算して保持し（`store_cn_weights` [D3:61-73]）、ペアのループでは掛けて足すだけにしています。ガウス重みの指数が $-4(\mathrm{CN}_i - \mathrm{ref}_a)^2 - 4(\mathrm{CN}_j - \mathrm{ref}_b)^2$ と**和**になるので、積 $w_i[a]\,w_j[b]$ に分解できるためです（A1.5 節）。
- $\partial C_6^{ij}/\partial \mathrm{CN}_i$ はコードで `dc6`（Z/W 形式）として求めます [D3:156-162]。

$$
\frac{\partial w_i[a]}{\partial \mathrm{CN}_i} = -8\,(\mathrm{CN}_i - \mathrm{ref}_a)\,w_i[a]\ \ [\text{D3:97}],\qquad
\frac{\partial C_6^{ij}}{\partial \mathrm{CN}_i} = \frac{\partial_i\mathcal{Z}}{\mathcal{W}} - C_6^{ij}\,\frac{\partial_i\mathcal{W}}{\mathcal{W}}
$$

（$\partial_i$ は $w_i[a] \to \partial w_i[a]/\partial \mathrm{CN}_i$ の置換、$\mathcal{W} = (\sum w_i)(\sum w_j)$。）

特別な場合 [D3:121-123, 154-162]：

- 両方の元素の参照が 1 個ずつなら $C_6 = C^{\mathrm{ref}}_{6,00}$、$\partial C_6/\partial\mathrm{CN} = 0$。
- `float` で $\mathcal{W} < 10^{-30}$（重みがアンダーフロー）なら、最後の参照同士の C6 に置き換え、微分は 0 にします。極端な配位数（密な H や Cu のクラスタ）では simple-dftd3 と大きくずれることがあり、upstream の `master` も同じ挙動です [UPSTREAM-0001、01 章 1.10 節]。

## 2.5 D3 の力とビリアル

エネルギーは $\mathbf{r}$ に二通りで依存します。(i) 距離 $r_{ij}$ への**直接**の依存、(ii) $C_6^{ij}$ が $\mathrm{CN}_i, \mathrm{CN}_j$ を通じて原子位置に依存する**間接**の依存。力はその両方を含みます。

$$
\mathbf{F}_i = \mathbf{F}_i^{\mathrm{direct}} + \mathbf{F}_i^{\mathrm{extra}}
$$

**直接項**（順序付きペアごと、$\mathbf{r}_{ij} = \mathbf{r}_j - \mathbf{r}_i$）：

$$
\mathbf{F}_i^{\mathrm{direct}} = \sum_{j} \left[\,6 s_6 C_6^{ij}\frac{r^4}{(r^6+f^6)^2} + 8 s_8 C_8^{ij}\frac{r^6}{(r^8+f^8)^2}\right]\mathbf{r}_{ij}
$$

コードは `f2 = s6*c6*3*r^4*damp_6^2 + s8*c8*4*r^6*damp_8^2`、`f12 = r12 * f2`、力 `+= 2 f12` と書きます [D3:599-604, 1135-1140]。順序付きペアが 2 回数えられるのではなく、上式の係数 6, 8 を 3, 4 の 2 倍として力に掛けています（導出は A1.2 節）。

**間接項（`extra`）**：まず各原子で

$$
D_i = \sum_{j} \frac{\partial C_6^{ij}}{\partial \mathrm{CN}_i}\left[s_6\,\frac{1}{r^6+f^6} + s_8\,\frac{C_8^{ij}}{C_6^{ij}}\,\frac{1}{r^8+f^8}\right] = -\frac{\partial E}{\partial \mathrm{CN}_i}
$$

をメインのカーネルが蓄積します（`dc6_sum + dc8_sum` [D3:614-615, 1150-1151]）。そのうえで

$$
\mathbf{F}_i^{\mathrm{extra}} = \sum_{j:\,r_{ij}<r_{c,\mathrm{cn}}} k_{12}\,(D_i + D_j)\,\mathbf{r}_{ij},
\qquad k_{12} = \frac{16\,(R_i + R_j)\,e}{(1+e)^2\, r^3}
$$

です [D3:726-733, 1294-1300]。$\mathbf{F}_i^{\mathrm{extra}}$ には、近傍の $D_j$ すべてが必要です（全原子の $D$ が確定してから計算する）。そのため、`extra` カーネルはメインのカーネルの後に別に走ります。この項は、配位数の計算に使うリスト（`rc_cn` 以内、コード内部の `angular` リスト）を回し、メインのカーネル（`rc_pot` 以内の `radial` リスト）とは別のループです。コード中の `// not 8.0f` [D3:728] は、係数 16（$= k_1$）が 8 ではないことを注記したものです。

**ビリアル**：

- 直接項：`s_sab -= r12_a * f12_b` [D3:605-613, 1141-1149]。ペアのビリアル $-\mathbf{r}\otimes\mathbf{F}_i$ を両原子に半分ずつ割り当てた形です。
- 間接項：`s_sab -= r12_a * r12_b * k12 * D_j` [D3:730-745, 1305-1316]。**原子ごとの割り当ては非対称**で、原子 $i$ には相手の $D_j$ だけが入り、自分の $D_i$ は相手側の原子に入ります。両原子を足せば $-\mathbf{r}\otimes\mathbf{r}\,k_{12}(D_i+D_j)$ で、ペアの $-\mathbf{r}\otimes\mathbf{F}^{\mathrm{extra}}_{\text{pair}}$ に等しいので、**全ビリアルは正しい**一方、原子ごとのビリアル（原子ごとの出力を使う解析）の割り当ては非対称です（コードのコメント「kept per atom as before」[D3:672-674]）。

これらの式が、エネルギーの有限差分（力）とひずみ微分（ビリアル）に一致することを A1.7 節のスクリプトで確認しました。

## 2.6 DFT-D2

### エネルギー

$$
\phi(r) = -\,s_6\,C_6^{ij}\,\frac{f(r)}{r^6},\qquad
f(r) = \frac{1}{1 + \exp\!\left[-d\left(r/R_0^{ij} - 1\right)\right]},\qquad d = 20
$$

$$
C_6^{ij} = \sqrt{C_6^i\,C_6^j},\qquad R_0^{ij} = R_0^i + R_0^j
$$

$C_6$ は eV Å$^6$、$R_0$ は Å です。元素ごとの $C_6^i$, $R_0^i$ は内蔵表（$Z \le 86$、H から Rn）から読みます [`dftd2para.cuh`、`initialize` D2:659-662]。**$s_6$ は表に含まれず、ユーザーが `dftd2` の第1引数で直接与えます**。$d = 20$ は定数 `D_DAMP` [D2:57] です。

### `table qe` と `trunc plain`（DR-004、ADR-0004）

`table qe`（Quantum ESPRESSO 7.2 の `dftd3_version = 2`）では、上の $C_6^{ij}$, $R_0^{ij}$ の代わりに
$C_6^{ij} = C_6^{\mathrm{D3}}(Z_i, Z_j;\ \text{両原子の最初の参照点})$、$R_0^{ij} = 1.1\, r_0^{\mathrm{D3},ij}$
を使います（$d = 20$、式の形は同じ）。QE の `pars.f90`（C6）と `core.f90`（`setr0ab`、`r0ab`）の
タグ `qe-7.2`（コミット `b4c5e8de`）から `tools/Miscellaneous/dftd2/generate_dftd2qe_table.py` で
`src/force/dftd2qe_table.cuh`（倍精度、Z ≤ 86）を生成します。C6 は $E_h a_0^6$ の値に、QE 7.2 の CODATA 2018
定数（$a_0$ = 0.529177210903 Å、$E_h$ = 27.211386245988 eV）をホストの倍精度で掛けて eV Å$^6$ にします
（float の `dftd3para.cuh` の定数は使いません。相対で 8.7×10⁻⁷ ずれるためです。VER-0025）。
`trunc plain` は $\phi(r_c) = \phi'(r_c) = 0$ とするだけで、カーネルは同じです。
$r_0$ の注意：QE 7.2 は `r0ab` を単精度リテラルで持つため（相対差は最大 5.9×10⁻⁸、減衰指数 $d(r/R_0 - 1)$ では最大 1.2×10⁻⁶）、
また autoang = 0.52917720859 で割る分（相対 4.4×10⁻⁹）は意図的に無視するため、QE 7.2 との一致は相対 1×10⁻⁶ 程度までです。
検証は依頼元の QE 7.2 データと QE 7.6-dev に対するもので、QE 7.2 のバイナリは実行していません。
QE の `INPUT_PW` は `dftd3_version = 2` を「Original Grimme-D2 parametrization」と説明しますが、実際は D3 の表に D2 の式を使うものなので、この説明は不正確です。
QE の本来の D2（`vdw_corr = 'grimme-d2'`）は再現しません。C6 の最初の参照点は固定のペア値で、H（CN 0.9118）と Ce（CN 2.7991）では自由原子の C6 ではありません。
`s6` の 0.75 以外の値は QE 7.2 では単精度リテラルで、`dsd-blyp`（alp = 60）は `table qe`（20 固定）では再現できません。
`trunc plain` の規約は $r < r_c$（ちょうど $r_c$ のペアは除外。QE は $r > r_{thr}$ のみ除外で、差は無視できます）。
カットオフを横切るペアごとにエネルギーは $|\phi(r_c)|$、力は約 $6|\phi(r_c)|/r_c$ だけ跳びます（C-C、12 Å で約 7×10⁻⁶ eV、約 3.7×10⁻⁶ eV/Å）。
平衡では横切りが相殺するため系統的なドリフトは想定しませんが、短い NVE 測定ではドリフトは見えませんでした：180 原子の PTFE 結晶、300 K、NVE 20 ps、NEP89 + `dftd2 0.75 12 table qe`、0.5 fs、各 3 シードでは、全エネルギーの傾きは `trunc plain` で |傾き| ≤ 2×10⁻⁴ meV/atom/ps、`trunc sf` で ≤ 1×10⁻⁴ meV/atom/ps、標準偏差は plain で 5.1–5.7×10⁻⁴ eV、sf で 4.9–5.1×10⁻⁴ eV でした。7×10⁻⁶ eV の個々の段差は積分の刻みごとのノイズ（中央値 9×10⁻⁵ eV）より小さく、分離できません。

### シフト付き力の打ち切り

距離がカットオフ $r_c$ 未満のペアについて、エネルギーと力が $r_c$ で連続に 0 となるよう、

$$
\phi_{\mathrm{sf}}(r) = \phi(r) - \phi(r_c) - \phi'(r_c)\,(r - r_c)\quad (r < r_c),\qquad \phi_{\mathrm{sf}}(r) = 0\quad (r \ge r_c)
$$

$$
E_{\mathrm{D2}} = \sum_{i<j} \phi_{\mathrm{sf}}(r_{ij})
$$

を使います [D2:19-20, 449-451]。$\phi(r_c)$ と $\phi'(r_c)$ はホスト側で倍精度に計算し、`float` にして保持します [D2:81-88, 662-665]。$r_c$ 自身も最初に `float` に丸め、シフト項、打ち切り判定、セル探索のすべてで同じ値を使います [D2:624]。

**力とビリアル**（導出は A2 節）：

$$
\mathbf{F}_i = \sum_j \left[\phi'(r) - \phi'(r_c)\right]\frac{\mathbf{r}_{ij}}{r},\qquad
\phi'(r) = -s_6C_6^{ij}\left[\frac{f'(r)}{r^6} - \frac{6f(r)}{r^7}\right],\quad f'(r) = \frac{d}{R_0}\,e\,f^2
$$

ここで $e = \exp[-d(r/R_0 - 1)]$ です。コードは `dphi_sf = dphi - par.w`、`g = dphi_sf * rinv`、`fp = g * dx` [D2:446-458]。ビリアルは `v_ab -= 0.5 * d_a * fp_b` [D2:459-467] で、D3 と同じ「ペアのビリアルを半分ずつ」の規約です。

**打ち切りの連続体近似**：シフトにより、一様密度 $\rho$ では、打ち切りなしの和（無限遠まで）に対し、1 原子あたり $\Delta E = 7\pi\rho\langle s_6 C_6\rangle/(3 r_c^3)$ のずれが生じます（$f \to 1$ の極限）。この係数 $7/3$ は、シフト項の寄与 $5/3$ と鋭い打ち切りの寄与 $2/3$ の和で、A2.6 節で数値積分により確認しました。このずれの圧力への換算（$\Delta P = \rho\,\Delta E$）は [DOC2: `dftd2.rst`、VER-0005 の測定] に基づく記述で、本マニュアルでは導出していません。

## 2.7 定数と単位換算

| 定数 | 値 | 場所 | 備考 |
|---|---|---|---|
| `Bohr`（Å/bohr） | 0.5291772575069165（`float` マクロ） | P3:22 | CODATA 2018 の 0.529177210903 に対し相対 +8.8e-8（導出値） |
| `Bohr2` | 0.280028569862541 | P3:23 | |
| `HartreeBohr6`（eV Å$^6$ / ($E_h\,a_0^6$)） | 0.597527426643772 | P3:24 | 27.2113957 eV × Bohr$^6$（作成者の割り算による推定）。定義の出典はコードに無い |
| $k_1$（配位数）, $k_3$（重み） | 16, 4 | D3:189,978 / D3:69 | コードに固定 |
| 参照 C6 の単位 | Hartree bohr$^6$ | P3:104- | 単位コメントは無く、`HartreeBohr6` の変換から推定 |
| D2 の $d$ | 20 | D2:57 | |
| D2 の換算（表の生成時） | 27.21138505 eV/$E_h$, 2625.4999 kJ/mol/$E_h$, 0.52917726 Å/bohr | `tests_pytest/fixtures/dftd2/generate_params.py` | |

**UPSTREAM-0002 B（`HartreeBohr6` の定数）**：テストの参照実装（simple-dftd3 側）の定数は CODATA 2018（$E_h$ = 27.211386245988 eV, $a_0$ = 0.529177210903 Å）で、そこから求めた $E_h a_0^6$ は 0.5975269043 eV Å$^6$ です。コードの値 0.597527426643772 はこれより**相対 +8.7e-7 大きく**、D3 のエネルギーはこの係数でそのままスケールされます。AC-1 のエネルギー許容（相対 2×10⁻⁶）のおよそ 44% に相当します。定数の出典はコードにも VER-0001/0011 にも書かれていません。値の変更は 1e-6 レベルの結果の変化を伴うので、このフォークでは変更せず、既知の制限として記録しています [UPSTREAM-0002 B]。

## 2.8 式とコードの対応一覧

| 内容 | 式 | D3 小箱 | D3 大箱 | D2 |
|---|---|---|---|---|
| 配位数 | 2.3 節 | 165-193 | 930-1003 | |
| ガウス重み | $w[a]$ | 61-73 | 同左 | |
| C6 と $\partial C_6/\partial\mathrm{CN}$ | 2.4 節 | 107-163 | 同左 | |
| 二体エネルギー | 2.2 節 | 588-598 | 1124-1134 | 444-451 |
| 直接の力 | 2.5 節 | 599-604 | 1135-1140 | 452-458 |
| $D_i$ の蓄積 | $D_i$ | 614-615 | 1150-1151 | |
| 間接の力（`extra`） | $k_{12}(D_i+D_j)$ | 726-733（カーネル 676-） | 1294-1300（カーネル 1213-） | |
| ビリアル | 2.5, 2.6 節 | 605-613, 730-745 | 1141-1149, 1305-1316 | 459-467 |
| BJ パラメータ | $s_6,s_8,a_1,a_2$ | `set_para` 1387-1400, 表 1818- | | |
| シフト項 | $\phi(r_c),\phi'(r_c)$ | | | 81-88, 659-665 |

小箱・大箱は D3 の二つの実装経路（セルの大きさとカットオフの関係で切り替え。04・05 章）で、数式は同じです。

## 2.9 コードに無いもの・確認できていないもの

- D3 の表データと汎関数パラメータの出典（コードにも記録にも無い）。
- `r2r4` の定義、共有半径の倍率 4/3（2.2 節）、$C_6$ 参照の単位、`HartreeBohr6` の来歴。
- D3 の打ち切り（`rc_pot`, `rc_cn`）の誤差の大きさ（記録に無い）。
- D2 の連続体近似の圧力への換算 $\Delta P = \rho\Delta E$（導出していない）。
- GPU コードの `float` 計算による誤差は本章の範囲外（06 章の測定値を参照）。
