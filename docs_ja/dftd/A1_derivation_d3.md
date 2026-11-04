# 付録 A1 DFT-D3（BJ）の力・ビリアルの導出

本付録は、[第2章](02_theory.md) 2.5 節の式を導きます。出発点は 2.2〜2.4 節のエネルギー（BJ 二体項、配位数、C6 補間）です。導出した式は、`scripts/check_d3_derivation.py`（CPU、NumPy、倍精度）で、エネルギーの有限差分と比べて確認しました（A1.7 節）。確認できなかった式は本文に含めていません（該当はありません。確認の範囲外にあるものは A1.8 節に列挙）。

## A1.1 出発点

原子位置 $\{\mathbf{r}_i\}$（周期イメージを含む）に対し、

$$
E = -\sum_{i<j}\Big[s_6\,C_6^{ij}\,d_6(r_{ij}) + s_8\,C_8^{ij}\,d_8(r_{ij})\Big],\quad
d_n(r) = \frac{1}{r^n + f_{ij}^n},\quad C_8^{ij} = \kappa_{ij}\,C_6^{ij},\quad \kappa_{ij} = 3\,\texttt{r2r4}_i\,\texttt{r2r4}_j
$$

$f_{ij} = a_1\sqrt{\kappa_{ij}} + a_2$ は距離に依存せず、元素のペアだけで決まります。$C_6^{ij} = C_6^{ij}(\mathrm{CN}_i, \mathrm{CN}_j)$ で、

$$
\mathrm{CN}_i = \sum_{j} c(r_{ij}),\qquad c(r) = \frac{1}{1+e(r)},\quad e(r) = \exp\!\left[-k_1\left(\frac{R_{ij}}{r} - 1\right)\right],\quad R_{ij} = R_i + R_j,\ k_1 = 16
$$

です。以下、$g_n \equiv s_n\,d_n$ と書きます（$g_6 = s_6 d_6$, $g_8 = s_8 d_8$）。

## A1.2 直接項（$C_6$ を固定した距離微分）

ペア $(i,j)$ のエネルギー $E_{ij} = -C_6^{ij}\,[\,s_6 d_6 + s_8\kappa_{ij}d_8\,]$ を、$C_6$ を固定して $r$ で微分します。$d_n' = -n\,r^{n-1}\,d_n^2$ なので

$$
\frac{\partial E_{ij}}{\partial r}\bigg|_{C_6} = C_6^{ij}\left[\,6 s_6\,r^5 d_6^2 + 8 s_8\,\kappa_{ij}\,r^7 d_8^2\right]
$$

です（$C_8 = \kappa C_6$）。$r = \lvert\mathbf{r}_j - \mathbf{r}_i\rvert$ なので $\partial r/\partial\mathbf{r}_i = -\mathbf{r}_{ij}/r$ です。よって原子 $i$ にかかる力は

$$
\mathbf{F}_i^{\mathrm{direct}} = -\sum_j \frac{\partial E_{ij}}{\partial r}\frac{\partial r}{\partial \mathbf{r}_i}
= \sum_j \Big[6 s_6 C_6^{ij}\,r^4 d_6^2 + 8 s_8 C_8^{ij}\,r^6 d_8^2\Big]\,\mathbf{r}_{ij}
$$

となります（引力）。ここで $r_{ij}\,/\,r \cdot r^5 = r^4\,\mathbf{r}_{ij}$ を使いました。総和は $i$ を含む全ペア（原子 $i$ から見た全近傍、順序付きペア）についてです。

コードは $f_2 = s_6 C_6\cdot 3 r^4 d_6^2 + s_8 C_8\cdot 4 r^6 d_8^2$ を作り、`f12 = r12 * f2`、力 `+= 2*f12` とします [D3:599-604]。$2 f_2 = 6 s_6 C_6 r^4 d_6^2 + 8 s_8 C_8 r^6 d_8^2$ なので、上式と一致します（係数 3, 4 に 2 を掛けるのは、微分の係数 6, 8 に等しい）。

## A1.3 配位数依存性による項

**ステップ 1：$C_6$ への連鎖律。** エネルギーが $\mathrm{CN}_k$ を通じて変わる分は

$$
\frac{\partial E}{\partial \mathrm{CN}_k} = -\sum_{j}\frac{\partial C_6^{kj}}{\partial \mathrm{CN}_k}\,\big[\,s_6 d_6(r_{kj}) + s_8\kappa_{kj}\,d_8(r_{kj})\,\big] \equiv -D_k
$$

です。$E$ の中でペア $\{k, j\}$ は一度しか現れず、$C_6^{kj}$ は $\mathrm{CN}_k$ と $\mathrm{CN}_j$ の**両方**に依存します。原子 $k$ に関する和は、$k$ を含むペアについて、$k$ の側の偏微分 $\partial C_6^{kj}/\partial\mathrm{CN}_k$ だけを取ります。これが 2.5 節の $D_k$ で、コードの `dc6_sum + dc8_sum`（`dc6 * s6 * damp_6` と `dc6 * c8_over_c6 * s8 * damp_8` の和）です [D3:614-615]。

**ステップ 2：配位数の位置微分。** $c(r)$ の導関数は

$$
c'(r) = -\frac{e'(r)}{(1+e)^2},\qquad e'(r) = e\cdot k_1\frac{R_{ij}}{r^2}\quad\Rightarrow\quad c'(r) = -\frac{k_1 R_{ij}\,e}{(1+e)^2 r^2}
$$

です。$k_{12} \equiv -c'(r)/r = k_1R_{ij}\,e/[(1+e)^2 r^3]$ と定義します。$k_1 = 16$ なので、コードの `k12 = e*16*(R1+R2)/((1+e)^2 d12^3)` です [D3:726-729]（コードのコメント `// not 8.0f` は係数が 16 であることを注記）。

**ステップ 3：和を取る。** 力 $\mathbf{F}_i = -\partial E/\partial\mathbf{r}_i$ のうち CN を通る部分は

$$
\mathbf{F}_i^{\mathrm{extra}} = -\sum_k \frac{\partial E}{\partial\mathrm{CN}_k}\frac{\partial\mathrm{CN}_k}{\partial\mathbf{r}_i} = \sum_k D_k\,\frac{\partial \mathrm{CN}_k}{\partial \mathbf{r}_i}
$$

です。$\mathbf{r}_i$ に依存する $\mathrm{CN}_k$ は、$k = i$（自分の配位数、$\sum_j c(r_{ij})$）と、$i$ を近傍にもつ $k = j$（$c(r_{ji})$ の項）だけです。どちらも同じ $c(r_{ij})$ を通ります。

$$
\frac{\partial c(r_{ij})}{\partial \mathbf{r}_i} = c'(r)\frac{\partial r}{\partial\mathbf{r}_i} = -c'(r)\,\frac{\mathbf{r}_{ij}}{r} = k_{12}\,\mathbf{r}_{ij}
$$

したがって

$$
\boxed{\mathbf{F}_i^{\mathrm{extra}} = \sum_{j:\,r_{ij}<r_{c,\mathrm{cn}}} k_{12}\,\big(D_i + D_j\big)\,\mathbf{r}_{ij}}
$$

です。ペア $(i,j)$ ごとに $D_i$ と $D_j$ の両方が掛かるのは、$\mathrm{CN}_i$ と $\mathrm{CN}_j$ がともに $\mathbf{r}_i$ に依存するからです。総和は配位数の打ち切り半径 $r_{c,\mathrm{cn}}$ 以内のペア（配位数に寄与するペア）で、ポテンシャルの打ち切り半径 $r_{c,\mathrm{pot}}$ 以内のペアではありません。これが `extra` カーネルが `angular` リストで走り、$D_j$ が全近傍で確定した後に実行される理由です [D3:675-、1213-]。

**この項を省くと力が合いません。** スクリプトでは、直接項だけの力はエネルギーの有限差分と最大成分で約 100% ずれ（相対誤差 1.05 と 0.99）、直接項と間接項の和は一致します（A1.7 節）。

## A1.4 ビリアル

変形 $\mathbf{x}\to(1+\varepsilon)\mathbf{x}$（格子ベクトルも同様）に対し、すべての距離ベクトルは $\mathbf{r}_{ij}\to(1+\varepsilon)\mathbf{r}_{ij}$ と変わります。エネルギーは $r_{ij}$ だけを通じて（直接の依存、および $\mathrm{CN}$ を通る依存）変形に依存するので、各ペアの寄与 $\partial E/\partial r\cdot\partial r/\partial\varepsilon_{ab} = (\partial E/\partial r)\,r_a r_b/r$ を足せば $W_{ab} = -\partial E/\partial\varepsilon_{ab}$ が得られます。一方、力は $\mathbf{F}_i = (\partial E/\partial r)\,\mathbf{r}_{ij}/r$（ペアの寄与）なので、ペアごとに

$$
W^{\mathrm{pair}}_{ab} = -\frac{\partial E_{\mathrm{pair}}}{\partial r}\frac{r_a r_b}{r} = -r_{ij,a}\,F^{\mathrm{pair}}_{i,b}
$$

です。全体では、**各ペアについて $-\mathbf{r}_{ij}\otimes\mathbf{F}_i^{\mathrm{pair}}$ を足す**だけです。これはそのまま $\sum_i x_{ia}F_{ib}$ に等しい（力の総和が 0 であることによる）ので、2.1 節の規約と一致します。

直接項：1 本のペアで $\mathbf{F}_i^{\mathrm{pair}} = 2 f_2\mathbf{r}_{ij}$（A1.2 節）。順序付きペア $(i,j)$ と $(j,i)$ の二つが各々 `s_sab -= r12_a * f12_b`（$f_{12} = f_2\mathbf{r}$）を加えるので、ペアの合計は $-2f_2\,\mathbf{r}\otimes\mathbf{r}$ で、ペアのビリアル $-\mathbf{r}\otimes(2f_2\mathbf{r})$ に等しくなります [D3:605-613]。

間接項：ペアの力は $k_{12}(D_i+D_j)\mathbf{r}_{ij}$ なので、ペアのビリアルは $-k_{12}(D_i+D_j)\,\mathbf{r}\otimes\mathbf{r}$ です。

**原子ごとの割り当てが非対称な点**：コードは `s_sab -= r12_a * (r12_b * k12 * D_j)`（$D_j$ は近傍側）を、順序付きペア $(i,j)$ の結果として原子 $i$ に加えます [D3:730-745（小箱）、1294-1316（大箱）]。したがって原子 $i$ の間接項のビリアルは $-\sum_j k_{12}D_j\,\mathbf{r}\otimes\mathbf{r}$ で、自分の $D_i$ は含みません。$(j,i)$ の順序では原子 $j$ に $-k_{12}D_i\,\mathbf{r}\otimes\mathbf{r}$ が入ります（$\mathbf{r}\otimes\mathbf{r}$ は $\mathbf{r}\to-\mathbf{r}$ で不変）。

$$
\sum_{\text{atoms}} W^{\mathrm{extra,code}} = -\sum_{\text{pairs}} k_{12}\,(D_j + D_i)\,\mathbf{r}\otimes\mathbf{r}
$$

となって、全ビリアルは正しく、**原子ごとの割り当ては対称な割り当て（各原子に $\tfrac12 k_{12}(D_i+D_j)$）とは異なる**ことになります。スクリプトでは、全ビリアルは両方の割り当てでひずみ微分と一致し（最大相対誤差 2.2e-8）、原子ごとの値の差は最大で最大ビリアル成分の約 61%（クラスタ）と 5.6%（周期系）でした（合成パラメータでの値で、一般の系の大きさを示すものではありません）。原子ごとのビリアル出力を使う解析では、この割り当ての非対称性に注意してください。

## A1.5 C6 補間：重みの分解（$W = \sum w_1\cdot\sum w_2$）

C6 は参照 C6 の重み付き平均です。定義どおりの二重和の形は

$$
C_6^{ij} = \frac{\sum_{a,b} C^{\mathrm{ref}}_{ab}\,L_{ab}}{\sum_{a,b}L_{ab}},\qquad L_{ab} = \exp\!\big[-k_3(\mathrm{CN}_i - \mathrm{ref}^{i}_a)^2 - k_3(\mathrm{CN}_j - \mathrm{ref}^{j}_b)^2\big]
$$

です。指数の中は $a$ だけの項と $b$ だけの項の**和**なので、指数関数の性質 $e^{x+y} = e^xe^y$ により

$$
L_{ab} = w_i[a]\,w_j[b],\qquad w_i[a] = e^{-k_3(\mathrm{CN}_i-\mathrm{ref}^i_a)^2}
$$

と分解します。このとき分母は

$$
\sum_{a,b}L_{ab} = \sum_a w_i[a]\sum_b w_j[b] \equiv \mathcal{W}
$$

（二重和 = 和の積）で、分子は $\mathcal{Z} = \sum_a w_i[a]\,t_a$、$t_a = \sum_b C^{\mathrm{ref}}_{ab}w_j[b]$ です。最適化版のコードは、原子ごとの重み $w[a]$ を一度だけ計算して保持し [D3:61-73]、ペアのループでは $t_a$ と $\mathcal{Z}, \mathcal{W}$ を作ります [D3:107-163]。以前の実装のように、ペアごとに指数関数を $5\times5$ 個計算する必要がなくなります。二重和と分解形が数学的に等しいので、この変更は結果を変えません（浮動小数点の丸め以外）。スクリプトでは、200 個のランダムな組で、二重和の形と Z/W の形の最大相対差が 6.4e-16 でした。

**$\mathrm{CN}_i$ による微分**：$w_i[a]$ の微分は $\partial w_i[a]/\partial\mathrm{CN}_i = -2k_3(\mathrm{CN}_i-\mathrm{ref}_a)w_i[a] = -8(\mathrm{CN}_i - \mathrm{ref}_a)\,w_i[a]$（$k_3=4$）で、コードの `dw = w * (-8.0f * diff)` [D3:97] です。商の微分から

$$
\frac{\partial C_6^{ij}}{\partial\mathrm{CN}_i} = \frac{\partial_i\mathcal{Z}}{\mathcal{W}} - \frac{\mathcal{Z}\,\partial_i\mathcal{W}}{\mathcal{W}^2} = \frac{\partial_i\mathcal{Z}}{\mathcal{W}} - C_6^{ij}\frac{\partial_i\mathcal{W}}{\mathcal{W}}
$$

で、$\partial_i\mathcal{Z} = \sum_a(\partial w_i[a])\,t_a$、$\partial_i\mathcal{W} = (\sum_a\partial w_i[a])\sum_b w_j[b]$ です。コードは `dc6 = dZ*W - c6*dW*W`（`W` は逆数に置き換え済み）です [D3:156-162]。スクリプトでは、中心差分との最大絶対誤差が、最大値に対して 1.7e-10 でした。

**特別な場合**（コードの挙動で、式の導出ではありません）：$\mathcal{W}$ が `float` の $10^{-30}$ を下回ると、最後の参照同士の値を返し、微分は 0 にします [D3:156-158]。この場合は式の微分と一致しません（UPSTREAM-0001）。

## A1.6 全体の流れ（コードの 2 段階）

1. **メインのカーネル**（`rc_pot` 以内の順序付きペア）：$C_6$, $\partial C_6/\partial\mathrm{CN}_i$ を求め、エネルギー（半分ずつ）、直接の力、直接項のビリアル、$D_i$ を原子ごとに書き出す。
2. **`extra` カーネル**（`rc_cn` 以内の順序付きペア）：$k_{12}$ を求め、$\mathbf{F}_i^{\mathrm{extra}} = \sum_j k_{12}(D_i+D_j)\mathbf{r}_{ij}$ と間接項のビリアルを加える。

どちらのカーネルも、各原子のスレッドが自分の力だけを書き、他の原子には書きません（全近傍リストにペアが 2 回現れるので、散布書き込みは不要）。

## A1.7 数値検証

`scripts/check_d3_derivation.py`（実行は 1 秒未満）。

- 条件：倍精度。**参照 C6 と参照 CN は合成した乱数・仮の値**（2 元素、参照 3 個と 2 個）で、D3 の実際の表ではない。代数的な式の検証が目的で、表の値の検証ではない。パラメータは PBE の $s_6, s_8, a_1, a_2$。式はコードと同じ形（順序付きペア、`k12`、ビリアルの原子ごとの割り当て）で書いた。
- 比較：エネルギー $E$ の中心差分（$h = 10^{-5}$ Å）から $-\partial E/\partial\mathbf{x}$、ひずみ $\varepsilon_{ab} = \pm h$ から $W_{ab} = -\partial E/\partial\varepsilon_{ab}$。ペア・配位数のリストは基準位置で固定し、カットオフ近傍にペアが無い（$>100h$）ことを確認した。
- 系：(a) 5 原子クラスタ（非周期）、(b) 三斜晶、周期境界、4 原子（周期イメージを含む、$r_{c,\mathrm{pot}}=9$ Å, $r_{c,\mathrm{cn}}=5.5$ Å）。

| 検証項目 | 最大誤差 |
|---|---|
| 力、直接項 + 間接項（クラスタ / 周期） | 相対 3.5e-9 / 1.8e-9（最大力成分に対して） |
| 力、直接項のみ（間接項を省いた場合） | 相対 1.05 / 0.99（一致しない。間接項が必要） |
| ビリアル、コードの割り当て（$D_j$ のみ） | 相対 2.2e-8 / 1.1e-8（最大成分に対して） |
| ビリアル、対称な割り当て | 同上（合計は同じ） |
| $\partial C_6/\partial\mathrm{CN}_i$ vs 中心差分 | 最大値に対し 1.7e-10 |
| Z/W 分解形 vs 二重和（200 組） | 相対 6.4e-16 |
| $c'(r) = -k_{12}r$（200 組） | 最大値に対し 1.2e-10 |

相対誤差は、いずれも有限差分の参照値の最大成分に対する最大誤差の比です。中心差分の打ち切り・丸め誤差の目安は $10^{-8}$〜$10^{-9}$ です（$h = 10^{-5}$）。**これらは式の代数の確認で、GPU コードの精度の測定ではありません**。GPU コードの精度は VER-0001, VER-0015 などの記録（06 章）を参照してください。

## A1.8 本付録の範囲外・未確認のもの

- D3 の表データ（`c6_ref_cpu`, `cn_ref`, `r2r4`, `covalent_radius`）の値。
- $\mathcal{W}$ のアンダーフロー領域（UPSTREAM-0001）での挙動（式の微分とは一致しない）。
- `float` 演算での丸め誤差と、`rc_pot`, `rc_cn` の鋭い打ち切りによる誤差の大きさ。
- 原子ごとのビリアルの割り当てが対称な割り当てと異なる件の、実際の GPU 出力での大きさ（スクリプトは合成パラメータでの代数的な確認）。
