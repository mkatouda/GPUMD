# 付録 A1 DFT-D3（BJ とゼロダンピング）の力・ビリアルの導出

本付録は、[第2章](02_theory.md) 2.5 節の式を導きます（A1.1〜A1.8 は BJ、A1.9 はゼロダンピング D3(0)）。出発点は 2.2〜2.4 節のエネルギー（BJ 二体項、配位数、C6 補間）です。導出した式は、`scripts/check_d3_derivation.py`（CPU、NumPy、倍精度）で、エネルギーの有限差分と比べて確認しました（A1.7 節）。確認できなかった式は本文に含めていません（該当はありません。確認の範囲外にあるものは A1.8 節に列挙）。ゼロダンピングの式は `scripts/check_d3zero_derivation.py` で同様に確認しました（A1.9.5 節）。

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

## A1.9 ゼロダンピング D3(0) の導出（DR-003）

ゼロダンピングは、BJ とは減衰関数だけが異なります。配位数 $\mathrm{CN}_i$、C6 補間（A1.5 節）、$C_8^{ij} = \kappa_{ij}C_6^{ij}$、配位数の位置微分（A1.3 節のステップ 2）、`extra` カーネルの構造は**そのまま共通**です。以下では違う部分だけを導きます。

### A1.9.1 エネルギー

$$
E = -\sum_{i<j}\Big[s_6\,C_6^{ij}\,g_6(r_{ij}) + s_8\,C_8^{ij}\,g_8(r_{ij})\Big],\qquad
g_n(r) = \frac{f_n(r)}{r^n},\quad
f_n(r) = \frac{1}{1 + 6\,t_n^{-\alpha_n}},\quad t_n = \frac{r}{s_{r,n}R_0^{ij}}
$$

$\alpha_6 = 14$、$\alpha_8 = 16$（$= n + 8$）です。$R_0^{ij}$ は `r0ab`（Å）で、元素のペアだけで決まり、距離に依存しません。BJ の $d_n = 1/(r^n + f^n)$ を $g_n$ に置き換えた形になっています（A1.1 節の $g_n \equiv s_nd_n$ という記法とは別物なので、本節では $g_n = f_n/r^n$ に $s_n$ を含めません）。

### A1.9.2 $g_n$ の距離微分

$1 - f_n = 6\,t^{-\alpha}/(1 + 6\,t^{-\alpha}) = 6\,t^{-\alpha}f_n$ です。$f_n$ の微分は $\mathrm{d}t/\mathrm{d}r = t/r$ から

$$
f_n' = \frac{6\alpha\,t^{-\alpha}}{(1 + 6t^{-\alpha})^2}\cdot\frac{1}{r} = \frac{\alpha}{r}\,f_n\,(1 - f_n)
$$

です。したがって

$$
\frac{\mathrm{d}g_n}{\mathrm{d}r} = \frac{f_n'}{r^n} - \frac{n\,f_n}{r^{n+1}} = -\frac{g_n}{r}\Big[\,n - \alpha_n\,(1 - f_n)\Big] \equiv -\frac{g_n\,h_n}{r},
\qquad h_n \equiv n - \alpha_n\,(1-f_n) = n - 6\alpha_n\,t^{-\alpha_n}f_n
$$

です。コードの `d3_zero_term<n>` はこの $g_n$ と $h_n$ を返します（$h_n$ の最後の等号が実際の評価式 `n - alpha * 6 * q * finv`）。$f_n \to 1$（$r \gg R_0$）では $h_n \to n$ で、$g_n \propto r^{-n}$ の微分に戻ります。$r \to 0$ では $f_n \to 0$ で、$h_n \to n - \alpha_n < 0$ です（$g_n \to 0$ で、力は有限）。

### A1.9.3 力と、C6・C8 の項

$E_{ij}$ の $C_6$ を固定した $r$ 微分は、BJ の A1.2 節と同じく

$$
\frac{\partial E_{ij}}{\partial r}\bigg|_{C_6} = C_6^{ij}\Big[s_6\,g_6h_6 + s_8\,\kappa_{ij}\,g_8h_8\Big]\frac{1}{r}
$$

ですから、原子 $i$ にかかる**直接項の力**は（$\partial r/\partial\mathbf{r}_i = -\mathbf{r}_{ij}/r$ から）

$$
\mathbf{F}_i^{\mathrm{direct}} = \sum_j \frac{1}{r^2}\Big[s_6C_6^{ij}\,g_6h_6 + s_8C_8^{ij}\,g_8h_8\Big]\mathbf{r}_{ij}
$$

です。コードは $f_2 = \tfrac12\,(s_6C_6g_6h_6 + s_8C_8g_8h_8)/r^2$、`f12 = r12 * f2`、力 `+= 2*f12` とします（BJ と同じ約束で、順序付きペアを 2 回数える分が $\tfrac12$）。$r = 0$ のとき（一致する原子）は 0/0 を避けて $f_2 = 0$ とします。ビリアルも BJ と同じ形（ペアごとに $-\mathbf{r}\otimes\mathbf{F}_i^{\mathrm{pair}}$、半分ずつ）です。

**$\mathrm{d}E/\mathrm{d}C_6$ と $\mathrm{d}E/\mathrm{d}C_8$ の項（間接項）**：$E$ は $C_6^{ij}$ と $C_8^{ij} = \kappa_{ij}C_6^{ij}$ を通じて $\mathrm{CN}_i$、$\mathrm{CN}_j$ に依存します。A1.3 節と同じ連鎖律で

$$
D_i = -\frac{\partial E}{\partial\mathrm{CN}_i} = \sum_j \frac{\partial C_6^{ij}}{\partial\mathrm{CN}_i}\Big[s_6\,g_6(r_{ij}) + s_8\,\kappa_{ij}\,g_8(r_{ij})\Big]
$$

となり、BJ の $D_i$（A1.3 節）で $d_n$ を $g_n$ に替えたものです（コードの `dc6 * s6 * g6 + dc6 * c8_over_c6 * s8 * g8`）。$\mathrm{d}C_6/\mathrm{d}\mathrm{CN}$（A1.5 節）、$k_{12}$ と $\mathbf{F}_i^{\mathrm{extra}} = \sum_j k_{12}(D_i + D_j)\mathbf{r}_{ij}$（A1.3 節）、間接項のビリアルとその原子ごとの割り当て（A1.4 節）は**変更なし**で、BJ と同じカーネルが使われます。違いは「$D_i$ に入る減衰関数」だけです。

### A1.9.4 桁あふれしない評価

素朴な $f_n = 1/(1 + 6\,t^{-\alpha})$ と $g_n = f_n/r^n$ は、$r \ll R_0$ で $t^{-\alpha}$ が桁あふれし、$r^{-n}$ も大きくなって、float では $\infty \times 0$ の NaN になります。コードは $t = r/(s_{r,n}R_0)$ に応じて次の二つの形を使い分けます。

- $t < 1$：$g_n = \dfrac{t^8\,a^{-n}}{t^{\alpha} + 6}$（$a = s_{r,n}R_0$）、$h_n = n - \dfrac{6\alpha}{t^{\alpha} + 6}$。$t^\alpha \le 1$ なので桁あふれせず、$r \to 0$ で $g_n \to 0$ です（$t^{-\alpha}f_n = 1/(t^\alpha + 6)$ と $r^{-n} = t^{-n}a^{-n}$ から、$t^{-n}/(1 + 6t^{-\alpha}) = t^{\alpha - n}/(t^\alpha + 6) = t^8/(t^\alpha + 6)$ と変形。$\alpha - n = 8$ を使った）。
- $t \ge 1$：$g_n = r^{-n}/(1 + 6\,t^{-\alpha})$、$h_n = n - 6\alpha\,t^{-\alpha}/(1 + 6\,t^{-\alpha})$。$t^{-\alpha} \le 1$ で、どの項も有限です。

どちらの形も同じ式の代数的な変形で、結果は同じです。この切り替えは `d3_zero_term` のコメントに書かれています。

### A1.9.5 数値検証

`scripts/check_d3zero_derivation.py`（CPU、NumPy、実行は数秒）。BJ の A1.7 節と同じ方針で、倍精度、**参照 C6・参照 CN と R0 は合成した値**（2 元素。R0 は 2.2、3.0、3.4 Å）、パラメータは PBE のゼロダンピングの値（$s_6 = 1$、$s_8 = 0.722$、$s_{r,6} = 1.217$、$s_{r,8} = 1$。VER-0023）です。表の値の検証ではなく、式の代数と桁あふれ対策の確認です。BJ の確認用の関数（C6 補間、配位数、ペアリスト）を `check_d3_derivation.py` から借りています。

- 微分：$g_n$ の素朴な式の中心差分（$h = 10^{-6}r$）と $-(g_n/r)h_n$ を、$n = 6, 8$、$t = 0.2$〜5 の 800 点で比較。
- 桁あふれ：$r = 10^{-9}$〜50 Å の 4000 点で、float32 の安全な形と倍精度の参照（$g > 10^{-30}$）を比較。素朴な float32 の式の非有限値の数を数える。
- 全体の力・ビリアル：エネルギー（素朴な式）の中心差分（$h = 10^{-5}$ Å、原子位置とひずみ）と解析式（直接項 + 間接項）を比較。(a) 5 原子クラスタ（非周期）、(b) 三斜晶・周期、4 原子（イメージを含む、$r_{c,\mathrm{pot}} = 9$ Å、$r_{c,\mathrm{cn}} = 5.5$ Å）。

| 検証項目 | 結果 |
|---|---|
| $-(g_n/r)h_n$ 対中心差分（800 点） | 最大相対誤差 2.6e-9 |
| 力、直接項 + 間接項（クラスタ / 周期） | 相対 5.1e-11 / 2.3e-10（最大力成分に対して） |
| ビリアル、直接項 + 間接項（クラスタ / 周期） | 相対 1.4e-10 / 5.9e-10（最大成分に対して） |
| 力、直接項のみ（間接項を省いた場合） | 相対 2.5e-2 / 1.2e-1（一致しない。間接項が必要） |
| float32 の安全な形対倍精度（4000 点） | 非有限 0 件、最大相対誤差 1.0e-6（float32 の丸めの水準） |
| 素朴な float32 の式 | 4000 点中 903 点が非有限（桁あふれ、$\infty\times0$） |

合格基準は、有限差分との一致が相対 1×10⁻⁷ 未満、安全な形の非有限 0 件で、結果は PASS でした（実行：`/home/katouda/miniforge3/envs/py312-gpumd/bin/python -I scripts/check_d3zero_derivation.py`、2026-11-07）。中心差分の打ち切り・丸め誤差の目安は 10⁻⁸〜10⁻⁹ です。**これらは式の代数と桁あふれ対策の確認で、GPU コードの精度の測定ではありません**。GPU コードの精度（simple-dftd3 との一致）は 06 章 6.10 節（VER-0024）を参照してください。

### A1.9.6 範囲外・未確認のもの

- D3(0) の表データ（`d3zero_r0ab`、$s_6, s_8, s_{r,6}, s_{r,8}$）の値（VER-0023 が dftd3-lib と simple-dftd3 に対して照合。論文は読んでいません）。
- Z ≥ 87 の値（実装間で不一致。D3(0) は Z ≤ 86 に限る）。
- `float` での丸め誤差、長さ定数の違い（約 1×10⁻⁶）、`rc_pot`・`rc_cn` の鋭い打ち切りの誤差。
- 三体項（ATM）。
