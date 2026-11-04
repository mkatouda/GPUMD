# 付録 A2 DFT-D2 の力・ビリアルの導出

本付録は、[第2章](02_theory.md) 2.6 節の D2 の式を導きます。導出した式は `scripts/check_d2_derivation.py` と `scripts/check_d2_tail.py`（CPU、NumPy、倍精度）で確認しました（A2.7 節）。

## A2.1 エネルギー

Grimme (2006) の D2 は、ペアごとの項

$$
\phi(r) = -\,s_6\,C_6^{ij}\,\frac{f(r)}{r^6},\qquad f(r) = \frac{1}{1+e(r)},\quad e(r) = \exp\!\left[-d\left(\frac{r}{R_0^{ij}} - 1\right)\right],\quad d = 20
$$

を $r<r_c$ のすべてのペアと周期イメージについて足します。$C_6^{ij} = \sqrt{C_6^iC_6^j}$（元素ごとの値の幾何平均）、$R_0^{ij} = R_0^i + R_0^j$ です [D2:22-23]。以下、$A \equiv s_6C_6^{ij}$ と書きます（コードの `par.x` = `s6c6`、`par.y` = $R_0$ [D2:659-662]）。

## A2.2 ダンピング関数の微分

$e'(r) = -(d/R_0)\,e$ なので

$$
f'(r) = -\frac{e'}{(1+e)^2} = \frac{d}{R_0}\,e\,f^2
$$

です（コード `dfdr = (D/R0) * e * f * f` [D2:85, 446]）。$\phi = -A f r^{-6}$ を微分すると

$$
\phi'(r) = -A\left[\frac{f'(r)}{r^6} - \frac{6\,f(r)}{r^7}\right]
$$

です [D2:87, 448]。第 1 項は、$f$ が $r$ とともに増える（短距離で引力が切られる）ことによる、斥力方向（$\phi'<0$）の寄与、第 2 項は $r^{-6}$ 項そのものの微分です。

スクリプトで 100 組のランダムな $(Z_i, Z_j, r)$ について中心差分と比べ、最大相対誤差は $1.2\times10^{-9}$ でした。

## A2.3 シフト付き力の打ち切り

エネルギーと力の両方が $r_c$ で 0 になるように、$r<r_c$ で

$$
\phi_{\mathrm{sf}}(r) = \phi(r) - \phi(r_c) - \phi'(r_c)\,(r - r_c)
$$

とし、$r\ge r_c$ では 0 とします [D2:25, 449]。これを $r$ で微分すると

$$
\phi_{\mathrm{sf}}'(r) = \phi'(r) - \phi'(r_c)
$$

です [D2:450]。$\phi_{\mathrm{sf}}(r_c) = 0$、$\phi_{\mathrm{sf}}'(r_c) = 0$ なので、エネルギーも力もカットオフで連続です（スクリプトで確認、値は $r = r_c$ で厳密に 0）。

**コードの実装**：`phi_sf = phi - par.z - par.w * (r - rc)`（`par.z` = $\phi(r_c)$、`par.w` = $\phi'(r_c)$、ホスト側で倍精度に計算し `float` で保持 [D2:660-665]）、`dphi_sf = dphi - par.w`。カットオフ $r_c$ は最初に `float` に丸めて以降すべてに同じ値を使います [D2:624]。

## A2.4 力

エネルギーは $E = \sum_{i<j}\phi_{\mathrm{sf}}(r_{ij})$ です。$\partial r_{ij}/\partial\mathbf{r}_i = -\mathbf{r}_{ij}/r_{ij}$（$\mathbf{r}_{ij} = \mathbf{r}_j - \mathbf{r}_i$）なので

$$
\mathbf{F}_i = -\frac{\partial E}{\partial \mathbf{r}_i} = \sum_{j}\phi_{\mathrm{sf}}'(r_{ij})\,\frac{\mathbf{r}_{ij}}{r_{ij}} = \sum_j\big[\phi'(r_{ij}) - \phi'(r_c)\big]\frac{\mathbf{r}_{ij}}{r_{ij}}
$$

です（$i$ を含む全ペア、順序付き）。コードは `g = dphi_sf * rinv; fp = g * dx`（`dx` $= \mathbf{r}_{ij}$ の成分）で $\mathbf{F}_i$ に加えます [D2:452-458]。符号：$\phi_{\mathrm{sf}}'>0$ なら $\mathbf{F}_i$ は $j$ の向きで、引力です。

原子エネルギーは各ペアの半分ずつ `e_i += 0.5 * phi_sf` [D2:451] で、全原子の和が $E$ です。

## A2.5 ビリアル

A1.4 節と同じ議論により（ペアが $r$ だけを通じて変形に依存する）、ペアのビリアルは $-\mathbf{r}_{ij}\otimes\mathbf{F}_i^{\mathrm{pair}}$ です。コードは順序付きペアごとに半分 `v_ab -= 0.5 * d_a * fp_b` を加えるので、ペア $(i,j)$ と $(j,i)$ の合計がペアのビリアルに等しくなります [D2:459-467]。

$$
W_{ab} = -\frac{\partial E}{\partial\varepsilon_{ab}} = -\sum_{i<j}\phi_{\mathrm{sf}}'(r_{ij})\,\frac{r_{ij,a}\,r_{ij,b}}{r_{ij}} = \sum_i x_{ia}F_{ib}
$$

（導出は A1.4 節と同じです。ペアの寄与は $-r_{ij,a}F^{\mathrm{pair}}_{i,b}$ で、中心力なので対称テンソルです。）

## A2.6 打ち切りの連続体近似（$7\pi\rho C/(3r_c^3)$）

$r$ が大きい領域では $f\to1$ なので $\phi\approx -c/r^6$（$c = s_6C_6$）です。数密度 $\rho$ の一様な系で、1 原子あたり、打ち切りなしの和（$r_c$ より先まで足す）と、シフト付き力の和との差は

$$
\Delta E = \frac{\rho}{2}\left[\int_0^{r_c}4\pi r^2\big(-\phi(r_c) - \phi'(r_c)(r - r_c)\big)\,dr \;-\; \int_{r_c}^{\infty}4\pi r^2\,\phi(r)\,dr\right]
$$

です。第 1 項は、シフトによる増分で、$\phi(r_c) = -c/r_c^6$、$\phi'(r_c) = 6c/r_c^7$、$\int_0^{r_c}r^2(r-r_c)dr = -r_c^4/12$ から $\tfrac{\rho}{2}\cdot4\pi\left[\frac{c}{3r_c^3} + \frac{c}{2r_c^3}\right]\cdot$ すなわち $\frac{5\pi\rho c}{3r_c^3}$ です。第 2 項（打ち切りなしの尾の符号を反転）は $\frac{\rho}{2}\cdot\frac{4\pi c}{3r_c^3} = \frac{2\pi\rho c}{3r_c^3}$ です。合計

$$
\Delta E = \frac{5\pi\rho c}{3r_c^3} + \frac{2\pi\rho c}{3r_c^3} = \frac{7\pi\rho\,c}{3\,r_c^3}
$$

で、DOC2（`dftd2.rst`）の式と一致します。$c$ には $\langle s_6C_6\rangle$（密度で重み付けした平均）を使います。`scripts/check_d2_tail.py` で、Gauss-Legendre 求積により、第 1 項が $5\pi\rho c/(3r_c^3)$ と相対誤差 $10^{-12}$ 未満で一致し、合計が $7\pi\rho c/(3r_c^3)$ と相対 $9\times10^{-16}$ で一致することを確認しました。

この見積もりは $f\to1$（長距離）、一様密度の極限です。短距離のダンピングと構造の不均一性は考慮していません。圧力のずれ $\Delta P = \rho\Delta E$ は導出していません（DOC2 の記述で、VER-0005 の測定が支持するとされています）。

## A2.7 数値検証

`scripts/check_d2_derivation.py`（実行は約 1 秒）。

- 条件：倍精度。C, H, O の $C_6$, $R_0$ は `tests_pytest/fixtures/dftd2/dftd2_params.csv` の実際の値（$R_0 = 1.1\,r_0$）、$s_6 = 0.75$、$r_c = 9$ Å（任意）。エネルギー関数は A2.3 節の $\phi_{\mathrm{sf}}$ をそのまま足したもの。
- 比較：エネルギーの中心差分（$h = 10^{-5}$ Å）から力、ひずみ $\varepsilon_{ab}=\pm h$（原子位置と格子の両方）からビリアル。

| 検証項目 | 最大誤差 |
|---|---|
| $\phi'(r)$ vs 中心差分（100 組） | 相対 1.2e-9 |
| $\phi_{\mathrm{sf}}(r_c)$, $\phi_{\mathrm{sf}}'(r_c)$ | 厳密に 0 |
| 力、5 原子クラスタ（C,H,O,H,C） | 相対 1.4e-10（最大力成分に対して） |
| ビリアル、同上 | 相対 4.8e-10 |
| 力、三斜晶・周期、4 原子（イメージを含む） | 相対 4.6e-10 |
| ビリアル、同上 | 相対 2.3e-9 |
| 連続体近似 $\Delta E$ vs 数値積分 | 相対 9e-16 |

これらは式の代数の確認で、GPU コード（`float` で計算）の精度の測定ではありません。GPU と NumPy 参照実装の一致、有限差分、ビリアルの整合性は VER-0003, VER-0005, VER-0006, VER-0010 に記録されています（06 章）。
