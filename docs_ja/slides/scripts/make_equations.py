"""Equation images for the appendix. Formulas transcribed from docs_ja/dftd/02_theory.md and docs_ja/qnep/02_theory.md."""
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'mathtext.fontset': 'dejavusans', 'savefig.dpi': 220})
OUT = sys.argv[1]


def eq(name, lines, size=20, gap=0.62):
    fig = plt.figure(figsize=(9, gap * len(lines) + 0.1))
    n = len(lines)
    for i, s in enumerate(lines):
        fig.text(0.5, 1 - (i + 0.5) / n, s, ha='center', va='center', fontsize=size)
    fig.savefig(f'{OUT}/{name}.png', bbox_inches='tight', pad_inches=0.05, transparent=True)
    plt.close(fig)


eq('E1_d3_bj', [
    r'$E_{\mathrm{D3}}=-\sum_{i<j}\left[s_6\,\frac{C_6^{ij}}{r_{ij}^6+f_{ij}^6}+s_8\,\frac{C_8^{ij}}{r_{ij}^8+f_{ij}^8}\right]$',
    r'$f_{ij}=a_1R_0^{ij}+a_2,\quad R_0^{ij}=\sqrt{C_8^{ij}/C_6^{ij}}$',
    r'$C_8^{ij}=3\,C_6^{ij}\sqrt{Q_iQ_j},\quad \sqrt{Q_i}\equiv \mathrm{r2r4}[Z_i]$'], size=19, gap=0.8)
eq('E2_d3_zero', [
    r'$E_{\mathrm{D3(0)}}=-\sum_{i<j}\left[s_6C_6^{ij}\,\frac{f_6(r_{ij})}{r_{ij}^6}+s_8C_8^{ij}\,\frac{f_8(r_{ij})}{r_{ij}^8}\right]$',
    r'$f_n(r)=\frac{1}{1+6\,\left(r/(s_{r,n}R_0^{ij})\right)^{-\alpha_n}},\quad \alpha_6=14,\ \alpha_8=16$'], size=19, gap=0.9)
eq('E3_cn_c6', [
    r'$\mathrm{CN}_i=\sum_{j:\,r_{ij}<r_{c,\mathrm{cn}}}\frac{1}{1+\exp\left[-k_1\left(\frac{R_i+R_j}{r_{ij}}-1\right)\right]},\quad k_1=16$',
    r'$w_i[a]=\exp\left[-k_3\left(\mathrm{CN}_i-\mathrm{CN}^{\mathrm{ref}}_{Z_i,a}\right)^2\right],\quad k_3=4$',
    r'$C_6^{ij}=\frac{\sum_{a,b}C^{\mathrm{ref}}_{6,ab}\,w_i[a]\,w_j[b]}{\left(\sum_a w_i[a]\right)\left(\sum_b w_j[b]\right)}$'], size=18, gap=0.95)
eq('E4_d2', [
    r'$\phi(r)=-s_6\,C_6^{ij}\,\frac{f(r)}{r^6},\quad f(r)=\frac{1}{1+\exp\left[-d\,(r/R_0^{ij}-1)\right]},\quad d=20$',
    r'Grimme 2006:  $C_6^{ij}=\sqrt{C_6^iC_6^j},\quad R_0^{ij}=R_0^i+R_0^j$',
    r'table qe:  $C_6^{ij}=C_6^{\mathrm{D3}}(Z_i,Z_j;\,\mathrm{first\ ref}),\quad R_0^{ij}=1.1\,r_0^{\mathrm{D3},ij}$'], size=18, gap=0.75)
eq('E5_d2_sf', [
    r'$\phi_{\mathrm{sf}}(r)=\phi(r)-\phi(r_c)-\phi^{\prime}(r_c)\,(r-r_c)\ \ (r<r_c),\quad 0\ \ (r\geq r_c)$',
    r'$E_{\mathrm{D2}}=\sum_{i<j}\phi_{\mathrm{sf}}(r_{ij}),\quad \mathbf{F}_i=\sum_j\left[\phi^{\prime}(r)-\phi^{\prime}(r_c)\right]\frac{\mathbf{r}_{ij}}{r}$'], size=18, gap=0.8)
eq('E6_q_P', [
    r'$q_a=\sum_b\phi_{ab}\,(h_a-h_b),\qquad \sum_a q_a=0$',
    r'$\mathbf{P}=\sum_a q_a\mathbf{r}_a=\sum_{a<b}\phi_{ab}\,(h_a-h_b)\,(\mathbf{r}_a-\mathbf{r}_b)=\sum_a h_a\mathbf{V}_a$',
    r'$\mathbf{V}_a=\sum_b\phi_{ab}\,(\mathbf{r}_a-\mathbf{r}_b)$'], size=19, gap=0.7)
eq('E7_bec', [
    r'$Z_{k,\beta\alpha}=\frac{\partial P_\beta}{\partial r_{k,\alpha}}=\sum_a V_{a,\beta}\frac{\partial h_a}{\partial r_{k,\alpha}}\ \ (\mathrm{term\ A})\ +\ \sum_a h_a\frac{\partial V_{a,\beta}}{\partial r_{k,\alpha}}\ \ (\mathrm{term\ B})$',
    r'$\mathrm{term\ B}_k=\sum_b M_{kb}\,(h_k-h_b),\quad M_{kb}=\phi^{\prime}(r)\,\hat{\mathbf{x}}\otimes\mathbf{x}+\phi(r)\,\mathbf{1},\quad \mathbf{x}=\mathbf{r}_k-\mathbf{r}_b$'], size=17, gap=0.85)
eq('E8_force', [
    r'$Dt_c=\frac{\partial E}{\partial h_c}=\sum_b\phi_{cb}\,(D_c-D_b),\qquad D_a=\frac{\partial E_{\mathrm{el}}}{\partial q_a}$',
    r'$g_{kb}=\phi^{\prime}(d)\,(h_k-h_b)\,(D_k-D_b),\qquad \mathbf{F}_k\leftarrow\mathbf{F}_k+g_{kb}\,\mathbf{v}/d,\quad \mathbf{v}=\mathbf{r}_b-\mathbf{r}_k$'], size=18, gap=0.8)
