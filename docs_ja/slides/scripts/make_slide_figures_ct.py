"""Slide versions of report figures F01-F12 (docs_ja/workreport/scripts/make_figures.py).
Same transcribed values, larger fonts, aspect ratio for a 4:3 slide body (about 8.6 x 5.7 in).
Panels of F01, F05, F06 are written as separate files (a/b). Nothing re-measured."""
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np
plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False, 'font.size': 14,
                     'savefig.dpi': 200, 'axes.spines.top': False, 'axes.spines.right': False})
OUT = sys.argv[1]
C_CT, C_OLD, C_GATE = '#1f77b4', '#d9822b', '#c0392b'
SZ = (9.5, 5.8)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f'{OUT}/{name}', bbox_inches='tight')
    plt.close(fig)


def grouped(ax, labels, series, colors, names, log=False):
    n = len(series); width = 0.8 / n; x = np.arange(len(labels))
    for i, (s, c, nm) in enumerate(zip(series, colors, names)):
        ax.bar(x + (i - (n - 1) / 2) * width, s, width, color=c, label=nm)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    if log:
        ax.set_yscale('log')


def s01a():  # VER-0028 1-d
    fig, ax = plt.subplots(figsize=SZ)
    cands = ['kernel\n(基準)', 'C1', 'C2', 'C3', 'C4']
    grouped(ax, cands, [[0, 0.122, 0.126, 0.425, 0.648], [0, 0.186, 0.186, 0.188, 0.905]],
            ['#4c72b0', '#dd8452'], ['オリゴマー 59 原子', 'アモルファス 248 原子'])
    ax.set_ylabel('RMS(候補 − kernel) [e]')
    ax.set_title('「厳密な勾配」の候補は kernel から 0.12〜0.9 e 離れる', fontsize=15)
    ax.legend(frameon=False, fontsize=13, loc='upper left')
    fig.text(0.5, -0.02, 'C1: qn·1+T2(生J)　C2: q·1+T2(生J)　C3: qn·1+T2(Jn)、最小イメージ　C4: qn·1+T2(Jn)、平面差　(VER-0028 1-d)',
             ha='center', fontsize=10)
    save(fig, 'S01a_diagnosis_candidates.png')


def s01b():  # VER-0028 1-a, 1-b
    fig, ax = plt.subplots(figsize=SZ)
    grouped(ax, ['オリゴマー\n中性化後 J (Jn)', 'アモルファス\n中性化後 J (Jn)', '両構造\n生の J'],
            [[0.682, 0.493, 0.500], [0.049, 0.483, 0.500]], ['#55a868', '#c44e52'],
            ['a（T1 の係数）', 'b（T2 の係数）'])
    ax.axhline(0.5, color='gray', ls=':', lw=1.5)
    ax.set_ylabel('BEC − q·1 = a T1 + b T2 のフィット係数')
    ax.set_title('(0.68, 0.05) は Jn を使った見かけ。生の J では a = b = 0.500', fontsize=15)
    ax.legend(frameon=False, fontsize=13, loc='upper right')
    save(fig, 'S01b_diagnosis_coeff.png')


def s02():  # schematic
    fig, ax = plt.subplots(figsize=SZ)
    ax.set_xlim(0, 10); ax.set_ylim(0, 6.2); ax.axis('off')
    xs = [1, 3, 5, 7, 9]; hs = [0.9, 0.2, -0.6, 0.5, -0.3]
    for x, h in zip(xs, hs):
        ax.add_patch(Circle((x, 4.6), 0.5, color=plt.cm.coolwarm((h + 1) / 2), ec='k'))
        ax.text(x, 4.6, f'h={h:+.1f}', ha='center', va='center', fontsize=12)
    for i in range(4):
        d = hs[i] - hs[i + 1]; x0, x1 = xs[i], xs[i + 1]
        a, b = ((x0 + 0.55, x1 - 0.55) if d > 0 else (x1 - 0.55, x0 + 0.55))
        ax.add_patch(FancyArrowPatch((a, 5.4), (b, 5.4), arrowstyle='->', mutation_scale=20, color='k'))
        ax.text((x0 + x1) / 2, 5.65, r'$\phi(h_a-h_b)$', ha='center', fontsize=13)
    ax.text(5, 3.55, '矢印: h が高い側 → 低い側。電荷を受け渡すので Σq = 0 が恒等的に成立', ha='center', fontsize=13)
    ax.text(5, 2.5, r'$q_a=\sum_b\phi_{ab}(h_a-h_b)$', ha='center', fontsize=20)
    ax.text(5, 1.5, r'$\mathbf{P}=\sum_a q_a\mathbf{r}_a=\sum_a h_a\mathbf{V}_a$,   $\mathbf{V}_a=\sum_b\phi_{ab}(\mathbf{r}_a-\mathbf{r}_b)$',
            ha='center', fontsize=17)
    ax.text(5, 0.5, 'r_ab のみに依存 → 並進不変、周期系でも一価。BEC = ∂P/∂r（項 A + 項 B）は厳密な勾配', ha='center', fontsize=13)
    save(fig, 'S02_model.png')


def s03():  # pipeline in two rows
    fig, ax = plt.subplots(figsize=SZ)
    ax.set_xlim(0, 10); ax.set_ylim(0, 6.4); ax.axis('off')
    def box(x, y, t, c, w=2.9, h=1.7):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.06', fc=c, ec='k'))
        ax.text(x + w / 2, y + h / 2, t, ha='center', va='center', fontsize=12)
    def arr(p, q): ax.add_patch(FancyArrowPatch(p, q, arrowstyle='->', mutation_scale=18, color='k'))
    top = 4.3; bot = 1.5
    box(0.1, top, 'find_descriptor\n記述子 → ANN\nh', '#dbe9f6')
    box(3.55, top, 'ct_find_charge\nq=Σφ(h_a−h_b)\nV=Σφ(r_a−r_b)', '#dbe9f6')
    box(7.0, top, 'BEC\n項 B（q·1 を含む）\n+ 項 A（V⊗∂h/∂r）', '#e8f3dc')
    arr((3.05, top + 0.85), (3.5, top + 0.85)); arr((6.5, top + 0.85), (6.95, top + 0.85))
    box(0.1, bot, 'Ewald / PPPM\nq → D_a = ∂E/∂q_a', '#f6e7c8')
    box(3.55, bot, 'ct_find_Dt_and_pair_force\nDt=Σφ(D_c−D_b)\nペア力・ビリアル', '#f6dcdc')
    box(7.0, bot, '既存の力カーネル\n（重み = Dt、無変更）', '#f6dcdc')
    arr((5.0, top), (1.55, bot + 1.75)); arr((3.05, bot + 0.85), (3.5, bot + 0.85)); arr((6.5, bot + 0.85), (6.95, bot + 0.85))
    ax.text(5, 0.55, '出力（dump_xyz bec / bec.out / compute_dpdt / add_efield bec）は同じ BEC 配列を読む', ha='center', fontsize=12)
    ax.text(5, 0.1, '従来の zero_total_charge と旧 BEC カーネル（0.5/0.5）は、このモデルでは走らない', ha='center', fontsize=11, color='#555')
    save(fig, 'S03_pipeline.png')


def s04():  # VER-0032
    labels = ['n8\nL13', 'n45\nL14', 'n59\nL16', 'n12\nL7.5', 'n10\nL6', 'n8 L13\n5.5/4', 'n59\nL18']
    err = [3.1e-6, 7.2e-5, 8.0e-5, 1.8e-4, 1.2e-4, 1.2e-5, 9.2e-5]
    rmsz = [8.3e-3, 0.143, 0.128, 0.34, 0.29, 0.040, 0.145]
    fig, ax = plt.subplots(figsize=SZ)
    x = np.arange(len(labels))
    ax.bar(x, err, color=C_CT, label='RMS(BEC/√ε∞ − Z_FD)')
    ax.plot(x, rmsz, 'ko-', ms=6, lw=1, label='RMS Z（参考）')
    ax.axhline(1e-3, color=C_GATE, ls='--', lw=2, label='ゲート 1×10$^{-3}$ e（A1）')
    ax.set_yscale('log'); ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel('[e]'); ax.set_ylim(1e-6, 60)
    ax.set_title('GPU の BEC は有限差分と 1.8×10$^{-4}$ e 以内で一致（δ = 0.02 Å）', fontsize=15)
    ax.legend(frameon=False, fontsize=12, loc='upper left')
    fig.text(0.5, -0.02, '誤差は δ$^{2}$ に比例 = 有限差分の打ち切り誤差（float の雑音ではない）。VER-0032', ha='center', fontsize=11)
    save(fig, 'S04_bec_fd.png')


def s05(kind):  # VER-0034
    labels = ['n8\nL13', 'n45\nL14', 'n59\nL16', 'n12\nL7.5', 'n10\nL6']
    fct = [5.6e-6, 1.7e-4, 9.8e-5, 1.2e-4, 1.5e-4]; fold = [1.0e-5, 3.3e-3, 3.0e-3, 7.9e-4, 5.2e-4]
    wct = [5.9e-5, 6.5e-4, 1.1e-3, 2.3e-3, 1.5e-3]; wold = [4.4e-5, 2.2e-2, 4.9e-3, 8.3e-3, 8.3e-3]
    fig, ax = plt.subplots(figsize=SZ)
    if kind == 'a':
        grouped(ax, labels, [fct, fold], [C_CT, C_OLD], ['charge_transfer', '旧 qNEP（雑音の下限）'], log=True)
        ax.axhline(1e-3, color=C_GATE, ls='--', lw=2, label='ゲート 1×10$^{-3}$ eV/Å')
        ax.set_ylabel('力の RMS 誤差 [eV/Å]')
        ax.set_title('力: 総エネルギーの中心有限差分（δ = 0.005 Å）との差', fontsize=15)
        ax.set_ylim(1e-6, 1e-1)
    else:
        grouped(ax, labels, [wct, wold], [C_CT, C_OLD], ['charge_transfer', '旧 qNEP（雑音の下限）'], log=True)
        ax.set_ylabel('ビリアルの RMS 誤差 [eV]')
        ax.set_title('ビリアル: 一様ひずみの有限差分との差', fontsize=15)
        ax.set_ylim(1e-5, 1e-1)
        fig.text(0.5, -0.02, 'ゲート: 旧の 3 倍以下 かつ 相対 3×10$^{-3}$ |W|（人間の決定）。VER-0034', ha='center', fontsize=11)
    ax.legend(frameon=False, fontsize=12, loc='upper left')
    save(fig, f'S05{kind}_force_virial.png')


def s06(kind):  # VER-0034 follow-up A
    fig, ax = plt.subplots(figsize=SZ)
    if kind == 'a':
        grouped(ax, ['薄い箱 (n12)', '厚い箱 (n45)'], [[4.7e-4, 3.1e-4], [9.4e-4, 2.5e-4], [1.20, 0.28]],
                [C_CT, '#7f7f7f', C_OLD], ['K+U−√ε∞E·P（電場あり）', 'K+U（電場なし）', 'K+U（電場あり）'], log=True)
        ax.set_ylabel('エネルギーの幅 (max−min) [eV]')
        ax.set_title('電場中の NVE（1000 ステップ）: 保存するのは K+U−√ε∞E·P', fontsize=14)
        ax.set_ylim(1e-4, 40)
    else:
        grouped(ax, ['E 方向 1', 'E 方向 2', 'E 方向 3'],
                [[1.4e-5, 2.6e-4, 8.3e-4], [2.2e-4, 2.7e-4, 2.8e-4], [0.50, 0.19, 0.16], [0.32, 0.04, 0.27]],
                [C_CT, '#7bb3dd', C_OLD, '#ecc08f'],
                ['新 薄い', '新 厚い', '旧 qNEP 薄い', '旧 qNEP 厚い'], log=True)
        ax.set_ylabel('|閉ループ積分| / 被積分関数のスケール')
        ax.set_title('閉じた経路での電場の仕事: 新は保存力（≈0）、旧は経路依存', fontsize=14)
        ax.set_ylim(1e-6, 30)
    ax.legend(frameon=False, fontsize=11, loc='upper left')
    save(fig, f'S06{kind}_field.png')


def s07():  # VER-0035 + requester report
    fig, ax = plt.subplots(1, 2, figsize=(11, 5.6))
    names = ['q2\nrc4', 'q3\nrc4', 'q2\nrc3', 'q2\n3型', 'q3\nn59', '学習済\nq2']
    bec = [4.5e-7, 4.5e-7, 1.8e-7, 7.5e-7, 2.4e-7, 1.5e-6]
    frc = [1.5e-6, 2.4e-6, 1.0e-6, 7.9e-6, 1.3e-6, 1.4e-5]
    for a, v, gate, title, unit in [(ax[0], bec, 5e-6, 'BEC', 'e'), (ax[1], frc, 1e-5, '力', 'eV/Å')]:
        x = np.arange(len(names))
        a.bar(x, v, color=C_CT, label='合成データの RMS 差')
        a.set_yscale('log'); a.set_xticks(x); a.set_xticklabels(names, fontsize=11)
        a.axhline(gate, color=C_GATE, ls='--', lw=2, label=f'A4 ゲート {gate:g} {unit}')
        a.set_ylabel(f'nep − gpumd [{unit}]'); a.set_title(title, fontsize=15)
    ax[0].axhspan(4.6e-6, 1.4e-5, color=C_OLD, alpha=0.3, label='実データ 14 構造の最大差（依頼元）')
    ax[1].axhspan(1.2e-5, 5.5e-5, color=C_OLD, alpha=0.3, label='実データ 14 構造の最大差（依頼元）')
    ax[0].set_ylim(1e-7, 1e-4); ax[1].set_ylim(5e-7, 3e-4)
    for a in ax:
        a.legend(frameon=False, fontsize=10, loc='upper left')
    fig.text(0.5, -0.02, '合成は RMS、実データは最大値で指標が異なる。学習済みの力のゲートは相対 1×10$^{-6}$ × RMS F。VER-0035、依頼元の報告',
             ha='center', fontsize=11)
    save(fig, 'S07_a4.png')


def s08():  # requester report 1
    q = ['ΔE\n[eV]', 'max|Δq|\n[e]', 'max|ΔBEC|\n[e]', 'max|ΔF|\n[eV/Å]']
    fig, ax = plt.subplots(figsize=SZ)
    grouped(ax, q, [[3.5e-5, 9.1e-7, 1.1e-6, 9.1e-6], [4.7e-5, 9.0e-7, 1.3e-6, 7.7e-6],
                    [3.7e-6, 8.9e-7, 1.2e-6, 6.9e-6], [1.5e-7, 8.9e-7, 1.1e-6, 6.9e-6]],
            ['#4c72b0', '#9cb6dc', '#dd8452', '#efc3a4'],
            ['アモルファス 248: 旧対旧', 'アモルファス 248: 旧対新', 'オリゴマー 59: 旧対旧', 'オリゴマー 59: 旧対新'], log=True)
    ax.set_ylim(1e-8, 1e-3)
    ax.set_title('新旧の差は、同じ旧バイナリ 2 回の差と同程度', fontsize=15)
    ax.legend(frameon=False, fontsize=11, loc='upper right', ncol=2)
    fig.text(0.5, -0.02, '実行ごとの雑音。依頼元の測定、float32、kspace pppm', ha='center', fontsize=11)
    save(fig, 'S08_old_noise.png')


def s09():
    g = [500, 1000, 1500, 2000]; v = [0.238, 0.192, 0.173, 0.164]
    fig, ax = plt.subplots(figsize=(9.5, 6.4))
    ax.plot(g, v, 'o-', color=C_CT, ms=8, lw=2, label='新 (charge_transfer 6)、loss.out の値')
    ax.axhline(0.159, color=C_CT, ls=':', lw=2, label='新: held-out BEC RMSE 0.159 e（最終評価）')
    ax.axhline(0.112, color=C_OLD, ls='--', lw=2, label='旧（同予算・同 seed）: 0.112 e')
    ax.axhline(0.224, color=C_GATE, ls='--', lw=1.5, label='5a ゲート ≤ 2×旧 = 0.224 e')
    ax.axhline(0.15, color='gray', ls='-.', lw=1.5, label='A6 の目安 約 0.15 e')
    ax.set_xlabel('世代'); ax.set_ylabel('held-out BEC RMSE [e]'); ax.set_ylim(0.08, 0.26)
    ax.set_title('5a（2000 世代、seed 1）: ゲートは満たすが未収束', fontsize=15)
    ax.legend(frameon=False, fontsize=11, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1)
    save(fig, 'S09_5a.png')


def s10():
    fig, ax = plt.subplots(1, 2, figsize=(11, 5.6))
    steps = np.arange(0, 201)
    ax[0].axhline(10.0, color=C_GATE, ls='--', lw=2, label='しきい値 10.0 Å\n(= 2.5 × rc 4 Å)')
    ax[0].plot(steps, 9.3 + 0.005 * steps, color='#1f77b4', lw=2.5, label='小 → 大 (9.3→10.3)')
    ax[0].plot(steps, 10.6 - 0.005 * steps, color='#2ca02c', lw=2.5, label='大 → 小 (10.6→9.6)')
    ax[0].set_xlabel('deform のステップ（time_step 0）'); ax[0].set_ylabel('箱の長さ [Å]')
    ax[0].set_title('(a) 各区間がしきい値をまたぐ', fontsize=14); ax[0].legend(frameon=False, fontsize=11, loc='upper center', bbox_to_anchor=(0.5, -0.18))
    grouped(ax[1], ['L=9.3 Å', 'L=10.3 Å'], [[3.9e-3, 3.2e-3], [3.7e-6, 2.6e-6]], [C_OLD, C_CT],
            ['1 ステップ動かしたときの\nE_D3 の変化', '合格の許容\n(2×10$^{-6}$ |E_D3|)'], log=True)
    ax[1].set_ylabel('[eV]'); ax[1].set_ylim(1e-6, 1)
    ax[1].set_title('(b) 感度: 約 1000 倍の余裕', fontsize=14); ax[1].legend(frameon=False, fontsize=11, loc='upper center', bbox_to_anchor=(0.5, -0.18))
    save(fig, 'S10_dr006.png')


def s11():
    fig, ax = plt.subplots(figsize=(10, 3.6))
    left = 0
    for lab, n, c in [('合格 216', 216, '#55a868'), ('既知 37\n(run-to-run)', 37, '#dd8452'), ('説明済\n4', 4, '#c44e52')]:
        ax.barh(0, n, left=left, color=c)
        if n > 30:
            ax.text(left + n / 2, 0, lab, ha='center', va='center', fontsize=13, color='white')
        else:
            ax.annotate(lab.replace('\n', ' '), (left + n / 2, 0.4), xytext=(left - 8, 0.75), fontsize=13, ha='center',
                        arrowprops=dict(arrowstyle='->'))
        left += n
    ax.set_xlim(0, 265); ax.set_ylim(-0.5, 1.0); ax.set_yticks([]); ax.spines['left'].set_visible(False)
    ax.set_xlabel('ケース数（計 257。関係テスト 25/25 合格は別）')
    ax.set_title('--suite full（基準: 古い master 基準 3b2c8db8）', fontsize=14)
    fig.text(0.5, -0.04, '説明済 4 = dftd2 キーワードが基準に無い 2 件 + 基準との 1×10$^{-10}$ 差 2 件（修正前 HEAD ビルドとはビット同一）',
             ha='center', fontsize=11)
    save(fig, 'S11_regression.png')


def s12():
    rows = [('DR-005 1 診断', 2), ('2a numpy 参照', 2), ('2b GPU 電荷・BEC', 2), ('2c 力・ビリアル・電場', 2),
            ('3a 学習側 nep', 2), ('3b replica コード', 1), ('3b replica GPU 検証', 0), ('4 文書（英・日）', 2),
            ('4 全回帰 --suite full', 0), ('5a 表現力の確認（依頼元）', 1), ('5b 本学習（依頼元）', 1),
            ('直接 P 出力', 0), ('DR-006 修正・試験・文書', 2), ('DR-006 HEAD 基準の全回帰・HIP', 0)]
    col = {2: '#55a868', 1: '#f0c14b', 0: '#c44e52'}; txt = {2: '完了', 1: '一部/進行中', 0: '未実施'}
    fig, ax = plt.subplots(figsize=(9.5, 6.4))
    for i, (r, s) in enumerate(rows[::-1]):
        ax.barh(i, 1, color=col[s]); ax.text(0.5, i, txt[s], ha='center', va='center', fontsize=12)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r for r, _ in rows[::-1]], fontsize=13)
    ax.set_xticks([]); ax.spines['bottom'].set_visible(False); ax.spines['left'].set_visible(False)
    ax.set_title('作業の進捗（2026-11-11 時点）', fontsize=15)
    save(fig, 'S12_status.png')


s01a(); s01b(); s02(); s03(); s04(); s05('a'); s05('b'); s06('a'); s06('b'); s07(); s08(); s09(); s10(); s11(); s12()
