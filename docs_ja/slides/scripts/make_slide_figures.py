"""Slide figures N1-N4 (D2/D3). Values transcribed from docs_ja/dftd/03_benchmark.md; nothing re-measured."""
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False,
                     'font.size': 14, 'savefig.dpi': 200, 'axes.spines.top': False,
                     'axes.spines.right': False})
OUT = sys.argv[1]
BLUE, ORANGE, GRAY, GREEN, RED = '#4472C4', '#ED7D31', '#A5A5A5', '#70AD47', '#C00000'

def n1():  # 03_benchmark 3.6.1: recorded per-phase speedups (metric differs per row)
    rows = [  # (label, factor, group)
        ('P4 小箱の近傍リスト構築 水648 [カーネル]', 7.2, 'P4'),
        ('P3 力カーネル Cu4,000 [カーネル]', 7.1, 'P3'),
        ('P3 力カーネル 水5,184 [カーネル]', 4.3, 'P3'),
        ('P3 力カーネル 水41,472 [カーネル]', 1.20, 'P3'),
        ('P7(i) 小箱 水81 [D3合計]', 15.1, 'P7i'),
        ('P7(i) 小箱 水648 [D3合計]', 3.84, 'P7i'),
        ('P7(i) 小箱 水2,187 [D3合計]', 3.0, 'P7i'),
        ('P7(i) 小箱 水8,748 [D3合計]', 1.22, 'P7i'),
        ('P7(v) C6因子化 水5,184 [D3合計]', 1.21, 'P7v'),
        ('P7(v) C6因子化 水41,472 [D3合計]', 1.10, 'P7v'),
        ('P7(vi) 走査共有 水41,472 extra [カーネル]', 3.8, 'P7vi'),
        ('P7(vi) 走査共有 水41,472 [D3合計]', 1.08, 'P7vi'),
    ]
    col = {'P4': BLUE, 'P3': GREEN, 'P7i': ORANGE, 'P7v': GRAY, 'P7vi': '#7F6000'}
    fig, ax = plt.subplots(figsize=(10, 6.2))
    y = np.arange(len(rows))[::-1]
    ax.barh(y, [r[1] for r in rows], color=[col[r[2]] for r in rows])
    for yi, r in zip(y, rows):
        ax.text(r[1] * 1.08, yi, f'{r[1]:g} 倍', va='center', fontsize=12)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=11)
    ax.set_xscale('log'); ax.set_xlim(1, 40); ax.set_xticks([1, 2, 5, 10, 20]); ax.set_xticklabels(['1', '2', '5', '10', '20']); ax.axvline(1, color='k', lw=0.8)
    ax.set_xlabel('段階の前後の比（記録値、nsys。指標は各行の [ ] を参照）')
    ax.set_title('D3 高速化の段階別の効果', fontsize=16)
    fig.tight_layout(); fig.savefig(f'{OUT}/N1_d3_stages.png'); plt.close(fig)

def n2():  # VER-0018 slope, ms/step
    sysn = ['水81\n小箱', '水648\n小箱', '水2,187\n小箱', '水5,184\n大箱', '水8,748\n小箱',
            '水41,472\n大箱', 'Cu4,000\n大箱', 'クラスター972\n大箱']
    base = [2.456, 4.417, 3.266, 4.568, 11.782, 8.398, 3.612, 1.717]
    opt = [0.152, 0.459, 0.750, 0.846, 8.274, 6.193, 0.554, 1.216]
    ratio = ['16.2', '9.6', '4.4', '5.4', '1.4', '1.36', '6.5', '1.4']
    x = np.arange(len(sysn)); w = 0.38
    fig, ax = plt.subplots(figsize=(11, 5.6))
    ax.bar(x - w/2, base, w, color=GRAY, label='master 基準 (3b2c8db8)')
    ax.bar(x + w/2, opt, w, color=BLUE, label='最適化版')
    for xi, b, o, r in zip(x, base, opt, ratio):
        ax.text(xi + w/2, o + 0.25, f'{r}倍', ha='center', fontsize=12, color=BLUE)
    ax.hlines(6.0, x[5] - 0.45, x[5] + 0.45, color=RED, ls='--', lw=2)
    ax.annotate('目標 6.0 ms\n(slope で未達、取り下げ)', (x[5] + 0.45, 6.0), xytext=(x[6] - 0.45, 8.6),
                fontsize=12, color=RED, arrowprops=dict(arrowstyle='->', color=RED))
    ax.set_xticks(x); ax.set_xticklabels(sysn, fontsize=11)
    ax.set_ylabel('D3 の extra [ms/ステップ]（slope、交互 3 回の中央値）')
    ax.set_title('D3 の計算時間: master 基準 対 最適化版（RTX A6000）', fontsize=16)
    ax.legend(frameon=False, loc='upper left')
    fig.tight_layout(); fig.savefig(f'{OUT}/N2_d3_slope.png'); plt.close(fig)

def n3():  # VER-0026 5 (R3 before/after); D3 extra measured in the same run
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 5.2), gridspec_kw={'width_ratios': [1, 1.5]})
    s = ['水41,472', 'PTFE 40,500']
    kb, ka = [6.41, 6.31], [1.41, 1.31]
    x = np.arange(2); w = 0.38
    ax[0].bar(x - w/2, kb, w, color=GRAY, label='R3 前')
    ax[0].bar(x + w/2, ka, w, color=BLUE, label='R3 後')
    for xi, a, p in zip(x, ka, ['−78 %', '−79 %']):
        ax[0].text(xi + w/2, a + 0.15, p, ha='center', fontsize=12, color=BLUE)
    ax[0].set_xticks(x); ax[0].set_xticklabels(s); ax[0].set_ylabel('D2 ペアカーネル [ms/ステップ]（nsys）')
    ax[0].set_title('(a) カーネル時間', fontsize=14); ax[0].set_ylim(0, 8.2); ax[0].legend(frameon=False, ncol=2, loc='upper center')
    s2 = ['水5,184', '水41,472', 'PTFE 40,500']
    eb, ea, d3 = [0.95, 7.06, 6.34], [0.54, 1.62, 1.39], [0.77, 6.66, 5.81]
    x = np.arange(3); w = 0.27
    ax[1].bar(x - w, eb, w, color=GRAY, label='D2 extra（R3 前）')
    ax[1].bar(x, ea, w, color=BLUE, label='D2 extra（R3 後）')
    ax[1].bar(x + w, d3, w, color=ORANGE, label='D3 extra（同じ測定）')
    ax[1].set_xticks(x); ax[1].set_xticklabels(s2); ax[1].set_ylabel('extra [ms/ステップ]（slope）')
    ax[1].set_title('(b) extra（補正あり − NEP のみ）', fontsize=14); ax[1].set_ylim(0, 9.5); ax[1].legend(frameon=False, fontsize=11, loc='upper left')
    fig.suptitle('D2 の補償付き float 累算（R3）の効果（NEP89、RTX A6000）', fontsize=16)
    fig.tight_layout(); fig.savefig(f'{OUT}/N3_d2_r3.png'); plt.close(fig)

def n4():  # VER-0026 5: R3 variants, time vs net-force residual (rc=30)
    pts = [('A7_1 (n=1)', 1.94, 1.6e-14, GREEN), ('A7_4 (n=4) 採用', 1.41, 7.6e-14, BLUE),
           ('A7_8 (n=8)', 1.38, 1.3e-13, GREEN), ('A3b 補償のみ(フラッシュなし)', 1.36, 3.7e-12, RED)]
    fig, ax = plt.subplots(figsize=(10, 5.6))
    for n, t, f, c in pts:
        ax.scatter(t, f, s=140, color=c, zorder=3)
        ax.annotate(n, (t, f), textcoords='offset points', xytext=(10, 6), fontsize=12)
    ax.text(1.62, 2.5e-15, '旧カーネルの誤差は 1.5×$10^{-14}$（範囲外）', fontsize=11)
    ax.axhline(1e-12, color=RED, ls='--'); ax.text(1.9, 1.4e-12, '試験の上限 $10^{-12}$', color=RED, fontsize=12)
    ax.set_yscale('log'); ax.set_ylim(1e-15, 1e-11); ax.set_xlim(0.8, 2.3)
    ax.set_xlabel('ペアカーネル時間 [ms]')
    ax.set_ylabel('|ΣF| / max|F|（rc = 30 Å）')
    ax.set_title('R3 の変種: 速度と正味の力の誤差', fontsize=16)
    ax.text(0.82, 6e-15, '時間は N=41,472 のカーネルと推測（記録に系の明記なし、未確認）', fontsize=10)
    fig.tight_layout(); fig.savefig(f'{OUT}/N4_d2_variants.png'); plt.close(fig)

def n5():  # VER-0026 5: R3-after D2 vs D3 extra, same measurement (slope, NEP89, 5 alternating repeats)
    s = ['水 5,184', '水 41,472', 'PTFE 40,500']
    d2, d3, r = [0.54, 1.62, 1.39], [0.77, 6.66, 5.81], ['0.70', '0.24', '0.24']
    x = np.arange(3); w = 0.36
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.bar(x - w/2, d3, w, color=ORANGE, label='D3 extra（dftd3 pbe 12 6）')
    ax.bar(x + w/2, d2, w, color=BLUE, label='D2 extra（R3 後、rc 12 Å）')
    for xi, a, b, q in zip(x, d3, d2, r):
        ax.text(xi + w/2, b + 0.15, f'{b:g}\nD2/D3 = {q}', ha='center', fontsize=13, color=BLUE)
        ax.text(xi - w/2, a + 0.15, f'{a:g}', ha='center', fontsize=12, color=ORANGE)
    ax.set_xticks(x); ax.set_xticklabels(s); ax.set_ylim(0, 8)
    ax.set_ylabel('extra（補正あり − NEP のみ）[ms/ステップ]')
    ax.set_title('D2（R3 後）と D3 の計算時間の比較（NEP89、RTX A6000）', fontsize=16)
    ax.legend(frameon=False, loc='upper left', fontsize=12)
    fig.text(0.99, 0.01, 'slope 法、交互 5 回の中央値。この 3 点のみの測定（VER-0026）', ha='right', fontsize=10)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(f'{OUT}/N5_d2_vs_d3.png'); plt.close(fig)

def n6():  # 1 + extra/NEP-only. extra: VER-0026 5 (R3 after, D3 same run); NEP-only: VER-0021 (different session) -> derived, reference only
    s = ['水 5,184', '水 41,472']
    nep = [1.784, 6.729]
    d3x, d2x = [0.77, 6.66], [0.54, 1.62]
    d3 = [1 + a / b for a, b in zip(d3x, nep)]
    d2 = [1 + a / b for a, b in zip(d2x, nep)]
    x = np.arange(2); w = 0.34
    fig, ax = plt.subplots(figsize=(9, 5.6))
    ax.bar(x - w/2, d3, w, color=ORANGE, label='NEP + D3（dftd3 pbe 12 6）')
    ax.bar(x + w/2, d2, w, color=BLUE, label='NEP + D2（R3 後、rc 12 Å）')
    ax.axhline(1, color='k', lw=1)
    for xi, a, b in zip(x, d3, d2):
        ax.text(xi - w/2, a + 0.03, f'{a:.2f} 倍', ha='center', fontsize=14, color=ORANGE)
        ax.text(xi + w/2, b + 0.03, f'{b:.2f} 倍', ha='center', fontsize=14, color=BLUE)
    ax.set_xticks(x); ax.set_xticklabels(s); ax.set_ylim(0, 2.9)
    ax.set_ylabel('1 ステップの時間 / NEP のみ（= 1 + extra / NEP）')
    ax.set_title('補正を付けたときの 1 ステップの時間（NEP 単独に対する倍率）', fontsize=15)
    ax.legend(frameon=False, loc='upper right', fontsize=12)
    fig.text(0.99, 0.01, '導出値: extra は VER-0026、NEP のみ（1.784 / 6.729 ms）は VER-0021（別セッション）。PTFE は NEP のみの記録なし',
             ha='right', fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(f'{OUT}/N6_ratio_to_nep.png'); plt.close(fig)

for f in (n1, n2, n3, n4, n5, n6): f()
