"""N7: D3 高速化の作業の流れ（スライド用、縦型）。値はすべて docs_ja/dftd と specs（DES-0001、VER-0001〜0020）の転記で、
新たに測定した値はない。全体図（横型、保留・不採用案つき）は docs_ja/workreport/scripts/make_d3_workflow.py。
使い方: python make_workflow_slide.py <出力ディレクトリ>   （出力: N7_d3_workflow.png）"""
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({'font.family': 'Noto Sans CJK JP'})
OUT = os.path.join(sys.argv[1] if len(sys.argv) > 1 else '.', 'N7_d3_workflow.png')
DARK, GRAY, RED = '#1F2937', '#6B7280', '#C0392B'
BLUE, LBLUE = '#1F5FA8', '#E6F0FB'
GREEN, LGREEN = '#1E8449', '#E4F4EA'
AMBER, LAMBER = '#B9770E', '#FCF1DC'

fig = plt.figure(figsize=(9.0, 5.8))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 90); ax.set_ylim(0, 58); ax.axis('off')


def box(x, y, w, h, fc, ec='none', r=0.9, lw=1.3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}', fc=fc, ec=ec, lw=lw))


def wrap(s, width, fs):
    """physical width in axis units (1 unit = 0.1 in); ASCII runs are not split"""
    full = fs / 72 * 10 * 1.04
    tokens, cur, kind = [], '', None
    for ch in s:
        k = 'a' if (ord(ch) < 0x300 and ch != ' ') or ch == '→' else 's' if ch == ' ' else 'c'
        if k == 'c' or k != kind:
            if cur:
                tokens.append(cur)
            cur = ''
        cur += ch; kind = k
    if cur:
        tokens.append(cur)
    lines, line, w = [], '', 0.0
    for t in tokens:
        tw = sum(full * (0.55 if ord(c) < 0x300 else 1.0) for c in t)
        if w + tw > width and line and t[0] not in '）。、」)':
            lines.append(line); line, w = '', 0.0
            if t == ' ':
                continue
        line += t; w += tw
    if line:
        lines.append(line)
    return lines


def para(x, y, s, width, fs, color=DARK, weight='normal', lh=2.15):
    for l in wrap(s, width, fs):
        ax.text(x, y, l, fontsize=fs, color=color, weight=weight, va='top', ha='left')
        y -= lh
    return y


rows = [
    ('1', '計測と基準づくり', 'Phase 0・1　VER-0001/0002', BLUE, LBLUE,
     'nsys で原因を絞る。D3 は 1 ステップの 56〜96 %。大箱は力カーネルが 80〜87 %、小箱は近傍リスト構築が 44〜65 %',
     '受入基準 AC-1〜6 と参照値の試験を先に決める', ''),
    ('2', '主な律速の解消', 'Phase 3・4　VER-0004', GREEN, LGREEN,
     '大箱：原子あたり複数スレッドと固定順の木縮約。小箱：近傍リストをブロック／原子で構築',
     'ゲート 2 倍以上。大箱 Cu 7.1 倍、水 5,184 4.3 倍。小箱 約 7.2 倍', ''),
    ('3', '追加の最適化', 'Phase 7　VER-0012〜0016', GREEN, LGREEN,
     '診断してから、候補ごとにゲート。採用：小箱の複数スレッド（水 81 で 15.1 倍）、CN リスト共有、C6 因数分解',
     'float 前段フィルタは 0.93 倍で不採用。C6 因数分解はゲートを変更して採用', ''),
    ('4', '最終検証', 'Stage 3　VER-0017〜0020', BLUE, LBLUE,
     '精度は simple-dftd3 と 7 ケースで一致。全回帰 216 合格・41 失敗（既知 37＋説明できる 4）。NVE は雑音の範囲で同等',
     '独立レビューで BLOCKER・MAJOR なし。結果は決定的', ''),
    ('5', '結果と判断', 'slope 法　VER-0018', AMBER, LAMBER,
     '水 81：16.2 倍、水 648：9.6 倍、Cu：6.5 倍、水 5,184：5.4 倍、水 41,472：1.36 倍（遅くなった系なし）',
     '', '水 41,472 の目標 6.0 ms は未達（6.19 ms）で、取り下げ'),
]
RH, GAP, Y0 = 9.0, 1.05, 57.4
CHIP_W, MID_X, MID_W, R_X, R_W = 20.5, 22.2, 40.5, 64.2, 25.0
for i, (num, name, src, fc, bg, mid, right, red) in enumerate(rows):
    y = Y0 - RH - i * (RH + GAP)
    box(0.4, y, 89.2, RH, bg, fc)
    box(0.4, y, CHIP_W, RH, fc)
    ax.text(2.0, y + RH / 2 + 0.2, num, fontsize=20, weight='bold', color='white', va='center', ha='left')
    ax.text(5.0, y + RH - 2.9, name, fontsize=11.5, weight='bold', color='white', va='center', ha='left')
    ax.text(5.0, y + 2.4, src, fontsize=8, color='white', va='center', ha='left')
    ty = y + RH - 1.2
    para(MID_X + 0.8, ty, mid, MID_W - 1.2, 10, lh=2.2)
    ry = y + RH - 1.2
    ax.plot([R_X - 0.7, R_X - 0.7], [y + 1.0, y + RH - 1.0], color=fc, lw=0.8)
    if right:
        ry = para(R_X, ry, right, R_W - 0.8, 10, lh=2.2)
    if red:
        para(R_X, ry, red, R_W - 0.8, 10, color=RED, weight='bold', lh=2.2)
    if i < len(rows) - 1:
        ax.add_patch(FancyArrowPatch((CHIP_W / 2 + 0.4, y - 0.05), (CHIP_W / 2 + 0.4, y - GAP + 0.05),
                                     arrowstyle='simple,head_width=1.7,head_length=0.8,tail_width=0.7', fc=GRAY, ec='none'))

# bottom band: the pattern repeated in every stage
box(0.4, 0.3, 89.2, 5.1, '#F9FAFB', '#9CA3AF')
ax.text(45, 3.9, '各段階の型：最も安い診断 → 数値ゲート → 実装（worktree）→ 精度・決定性・回帰 → 独立レビュー → 採否',
        fontsize=9.6, ha='center', va='center', color=DARK, weight='bold')
ax.text(45, 1.75, '許容値は緩めない', fontsize=9.6, ha='center', va='center', color=RED)
fig.savefig(OUT, dpi=200)
print('saved', OUT)
