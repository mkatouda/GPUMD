"""Infographic: workflow of the DFT-D3 GPU optimization (DR-001, DES-0001).
All numbers are transcriptions from docs_ja/dftd (05_implementation, 06_verification) and specs (DES-0001, VER-0002/0004/0012-0020).
Nothing is newly measured.  使い方: python make_d3_workflow.py   (出力: ../figures/fig13_d3_workflow.png)
(スライド用の縦型は docs_ja/slides/scripts/make_workflow_slide.py)"""
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Polygon

plt.rcParams.update({'font.family': 'Noto Sans CJK JP'})
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures', 'fig13_d3_workflow.png')
DPI = 150
DARK, GRAY = '#1F2937', '#6B7280'
BLUE, LBLUE = '#1F5FA8', '#E6F0FB'
GREEN, LGREEN = '#1E8449', '#E4F4EA'
AMBER, LAMBER = '#B9770E', '#FCF1DC'
SLATE, LSLATE = '#5D6D7E', '#EEF1F4'
RED = '#C0392B'

fig = plt.figure(figsize=(14, 7.9))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 140); ax.set_ylim(0, 79); ax.axis('off')


def box(x, y, w, h, fc, ec='none', r=1.0, lw=1.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}', fc=fc, ec=ec, lw=lw))


def wrap(s, width, fs=9.6):
    """wrap by physical width in axis units (1 unit = 0.1 in); CJK char = fs/72*10, ASCII = 0.55 of it.
    Runs of ASCII / digits / arrows are kept together (no break inside a number or a name)."""
    full = fs / 72 * 10 * 1.04
    def cw(ch):
        return full * (0.55 if ord(ch) < 0x300 or ch in '→～' and False else 1.0)
    tokens, cur, kind = [], '', None
    for ch in s:
        k = 'a' if (ord(ch) < 0x300 and ch != ' ') or ch in '→' else 's' if ch == ' ' else 'c'
        if k == 'c' or k != kind:
            if cur:
                tokens.append(cur)
            cur = ''
        cur += ch; kind = k
    if cur:
        tokens.append(cur)
    lines, line, w = [], '', 0.0
    for t in tokens:
        tw = sum(cw(c) for c in t)
        if w + tw > width and line and t[0] not in '）。、」)':
            lines.append(line); line, w = '', 0.0
            if t == ' ':
                continue
        line += t; w += tw
    if line:
        lines.append(line)
    return lines


def bullets(x, y, items, width, fs=10, lh=1.95, color=DARK):
    """items: list of (text, bold?) ; returns y after the last line"""
    for it in items:
        t, b = (it, False) if isinstance(it, str) else it
        ls = wrap(t, width, fs)
        for i, l in enumerate(ls):
            ax.text(x + (0 if i == 0 else 1.2), y, ('・' if i == 0 else '') + l, fontsize=fs, color=color,
                    va='top', ha='left', weight='bold' if b else 'normal')
            y -= lh
        y -= 0.45
    return y


# ---------------- title
ax.add_patch(Rectangle((0, 69.5), 140, 9.5, fc=DARK))
ax.text(70, 75.7, 'DFT-D3 の GPU 高速化：作業の流れ', ha='center', va='center', fontsize=25, color='white', weight='bold')
ax.text(70, 71.8, '計測で原因を絞る → 段階ごとに数値ゲートで採否 → 最後に精度・回帰・保存則を確認　（結果は変えず、時間を短くする）',
        ha='center', va='center', fontsize=12.5, color='#D1D5DB')

# ---------------- five stages
cols = [
    ('1', '計測と基準づくり', 'DR-001・VER-0001/0002', BLUE, LBLUE, [
        ('Phase 0：nsys でカーネル別に計測', True),
        '原子数 8 倍で時間は 1.8 倍：GPU が埋まらない',
        'D3 は 1 ステップの 56〜96 %',
        '大箱は力カーネルが 80〜87 %',
        '小箱は近傍リスト構築が 44〜65 %',
        ('Phase 1：基準を先に決める', True),
        '受入基準 AC-1〜6、simple-dftd3 の参照値試験',
    ]),
    ('2', '主な律速の解消', 'Phase 3・4　VER-0004', GREEN, LGREEN, [
        ('大箱（Phase 3）', True),
        '原子あたり複数スレッド＋固定順の木縮約（atomic なし）',
        'ゲート：力カーネル 2 倍以上',
        '→ Cu 4,000：7.1 倍、水 5,184：4.3 倍',
        ('小箱（Phase 4）', True),
        '近傍リストを原子ごとのブロックで構築',
        'ゲート：2 倍以上',
        '→ 水 648：約 7.2 倍',
    ]),
    ('3', '追加の最適化', 'Phase 7　VER-0012〜0016', GREEN, LGREEN, [
        ('Stage 1：まず診断（コード変更なし）', True),
        ('Stage 2：候補ごとにゲート', True),
        '(i) 小箱の複数スレッド：水 81 で 15.1 倍　採用',
        '(vi) CN リスト共有：extra 0.760→0.200 ms　採用',
        '(v) C6 因数分解：1.10 倍　ゲートを変更して採用',
        '(vii) float 前段フィルタ：0.93 倍で遅化　不採用',
    ]),
    ('4', '最終検証', 'Stage 3　VER-0017〜0020', BLUE, LBLUE, [
        'AC-1 精度：simple-dftd3 と 7 ケースで一致（極端な配位数は対象外）',
        'AC-2 回帰：全回帰 216 合格・41 失敗（既知 37＋説明できる 4）',
        'AC-3 NVE：基準と雑音の範囲で区別不能',
        '独立レビュー：BLOCKER・MAJOR なし',
        '決定的（固定順の縮約）',
    ]),
    ('5', '結果と判断', 'slope 法　VER-0018', AMBER, LAMBER, [
        ('master → 最適化版（D3 extra）', True),
        '水 81：16.2 倍（2.456→0.152 ms）',
        '水 648：9.6 倍、Cu：6.5 倍',
        '水 5,184：5.4 倍',
        '水 41,472：1.36 倍（8.398→6.193 ms）',
        ('目標 6.0 ms は未達で、取り下げ', True),
        '遅くなった系はなし',
    ]),
]
X0, W, GAP = 2.0, 25.2, 2.5
Y_TOP, Y_BOT = 67.0, 35.0
for i, (num, name, src, fc, bg, items) in enumerate(cols):
    x = X0 + i * (W + GAP)
    box(x, Y_BOT, W, Y_TOP - Y_BOT, bg, fc, lw=1.6)
    box(x, Y_TOP - 6.2, W, 6.2, fc, r=1.0)
    ax.add_patch(Rectangle((x, Y_TOP - 6.2), W, 2.5, fc=fc, ec='none'))
    ax.text(x + 1.6, Y_TOP - 3.1, num, fontsize=17, color='white', weight='bold', va='center', ha='left')
    ax.text(x + 4.6, Y_TOP - 2.6, name, fontsize=12.5, color='white', weight='bold', va='center', ha='left')
    ax.text(x + 4.6, Y_TOP - 5.0, src, fontsize=8.5, color='white', va='center', ha='left')
    bullets(x + 1.2, Y_TOP - 7.6, items, W - 3.0, fs=9.6, lh=1.85)
    if i < len(cols) - 1:
        ax.add_patch(FancyArrowPatch((x + W + 0.15, 50), (x + W + GAP - 0.15, 50),
                                     arrowstyle='simple,head_width=6,head_length=2.2,tail_width=2.6', fc=GRAY, ec='none'))

# ---------------- lane: held / dropped / rejected
LY, LH = 17.2, 13.6
ax.text(2.2, LY + LH + 1.3, '保留・脱落・不採用になった案（理由つき）', fontsize=11.5, weight='bold', color=DARK, va='center')
ax.text(46, LY + LH + 1.3, '色の意味：', fontsize=9.5, color=GRAY, va='center')
for lx, lc, lt in [(54, GREEN, '採用'), (62, AMBER, '保留・条件変更'), (78, SLATE, '不採用・脱落')]:
    ax.add_patch(Rectangle((lx, LY + LH + 0.6), 1.6, 1.4, fc=lc, ec='none'))
    ax.text(lx + 2.3, LY + LH + 1.3, lt, fontsize=9.5, color=GRAY, va='center')
lane = [
    (AMBER, LAMBER, 'Phase 2 C6 因数分解の試作', 'Cu 4,000 で 1.04 倍（ゲート 1.2 倍）。大箱がスレッド数と走査に律速されていた。保留'),
    (AMBER, LAMBER, 'Phase 2′ 再診断', '力カーネルに占める C6 の割合 Cu 13.8 %（必要 16.7 %）。ゲートを変えて (v) で再開'),
    (SLATE, LSLATE, '(vii) float 前段フィルタ', '約 22 % が受理され、ほぼ全ワープが倍精度の経路を通る。0.93 倍で遅化。変更は残さず'),
    (AMBER, LAMBER, '(ii) 小箱のセルリスト', '小箱は AC-4 の目標を (i) で達成済み。「AC-4 に必要ない」として保留'),
    (SLATE, LSLATE, 'Phase 5 半リスト・別ストリーム', '条件付きの案。実施した記録なし（半リストは atomic が必要で、決定性を失う）'),
]
LW = (136 - 4 * 1.5) / 5
for i, (ec, bg, head, body) in enumerate(lane):
    x = 2.0 + i * (LW + 1.5)
    box(x, LY, LW, LH, bg, ec, lw=1.3)
    ax.add_patch(Rectangle((x, LY + LH - 0.9), LW, 0.9, fc='none', ec='none'))
    ax.text(x + 1.0, LY + LH - 1.1, head, fontsize=9.6, weight='bold', color=ec, va='top')
    yy = LY + LH - 3.9
    for l in wrap(body, LW - 2.0, 8.8):
        ax.text(x + 1.0, yy, l, fontsize=8.8, color=DARK, va='top'); yy -= 1.65

# ---------------- bottom: the pattern of every stage
BY = 2.4
box(2, BY, 136, 11.6, '#F9FAFB', '#9CA3AF', lw=1.2)
ax.text(4.2, BY + 10.1, '各段階の型', fontsize=12, weight='bold', color=DARK, va='center')
steps = [
    ('最も安い診断', 'コード変更なし。\n結果が悪ければ中止'),
    ('数値ゲート', '事前に決めた倍率・\n許容値で判定'),
    ('実装', 'worktree で隔離。\n差分を確認して採用'),
    ('精度・決定性・回帰', 'simple-dftd3 と比較。\n許容値は緩めない'),
    ('独立レビュー', '別系統の\nレビュー担当'),
    ('採否の判断', '採用・保留・\n取り下げを決める'),
]
SW = 19.4
for i, (h, b) in enumerate(steps):
    x = 4.0 + i * (SW + 2.4)
    last = i == len(steps) - 1
    box(x, BY + 0.6, SW, 7.2, '#FDECEA' if last else 'white', RED if last else BLUE, lw=1.3)
    ax.text(x + SW / 2, BY + 6.6, h, fontsize=10.2, weight='bold', color=RED if last else BLUE, ha='center', va='center')
    ax.text(x + SW / 2, BY + 3.4, b, fontsize=8.8, color=DARK, ha='center', va='center', linespacing=1.25)
    if i < len(steps) - 1:
        ax.add_patch(FancyArrowPatch((x + SW + 0.2, BY + 4.2), (x + SW + 2.2, BY + 4.2),
                                     arrowstyle='simple,head_width=2.6,head_length=1.4,tail_width=0.9', fc=GRAY, ec='none'))
ax.text(138.5, 0.3, '値はすべて記録（DES-0001、VER-0001〜0020、docs_ja/dftd）の転記。測定環境：RTX A6000（sm_86）、CUDA 12.3、NEP89、dftd3 pbe 12 6',
        fontsize=7.6, color=GRAY, ha='right', va='bottom')
fig.savefig(OUT, dpi=DPI)
print('saved', OUT)
