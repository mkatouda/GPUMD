"""Main deck (build_slides.py, 26 slides) + appendix/reference slides + internal links.
Appendix content is transcribed from docs_ja/dftd and docs_ja/qnep; nothing is newly measured.
Usage: [RIST_TEMPLATE=/path/RIST-template.pptx] python build_all.py <figures_dir> <out.pptx>"""
import sys
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

import os
_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'build_slides.py')).read()
exec(compile(_src.split('prs.save(OUTFILE)')[0], 'build_slides.py', 'exec'))

MAIN_SLIDES = list(prs.slides)
APP = {}
L_SECTION = prs.slide_layouts[2]


# ---------- helpers for appendix ----------
def textbox(slide, x, y, w, h, lines, size, color=None, bullet=True):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    for i, t in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        r = p.add_run()
        r.text = ('• ' + t) if bullet else t
        r.font.size = Pt(size)
        if color:
            r.font.color.rgb = color
        p.space_after = Pt(3)
    return tb


def table(slide, x, y, rows, colw, size):
    nr, nc = len(rows), len(rows[0])
    heights = []
    for r in rows:
        mx = 1
        for t, w in zip(r, colw):
            cpl = max(1.0, (w - 0.12) * 72 / size)
            mx = max(mx, math.ceil(width_chars(str(t)) / cpl))
        heights.append(mx * size * 1.25 / 72 + 0.09)
    shp = slide.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(sum(colw)), Inches(sum(heights)))
    tbl = shp.table
    for i, w in enumerate(colw):
        tbl.columns[i].width = Inches(w)
    for ri, r in enumerate(rows):
        tbl.rows[ri].height = Inches(heights[ri])
        for ci, t in enumerate(r):
            c = tbl.cell(ri, ci)
            c.margin_left = c.margin_right = Inches(0.05)
            c.margin_top = c.margin_bottom = Inches(0.03)
            c.text_frame.word_wrap = True
            c.text_frame.text = str(t)
            for p in c.text_frame.paragraphs:
                for run in p.runs:
                    run.font.size = Pt(size)
                    run.font.bold = (ri == 0)
    return sum(heights)


def _tbl_h(rows, colw, size):
    tot = 0
    for r in rows:
        mx = 1
        for t, w in zip(r, colw):
            cpl = max(1.0, (w - 0.12) * 72 / size)
            mx = max(mx, math.ceil(width_chars(str(t)) / cpl))
        tot += mx * size * 1.25 / 72 + 0.09
    return tot


def _block_h(b, f):
    kind = b[0]
    if kind == 'eq':
        im = Image.open(f'{FIG}/{b[1]}')
        w = min(8.8, im.width / 220 * (b[2] if len(b) > 2 else 1.0))
        return w * im.height / im.width + 0.1
    if kind == 'img':
        return b[2] + 0.1
    if kind == 'bul':
        return est_height([(t, 0) for t in b[1]], round(b[2] * f), 8.6) + 0.1
    if kind == 'txt':
        return est_height([(b[1], 0)], round(b[2] * f), 8.6) + 0.05
    if kind == 'tbl':
        return _tbl_h(b[1], b[2], round(b[3] * f)) + 0.12


def app(key, title, blocks, src, notes=''):
    """blocks: ('eq', image) | ('bul', [text], size) | ('tbl', rows, colw, size) | ('img', image, h) | ('txt', text, size)"""
    no = f'app {key}'
    f = 1.0
    for cand in (1.4, 1.35, 1.3, 1.25, 1.2, 1.15, 1.1, 1.05):
        if 0.85 + sum(_block_h(b, cand) for b in blocks) <= 6.5:
            f = cand
            break
    slide = prs.slides.add_slide(L_TONLY)
    set_title(slide, title, no)
    y = 0.85
    for b in blocks:
        kind = b[0]
        if kind == 'eq':
            im = Image.open(f'{FIG}/{b[1]}')
            w = min(8.8, im.width / 220 * (b[2] if len(b) > 2 else 1.0))
            h = w * im.height / im.width
            slide.shapes.add_picture(f'{FIG}/{b[1]}', Inches(0.5 + (9.0 - w) / 2), Inches(y), Inches(w), Inches(h))
            y += h + 0.1
        elif kind == 'img':
            im = Image.open(f'{FIG}/{b[1]}')
            h = b[2]
            w = h * im.width / im.height
            if w > 9.0:
                w, h = 9.0, 9.0 * im.height / im.width
            slide.shapes.add_picture(f'{FIG}/{b[1]}', Inches(0.5 + (9.0 - w) / 2), Inches(y), Inches(w), Inches(h))
            y += h + 0.1
        elif kind == 'bul':
            h = _block_h(b, f)
            textbox(slide, 0.5, y, 9.0, h, b[1], round(b[2] * f))
            y += h
        elif kind == 'txt':
            h = _block_h(b, f)
            textbox(slide, 0.5, y, 9.0, h, [b[1]], round(b[2] * f), bullet=False)
            y += h
        elif kind == 'tbl':
            h = table(slide, 0.5, y, b[1], b[2], round(b[3] * f))
            y += h + 0.12
    if y > 6.55:
        warnings.append(f'{no}: content bottom {y:.2f} in > 6.55 (scale {f})')
    textbox(slide, 0.5, 6.62, 6.4, 0.3, ['出典: ' + src], 9, RGBColor(0x59, 0x59, 0x59), bullet=False)
    add_slide_number(slide, L_TONLY)
    slide.notes_slide.notes_text_frame.text = notes or ('出典: ' + src)
    APP[key] = slide
    SCALES[key] = f
    return slide


SCALES = {}


# ---------- divider ----------
s = prs.slides.add_slide(L_SECTION)
s.shapes.title.text_frame.text = '付録・参考資料'
for r in s.shapes.title.text_frame.paragraphs[0].runs:
    r.font.size = Pt(32)
body = [p for p in s.placeholders if p.placeholder_format.idx == 1][0]
body.text_frame.text = 'A: DFT-D3 / D2（docs_ja/dftd）\nB: charge_transfer qNEP（docs_ja/qnep）\nC: 記録の対応'
for p in body.text_frame.paragraphs:
    for r in p.runs:
        r.font.size = Pt(18)
add_slide_number(s, L_SECTION)
s.notes_slide.notes_text_frame.text = ('本編の各スライド下部の「→ A-n / B-n」から、対応する付録へ移動できる。付録は説明書の記述の転記で、新たな測定・推定は含まない。'
                                       '説明書どうしの不整合は直さずに記録する（例: VER-0035 の表見出しの力のゲート 1e-4 と、人間が決めた A4 の 1e-5）。')

# =============== A: DFT-D3 / D2 ===============
app('A-1', 'A-1 dftd3 / dftd2 の構文と引数', [
    ('tbl', [
        ['キーワード', '引数', '意味・既定値・検査'],
        ['dftd3', 'functional', 'BJ パラメータを選ぶ汎関数名（必須）。NEP の訓練データを作った汎関数に合わせる'],
        ['dftd3', 'potential_cutoff (rc_pot)', 'D3 ポテンシャルのカットオフ [Å]（必須）'],
        ['dftd3', 'coordination_number_cutoff (rc_cn)', '配位数のカットオフ [Å]（必須）。二つの値域はコードで検査されない'],
        ['dftd3', 'damping（任意）', 'bj（既定）または zero（D3(0)）。小文字のみ。トークン数は 4 または 5'],
        ['dftd2', 's6', '大域スケール係数（必須、正の有限値）。汎関数の表はなく、値の選択はユーザーの責任'],
        ['dftd2', 'cutoff', 'カットオフ [Å]。既定 30 Å。200 Å 超は警告のみ'],
        ['dftd2', 'table grimme2006|qe', '既定 grimme2006。qe は QE 7.2 の dftd3_version = 2 と同じパラメータ'],
        ['dftd2', 'trunc sf|plain', '既定 sf（シフト付き力）。table qe または trunc plain では cutoff を明示'],
    ], [1.0, 2.5, 5.5], 11),
    ('bul', ['書式: dftd3 <functional> <rc_pot> <rc_cn> [zero|bj]　　dftd2 <s6> [<cutoff>] [table grimme2006|qe] [trunc sf|plain]',
             'どちらも run.in の任意の位置に書ける。単位: エネルギー eV、力 eV/Å、ビリアル eV'], 11),
], 'docs_ja/dftd 01 章 1.1 節')

app('A-2', 'A-2 D3（BJ ダンピング）の式', [
    ('eq', 'E1_d3_bj.png'),
    ('bul', ['dftd3 pbe: s6 = 1.000、a1 = 0.4289、s8 = 0.7875、a2 = 4.4407 bohr（Å へ変換して保持）',
             '総和は rc_pot 未満のすべてのペア（周期イメージを含む）。打ち切りは鋭く、平滑化はない',
             '三体項（ATM）は含まない。C6 は配位数に依存（A-4）',
             'r2r4 と共有半径の定義はコードにない（文献上の定義は未確認）'], 13),
], 'docs_ja/dftd 02 章 2.2 節（D3:1976、VER-0001）')

app('A-3', 'A-3 D3(0)（ゼロダンピング）の式', [
    ('eq', 'E2_d3_zero.png'),
    ('bul', ['C6、C8、配位数、C6 補間は BJ と共通。違いは減衰関数だけ',
             'R0 は D3 の r0ab（Å、表から読む）で、BJ の √(C8/C6) とは別の量',
             'パラメータ表: def2-QZVP 版、dftd3-lib 3c99ee9 から生成（VER-0023）。PBE は s6 = 1、s8 = 0.722、s_r,6 = 1.217',
             '小箱の配位数だけは double で累積（zero のときのみ。BJ は float のままビット不変）。Z ≤ 86',
             '長さ定数の差で、simple-dftd3 との間にダンピング領域で相対 1.5×10⁻⁶ 程度の差が残る'], 13),
], 'docs_ja/dftd 02 章 2.2 節（DR-003、VER-0023、VER-0024、ADR-0003）')

app('A-4', 'A-4 配位数と C6 の補間（D3）', [
    ('eq', 'E3_cn_c6.png'),
    ('bul', ['k1 = 16、k3 = 4 はコードに固定。参照配位数は元素ごとに最大 5 個、参照 C6 は最大 5×5',
             '重みは原子の配位数の確定後に原子ごとに一度だけ計算して保持する',
             'float で W < 10⁻³⁰ なら最後の参照同士の C6 に置換し、微分は 0（upstream master も同じ挙動）',
             '極端な配位数（密な H や Cu のクラスタ）では simple-dftd3 と大きくずれる（UPSTREAM-0001）'], 12),
], 'docs_ja/dftd 02 章 2.3〜2.4 節')

app('A-5', 'A-5 D2 の式と table qe', [
    ('eq', 'E4_d2.png'),
    ('bul', ['元素ごとの C6、R0 は内蔵表（Z ≤ 86）。s6 は表に含まれずユーザーが指定。d = 20',
             'table qe: C6 は QE 7.2 の c6ab(Zi, Zj, 1, 1, 1)（両原子の最初の参照点）。D2 の式に D3 の表を使うもので、Grimme 2006 の D2 ではない',
             '定数は QE 7.2 の CODATA 2018（a0 = 0.529177210903 Å、Eh = 27.211386245988 eV）。dftd3para.cuh の float 定数は相対 8.7×10⁻⁷ ずれるので使わない',
             'QE の本来の D2（vdw_corr = grimme-d2）は再現しない。QE 7.2 との一致は相対 1×10⁻⁶ 程度',
             'QE 7.2 のバイナリは実行していない（ソース読解、7.6-dev での実行、依頼元のデータとの比較のみ）'], 12),
], 'docs_ja/dftd 02 章 2.6 節（DR-004、ADR-0004、VER-0025）')

app('A-6', 'A-6 D2 の打ち切り: シフト付き力と plain', [
    ('eq', 'E5_d2_sf.png'),
    ('bul', ['φ(rc)、φ′(rc) はホストで倍精度に計算して float で保持。rc 自身も float に丸め、シフト・判定・セル探索で同じ値を使う',
             'trunc plain はシフトなしの打ち切り（φ(rc) = φ′(rc) = 0 とするだけ）。ペアが rc を横切るたびにエネルギーが |φ(rc)|、力が約 6|φ(rc)|/rc 跳ぶ',
             '例（C-C、12 Å、table qe、s6 = 0.75）: 約 7×10⁻⁶ eV、約 3.7×10⁻⁶ eV/Å。短い NVE（180 原子 PTFE、20 ps、3 シード）で系統的なドリフトは見えず（証明ではない）',
             '連続体近似: 一様密度で、打ち切りなしの和に対し 1 原子あたり ΔE = 7πρ⟨s6C6⟩/(3 rc³) のずれ'], 12),
], 'docs_ja/dftd 02 章 2.6 節、VER-0026 3 節')

app('A-7', 'A-7 定数と単位換算', [
    ('tbl', [
        ['定数', '値', '備考'],
        ['Bohr（Å/bohr）', '0.5291772575069165（float マクロ）', 'CODATA 2018 の 0.529177210903 に対し相対 +8.8×10⁻⁸（導出値）'],
        ['Bohr2', '0.280028569862541', ''],
        ['HartreeBohr6（eV Å⁶ / (Eh a0⁶)）', '0.597527426643772', '定義の出典はコードにない。CODATA 2018 からの 0.5975269043 に対し相対 +8.7×10⁻⁷（UPSTREAM-0002 B）'],
        ['k1（配位数）、k3（重み）', '16、4', 'コードに固定'],
        ['参照 C6 の単位', 'Hartree bohr⁶', '単位コメントはなく、HartreeBohr6 の変換から推定'],
        ['D2 の d', '20', ''],
        ['D2 の表生成時の換算', '27.21138505 eV/Eh、0.52917726 Å/bohr', 'tests_pytest/fixtures/dftd2/generate_params.py'],
    ], [2.6, 2.9, 3.5], 11),
], 'docs_ja/dftd 02 章 2.7 節')

app('A-8', 'A-8 D3 の計算フロー', [
    ('tbl', [
        ['段階', 'カーネル（小箱 / 大箱）', '入力 → 出力'],
        ['1. CN', 'find_dftd3_coordination_number_small_box / _large_box', 'rc_cn 内の近傍 → cn、cn_weight（5 個/原子）、大箱は CN リスト'],
        ['2. 力', 'add_dftd3_force_small_box / find_dftd3_force_large_box', 'cn、cn_weight、rc_pot 内の近傍 → エネルギー、ペア項の力、ビリアル、dc6_sum、dc8_sum'],
        ['3. extra', 'add_dftd3_force_extra_small_box / find_dftd3_force_extra_large_box', 'dc6_sum、dc8_sum、rc_cn 内の近傍 → CN 微分による力とビリアル'],
    ], [1.0, 3.9, 4.1], 11),
    ('bul', ['小箱の前に近傍リストの構築、大箱の前にセルリストの構築が入る',
             '力カーネルは両端の CN を必要とし、extra は力カーネルの dc6_sum / dc8_sum を必要とする。この依存のためカーネルは融合できない（VER-0015）',
             'extra: エネルギーは r だけでなく C6 を通じて CN_i、CN_j にも依存するため、力にはペア項（r への明示依存）のほかに CN 微分項が加わる'], 12),
], 'docs_ja/dftd 05 章 5.1 節')

app('A-9', 'A-9 master と最適化後の構造', [
    ('tbl', [
        ['項目', 'master（upstream 3b2c8db8）', '本フォーク'],
        ['小箱の近傍リスト', '原子あたり 1 スレッド、全 N×像を直列に走査（O(N²)）', '原子あたり 1 ブロック（128 スレッド）、128 候補ずつ走査し共有メモリのスキャンで書き込み位置を決める。順序・値は master と同一'],
        ['小箱の力・extra', '原子あたり 1 スレッド、ペアごとに倍精度 atomicAdd を 15 回', 'num_sub（1〜128）スレッドが 1 原子を分担、固定順序の木縮約、atomicAdd なし'],
        ['大箱', '原子あたり 1 スレッド（累積はレジスタ）', 'num_sub（1〜16）スレッドが近傍セルを分担し、同じ縮約'],
        ['C6 補間', 'ペアごとに最大 25 回の exp', '原子ごとのガウス重みを cn_weight に保存し、ペアは積と和のみ'],
        ['extra（大箱）', 'CN と同じセル走査をもう一度', 'CN カーネルが受理した相手を保存して再利用'],
        ['atomicAdd', 'あり（sm < 60 向けフォールバックも）', 'D3 内になし（フォールバックも削除）'],
    ], [1.5, 3.2, 4.3], 10),
], 'docs_ja/dftd 05 章 5.2 節')

app('A-10', 'A-10 複数スレッド/原子・木縮約と決定性', [
    ('bul', ['num_sub 個（2 の冪）の連続スレッドが 1 原子を担当。部分和を共有メモリに置き、offset を num_sub/2 から 1 まで半分にして足す。足す順序は num_sub だけで決まる',
             '出力要素ごとの書き手は 1 スレッド（sub == 0）で、D3 に atomicAdd はない。num_sub = 1 のときは累積順序が master と同じ',
             '小箱経路は、同じバイナリ・GPU・入力で実行ごとにビット一致。大箱経路はセルリスト構築の整数 atomicAdd でセル内の原子順序がスケジューリング依存となり、float の累積順序が実行ごとに変わり得る',
             '大箱のビット一致テストは 24 原子・1 ワープの系のみ。大きな大箱系のビット一致は主張していない',
             'num_sub は GPU の SM 数から決まる。異なる GPU 間では num_sub が変わり得て、丸めレベルの差が出る（導出）'], 12),
    ('txt', '小箱の力カーネルの時間 [ms]（VER-0013）: N が数千を超えると num_sub を大きくするほど遅くなる', 11),
    ('tbl', [
        ['N', '既定', 'num_sub = 1', '4', '16', '128'],
        ['81', '0.019', '1.099', '0.284', '0.081', '0.019'],
        ['2,187', '0.144', '1.384', '0.381', '0.144', '0.188'],
        ['8,748', '0.442', '1.323', '0.419', '0.952', '0.686'],
    ], [1.5, 1.5, 1.5, 1.5, 1.5, 1.5], 11),
], 'docs_ja/dftd 05 章 5.3 節（VER-0004、VER-0013）')

app('A-11', 'A-11 C6 重みの因数分解と float 累積', [
    ('bul', ['重みは w[k] = exp(−4(CN − CNref[k])²) と因数分解でき、W = (Σw1)(Σw2)。ペアごとの二重ループの exp（最大 25 回）が不要になる',
             'CN カーネルが原子ごとに 5 個の重みを cn_weight に保存（1 原子 20 B。N = 41,472 で約 0.83 MB）。力カーネルは中心原子の重みと dw を 1 回だけ読む',
             'c6 = Z/W、dc6 = dZ/W − c6·dW/W²。両原子の num_cn が 1 のときは c6 = 先頭要素、dc6 = 0 の近道',
             '効果（D3 合計）: 水 5,184 が 1.21 倍、クラスター 972 が 1.23 倍、水 41,472 が 1.10 倍、他は 1.02〜1.08 倍。遅くなる例なし（VER-0014）',
             'float 累積: ペアの和（エネルギー、力、ビリアル、dc6_sum、dc8_sum）はレジスタ内で float。sub == 0 のスレッドが double の原子ごとの値へ加える。AC-1 の許容内を確認',
             'D2 は R3 の前は、ペア項が float で累積は double。R3 以降は補償付き float（A-13）'], 12),
], 'docs_ja/dftd 05 章 5.4 節（VER-0014）')

app('A-12', 'A-12 D2 の実装: セル数、スレッド、float 候補位置', [
    ('bul', ['近傍リストを保存せず、分数座標のセルリストを直接走査。セル数はコスト cost = N·cells_visited/num_cells + 3.0·cells_visited を最小にする組から選ぶ（CELL_VISIT_COST = 3.0。40 点の実測にフィット、rms 対数誤差 0.14）',
             '旧規則 n = floor(2·width/rc) では、水 648（箱 18.6 Å）で rc ≥ 18.6 Å でセル数が 2 から 1 に落ち、16 → 20 Å で時間が 4〜4.8 倍に跳ねた',
             'NUM_SUB（最大 32、2 の冪）スレッドが 1 原子を共有。縮約は固定順序の木で atomicAdd なし。place_atoms がセル内を元の番号の昇順に並べるので決定的'], 12),
    ('txt', '候補位置の float 化（VER-0009 §2）: AC-1 許容値に対する最大誤差の割合', 11),
    ('tbl', [
        ['変形', '割合', '採否'],
        ['候補を double（旧）', '0.03〜0.14', '－'],
        ['絶対座標の float', '0.29〜0.73。原点から +100 Å の系で 2.53（不合格）', '不採用'],
        ['セル中心相対の float', '最大 0.27（ゲート 1/3 以下）。合力は Fmax の 3×10⁻¹⁶〜6×10⁻¹⁵', '採用（ペアカーネル 1.49〜1.86 倍、35 点）'],
    ], [2.6, 4.2, 2.2], 11),
], 'docs_ja/dftd 05 章 5.6.1〜5.6.4 節（VER-0008、VER-0009）')

app('A-13', 'A-13 D2 の R3: 補償付き float 累算', [
    ('bul', ['ペアカーネルは 1 ペアあたりエネルギー 1、力 3、ビリアル 9 の計 13 個の和を作る。R3 の前は double 累算だったが、GA102 では FP64 が FP32 の 1/64',
             '採用: TwoSum で補償した float の和（CompensatedSum）を 4 セルごと（FLUSH_CELLS = 4）に double へ足す。積は __fmul_rn で丸め FMA 縮約を避ける。atomic なし、結果は決定的',
             '採用基準（ADR-0005 項目 6）: N = 41,472 でカーネル時間と D2 の extra がともに 20 % 以上短縮。満たした'], 12),
    ('tbl', [
        ['変種', '正味の力（水、rc = 30 Å）', 'カーネル時間 [ms]'],
        ['旧カーネル（double 累算）', '1.5×10⁻¹⁴', '6.41（水 41,472）'],
        ['TwoSum、フラッシュなし（A3b）→ 棄却', '3.7×10⁻¹²（試験の上限 1×10⁻¹² に不合格）', '1.36'],
        ['毎セルでフラッシュ', '1.6×10⁻¹⁴', '1.94'],
        ['4 セルごと → 採用', '7.6×10⁻¹⁴', '1.41'],
        ['8 セルごと', '1.3×10⁻¹³', '1.38'],
    ], [3.2, 3.6, 2.2], 11),
    ('txt', '注: 変種の時間は VER-0026 5 節の記載で、測定系の明記がない（旧と採用の値は水 41,472 の値と一致）。採用変種の旧カーネルとの差は AC-1 上限の 5×10⁻¹⁰ 以下', 10),
], 'docs_ja/dftd 05 章 5.6.6 節（ADR-0005、VER-0026 5 節）')

app('A-14', 'A-14 D3 の高速化履歴', [
    ('tbl', [
        ['Phase', '内容', '変化（nsys、前 → 後）', '出典'],
        ['0', 'プロファイル（基準 master）', 'D3 は NEP + D3 の GPU 時間の 56〜96 %。基準の D3 合計: 水 81 が 2.55、水 648 が 4.37、水 5,184 が 3.86、水 41,472 が 8.17、Cu が 3.65 ms', 'VER-0002'],
        ['4', '小箱の近傍リスト（原子ごとのブロック）', '水 648 のカーネル 2.79〜2.86 → 0.38〜0.40 ms（約 7.2 倍）。リストは 13 ケースでビット一致', 'VER-0004'],
        ['3', '大箱の複数スレッド/原子＋木縮約', '力カーネル: Cu 3.087 → 0.436（7.1 倍）、水 5,184 3.321 → 0.766（4.3 倍）、水 41,472 6.559 → 5.458（1.20 倍）', 'VER-0004'],
        ['7(i)', '小箱の複数スレッド、atomic なし', 'D3 合計: 水 81 が 1.516 → 0.101（15.1 倍）、水 648 が 3.84 倍、水 2,187 が 3.0 倍、水 8,748 が 1.22 倍', 'VER-0013'],
        ['7(v)', 'C6 重みの因数分解', 'D3 合計: 水 41,472 が 6.931 → 6.316（1.10 倍）、水 5,184 が 1.21 倍', 'VER-0014'],
        ['7(vi)(3)', 'CN と extra の走査共有', 'extra カーネル（水 41,472）0.760 → 0.200 ms。D3 合計 6.346 → 5.853 ms', 'VER-0015'],
        ['7(vii)', 'float 前段フィルタ', '不採用: CN カーネル 0.706 → 0.759 ms（0.93 倍）。受理率 約 22 %、99.9 % のワープが double 経路も実行', 'VER-0016'],
    ], [0.8, 2.3, 4.9, 1.0], 10),
    ('txt', '最終（slope）: 水 81 が 2.456 → 0.152、水 648 が 4.417 → 0.459、水 5,184 が 4.568 → 0.846、水 41,472 が 8.398 → 6.193、Cu が 3.612 → 0.554 ms（VER-0018）。値は各時点のコードに対するもので、指標が行ごとに違う', 10),
], 'docs_ja/dftd 03 章 3.6.1 節')

app('A-15', 'A-15 D2 の高速化履歴', [
    ('tbl', [
        ['Phase', '内容', '効果', '種別・出典'],
        ['基準', '原子ごと 1 スレッド、セルリスト、シフト力', '水 648・rc 30 Å の extra が 55.4 ms（D2/D3 = 22.6〜54.1）', 'slope、VER-0005/0006'],
        ['6(i)', 'コスト基準のセル数', '水 648 で 1.62 倍（rc 30）、2.28 倍（rc 20）。16 → 20 Å の飛びが解消', 'カーネル、VER-0008'],
        ['6(ii)', 'NUM_SUB スレッド＋木縮約', '水 648 で (i) の上にさらに 9.4 倍（合計 15.2 倍、47.10 → 3.09 ms）。水 5,184 は 3.4 倍、Cu は 2.9 倍', 'カーネル、VER-0008'],
        ['6(2b)', 'セル中心相対の float 候補位置', '35 点でカーネル 1.49〜1.86 倍。精度ゲートの最大は 0.27', 'カーネル、VER-0009'],
        ['R3 stage 1', 'float 累算の試作（精度未評価）', 'N = 41,472 のカーネル 6.44 → 1.09 ms（−83 %）', 'nsys、VER-0026 4 節'],
        ['R3 stage 2', '補償付き float＋4 セルごとのフラッシュ', 'カーネル 6.41 → 1.41 ms（−78 %）、D2 extra 7.06 → 1.62 ms（水 41,472）。決定的', 'nsys と slope、VER-0026 5 節'],
    ], [1.0, 2.5, 3.9, 1.6], 10),
    ('txt', 'R3 前までの slope 全体の改善（VER-0006 → VER-0010、導出値）: rc 30 Å で 34 倍（N = 81）、30 倍（648）、5.6 倍（5,184）、3.1 倍（41,472）。R3 の効果とは別の測定', 10),
], 'docs_ja/dftd 03 章 3.6.2 節')

app('A-16', 'A-16 D3 の精度（DR-001 AC-1、ケース別）', [
    ('tbl', [
        ['ケース', 'dE [eV]：誤差 / 許容', 'dF [eV/Å]：誤差 / 許容', 'dV [eV]：誤差 / 許容'],
        ['benzene_dimer（非周期）', '3.68e-07 / 1.53e-06', '2.12e-08 / 1.00e-06', '3.66e-07 / 1.27e-06'],
        ['graphite', '4.60e-06 / 7.88e-06', '5.16e-07 / 1.00e-06', '5.32e-06 / 1.31e-05'],
        ['water27', '1.67e-06 / 3.30e-06', '1.89e-08 / 1.00e-06', '1.81e-06 / 6.07e-06'],
        ['water27_rc25', '8.07e-07 / 3.46e-06', '7.08e-08 / 1.00e-06', '1.07e-06 / 6.18e-06'],
        ['water27_tri', '2.22e-06 / 4.32e-06', '4.94e-08 / 1.00e-06', '2.16e-06 / 6.62e-06'],
        ['water_large', '4.81e-05 / 9.64e-05', '2.09e-07 / 1.00e-06', '4.64e-05 / 9.22e-05'],
        ['cu_large', '6.13e-04 / 1.23e-03', '4.61e-08 / 1.00e-06', '5.70e-04 / 1.13e-03'],
    ], [2.4, 2.2, 2.2, 2.2], 10),
    ('bul', ['基準: simple-dftd3 1.7.0（PBE、BJ、二体項のみ、倍精度）。規則: |ΔE| ≤ 2×10⁻⁶|E|、max|ΔV| ≤ 2×10⁻⁶ max|V_ref|、max|ΔF| ≤ 1×10⁻⁵ eV/Å、かつ各誤差が max(未変更コードの 2 倍, float 雑音の下限) 以下',
             '許容は各ケースの「未変更コードの誤差の 2 倍（下限付き）」で、余裕は約 1.7〜1.9 倍。最も小さいのは graphite の dE（許容の約 59 %）',
             '値は VER-0015 時点（最終コード 0c353281 の accuracy.txt は残っていない）。極端な配位数の系は対象外（UPSTREAM-0001）'], 11),
], 'docs_ja/dftd 06 章 6.2 節（VER-0015）')

app('A-17', 'A-17 受入基準の状況: DR-001（D3 高速化）', [
    ('tbl', [
        ['基準', '内容', '判定'],
        ['AC-1 精度', '6.2 節の規則、両経路', '達成（7 ケース、全 num_sub）。極端な配位数は対象外'],
        ['AC-2 回帰なし', 'pytest 新規失敗なし、--suite full は既知ケースのみ', '達成（厳密許容値では 2 ケースが相対 1.46×10⁻⁶ の差で不合格）'],
        ['AC-3 エネルギー保存', '基準バイナリより悪くない', '雑音の範囲でのみ達成（有意差なし。水 5,184 で最大約 2 倍/4 倍の悪化は排除できない）'],
        ['AC-4 性能（slope）', '81 ≤ 1.0、648 ≤ 2.0、Cu ≤ 2.0、5,184 ≤ 2.0、41,472 ≤ 6.0 ms。基準より 3 % を超えて遅くならない', '81（0.152）、648（0.459）、5,184（0.846）、Cu（0.554）は達成。41,472 は 6.19 ms で未達（目標は人間の判断で取り下げ）'],
        ['AC-5 移植性', 'CUDA で警告なし、HIP を壊さない', 'CUDA は警告なし。HIP と sm < 60 のビルドは未検証'],
        ['AC-6 文書', 'dftd3.rst にカットオフ依存のコストを記載', '達成（rst は目視確認のみ）'],
    ], [1.6, 3.4, 4.0], 10),
    ('txt', 'AC-4 補足: 水 41,472 の nsys による D3 カーネル時間の和は 5.875 ms だが、AC-4 の定義は slope 法であり、slope 法では未達', 10),
], 'docs_ja/dftd 06 章 6.7 節（VER-0011、0013、0018、0020）')

app('A-18', 'A-18 受入基準の状況: DR-002（D2）', [
    ('tbl', [
        ['基準', '内容', '判定'],
        ['AC-1 精度', '独立 D2 と E・W 2e-6、F 1e-5', '達成（17 ケースの最大 E 3.3e-7、W 5.9e-7、F 1.7e-6）'],
        ['AC-2 パラメータ', '組込み表（Z ≤ 86）が出典に追跡可能', '達成。ただし H-Xe と公表表（Turbomole 版）の照合テストはスキップされ、Z = 55〜86 は単一出典'],
        ['AC-3 カットオフ', '既定値は事前登録の基準を満たす最小値', '達成（既定 30 Å、人間が承認）。重元素・高密度系では未収束'],
        ['AC-4 整合性', '力 = −∇E、ビリアル = 歪み微分、NVE ドリフトが悪化しない', '達成（rc = 12 と 30 Å、s6 = 75）。NVE は単一軌道'],
        ['AC-5 入力検査', '6 種のエラーが固有メッセージで停止', '達成'],
        ['AC-6 回帰なし', 'dftd2 無しで結果不変', '達成'],
        ['AC-7 性能（参考値）', 'D2 ≤ 0.5 × D3', 'R3 前は未達（10 点中 水 648・rc 12 Å の 1 点のみ達成）。R3 後は N ≈ 40,000・rc 12 Å で D2/D3 ≈ 0.24（3 点の再測定のみ）'],
        ['AC-8 文書', '式・単位・シフト・既定カットオフ・注意', '達成'],
    ], [1.6, 3.3, 4.1], 10),
], 'docs_ja/dftd 06 章 6.7 節（VER-0003、0005、0007、0010、0021、0026）')

app('A-19', 'A-19 受入基準の状況: DR-003（D3(0)）', [
    ('tbl', [
        ['基準', '内容', '判定'],
        ['AC-1 精度', 'simple-dftd3（zero damping、二体項）と一致', '達成（20 ケース中 20 が許容内）'],
        ['AC-2 有限差分・NVE', '有限差分と一致、NVE が BJ より悪くない', '達成（NVE 45 本で zero は BJ と区別できず p ≥ 0.31。同等の証明ではない）'],
        ['AC-3 BJ のビット一致', 'bj の明示と省略が従来のバイナリとビット同一', '達成（凍結出力のテスト。小箱 81 原子でバイト一致。大箱は旧バイナリ自体が再現しない）'],
        ['AC-4 / 4b', '表に無い汎関数、Z ≥ 87 はエラー（BJ へのフォールバックなし）', '達成'],
        ['AC-5 桁あふれなし・小箱と大箱の一致', 'r ≪ R0 と float で NaN なし。両経路が一致', '一部未達（桁あふれなしは達成）。人間が例外として承認'],
        ['AC-6 性能（参考）', 'zero は BJ の 10 % 以内', '1 ステップ全体の比 1.04〜1.08（達成、人間の判断）。D3 の extra の比は 1.10〜1.29。BJ の非退行は境界上'],
        ['AC-7 表の出典', 'r0ab とパラメータはスクリプトで生成し照合', '達成（Z ≤ 86）。Z = 87〜94 は出典間で不一致のため対象外（エラー）'],
    ], [1.9, 3.0, 4.1], 10),
], 'docs_ja/dftd 06 章 6.7 節（VER-0023、VER-0024）')

app('A-20', 'A-20 受入基準の状況: DR-004（D2 QE 型）と DR-006', [
    ('tbl', [
        ['基準', '判定'],
        ['DR-004 AC-1 精度（NumPy 参照と依頼元の QE 7.2 データ）', '達成。QE 7.2 のバイナリは実行していない'],
        ['DR-004 AC-2 既定のビット一致', 'R1/R2 では達成。R3 後は ADR-0005 により AC-1 許容値での一致（旧凍結出力に対し力 ≤ 9.4×10⁻¹⁵ 相対）'],
        ['DR-004 AC-3 パラメータ（QE 7.2 のソースと項目ごとに一致）', '達成（3741 ペアで不一致 0）'],
        ['DR-004 AC-4〜6（複製・薄い箱、有限差分・NVE、入力エラー）', '達成（NVE は参考値で系統的なドリフトは見えず。証明ではない）'],
        ['DR-004 AC-7 性能（参考）', 'sm_86 で達成: N ≥ 40,000 で D2 extra < D3 extra。A100 は未測定'],
        ['DR-006 AC-1〜3、AC-5（修正、試験、遷移のない実行の不変）', '達成'],
        ['DR-006 AC-4（--suite full）', 'HEAD 基準では未実施。HIP は未検証'],
    ], [5.4, 3.6], 10),
    ('txt', 'DR-006 の状態は partial。DR-004 の文書は、本マニュアルの 04 章が D2 QE を未反映（upstream マージ後に更新予定）', 10),
], 'docs_ja/dftd 06 章 6.7 節（VER-0025、VER-0026、VER-0030）')

app('A-21', 'A-21 未測定の事項（D2/D3）', [
    ('tbl', [
        ['区分', '未測定・未検証の内容'],
        ['ハードウェア・環境', 'HIP（AMD）ビルド、sm < 60、RTX A6000 以外の GPU、CUDA 12.3 以外。他 GPU での定数（num_sub 規則、CELL_VISIT_COST）の最適性'],
        ['精度', '妥当性確認（DFT・実験との比較）は全体に無し。D3 の float と double の比較。D3 の参照試験は H、C、O、Cu・PBE のみ。極端な配位数の D3'],
        ['エネルギー保存', 'HIP、他の GPU、10 ps を超える長時間、各ケース 5〜20 本を超える実行数。D2 は単一軌道のみ'],
        ['性能', '最適化後の D2 の rc 16/20/25/40 Å、D3 の rc 20 Å 超。D2 の N = 2,187/8,748 の NEP 比。D3 の slope 表の最終 HEAD での再測定。R3 後の D2(30)、水 648、Cu'],
        ['再現性', '大箱経路の D3 の実行ごとのビット一致は保証されない（セルリストの整数 atomicAdd）。小箱経路と D2 は一致'],
        ['D2 QE 型', 'QE 7.2 のバイナリでの実行、A100（sm_80）での速度と精度、Z ≥ 87'],
        ['三体項', 'D3(0) の三体項（QE の既定では有効）。下流の見積もりは D3 の約 1 %（60 原子で 5 meV）だが、別には測定されていない'],
    ], [2.0, 7.0], 10),
], 'docs_ja/dftd 06 章 6.9 節')

app('A-22', 'A-22 D3 の NEP 単独に対するコスト', [
    ('tbl', [
        ['系', 'NEP のみ（基準 / 最適化版）[ms]', 'master 基準の extra/NEP', '最適化版の extra/NEP'],
        ['水 81', '1.417 / 1.320', '1.73', '0.12'],
        ['水 648', '1.545 / 1.626', '2.86', '0.28'],
        ['水 2,187', '1.707 / 1.726', '1.91', '0.44'],
        ['水 5,184', '1.740 / 1.775', '2.63', '0.48'],
        ['水 8,748', '1.999 / 2.087', '5.89', '3.96'],
        ['水 41,472', '6.782 / 6.830', '1.24', '0.91'],
        ['Cu 4,000', '1.337 / 1.295', '2.70', '0.43'],
        ['クラスター 972', '1.408 / 1.411', '1.22', '0.86'],
    ], [2.0, 3.0, 2.0, 2.0], 11),
    ('bul', ['extra/NEP は導出値（slope の extra を同じ JSON の NEP のみの値で割ったもの）。1 ステップ全体の比は 1 + extra/NEP',
             '最初の調査（VER-0001）では NEP89 + dftd3 pbe 12 6 で 1 ステップが 2.2〜3.5 倍。最適化版は水 81 で 1.12 倍、水 41,472 で 1.91 倍',
             '水 8,748 は小箱経路の O(N²) 近傍リスト構築が支配的で、NEP 単独の約 4 倍'], 11),
], 'docs_ja/dftd 03 章 3.5.1 節（VER-0018）')

# =============== B: qNEP ===============
app('B-1', 'B-1 nep.in と nep.txt の形式', [
    ('bul', ['nep.in: type 2 Na Cl / model qnep2（または charge_mode 2・3）/ cutoff 6 5 / charge_transfer 6',
             'charge_transfer <rc_phi>: 必須（既定値なし）、1 回のみ。0 < rc_phi ≤ 動径カットオフ かつ ≤ 100 Å。qnep2・qnep3 のみ',
             '追加パラメータ: N_pair × (basis_size_radial + 1) 個（N_pair = N_type(N_type+1)/2）。末尾に付き、初期値は [0, 0.1] の一様乱数。他の乱数列は乱さない',
             'ネットワークの電荷出力は q ではなく h を意味する。charge_transfer なしで学習したネットワークは流用できず、再学習が必要',
             '学習データ: 電荷の目標値は不要。0 でない charge= ラベル（絶対値 1×10⁻⁶ 超）は入力エラー。BEC は従来どおりラベルで学習'], 12),
    ('txt', 'nep.txt（legacy テキスト）: 1 行目 nep4_charge2 … / cutoff 行 / 「charge_transfer rc_phi」（cutoff の直後の追加行）/ n_max、basis_size、l_max、ANN … / ANN ブロック → 記述子の係数 c → q_scaler → φ ブロック（最後。num_pairs × (basis_size_radial + 1) 個、ペア主順）', 11),
    ('bul', ['ペア番号 p = t1·nt − t1(t1−1)/2 + (t2 − t1)（t1 ≤ t2）。φ(r) = Σ_k c_k f_k(r) fc(r)、カットオフ rc_phi',
             '新形式の nep.txt はヘッダーに charge_transfer rc_phi、@ q_scaler の後に @ phi_parameters N のブロック。charge_transfer 行の無いファイルは従来どおり',
             '新しい nep.txt を古いバイナリで読むと、明確なメッセージを出して停止する'], 11),
], 'docs_ja/qnep 01 章 1.1〜1.2 節（VER-0032、ADR-0006、VER-0035）')

app('B-2', 'B-2 電荷と分極 P', [
    ('eq', 'E6_q_P.png'),
    ('bul', ['φ は対称（φ_ab = φ_ba）で、b を rc_phi 以内の最小イメージで数える。b から a への移動量は a から b への符号反転なので Σq = 0 がどの配置でも厳密に成り立つ（平均の差し引きは不要。zero_total_charge は走らない）',
             'q_a は rc_phi + rc_desc 以内の原子に依存するが、BEC の導出では a 中心の和に書き換えるため二段の最小イメージは不要',
             'P は r_ab だけに依存するので、剛体並進で不変かつ周期境界でも一価（φ は rc_phi で滑らかに 0 になる）',
             '従来のモデルが勾配になれなかった理由は、平均の差し引き（S_k ≠ 0）で非局所項が入ること。新モデルは S_k = 0'], 12),
], 'docs_ja/qnep 02 章 2.1〜2.2 節')

app('B-3', 'B-3 厳密な BEC: 項 A + 項 B', [
    ('eq', 'E7_bec.png'),
    ('bul', ['項 A: ネットワークの連鎖律による G_ab = ∂h_a/∂r_b（動径部分と角度部分）。項 B: φ の位置依存性',
             '対角の q_k·1 は項 B の φ·1 の部分に含まれる（別に加えると二重計上）。find_bec_diagonal は走らない',
             '「T2 のみ」の簡易形が厳密になるのは rc_desc + rc_phi < L/2 のときだけ。項 A + B の形は、通常のカットオフごとの箱の検査を通る任意の箱で厳密',
             'Z は非対称。9 成分を保存し、添字は [3β + α] = ∂P_β/∂r_α。出力される BEC は s × Z（scale_bec）',
             'ΣZ_k = 0（並進不変性）は float64 の参照実装で 4×10⁻¹⁵ 以下（6 ケース、VER-0029）'], 12),
], 'docs_ja/qnep 02 章 2.3 節（VER-0029）')

app('B-4', 'B-4 力・ビリアル・電場中のエネルギー', [
    ('eq', 'E8_force.png'),
    ('bul', ['記述子の重み: Dt_c = ∂E/∂h_c（rc_phi のループ）。記述子の力: F_k −= Σ_a Dt_a ∂h_a/∂r_k。既存の力カーネルの重みを D − mean(D) から Dt に替えたもの',
             'ペア項（φ(r_ab) の位置依存性、h と D 固定）: 作用・反作用が成り立ち、各原子は自分にだけ加える。q 固定の明示力は Ewald/PPPM に含まれる',
             'ビリアル: W_ab = −Σ v_a Dt_a G_ab − 0.5 Σ_{順序ペア} (g + g^K) v_a v_b/d。ソルバーの k 空間ビリアルは変わらない',
             '従来の重み D − mean(D) はこのモデルでは勾配にならない（VER-0033）',
             'add_efield … bec の力は F = ∇_r(s E·P)（保守力）。保存するのは K + U − s E·P_direct で、K+U は保存しない'], 12),
], 'docs_ja/qnep 02 章 2.4〜2.5 節（VER-0033、VER-0034）')

app('B-5', 'B-5 gpumd の 1 ステップの流れ', [
    ('tbl', [
        ['段', '処理', '内容'],
        ['1', 'find_descriptor', '記述子とネットワーク（h）を計算し nep_data.ct_h にコピー'],
        ['2', 'ct_find_charge_*', 'q_a = Σ_b φ_ab(h_a − h_b)（原子ごとに double で累算、結果は float）と V_a。zero_total_charge の代わり'],
        ['3', 'BEC', '項 B のカーネルが 9 成分を書き（q·1 を含む）、項 A（動径・角度）が V⊗∂h/∂r を atomicAdd で加え、最後に scale_bec'],
        ['4', 'Ewald / PPPM', 'q を使い D_a = ∂E/∂q_a（nep_data.D_real）を得る'],
        ['5', 'ct_find_Dt_and_pair_force_*', 'Dt_k = Σ_b φ_kb(D_k − D_b)、明示的なペア力・ビリアル（G = φ′(d)(h_k − h_b)(D_k − D_b)）'],
        ['6', '既存の力カーネル', 'ct_Dt を重み g_D_real として渡す（カーネル自体は無変更）'],
    ], [0.5, 2.8, 5.7], 11),
    ('bul', ['大箱・小箱の両経路で同じ流れ。小箱経路は箱の厚さが 2.5(rc_radial + 1) 以下のとき（rc 4 で 12.5 Å）',
             '新しい総和は double で累算。近傍は動径リストで a 中心の和だけなので、新たな近傍リストや二段の最小イメージは不要。新カーネルは汎用経路のみ'], 12),
], 'docs_ja/qnep 03 章 3.1〜3.2 節（VER-0032、VER-0034）')

app('B-6', 'B-6 学習側（nep）と gpumd_replica', [
    ('bul', ['学習側: number_of_variables += pairs × (basis_size_radial + 1)。φ ブロックはベクトルの末尾で、nep.restart に含まれる',
             '流れ: 記述子 → ANN（h）→ ct_find_charge → BEC（項 B + 項 A、scale_bec）→ 長距離（q）→ ct_find_Dt_and_pair_force。算術は gpumd と同じ（型が別のためヘッダーは共有しない）',
             'charge_transfer モデルでは enforce_nep_total_charge と subtract_nep_charge_derivative_mean は走らない。0 でない charge= ラベルは入力エラー',
             '学習側の長距離は Ewald（α = π/rc_radial、|k|² < (2πα)²）。薄い箱（L < 2 rc）は学習側の内部の複製を使う（gpumd との一致で確認、VER-0035）',
             'gpumd_replica: モード 2 のみ。BEC カーネルは移植せず、zero_total_charge を h のコピーと ct_find_charge_* に置換。バッチ積分には入らない。GPU 検証は未実施（VER-0036）',
             'charge_transfer なしのモデルは旧経路を通る。旧カーネルの SASS は 22 本中 20 本が同一（定数バンクの正規化後）'], 12),
], 'docs_ja/qnep 03 章 3.3〜3.4 節（VER-0035、VER-0036）')

app('B-7', 'B-7 検証の段階の構成（DES-0005 6 節）', [
    ('tbl', [
        ['段階', '内容', '記録', '状態'],
        ['1', '診断（src/ の変更なし）: 従来の BEC が勾配にならない理由', 'VER-0028', '完了'],
        ['2a', 'float64 の numpy 参照実装（q、P、厳密な Z_k）', 'VER-0029', '完了'],
        ['2b', 'GPU の電荷・BEC カーネル（大箱・小箱）。FD と参照との比較', 'VER-0032', '完了'],
        ['2c', '力・ビリアルの参照（VER-0033）と GPU（VER-0034）。NVE、電場、閉ループ、モード 3、三斜晶', 'VER-0033/0034', '完了'],
        ['3a', '学習側 nep。A4（nep 対 gpumd）、変異試験、学習の動作確認', 'VER-0035', '完了'],
        ['3b', 'gpumd_replica（モード 2）', 'VER-0036', 'コードのみ。GPU 検証は未実施'],
        ['4', '文書（doc/ と本マニュアル）。全回帰', '—', '文書は完了、全回帰は未実施'],
        ['5', '再学習（5a 表現力の短い確認、5b 本学習）', '—', '依頼元側で実施中'],
    ], [0.8, 5.0, 1.5, 1.7], 10),
], 'docs_ja/qnep 04 章 4.1 節')

app('B-8', 'B-8 GPU の電荷と BEC（VER-0032）', [
    ('tbl', [
        ['構造（rc_radial/rc_phi、経路）', 'RMS(BEC/√ε∞ − Z_FD) [e]', 'RMS Z [e]', 'Σq（float ダンプ）'],
        ['n8、L13、4/4、大箱', '3.1×10⁻⁶', '8.3×10⁻³', '0'],
        ['n45、L14、4/4、大箱', '7.2×10⁻⁵', '0.143', '−1.7×10⁻⁸'],
        ['n59、L16、4/4、大箱', '8.0×10⁻⁵', '0.128', '4.9×10⁻⁹'],
        ['n12、L7.5、4/4、小箱', '1.8×10⁻⁴（δ 0.01 では 4.6×10⁻⁵）', '0.34', '−3.0×10⁻⁷'],
        ['n10、L6、4/4、小箱', '1.2×10⁻⁴', '0.29', '3.2×10⁻⁷'],
        ['n8、L13、5.5/4、小箱（強制）', '1.2×10⁻⁵', '0.040', '0'],
        ['広い箱 n59、L18、4/4', '9.2×10⁻⁵', '0.145', '2.4×10⁻⁸'],
    ], [3.0, 2.9, 1.3, 1.8], 11),
    ('bul', ['基準: gpumd のモデルから作った単一価の P の有限差分（ct_bec_fd.py、δ = 0.02 Å）。ゲート: Σq ≤ 1×10⁻⁶ e、q の差 ≤ 3×10⁻⁷ e、RMS(BEC − FD) ≤ 1×10⁻³ e（A1）。全行で合格',
             '誤差は δ² に比例（n12: δ 0.02 で 1.8×10⁻⁴、0.01 で 4.6×10⁻⁵）で、有限差分の打ち切り誤差',
             '最初の生成器は重みが大きく、薄い箱で |q| が最大 5.7 e、float の Σq が 2×10⁻⁶ に達した。ゲートは緩めずモデルのスケールを下げて対応',
             '試験が空虚でないこと: RMS Z > 0.02 を要求。原子の入れ替えに対する q・BEC の不変性（≤ 1×10⁻⁵）も確認'], 11),
], 'docs_ja/qnep 04 章 4.4 節（VER-0032）')

app('B-9', 'B-9 nep 対 gpumd（A4、VER-0035）', [
    ('tbl', [
        ['ケース', '構造', 'BEC RMS [e]', '力 RMS [eV/Å]', 'max ΔE/原子 [eV]'],
        ['qnep2、rc_phi 4', '4 構造', '4.5×10⁻⁷', '1.5×10⁻⁶', '1.9×10⁻⁷'],
        ['qnep3、rc_phi 4', '同', '4.5×10⁻⁷', '2.4×10⁻⁶', '4.2×10⁻⁷'],
        ['qnep2、rc_phi 3 < rc_radial', '同', '1.8×10⁻⁷', '1.0×10⁻⁶', '1.6×10⁻⁷'],
        ['qnep2、3 型（φ 6 ペア）', '3 型', '7.5×10⁻⁷', '7.9×10⁻⁶', '1.6×10⁻⁶'],
        ['学習済み qnep2（100 世代、RMS F 26 eV/Å）', '薄い 6 構造', '1.5×10⁻⁶', '1.4×10⁻⁵（最大 6.2×10⁻⁵）', '1.9×10⁻⁶'],
    ], [3.3, 1.2, 1.4, 1.7, 1.4], 10),
    ('bul', ['ゲート: BEC ≤ 5×10⁻⁶ e、力 ≤ 1×10⁻⁵ eV/Å（学習済みモデルは相対 1×10⁻⁶ × RMS F）、エネルギー ≤ 1×10⁻⁵ eV/原子。比較する量の RMS は BEC 0.01〜1.5 e、力 0.1〜26 eV/Å。10〜100 倍の余裕',
             '変異試験（学習側）: 項 A 削除（BEC 差 0.185）、項 B 削除（0.122）、Dt = 0（力の差 0.48）、ペア力・ビリアル削除（力の差 0.57）。4 件すべて A4 で検出',
             '学習の動作確認（教師-生徒、薄い周期構造 6 個）: 30 世代で損失 21.64 → 9.91。書き出した nep.txt の予測が loss.out と一致',
             '旧モデル: 合成 4 構造で qnep1/2/3 と通常の nep の loss.out、nep.txt、nep.restart がバイト同一'], 11),
    ('txt', '記録間の不整合: VER-0035 の表見出しの力のゲートは 1e-4 で、人間が決めた A4 の値は 1e-5（未解決）', 10),
], 'docs_ja/qnep 04 章 4.6 節（VER-0035）')

app('B-10', 'B-10 ビリアルの検証（VER-0034）', [
    ('img', 'S05b_force_virial.png', 3.7),
    ('bul', ['ビリアル: 一様ひずみの有限差分との RMS 誤差。「旧」は同じ生成器で charge_transfer なしの通常の qNEP（雑音の下限）。5 構造中 4 構造で旧以下（n8 は 1.3 倍でどちらも微小）',
             'ゲートは、ビリアルの大きさが 4〜13 eV のため絶対 1×10⁻³ が意味を持たず、「旧の雑音の 3 倍以下」かつ「相対 3×10⁻³ |W|_RMS」とした（人間の決定。絶対 1×10⁻³ は力にのみ適用）',
             'float64 の参照（VER-0033）では解析と有限差分が 5 ケースで 1×10⁻⁸ 以内。2 次の中心差分（δ = 1×10⁻⁵）は 3 ケースで打ち切り誤差のため 4 次（5 点）にした'], 11),
], 'docs_ja/workreport 3.4(c)、docs_ja/qnep 04 章 4.5 節（VER-0033、VER-0034）')

app('B-11', 'B-11 再現手順', [
    ('txt', 'PY=/home/katouda/miniforge3/envs/py312-gpumd/bin/python（ase calorine numpy pytest hypothesis）。生成物はリポジトリの外で作る。実行前に nvidia-smi を確認（GPU は共有）', 11),
    ('txt', 'cd src && make CUDA_ARCH="-arch=sm_86" -j4 2>&1 | tail -30', 11),
    ('txt', '$PY specs/verification/VER-0029-qnep-charge-transfer-reference/scripts/ct_reference.py    # 参照実装（CPU、約 25 秒）', 10),
    ('txt', 'S=specs/verification/VER-0032-qnep-charge-transfer-gpu/scripts', 10),
    ('txt', '$PY $S/make_ct_model.py OUTDIR [--seed N] [--rc-radial 4] [--rc-phi 4] [--no-ct]    # 合成の nep.txt と 5 構造', 10),
    ('txt', '$PY $S/ct_bec_fd.py src/gpumd /scratch/fd --structure thick_n45_L14 [--max-atoms 12] [--delta 0.02]    # BEC と有限差分', 10),
    ('txt', 'VER-0035/scripts: ct_a4_compare.py src/nep src/gpumd /scratch/a4 [--charge-mode 3] [--rc-phi 3] [--nt 3]　/　ct_train_smoke.py src/nep src/gpumd /scratch/smoke --generations 100', 10),
    ('txt', 'cd tests_pytest && $PY -m pytest -q test_qnep_charge_transfer_parse.py test_qnep_charge_transfer_bec.py test_qnep_charge_transfer_forces.py test_qnep_charge_transfer_field.py test_nep_charge_transfer_train.py', 10),
    ('txt', '3b（未実行。GPU 検証は再学習の終了後）: $PY -m pytest -q test_replica_charge_transfer.py', 10),
], 'docs_ja/qnep 05 章 5.6 節')

app('B-12', 'B-12 理論面の限界、代替案、未完了の作業', [
    ('bul', ['q = L h（L はラプラシアン型の演算子、h は局所）の形は、表せる中性の電荷パターンを制限する。実データでの BEC の精度が落ちる可能性（DR-005 A6）。5a の短い学習で確認中',
             'φ の正値性は強制していない。電荷は目標値で学習せず、BEC・エネルギー・力・ビリアルのラベルで学習する',
             '検討した代替案（採用せず、5a が不合格のときに再検討）: ペアのネットワークによる環境依存の φ、ボンド電荷増分のみのモデル（環境依存が弱すぎる）、直接の双極子モデル（nep_dipole）',
             '未完了: gpumd_replica の GPU 検証、全回帰（人間の合図）、nep 対 gpumd の差の床（gpumd 同士）の測定、直接の P 出力（DES-0005 3b）、HIP ビルド',
             '依頼元側: 5b（10000 世代、seed 1〜3）の結果。C の対角 BEC のずれ（−0.28 e）の見立ては 5b の後',
             '実データの 5a: held-out BEC RMSE 0.159 e（旧 0.112 e、比 1.4、ゲート ≤ 2×）。未収束・1 seed・20 構造のみ'], 12),
], 'docs_ja/qnep 02 章 2.6 節、05 章（DES-0005 7 節）')

# =============== C: records ===============
app('C-1', 'C-1 記録の対応表（DR / ADR / VER）', [
    ('tbl', [
        ['ID', '内容（記録名による分類）'],
        ['DR-001 / 002 / 003 / 004', 'D3 の GPU 高速化 / D2 外付け補正 / D3 ゼロダンピング / D2 QE 互換'],
        ['DR-005 / 006', 'qNEP の BEC と P の勾配整合（charge_transfer）/ dftd3 のセルバッファ（小箱 ↔ 大箱の遷移）'],
        ['DES-0001 / 0002 / 0005', 'D3 高速化の設計（Phase）/ D2 の設計 / 電荷保存型 qNEP の設計'],
        ['ADR-0004 / 0005 / 0006', 'D2 QE 互換テーブル / D2 の補償付き float 累算（R3）/ charge_transfer の契約（互換性）'],
        ['VER-0001〜0002', 'D3 の simple-dftd3 との比較、GPU プロファイル'],
        ['VER-0003、0005〜0010', 'D2 のパラメータと参照、カットオフ、整合性・性能、最適化、float 候補、最終検証'],
        ['VER-0004、0011〜0020', 'D3 の最適化結果、最終検証、小箱の診断、複数スレッド、因数分解、CN の追加カーネル、float 前段フィルタ、stage 3、NVE'],
        ['VER-0021', 'D2 対 D3 の時間'],
        ['VER-0022、0027', 'upstream マージの確認'],
        ['VER-0023 / 0024', 'D3(0) のパラメータ / D3(0) の検証'],
        ['VER-0025 / 0026', 'D2 QE のパラメータ / D2 QE の互換と性能（R3 を含む）'],
        ['VER-0028 / 0029 / 0031', 'BEC の診断 / charge_transfer の参照実装 / 非保存 BEC の分解'],
        ['VER-0030', 'dftd3 のセルバッファ（DR-006）'],
        ['VER-0032〜0036', 'charge_transfer の GPU（BEC、力・ビリアル）、参照（力）、学習側（nep）、replica'],
    ], [2.9, 6.1], 10),
    ('txt', 'upstream 側の記録: UPSTREAM-0001（極端な配位数）、UPSTREAM-0002（コード読解のみの注意 A〜F）、UPSTREAM-0003（dftd3 のセルバッファ。報告・PR は送っていない）', 10),
], 'specs/（requirements、design、decisions、verification、upstream-issues）、specs/traceability.yaml')

app('C-2', 'C-2 用語・略語', [
    ('tbl', [
        ['用語', '意味'],
        ['BEC（Born 有効電荷）', 'Z_k,βα = ∂P_β/∂r_k,α。出力は s × Z（scale_bec）'],
        ['slope 法', '(t500 − t100)/400 などの時間差による 1 ステップあたりの時間。extra = 補正ありの時間 − NEP のみの時間'],
        ['nsys 合計', 'nsys で測ったカーネル時間の和。slope とは測り方が違い、混同しない'],
        ['小箱 / 大箱', '周期方向の厚さが 2.5 rc 以下 / 超。小箱は近傍リスト、大箱はセルリスト'],
        ['CN / extra', '配位数 / CN 微分による力・ビリアルの項'],
        ['R3', 'DR-004 の D2 の補償付き float 累算（ADR-0005）'],
        ['A1〜A6', 'DR-005 の受入基準（A1 BEC と勾配、A4 nep と gpumd、A5 既存結果不変、A6 再学習後 BEC RMSE）'],
        ['Tier P', '依頼元の試作データの区分。内部の開発・検証用に限る'],
        ['検証 / 回帰 / 妥当性確認', 'verification（式は正しいか）/ regression（変更前と同じか）/ validation（実験・DFT と合うか。行っていない）'],
    ], [2.6, 6.4], 11),
], 'docs_ja/dftd 06 章 6.1 節、docs_ja/dftd 03 章、docs_ja/workreport')

# ---------- internal links from the main slides ----------
LINKS = {4: ['A-1', 'A-2', 'A-3', 'A-5'], 5: ['A-8', 'A-9', 'A-10', 'A-11', 'A-14'], 6: ['A-14', 'A-16', 'A-22'],
         7: ['A-12', 'A-13', 'A-15'], 8: ['A-13', 'A-15'], 9: ['A-15', 'A-18', 'A-20'], 10: ['A-22'],
         11: ['A-13', 'A-16', 'A-21'], 12: ['B-7'], 13: ['B-2'], 14: ['B-2', 'B-3'], 15: ['B-4'],
         16: ['B-1', 'B-5', 'B-6'], 17: ['B-8', 'B-10'], 18: ['B-7'], 19: ['B-9'], 20: ['B-12'], 21: ['B-12'],
         22: ['A-20', 'C-1'], 23: ['A-20'], 24: ['C-1'], 25: ['A-21', 'B-12'], 26: ['C-1']}
for no, keys in LINKS.items():
    sl = MAIN_SLIDES[no - 1]
    x = 0.5
    lab = textbox(sl, x, 6.88, 0.9, 0.3, ['参考:'], 11, RGBColor(0x59, 0x59, 0x59), bullet=False)
    x += 0.65
    for k in keys:
        tb = textbox(sl, x, 6.88, 0.8, 0.3, ['→ ' + k], 11, RGBColor(0x05, 0x63, 0xC1), bullet=False)
        tb.click_action.target_slide = APP[k]
        x += 0.85

print('scales', {k: v for k, v in SCALES.items()})
prs.save(OUTFILE)
print('saved', OUTFILE, len(prs.slides), 'slides')
for w in warnings:
    print('WARN', w)
