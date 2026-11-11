"""Build the 26-slide deck from /home/katouda/data/RIST-template.pptx.
All numbers are transcribed from docs_ja/{workreport,dftd,qnep} and specs/; nothing is newly measured.
Usage: [RIST_TEMPLATE=/path/RIST-template.pptx] python build_slides.py <figures_dir> <out.pptx>"""
import copy
import os
import math
import sys
import unicodedata
from PIL import Image
from pptx import Presentation
from pptx.util import Inches, Pt

FIG = sys.argv[1]
OUTFILE = sys.argv[2]
prs = Presentation(os.environ.get('RIST_TEMPLATE', '/home/katouda/data/RIST-template.pptx'))
L_TITLE, L_BODY, L_TONLY = prs.slide_layouts[0], prs.slide_layouts[1], prs.slide_layouts[5]
warnings = []

# drop the two template slides
ids = prs.slides._sldIdLst
for sid in list(ids):
    prs.part.drop_rel(sid.rId)
    ids.remove(sid)


def width_chars(s):
    return sum(1.0 if unicodedata.east_asian_width(c) in 'WF' else 0.55 for c in s)


def est_height(paras, size, width_in):
    cpl = width_in * 72 / size
    lines = sum(max(1, math.ceil(width_chars(t) / cpl)) for t, _ in paras)
    return lines * size * 1.3 / 72 + len(paras) * size * 0.35 / 72


def add_slide_number(slide, layout):
    for sh in layout.placeholders:
        if sh.placeholder_format.idx == 12:
            slide.shapes._spTree.append(copy.deepcopy(sh._element))


def set_title(slide, text, no):
    t = slide.shapes.title
    t.text_frame.text = text
    n = width_chars(text)
    size = 24 if n <= 26 else 20 if n <= 32 else 18
    if n > 38:
        warnings.append(f'slide {no}: title long ({n:.0f})')
    for p in t.text_frame.paragraphs:
        for r in p.runs:
            r.font.size = Pt(size)


def fill_body(ph, bullets, size, no, left=None, top=None, width=None, height=None):
    if left is not None:
        ph.left, ph.top, ph.width, ph.height = Inches(left), Inches(top), Inches(width), Inches(height)
    tf = ph.text_frame
    tf.word_wrap = True
    paras = []
    for i, b in enumerate(bullets):
        text, lvl = (b, 0) if isinstance(b, str) else b
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.level = lvl
        r = p.add_run()
        r.text = text
        r.font.size = Pt(size - 2 * lvl)
        paras.append((text, lvl))
    w = ph.width / 914400 - 0.6
    h = ph.height / 914400
    need = est_height(paras, size, w)
    if need > h:
        warnings.append(f'slide {no}: body may overflow (need {need:.2f} in > {h:.2f} in, {size} pt)')


def put_image(slide, name, x, y, w, h):
    im = Image.open(f'{FIG}/{name}')
    ar = im.width / im.height
    if w / h > ar:
        ph_, pw_ = h, h * ar
    else:
        pw_, ph_ = w, w / ar
    slide.shapes.add_picture(f'{FIG}/{name}', Inches(x + (w - pw_) / 2), Inches(y + (h - ph_) / 2),
                             Inches(pw_), Inches(ph_))


def add(no, title, bullets, images=(), notes='', size=None):
    slide = prs.slides.add_slide(L_BODY)
    set_title(slide, title, no)
    body = [s for s in slide.placeholders if s.placeholder_format.idx == 1][0]
    if images:
        fill_body(body, bullets, size or 14, no, 0.5, 4.9, 9.0, 1.95)
        n = len(images)
        gap = 0.1
        w = (9.0 - gap * (n - 1)) / n
        for i, im in enumerate(images):
            put_image(slide, im, 0.5 + i * (w + gap), 0.85, w, 3.95)
    else:
        fill_body(body, bullets, size or 20, no)
    add_slide_number(slide, L_BODY)
    slide.notes_slide.notes_text_frame.text = notes
    return slide


# ---- 1. title ----
s = prs.slides.add_slide(L_TITLE)
s.shapes.title.text_frame.text = 'GPUMD フォークにおける\n分散力補正の高速化と\nqNEP の Born 有効電荷の厳密化'
for p in s.shapes.title.text_frame.paragraphs:
    for r in p.runs:
        r.font.size = Pt(30)
s.placeholders[1].text_frame.text = '2026.11.12'
s.notes_slide.notes_text_frame.text = ('対象: mkatouda/GPUMD develop。出典: docs_ja/workreport/README.md、docs_ja/dftd、docs_ja/qnep、specs/。'
                                       '数値はすべて記録からの転記で、新たな測定はない。')

add(2, '要旨', [
    '作業 1: DFT-D3 / D2 の実装と GPU 高速化（結果を変えずに高速化）',
    '作業 2: qNEP の BEC を分極 P の厳密な勾配にする charge_transfer モデル（DR-005）',
    '作業 3: dftd3 の箱経路遷移での異常終了の修正（DR-006）',
    '状態: 合成データで検証済み。実データでの表現力、疑似散逸の解消、全回帰は未確認',
], notes='要旨。DR-005/006 は workreport 1 章、D2/D3 は docs_ja/dftd 0 章と 3 章。')

add(3, '範囲・データの扱い・位置づけ', [
    '対象: mkatouda/GPUMD の develop。RTX A6000（sm_86）、CUDA 12.3、float32 ビルド',
    '検証は合成データが中心。DFT・実験との比較（validation）は含まない',
    '依頼元の実データ（PTFE、Tier P：試作）の結果は、依頼元の測定として区別する',
    '依頼元の学習データ・モデルは内部の開発・検証用に限る（再配布・商用モデルの学習には使わない）',
], notes='workreport 冒頭と 3.6 節、依頼元の reply-to-gpumd-bec-rev0-checks.md。')

add(4, '提供する機能: DFT-D3 / D2', [
    'dftd3: BJ ダンピング（既定）と zero（D3(0)）。二体項のみ',
    'dftd2: Grimme 2006（既定）と table qe（QE の dftd3_version = 2 と同じパラメータ）',
    'NEP の力・エネルギー・ビリアルに外付けで加算。訓練データに分散力が含まれる場合は二重計上に注意',
    'dftd3 は upstream に既存。フォークは結果を変えずに高速化。zero と dftd2 はフォークでの追加',
], size=19, notes='docs_ja/dftd 0 章 0.1〜0.3 節（DR-001〜004）。D3 は二体項のみで ATM 三体項なし。')

add(5, 'D3 の高速化技法', [
    'プロファイル: 大箱の力カーネルが D3 の 80〜87 %、小箱の近傍リスト構築が 44〜65 %（基準: master 3b2c8db8）',
    '小箱: 近傍リスト構築を原子ごとのブロックに（水 648 で約 7.2 倍）。原子あたり複数スレッドと固定順の木縮約（atomic なし、決定的）',
    'C6 重みの因子化、CN カーネルと extra カーネルでの近傍走査の共有',
], images=['N1_d3_stages.png'],
    notes='docs_ja/dftd 3.6.1 節（VER-0002, 0004, 0011, 0014, 0015）、5 章。図の指標（カーネルか D3 合計か）は行ごとに違い、累積倍率ではない。')

add(6, 'D3 の成果: master 基準との比較', [
    'slope（ms/ステップ）: 水 81 は 2.456→0.152（16.2 倍）、水 648 は 9.6 倍、水 5,184 は 5.4 倍、Cu 4,000 は 6.5 倍',
    '水 41,472 は 8.398→6.193（1.36 倍）。目標 ≤ 6.0 ms は slope で未達（nsys 合計 5.875 ms）。目標は人間の判断で取り下げ',
    'どの系も master より遅くなっていない',
], images=['N2_d3_slope.png'],
    notes='VER-0018 S2（docs_ja/dftd 3.2 節）。交互 3 回の中央値。slope の最終 HEAD での再測定はなく、VER-0019 は nsys のみ再確認。')

add(7, 'D2 の高速化技法', [
    '近傍リストなしのセルリスト。コスト基準のセル数、原子あたり NUM_SUB スレッド、固定順リダクション',
    'セル中心相対の float 候補位置（単純な float 化は精度で不採用）',
    'R3: 補償付き float 累算（TwoSum）を 4 セルごとに double へ。GA102 で FP64 が FP32 の 1/64 であることに対応',
], images=['N3_d2_r3.png'],
    notes='docs_ja/dftd 3.4.4 節、3.6.2 節、5.6 節（DR-004 R3、ADR-0005、VER-0026 4・5 節、VER-0008、0009）。')

add(8, 'D2 の成果', [
    'R3 後、N = 41,472 でカーネル 6.41→1.41 ms（−78 %）、PTFE 40,500 は 6.31→1.31 ms（−79 %）',
    'D2 の extra が 7.06→1.62 ms（水 41,472）、6.34→1.39 ms（PTFE 40,500）、0.95→0.54 ms（水 5,184）',
    'R3 前までの D2 slope の改善（VER-0006→0010、rc 30 Å）は 3.1〜34 倍。R3 の効果とは別の測定',
    '結果は決定的（3 回ビット一致、5 系で確認。atomic は使わない）',
], size=18, notes='VER-0026 5 節、docs_ja/dftd 3.4.4 節と 3.6.2 節。R3 の効果（−78 %）と、R3 前までの改善（3.1〜34 倍）は別の測定。')

add(9, 'D2（R3 後）と D3 の計算時間', [
    'D2/D3 の extra 比（rc 12 Å、同じ測定）: 水 41,472 と PTFE 40,500 で 0.24、水 5,184 で 0.70（導出値）',
    'R3 前の D2 は大箱で D3 以上（D2(12)/D3 = 1.1〜1.9）。R3 で逆転（DR-004 R3 の目標を sm_86 で達成）',
    '範囲の限定: R3 後の再測定はこの 3 点のみ。水 648、Cu、rc 30 Å は未測定。A100 など他の GPU は未測定',
], images=['N5_d2_vs_d3.png'],
    notes='VER-0026 5 節、VER-0021（R3 前の比）。再測定は 3 点のみ。FP64 比の小さい GPU 向けの効果で、A100 の利得は小さいと予想されるが未確認。')

add(10, '補正を付けたときの 1 ステップの時間', [
    '1 ステップの時間 / NEP のみ（導出値）: 水 41,472 は D3 が 1.99 倍、D2（R3 後）が 1.24 倍。水 5,184 は 1.43 倍と 1.30 倍',
    '注意: R3 後の測定に同じセッションの NEP のみの記録がなく、NEP のみは VER-0021（別セッション）の値を使った参考値',
    'PTFE 40,500 は NEP のみの記録がないため省略',
], images=['N6_ratio_to_nep.png'],
    notes='extra は VER-0026、NEP のみは VER-0021（水 5,184: 1.784 ms、水 41,472: 6.729 ms）。別セッションの値の組み合わせであり、参考値として「約」を付けて説明する。')

add(11, '精度・不採用案・限界', [
    'A7_4（採用）は旧カーネルとの差が AC-1 の上限の 5×10⁻¹⁰ 以下。補償のみの A3b は正味の力 3.7×10⁻¹² が上限 10⁻¹² を超え不採用',
    'D3 の float 前段フィルタは CN カーネルが 0.93 倍で遅くなり不採用。dftd3 zero の追加後も BJ（既定）は従来とビット同一',
    '限界: RTX A6000 のみ。A100・HIP は未測定。DFT・実験に対する検証なし。三体項なし',
], images=['N4_d2_variants.png'],
    notes='VER-0026 5 節（変種の表）、VER-0016、DR-003 AC-3。N4 の時間軸は、記録に系の明記がなく N=41,472 のカーネル時間と推測（未確認）。')

add(12, 'qNEP の BEC: 背景と受入基準', [
    '現象（依頼元）: PTFE の AC 電場 MD で ω·χ″ が 20〜500 GHz でほぼ一定。ゼロ電場でも ∫ΣBEC·v dt が非有界',
    '要求（DR-005）: dump_xyz bec、compute_dpdt、add_efield bec が出す BEC を、モデルの単一価な P(r) の微分にする',
    'A1: BEC と P の有限差分勾配の RMS 差 ≤ 1×10⁻³ e',
    'A4: nep と gpumd の一致（BEC ≤ 5×10⁻⁶ e、力 ≤ 1×10⁻⁵ eV/Å、エネルギー ≤ 1×10⁻⁵ eV/原子）',
    'A5: 既存モデルの結果を変えない。A6: 再学習後の BEC RMSE を報告（目安 約 0.15 e）',
], size=18, notes='workreport 2 章。疑似散逸の原因の一つの可能性として BEC の非保存が疑われた。')

add(13, '診断: 従来の BEC は P の勾配でない', [
    '従来式 s[qn·1 + 0.5 T1 + 0.5 T2] は並進不変だが、P = Σ qn r の勾配ではない',
    '局所 BEC が厳密勾配になれるのは S_k = ∂ΣQ/∂r_k = 0（総電荷の局所保存）のときだけ。平均を引く中性化が壊す',
    '係数の調整では解決せず、モデルの変更（再学習）が必要。人間が D1(a) を決定',
], images=['S01a_diagnosis_candidates.png', 'S01b_diagnosis_coeff.png'],
    notes='VER-0028 stage 1（1-a, 1-b, 1-d）。生の J では a = b = 0.500。S_k の RMS は 0.140 / 0.156 e/Å。')

add(14, 'charge_transfer モデル', [
    '電荷 q_a = Σ_b φ_ab (h_a − h_b)（φ は対称なペア関数、h はネットワーク出力）。Σq = 0 が構成で厳密',
    'P = Σ_a h_a V_a。BEC = 項 A + 項 B は P の厳密な勾配。a 中心の和だけで書け、新たな近傍リストは不要',
    '設計上のリスク: q = L h の形は表せる中性の電荷パターンを制限する（実データでの精度は段階 5 で確認）',
], images=['S02_model.png'],
    notes='DES-0005、ADR-0006。簡易形（T2 のみ）が厳密になるのは rc_desc + rc_phi < L/2 のときだけ。項 A + B は通常の箱の検査を通る任意の箱で厳密。')

add(15, '力・ビリアル・電場', [
    '力: 既存の力カーネルの重みを Dt = Σ_b φ_cb (D_c − D_b) に差し替え、ペア項 φ′(r)(h_a − h_b)(D_a − D_b) を加える',
    'D_a = ∂E_el/∂q_a は Ewald / PPPM が与える。ビリアルも同じエネルギーと整合する',
    'add_efield … bec の力は ∇_r(√ε∞ E·P) で保守力',
    'thermo のエネルギーには −E·P が含まれない。電場中に保存するのは K+U ではなく K+U−√ε∞E·P',
], size=18, notes='workreport 3.2 節。周期系の P の直接出力（DES-0005 3b）は未実装。')

add(16, '実装', [
    'nep.in: charge_transfer <rc_phi>（qnep2 / qnep3 のみ）。nep.txt の cutoff 行の直後に行、φ の係数は末尾',
    'gpumd、学習側 nep、gpumd_replica（モード 2）に実装。新カーネルは nep_charge_transfer.cuh に集約',
    '旧モデル・旧入力は、電荷・力・エネルギーがビット同一（小さい箱の静的計算）',
], images=['S03_pipeline.png'],
    notes='workreport 3.3 節。gpumd_replica は BEC カーネルを移植せず、GPU 検証は未実施。')

add(17, '検証 1: BEC と力（合成データ）', [
    'float64 参照: 解析と有限差分の差は最大 2.3×10⁻⁹（ゲート 1×10⁻⁸）。GPU の BEC は有限差分と最大 1.8×10⁻⁴ e（A1 は 1×10⁻³）',
    '誤差は δ² に比例（有限差分の打ち切り誤差）。力は旧 qNEP 以下',
    'ビリアルは 5 構造中 4 構造で旧以下。ゲートは人間の決定（旧の 3 倍以下かつ相対 3×10⁻³）。ビリアルの図は付録',
], images=['S04_bec_fd.png', 'S05a_force_virial.png'],
    notes='VER-0029、0032、0033、0034。ビリアルの図は S05b_force_virial.png（付録で使用可）。n8 のビリアルは旧の 1.3 倍でどちらも微小。')

add(18, '検証 2: 電場中の保存量と閉ループの仕事', [
    '電場中の NVE（1000 ステップ）で保存するのは K+U−√ε∞E·P。K+U 単独は保存しない',
    '閉じた経路での電場の仕事: 新は 1.4×10⁻⁵〜8.3×10⁻⁴（保存力）、旧 qNEP は 0.04〜0.50（経路依存）',
    '疑似散逸が消えることの証明ではない（再学習後に依頼元が確認する範囲）',
], images=['S06a_field.png', 'S06b_field.png'],
    notes='VER-0034（review follow-up A）。DR-005 の範囲外である疑似散逸の解消は未確認。')

add(19, '検証 3: nep 対 gpumd と変異試験', [
    '合成データでは nep と gpumd が A4 のゲートの 10〜100 倍の余裕で一致（BEC RMS ≤ 1.5×10⁻⁶ e）',
    '変異試験（意図的な誤りの注入）: 学習側 4 件、gpumd 側 3 件が検出された。独立レビュー 3 種で BLOCKER なし',
    '実データ（依頼元の 14 構造）は最大値でゲートの数倍。指標が RMS と最大値で異なる点に注意',
], images=['S07_a4.png'],
    notes='VER-0035、workreport 3.4(e)(g)。学習の動作確認（教師-生徒、30/100 世代）で損失 21.64→9.91/7.72。')

add(20, '依頼元の実データ確認で見つかった事項', [
    '「旧モデルでビット同一」は小さい箱の静的計算に限られていた。248 原子では旧バイナリ同士でも ΔE 3.5×10⁻⁵ eV、ΔF 9.1×10⁻⁶ eV/Å の差。主張を訂正',
    '実データでは nep 対 gpumd の差が A4 の数倍（BEC 最大 1.4×10⁻⁵ e、力 最大 5.5×10⁻⁵ eV/Å）。原因は未切り分け',
], images=['S08_old_noise.png'],
    notes='依頼元の reply-to-gpumd-bec-rev0-checks.md 1・2 章（依頼元の測定。GPUMD 側では再現していない）。atomicAdd の順序は仮説で未検証。')

add(21, '再学習 5a の結果（依頼元）', [
    'held-out BEC RMSE 0.159 e（旧 0.112 e、比 1.4）。ゲート ≤ 2× は満たす。ただし未収束・1 seed・20 構造',
    'C の対角 BEC に系統的なずれ −0.28 e（旧 +0.003）。原因（学習途中かモデルの制約か）は未確定',
    '5b（10000 世代、seed 1〜3）が実行中。疑似散逸の解消は未確認',
], images=['S09_5a.png'],
    notes='依頼元の報告 3 章。E/F RMSE は並記のみ（力 0.330 対 0.409 eV/Å、エネルギー 0.045 対 0.030 eV/原子）。')

add(22, 'dftd3 の箱経路遷移での異常終了', [
    '問題: 周期方向の厚さが 2.5 rc を超えると大箱経路へ移る。小箱から始めると neighbor.cu:193 で異常終了（PTFE 368 原子、npt_mttk で 8 回中 8 回。v5.9.1 でも同じ）',
    '修正: セルリスト 6 配列の確保条件を cn 系と分離（dftd3.cu の 2 行）。数値は変えない',
    '試験: 水 27、deform で各区間がしきい値をまたぐ 4 経路。許容は DR-001 AC-1 の相対 2×10⁻⁶',
], images=['S10_dr006.png'],
    notes='DR-006、VER-0030、UPSTREAM-0003。感度: 箱を 1 ステップ動かすと E_D3 が 3.9×10⁻³ / 3.2×10⁻³ eV 変わり、許容は 3.7×10⁻⁶ / 2.6×10⁻⁶ eV。')

add(23, 'DR-006 の検証と回帰', [
    '修正前は 4 本中 2 本が失敗、修正後は 4/4 合格（一致 約 5×10⁻⁹ eV）。10 回繰り返して 10/10 合格。遷移のない実行は修正前とビット同一',
    '全回帰: 216 合格 / 41 失敗（既知の非決定性 37 件 + 説明できる 4 件）。HEAD 基準の全回帰は未実施',
    'upstream への報告・PR は人間の判断で送っていない（記録のみ）',
], images=['S11_regression.png'],
    notes='VER-0030。説明済 4 件は、基準に dftd2 キーワードが無い 2 件と、基準との差 約 1×10⁻¹⁰ の 2 件（修正前 HEAD ビルドとはビット同一）。')

add(24, '進め方と判断の記録', [
    '段階的・数値ゲート: 診断 → numpy 参照 → GPU の BEC → 力・ビリアル → 学習側 → replica。満たさなければ停止して報告',
    '許容値は緩めない。ビリアルが絶対 1×10⁻³ を超えたときは、緩めずに報告して人間が基準を決めた',
    '人間の決定: D1(a)（モデルを新設して再学習）、ビリアルのゲート、A4 の力のゲート、φ の係数数、replica のゲート',
    '全回帰は人間の合図まで、再学習の終了まで開始しない。コミット・push は指示ごと',
], size=18, notes='workreport 6 章。AGENTS.md の段階的測定。')

add(25, '限界・未了', [
    '未実施: replica の GPU 検証、全回帰（DR-005、DR-006 とも）、HIP ビルド、周期系の P の直接出力',
    '未確認: 実データでの表現力（5a は途中、5b 実行中）、疑似散逸の解消、nep 対 gpumd の差の原因',
], images=['S12_status.png'],
    notes='workreport 7 章と図 12（2026-11-11 時点）。')

add(26, 'まとめと今後', [
    'D3/D2: 結果を変えずに GPU 高速化。D2 は R3 で大きな系（rc 12 Å）が D3 より速い（測定した 3 点の範囲）',
    'BEC: charge_transfer で BEC が P の厳密な勾配に。合成データで検証済み。実データの表現力は未確認',
    'DR-006: 箱経路遷移の異常終了を 2 行で修正。upstream へは未報告',
    '今後: replica の GPU 検証 → 全回帰（人間の合図）→ nep/gpumd の差の床の測定 → 5b の結果の見立て → P の直接出力',
], size=18, notes='workreport 1 章と 7 章。')

prs.save(OUTFILE)
print('saved', OUTFILE, len(prs.slides), 'slides')
for w in warnings:
    print('WARN', w)
