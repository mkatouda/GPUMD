"""作業報告書の図を作る。数値はすべて specs/verification/VER-xxxx の記録、または依頼元の報告
(GPUMD-polymer-apps docs/work-requests/reply-to-gpumd-bec-rev0-checks.md) からの転記で、
ここで新たに測定・計算した値はない。出典は各図の関数のコメントと報告書の本文に示す。
使い方: python make_figures.py  (出力: ../figures/*.png)"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 110
plt.rcParams['savefig.dpi'] = 150
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures')
os.makedirs(OUT, exist_ok=True)
C_CT, C_OLD, C_GATE = '#1f77b4', '#d9822b', '#c0392b'


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), bbox_inches='tight')
    plt.close(fig)


def grouped(ax, labels, series, colors, names, log=False, width=None):
    n = len(series)
    width = width or 0.8 / n
    x = np.arange(len(labels))
    for i, (s, c, nm) in enumerate(zip(series, colors, names)):
        ax.bar(x + (i - (n - 1) / 2) * width, s, width, color=c, label=nm)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    if log:
        ax.set_yscale('log')


# ---- 図 1: 診断 (VER-0028 stage 1) ----
def fig01():
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    cands = ['kernel\n(基準)', 'C1\nqn·1+T2(生J)', 'C2\nq·1+T2(生J)', 'C3\nqn·1+T2(Jn),MIC', 'C4\nqn·1+T2(Jn),平面差']
    oligo = [0, 0.122, 0.126, 0.425, 0.648]
    amor = [0, 0.186, 0.186, 0.188, 0.905]
    grouped(ax[0], cands, [oligo, amor], ['#4c72b0', '#dd8452'], ['オリゴマー 59 原子', 'アモルファス 248 原子'])
    ax[0].set_ylabel('RMS(候補 − kernel)  [e]')
    ax[0].set_title('(a) 「厳密な勾配」の候補は kernel から 0.12〜0.9 e 離れる\n(VER-0028 1-d)', fontsize=10)
    ax[0].legend(fontsize=8)
    ax[0].tick_params(axis='x', labelsize=7)
    lab = ['オリゴマー\n中性化後J(Jn)', 'アモルファス\n中性化後J(Jn)', '両構造\n生のJ']
    a = [0.682, 0.493, 0.500]
    b = [0.049, 0.483, 0.500]
    grouped(ax[1], lab, [a, b], ['#55a868', '#c44e52'], ['a (T1 の係数)', 'b (T2 の係数)'])
    ax[1].axhline(0.5, color='gray', ls=':', lw=1)
    ax[1].set_ylabel('フィット係数 (BEC − q·1 = a T1 + b T2)')
    ax[1].set_title('(b) 係数が (0.68, 0.05) になったのは Jn を使った見かけ\n生の J では両構造とも a = b = 0.500 (VER-0028 1-a/1-b)', fontsize=10)
    ax[1].legend(fontsize=8)
    save(fig, 'fig01_diagnosis.png')


# ---- 図 2: モデルの概念図 ----
def fig02():
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis('off')
    xs = [1, 3, 5, 7, 9]
    hs = [0.9, 0.2, -0.6, 0.5, -0.3]
    for x, h in zip(xs, hs):
        ax.add_patch(Circle((x, 2), 0.32, color=plt.cm.coolwarm((h + 1) / 2), ec='k'))
        ax.text(x, 2, f'h={h:+.1f}', ha='center', va='center', fontsize=8)
    for i in range(4):
        d = hs[i] - hs[i + 1]
        x0, x1 = xs[i], xs[i + 1]
        if d > 0:
            ax.add_patch(FancyArrowPatch((x0 + 0.4, 2.55), (x1 - 0.4, 2.55), arrowstyle='->', mutation_scale=14, color='k'))
        else:
            ax.add_patch(FancyArrowPatch((x1 - 0.4, 2.55), (x0 + 0.4, 2.55), arrowstyle='->', mutation_scale=14, color='k'))
        ax.text((x0 + x1) / 2, 2.85, 'φ(h_a−h_b)', ha='center', fontsize=8)
    ax.text(5, 3.6, '電荷移動: q_a = Σ_b φ_ab (h_a − h_b)    (矢印: h が高い側 → 低い側。原子間で受け渡すので Σ_a q_a = 0 が恒等的に成立)',
            ha='center', fontsize=9)
    ax.text(5, 1.0, 'P = Σ_a q_a r_a = Σ_{a<b} φ_ab (h_a−h_b) r_ab = Σ_a h_a V_a      (r_ab のみに依存 → 並進不変、周期系でも一価)',
            ha='center', fontsize=9)
    ax.text(5, 0.45, 'BEC  Z_k = ∂P/∂r_k = 項A (Σ_a V_a ⊗ ∂h_a/∂r_k) + 項B (Σ_b M_kb (h_k − h_b))      → 厳密な勾配', ha='center', fontsize=9)
    save(fig, 'fig02_model.png')


# ---- 図 3: 1 ステップの流れ (VER-0032/0034, 03 章) ----
def fig03():
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis('off')
    boxes = [
        ('find_descriptor\n記述子→ANN\nh', 0.1, 2.3, '#dbe9f6'),
        ('ct_find_charge\nq=Σφ(h_a−h_b)\nV=Σφ(r_a−r_b)\n(zero_total_charge の代わり)', 2.3, 2.3, '#dbe9f6'),
        ('BEC\n項B(9成分,q·1含む)\n+項A(V⊗∂h/∂r)\n→ scale_bec', 4.6, 2.3, '#e8f3dc'),
        ('Ewald / PPPM\nq → D_a=∂E/∂q_a', 6.9, 2.3, '#f6e7c8'),
        ('ct_find_Dt_and_pair_force\nDt=Σφ(D_c−D_b)\nペア力・ビリアル', 9.0, 2.3, '#f6dcdc'),
    ]
    for t, x, y, c in boxes:
        ax.add_patch(FancyBboxPatch((x, y), 2.0, 1.4, boxstyle='round,pad=0.05', fc=c, ec='k'))
        ax.text(x + 1.0, y + 0.7, t, ha='center', va='center', fontsize=7.5)
    for i in range(4):
        x = boxes[i][1] + 2.05
        ax.add_patch(FancyArrowPatch((x, 3.0), (boxes[i + 1][1] - 0.02, 3.0), arrowstyle='->', mutation_scale=12))
    ax.add_patch(FancyBboxPatch((9.0, 0.3), 2.0, 1.2, boxstyle='round,pad=0.05', fc='#f6dcdc', ec='k'))
    ax.text(10.0, 0.9, '既存の力カーネル\n(重み = Dt、無変更)\n記述子の力・ビリアル', ha='center', va='center', fontsize=7.5)
    ax.add_patch(FancyArrowPatch((10.0, 2.3), (10.0, 1.55), arrowstyle='->', mutation_scale=12))
    ax.text(5.6, 0.9, '出力: dump_xyz bec / bec.out / compute_dpdt / add_efield bec\nはすべて同じ BEC 配列(9 成分, 添字 3β+α)を読む', fontsize=8, ha='center')
    ax.text(0.1, 0.3, '従来の zero_total_charge, find_bec_diagonal, 旧 BEC カーネル(0.5/0.5), zero_mean_D_real は\ncharge_transfer モデルでは走らない', fontsize=7.5, color='#555')
    save(fig, 'fig03_pipeline.png')


# ---- 図 4: GPU の BEC と有限差分 (VER-0032) ----
def fig04():
    labels = ['n8\nL13\n大箱', 'n45\nL14\n大箱', 'n59\nL16\n大箱', 'n12\nL7.5\n小箱', 'n10\nL6\n小箱', 'n8 L13\n5.5/4\n小箱', 'n59 L18\n広い箱']
    err = [3.1e-6, 7.2e-5, 8.0e-5, 1.8e-4, 1.2e-4, 1.2e-5, 9.2e-5]
    rmsz = [8.3e-3, 0.143, 0.128, 0.34, 0.29, 0.040, 0.145]
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(labels))
    ax.bar(x, err, color=C_CT, label='RMS(BEC/√ε∞ − Z_FD)')
    ax.plot(x, rmsz, 'ko-', ms=4, lw=0.8, label='RMS Z (参考)')
    ax.axhline(1e-3, color=C_GATE, ls='--', label='ゲート 1e-3 e (A1)')
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('[e]')
    ax.set_title('GPU の BEC は、単一価 P の有限差分と 1.8e-4 e 以内で一致 (VER-0032, δ=0.02 Å)\n誤差は δ² に比例 = FD の打ち切り誤差(float 雑音ではない)', fontsize=10)
    ax.legend(fontsize=8)
    save(fig, 'fig04_bec_fd.png')


# ---- 図 5: 力・ビリアル (VER-0034) ----
def fig05():
    labels = ['n8\nL13', 'n45\nL14', 'n59\nL16', 'n12\nL7.5', 'n10\nL6']
    fct = [5.6e-6, 1.7e-4, 9.8e-5, 1.2e-4, 1.5e-4]
    fold = [1.0e-5, 3.3e-3, 3.0e-3, 7.9e-4, 5.2e-4]
    wct = [5.9e-5, 6.5e-4, 1.1e-3, 2.3e-3, 1.5e-3]
    wold = [4.4e-5, 2.2e-2, 4.9e-3, 8.3e-3, 8.3e-3]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    grouped(ax[0], labels, [fct, fold], [C_CT, C_OLD], ['charge_transfer', '旧 qNEP(雑音の下限)'], log=True)
    ax[0].axhline(1e-3, color=C_GATE, ls='--', label='ゲート 1e-3 eV/Å')
    ax[0].set_ylabel('力の RMS 誤差 [eV/Å]')
    ax[0].set_title('(a) 力 vs 総エネルギーの中心 FD (δ=0.005 Å)', fontsize=10)
    ax[0].legend(fontsize=8)
    grouped(ax[1], labels, [wct, wold], [C_CT, C_OLD], ['charge_transfer', '旧 qNEP(雑音の下限)'], log=True)
    ax[1].set_ylabel('ビリアルの RMS 誤差 [eV]')
    ax[1].set_title('(b) ビリアル vs 一様ひずみの FD\n(ゲート: 旧の 3 倍以下かつ相対 3e-3 |W|)', fontsize=10)
    ax[1].legend(fontsize=8)
    save(fig, 'fig05_force_virial.png')


# ---- 図 6: 電場 (VER-0034 review follow-up A) ----
def fig06():
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    grouped(ax[0], ['薄い箱 (n12)', '厚い箱 (n45)'],
            [[4.7e-4, 3.1e-4], [9.4e-4, 2.5e-4], [1.20, 0.28]],
            [C_CT, '#7f7f7f', C_OLD], ['K+U−√ε∞E·P (電場あり)', 'K+U (電場なし)', 'K+U (電場あり)'], log=True)
    ax[0].set_ylabel('エネルギーの幅 (max−min) [eV]')
    ax[0].set_title('(a) 電場中の NVE (1000 ステップ): 保存するのは K+U−√ε∞E·P', fontsize=10)
    ax[0].legend(fontsize=7.5)
    lab = ['E 方向 1', 'E 方向 2', 'E 方向 3']
    grouped(ax[1], lab,
            [[1.4e-5, 2.6e-4, 8.3e-4], [2.2e-4, 2.7e-4, 2.8e-4], [0.50, 0.19, 0.16], [0.32, 0.04, 0.27]],
            [C_CT, '#7bb3dd', C_OLD, '#ecc08f'],
            ['charge_transfer 薄い', 'charge_transfer 厚い', '旧 qNEP 薄い', '旧 qNEP 厚い'], log=True)
    ax[1].set_ylabel('|閉ループ積分| / 被積分関数のスケール')
    ax[1].set_title('(b) 閉じた経路での電場の仕事: 新モデルは保存力(≈0)、旧は経路依存', fontsize=10)
    ax[1].legend(fontsize=7.5)
    save(fig, 'fig06_field.png')


# ---- 図 7: A4 nep vs gpumd (VER-0035 と依頼元の報告) ----
def fig07():
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    names = ['qnep2\nrc_phi4', 'qnep3\nrc_phi4', 'qnep2\nrc_phi3', 'qnep2\n3型', 'qnep3\nn59,n45', '学習済み\nqnep2']
    bec = [4.5e-7, 4.5e-7, 1.8e-7, 7.5e-7, 2.4e-7, 1.5e-6]
    frc = [1.5e-6, 2.4e-6, 1.0e-6, 7.9e-6, 1.3e-6, 1.4e-5]
    for a, v, gate, title, unit in [(ax[0], bec, 5e-6, '(a) BEC', 'e'), (ax[1], frc, 1e-5, '(b) 力', 'eV/Å')]:
        x = np.arange(len(names))
        a.bar(x, v, color=C_CT, label='合成データの RMS 差 (VER-0035)')
        a.set_yscale('log')
        a.set_xticks(x)
        a.set_xticklabels(names, fontsize=8)
        a.axhline(gate, color=C_GATE, ls='--', label=f'A4 ゲート {gate:g} {unit}')
        a.set_ylabel(f'nep − gpumd [{unit}]')
        a.set_title(title + '  (nep 予測 対 gpumd 静的)', fontsize=10)
    ax[0].axhspan(4.6e-6, 1.4e-5, color=C_OLD, alpha=0.3, label='実データ 14 構造の最大差の範囲(依頼元)')
    ax[1].axhspan(1.2e-5, 5.5e-5, color=C_OLD, alpha=0.3, label='実データ 14 構造の最大差の範囲(依頼元)')
    ax[1].text(5, 2.2e-5, '※学習済みは相対 1e-6×RMS F がゲート', fontsize=7, ha='center')
    for a in ax:
        a.legend(fontsize=7.5, loc='upper left')
    fig.suptitle('実データでは差が A4 ゲートの数倍(ただし依頼元の許容より 4 桁小さい)。合成は RMS、実データは最大値で指標が異なる', fontsize=9, y=1.0)
    save(fig, 'fig07_a4.png')


# ---- 図 8: 旧モデルの新旧差 = 実行ごとの雑音 (依頼元の報告 1 章) ----
def fig08():
    q = ['ΔE [eV]', 'max|Δq| [e]', 'max|ΔBEC| [e]', 'max|ΔF| [eV/Å]']
    amor_oo = [3.5e-5, 9.1e-7, 1.1e-6, 9.1e-6]
    amor_on = [4.7e-5, 9.0e-7, 1.3e-6, 7.7e-6]
    oli_oo = [3.7e-6, 8.9e-7, 1.2e-6, 6.9e-6]
    oli_on = [1.5e-7, 8.9e-7, 1.1e-6, 6.9e-6]
    fig, ax = plt.subplots(figsize=(9, 4))
    grouped(ax, q, [amor_oo, amor_on, oli_oo, oli_on], ['#4c72b0', '#9cb6dc', '#dd8452', '#efc3a4'],
            ['アモルファス 248: 旧対旧', 'アモルファス 248: 旧対新', 'オリゴマー 59: 旧対旧', 'オリゴマー 59: 旧対新'], log=True)
    ax.set_title('旧モデルの静的計算: 新旧バイナリの差は、同じ旧バイナリ 2 回の差と同程度\n(= 実行ごとの雑音。依頼元の測定、float32、kspace pppm)', fontsize=10)
    ax.legend(fontsize=8)
    save(fig, 'fig08_old_noise.png')


# ---- 図 9: 5a の収束 (依頼元の報告 3 章) ----
def fig09():
    g = [500, 1000, 1500, 2000]
    v = [0.238, 0.192, 0.173, 0.164]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(g, v, 'o-', color=C_CT, label='新 (charge_transfer 6), loss.out の値')
    ax.axhline(0.159, color=C_CT, ls=':', label='新: held-out BEC RMSE 0.159 e (最終評価)')
    ax.axhline(0.112, color=C_OLD, ls='--', label='旧, 同予算・同 seed: 0.112 e')
    ax.axhline(0.224, color=C_GATE, ls='--', lw=1, label='5a ゲート ≤ 2×旧(同予算) = 0.224 e')
    ax.axhline(0.15, color='gray', ls='-.', lw=1, label='DR-005 A6 の目安 約 0.15 e (旧 0.073 e の 2 倍)')
    ax.set_xlabel('世代')
    ax.set_ylabel('held-out BEC RMSE (全成分) [e]')
    ax.set_title('5a (generation 2000, seed 1): ゲートは満たすが未収束 (依頼元の報告)', fontsize=10)
    ax.legend(fontsize=7.5)
    ax.set_ylim(0.08, 0.26)
    save(fig, 'fig09_5a.png')


# ---- 図 10: DR-006 ----
def fig10():
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    steps = np.arange(0, 201)
    ax[0].axhline(10.0, color=C_GATE, ls='--', label='しきい値 10.0 Å (= 2.5×rc 4 Å)')
    cases = [('small→large (9.3→10.3)', 9.3, 0.005, '#1f77b4'), ('large→small (10.6→9.6)', 10.6, -0.005, '#2ca02c')]
    for lab, L0, r, c in cases:
        ax[0].plot(steps, L0 + r * steps, color=c, label=lab)
    ax[0].set_xlabel('deform のステップ (0.005 Å/step 、time_step 0)')
    ax[0].set_ylabel('箱の長さ [Å]')
    ax[0].set_title('(a) 試験: 各区間が小箱/大箱のしきい値をまたぐ\n(往復 2 ケースは区間を連結)', fontsize=10)
    ax[0].legend(fontsize=8)
    cap = [3.7e-6, 2.6e-6]
    shift = [3.9e-3, 3.2e-3]
    grouped(ax[1], ['L=9.3 Å', 'L=10.3 Å'], [shift, cap], [C_OLD, C_CT],
            ['箱を 1 ステップ(0.005 Å)動かしたときの E_D3 の変化', '合格の許容 (2e-6 |E_D3|)'], log=True)
    ax[1].set_ylabel('[eV]')
    ax[1].set_title('(b) 試験の感度: 1 ステップの取り違えを約 1000 倍の余裕で検出', fontsize=10)
    ax[1].set_ylim(1e-6, 1e-1)
    ax[1].legend(fontsize=7.5, loc='upper right')
    save(fig, 'fig10_dr006.png')


# ---- 図 11: 回帰の内訳 (VER-0030) ----
def fig11():
    fig, ax = plt.subplots(figsize=(9, 2.4))
    parts = [('合格 216', 216, '#55a868'), ('既知の 37 件\n(run-to-run)', 37, '#dd8452'),
             ('説明済 4 件', 4, '#c44e52')]
    left = 0
    for lab, n, c in parts:
        ax.barh(0, n, left=left, color=c)
        ax.text(left + n / 2, 0, lab, ha='center', va='center', fontsize=8, color='white' if n > 30 else 'black')
        left += n
    ax.set_xlim(0, 257)
    ax.set_yticks([])
    ax.set_xlabel('ケース数 (計 257、関係テスト 25/25 合格は別)')
    ax.set_title('DR-006 の `--suite full` (基準: 古い master 基準 3b2c8db8)\n説明済 4 件 = dftd2 キーワードが基準に無い 2 件 + 基準との 1e-10 差 2 件 (HEAD 修正前ビルドとはビット同一)', fontsize=9)
    save(fig, 'fig11_regression.png')


# ---- 図 12: 進捗 ----
def fig12():
    rows = [('DR-005 1 診断 (VER-0028)', 2), ('2a numpy 参照 (VER-0029)', 2), ('2b GPU 電荷・BEC (VER-0032)', 2),
            ('2c 力・ビリアル・電場 (VER-0033/34)', 2), ('3a 学習側 nep (VER-0035)', 2), ('3b replica コード', 1),
            ('3b replica GPU 検証', 0), ('4 文書 (英/日)', 2), ('4 全回帰 --suite full', 0),
            ('5a 表現力の短い確認(依頼元)', 1), ('5b 本学習(依頼元)', 1), ('直接 P 出力', 0),
            ('DR-006 修正・試験・文書', 2), ('DR-006 HEAD基準の全回帰・HIP', 0)]
    col = {2: '#55a868', 1: '#f0c14b', 0: '#c44e52'}
    txt = {2: '完了', 1: '一部/進行中', 0: '未実施'}
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (r, s) in enumerate(rows[::-1]):
        ax.barh(i, 1, color=col[s])
        ax.text(0.5, i, txt[s], ha='center', va='center', fontsize=8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r for r, _ in rows[::-1]], fontsize=8)
    ax.set_xticks([])
    ax.set_title('作業の進捗 (2026-11-11 時点)', fontsize=10)
    save(fig, 'fig12_status.png')


if __name__ == '__main__':
    for f in (fig01, fig02, fig03, fig04, fig05, fig06, fig07, fig08, fig09, fig10, fig11, fig12):
        f()
    print('done')
