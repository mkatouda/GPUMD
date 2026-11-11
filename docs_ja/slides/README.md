# 発表スライド（pptx）: DFT-D3/D2 の高速化と qNEP の Born 有効電荷（BEC）の厳密化

- `GPUMD_dftd_qnep_report.pptx`: 本編 26 枚 + 付録・参考資料 37 枚（計 63 枚）。想定読者は材料シミュレーションの専門家。テンプレートは RIST-template.pptx（このリポジトリには含めない）。
- 内容は `docs_ja/workreport`、`docs_ja/dftd`、`docs_ja/qnep` と `specs/` の転記で、新たな測定値はない。各スライドの発表者ノートに出典（VER/DR/ADR 番号、節）を入れた。本編の下部の「→ A-n / B-n / C-1」は付録へのリンク。
- 図は `figures/`（N1〜N6: D2/D3、S01〜S12: `docs_ja/workreport` の図のスライド用再描画、E1〜E8: 付録の式）。

## 再生成
```
PY=/home/katouda/miniforge3/envs/py312-gpumd/bin/python        # matplotlib、フォント Noto Sans CJK JP
cd docs_ja/slides
$PY scripts/make_slide_figures.py figures          # N1〜N6
$PY scripts/make_slide_figures_ct.py figures       # S01〜S12
$PY scripts/make_equations.py figures              # E1〜E8
RIST_TEMPLATE=/home/katouda/data/RIST-template.pptx \
  /home/katouda/miniforge3/envs/py312-pptx/bin/python scripts/build_all.py figures out.pptx   # python-pptx
```
`scripts/build_slides.py` は本編 26 枚のみを作る（`build_all.py` が読み込む）。

## 未確認・限界
- レイアウト（文字のあふれ、表の折り返し、図中の文字の読みやすさ）は、この環境で LibreOffice が動かないため、PowerPoint で確認していない。文字量は見積りで抑えている。
- 付録 A-13 の R3 変種の時間は、記録に測定系の明記がない。スライド 10 の「NEP 単独に対する倍率」は別セッションの NEP のみの値を使った導出値。
- VER-0035 の表見出しの力のゲート（1e-4）と、人間が決めた A4 の値（1e-5）の不整合は、付録 B-9 に注記した（未解決）。
