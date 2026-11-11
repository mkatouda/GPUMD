# 第 1 章 使い方

## 1.1 `nep.in`

```
type 2 Na Cl
model qnep2            # または charge_mode 2 / 3（qnep2, qnep3 のみ）
cutoff 6 5
charge_transfer 6      # rc_phi（Å）。必須、既定値なし
```

- 構文：`charge_transfer <rc_phi>`（数値 1 個）。1 回だけ指定できます。
- 値域：`0 < rc_phi ≤ 動径カットオフ`（`cutoff` の第 1 値）かつ `rc_phi ≤ 100` Å。NaN・inf・非数値・0・負値・極端に小さい値は拒否されます。
- 対象モデル：`qnep2`、`qnep3`（旧 `charge_mode 2`、`3`）。
- 追加される最適化パラメータ：`N_pair × (basis_size_radial + 1)` 個。`N_pair = N_type(N_type+1)/2`、`basis_size_radial` は `basis_size` の第 1 値。パラメータ全体の**末尾**に付き、初期値は [0, 0.1] の一様乱数（他の変数の乱数列を乱さないよう、他の全変数の**後**に引く）、他の重みと同様に正則化（`lambda_1`、`lambda_2`）されます。SNES が他の変数と同様に更新します。
- φ の係数は、型対の**最初の型のグループ**に属するものとして、型ごと・全体の L1/L2 項に入ります [VER-0035]。
- 推奨の初期値の目安：依頼元の計画では `rc_phi` = 動径カットオフ（例：`cutoff 6 5` なら 6）を既定案としています。他の値は 5a の結果が悪かった場合の代替として扱う、としています（保証ではありません）。

### ネットワークの意味の変更

ネットワークの電荷出力は、従来の「電荷 q」ではなく**ポテンシャル h** を意味します。したがって、`charge_transfer` なしで学習したネットワークは流用できず、**最初から学習し直す**必要があります。

### 学習データ

- 電荷の目標値は不要です。総電荷は常に 0 なので、`train.xyz`/`test.xyz` に **0 でない `charge=` ラベル**（絶対値 1×10⁻⁶ 超）があると入力エラーです（`charge=0` は可）。
- BEC は従来どおり BEC ラベル（`bec_train.out` など）で学習します。

## 1.2 `nep.txt` の形式

legacy テキスト形式（`gpumd` が読む形）の構成 [VER-0032]：

```
nep4_charge2 2 Na Cl
cutoff rc_radial rc_angular MN_radial MN_angular
charge_transfer rc_phi          <- cutoff の「直後」の追加行（旧バイナリはここで停止する）
n_max / basis_size / l_max / ANN ...
（数値が 1 行 1 個）
  1. ANN ブロック: (dim+3)×neurons×types + 2
  2. 記述子の係数 c
  3. q_scaler（dim 個）
  4. φ ブロック（charge_transfer 行がある場合のみ、ファイルの最後）:
     num_pairs × (basis_size_radial + 1) 個、ペア主順
```

- φ のペア番号（型の上三角、行優先、t1 ≤ t2）：`p = t1·nt − t1(t1−1)/2 + (t2 − t1)`。2 型なら p(0,0)=0、p(0,1)=1、p(1,1)=2。
- ペア p の係数 k は φ ブロックの `p·(basis_size_radial+1) + k` 番目。`φ(r) = Σ_k c_k f_k(r) fc(r)`、カットオフは `rc_phi`（既存の動径基底と同じ形）。
- `nep` が書く新形式の `nep.txt` では、ヘッダーに `charge_transfer rc_phi` のフィールド、`@ q_scaler` の後に `@ phi_parameters N` のブロックが付きます。`gpumd` が読む legacy テキストへは変換されます。
- `charge_transfer` 行の無いファイルは従来どおり読まれ、結果は変わりません。
- 新しい `nep.txt` を**古いバイナリ**で読むと、不明な行で明確なメッセージを出して停止します（黙って無視しません）[ADR-0006 項目 5]。
- 型の絞り込み（`active`、化学種の圧縮）は、φ ブロックのペアの部分選択に対応しています（実装済みだが、`gpumd` 側の実行は未確認 [VER-0035 Not verified]）。

## 1.3 `gpumd` での使い方

```
potential nep.txt
kspace pppm               # または ewald
ensemble nve              # 例
add_efield ... bec        # BEC を使う電場（任意）
compute_dpdt ...          # dP/dt（任意）
dump_xyz 1 d.xyz precision double force charge bec
run 1000
```

- キーワードの変更はありません。`charge_transfer` モデルでも `potential nep.txt` で読みます。
- `dump_xyz ... bec`、`bec.out`、`compute_dpdt`、`add_efield ... bec` は同じ BEC 配列（9 成分、添字 `[3β+α]`）を読みます。`scale_bec` の因子 √ε∞ は従来どおり掛かります（定数倍で、積分可能性を変えません）。
- `dump_xyz` の `charge` は q（中性）です。平均は引かれません。
- **電場中のエネルギー**：`add_efield` は力だけを加えます。`thermo.out` のエネルギーに −E·P は含まれないため、電場中では **K+U は保存せず、K+U−√ε∞ E·P が保存**します（1000 ステップの幅：K+U−√ε∞E·P は 4.7×10⁻⁴ / 3.1×10⁻⁴ eV、K+U のみは 1.20 / 0.28 eV [VER-0034]）。
- 周期系の P そのもの（直接の分極）の出力は**未実装**です（DES-0005 3b）。

## 1.4 エラーとなる場合（設計どおり）

`nep`（学習側）で入力エラーになるもの [`charge_transfer.rst`, VER-0035]：

| 組み合わせ・内容 | メッセージの方向性 |
|---|---|
| `qnep1`（`charge_mode 1`）、通常の `nep`、`qnep3_vdw`（`charge_vdw`）、`nep_vdw`、spin、`model_type` 系（dipole、polarizability、temperature など） | "charge_transfer requires charge_mode 2 or 3 (qnep2/qnep3)." または "not supported by this model" |
| `fine_tune` との併用、`USE_CJ` ビルド | エラー |
| `rc_phi` の欠落、非数値、0 以下、極端に小さい値、100 超、動径カットオフ超 | エラー |
| `charge_transfer` の重複指定 | エラー |
| 0 でない `charge=` ラベル | "charge_transfer models are neutral by construction, but a structure has a non-zero charge= label" |
| 既存の `nep.txt` を読む予測モードで、`nep.in` の `charge_transfer` と `nep.txt` の設定が不一致 | エラー |

`gpumd` / `gpumd_replica`（`nep.txt` を読む側）：`charge_mode` が 2/3 以外、`rc_phi ≤ 0`、`rc_phi > rc_radial`、φ ブロックの個数の過不足はエラーです。`vdw`/`charge_vdw`/spin のモデルは新形式の読み取りで `unhandled field 'charge_transfer'` として拒否されます。

## 1.5 `gpumd_replica`

- `qnep2`（`nep4_charge2`）のみ対応です。`charge_mode 3`（`nep4_charge3`）は、`gpumd_replica` が元々知らないモデル名のため既存の検査で拒否されます。
- `gpumd_replica` では BEC を計算しません（`charge_transfer` モデルでは BEC のカーネルを省略）。
- 可変 ZBL（flexible ZBL）と `charge_transfer` の併用は、配置が未定義のため拒否されます。
- **GPU での検証は未実施**です（05 章）。配布済みの `bec_rev0` の `gpumd_replica` は、`charge_transfer` を受け付けない版（3b より前）です。

## 1.6 利用上の注意

1. **総電荷の和の浮動小数点の限界**：`Σq` は q の float 表現で制限され、約 1×10⁻⁷ √N |q|_typ です（例：|q| ≈ 1、N = 10 で 3×10⁻⁷）[VER-0032]。テストの合格基準（1×10⁻⁶ e）は |q| ≲ 1 の範囲で成り立ちます。
2. **GPU の実行ごとの雑音**：float の `atomicAdd` の順序のため、BEC の下位桁は実行ごとに変わります（既存の仕様。旧モデルでも同様）。特に、大きな系では旧バイナリ同士でも ΔE 3.5×10⁻⁵ eV、Δq 9×10⁻⁷、ΔF 9×10⁻⁶ eV/Å の差が出ます [依頼元の報告 2026-11-11、05 章]。
3. **`nep` と `gpumd` の差**：合成データでは BEC 4.5×10⁻⁷ e（RMS）、力 1.5×10⁻⁶ eV/Å で一致しますが（04 章）、実データでは BEC 最大 1.4×10⁻⁵ e、力 最大 5.5×10⁻⁵ eV/Å（原子数が多いほど大きい傾向）の差が報告されています（05 章）。
4. **GPU の共有**：実行前に `nvidia-smi` を確認してください。
5. **旧バイナリ・旧モデルは変更しない**：新バイナリは `/home/katouda/bin/GPUMD/bec_rev0/` にあり、旧バイナリ（`/home/katouda/bin/GPUMD/`）は変更していません。
