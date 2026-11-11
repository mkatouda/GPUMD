# 第 3 章 実装

## 3.1 ソース構成

| 場所 | 内容 |
|---|---|
| `src/force/nep_charge/nep_charge_transfer.cuh`（新規） | `gpumd` 側のカーネル：`ct_find_phi`、`ct_find_charge_{large,small}_box`、`ct_find_bec_B_{large,small}_box`、`ct_find_bec_A_{radial,angular}_{large,small}_box`、`ct_find_Dt_and_pair_force_{large,small}_box`、補助（`ct_add_term_B`、`ct_scatter_term_A`、`ct_radial_dh`、`ct_angular_dh`、`ct_pair_force`、`ct_store_force`） |
| `src/force/nep_charge/nep_charge.cu`、`nep_charge.cuh` | `nep.txt` の解析（`charge_transfer` 行、φ ブロック）、`ParaMB`/`ANN`/`NEP_Charge_Data` の拡張、大箱・小箱の経路への組み込み。`paramb.charge_transfer` で分岐 |
| `src/main_nep/nep_charge/nep_charge_transfer.cuh`（新規）、`nep_charge.cu` | 学習側（`nep`）のカーネル。gpumd の移植（算術は同じ。型 `NEP_Charge::ParaMB/ANN` が `force/` と `main_nep/` で別で、近傍リストの配置も違うため、ヘッダーの共有は不可能） |
| `src/main_nep/common/`（`nep_parameters.cu`、`nep_charge_parameters.cu`、`nep_charge_reader.cu`、`nep_io.cu`） | キーワードの解析と検査、パラメータ数、`nep.txt` の入出力、`charge=` ラベルの検査 |
| `src/utilities/nep_io/`（`nep_model_file*.cu*`、`compact_nep.cu`） | `nep.txt` の新形式 ↔ legacy テキストの変換、化学種の圧縮（φ ブロックのペアの部分選択）。旧経路の不具合を 1 件修正（VER-0035） |
| `src/main_replica/force/nep_charge_transfer.cuh`（新規）、`nep_charge.cu`、`nep_charge.cuh` | `gpumd_replica` 側。gpumd のヘッダーのコピー（共有しない。BEC のカーネルは移植しない）。モード 2 のみ |
| テスト | `tests_pytest/test_qnep_charge_transfer_{parse,bec,forces,field}.py`、`test_nep_charge_transfer_train.py`、`test_replica_charge_transfer.py`。生成器・参照実装・検証スクリプトは `specs/verification/VER-0029/0032〜0036/scripts/` |

仕様書の置き場所は `specs/`（フォーク専用。upstream への PR には含めない）です。

## 3.2 `gpumd` の 1 ステップの流れ（`charge_transfer` モデル）

[VER-0032、VER-0034]

1. `find_descriptor`：記述子とネットワーク（h = 電荷出力）を計算し、`nep_data.ct_h` にコピー。
2. `ct_find_charge_*`：`q_a = Σ_b φ_ab (h_a − h_b)`（原子ごとに **double で累算**、結果は float で `nep_data.charge` へ）と `V_a = Σ_b φ_ab (r_a − r_b)`（`nep_data.ct_V`）。**`zero_total_charge` の代わり**です。φ は既存の `find_fc_and_fcp`/`find_fn_and_fnp` を `rc_phi` で使います。近傍は動径リストです。
3. BEC：項 B のカーネルが 9 成分を書き（q·1 を含む。`find_bec_diagonal` は走らない）、項 A のカーネル（動径、角度）が `V ⊗ ∂h/∂r` を `atomicAdd` で加え、最後に従来の `scale_bec`。従来の `find_bec_radial/angular` は走りません。
4. 長距離（Ewald/PPPM）：q を使い、D_a = ∂E/∂q_a（`nep_data.D_real`）を得る。モード 1 の実空間部分、モード 3 の自己エネルギーは従来どおり。
5. `ct_find_Dt_and_pair_force_*`：`Dt_k = Σ_b φ_kb (D_k − D_b)`（`nep_data.ct_Dt`。近傍の D を読む）と、明示的なペア力・ビリアル（`G = φ'(d)(h_k − h_b)(D_k − D_b)`、`F_k += G v/d`、`W −= 0.5 G v⊗v/d`）を **double で**計算し、`force_per_atom` と `virial_per_atom` へ加えます（ビリアルの添字の対応 0 3 4 / 6 1 5 / 7 8 2）。`zero_mean_D_real` は**走りません**。
6. `ct_Dt` を、既存の `find_force_radial`、`find_partial_force_angular`（`find_properties_many_body` が角度部分の力とビリアルを f12 から出す）、小箱用の `find_force_*_small_box` の重み `g_D_real` として渡す（カーネルは無変更）。

大箱と小箱のどちらのカーネルでも同じ流れです（小箱経路は箱の厚さが `2.5 (rc_radial + 1)` 以下のとき。`rc 4` で 12.5 Å）。

### 数値上の工夫

- 新しい総和は double で累算し、float の丸めの蓄積を避けています。
- 近傍は動径リストを使い、**a 中心の和**だけなので、新たな近傍リストや二段の最小イメージは不要です。
- 新カーネルは**汎用経路のみ**（学習側でも、これらのモデル用に特殊化したコンパイル済みカーネルは作っていません）。

## 3.3 学習側（`nep`）

[VER-0035]

- パラメータ：`number_of_variables += pairs × (basis_size_radial + 1)`。φ ブロックは**ベクトルの末尾**。`nep.restart` に含まれ、個数の検査に入ります。
- 流れ（`find_force`）：記述子 → ANN（h）→ h のコピー → `ct_find_charge`（`charge` と `charge_shifted` に q、V）→ BEC（項 B + 項 A、`scale_bec`）→ 長距離（q を使う）→ `ct_find_Dt_and_pair_force` → 従来の記述子の力カーネル（重み Dt）。
- **中性化の処理**：学習側には従来、(1) `enforce_nep_total_charge`（`charge=` ラベルに合わせて q をシフト）と (2) `subtract_nep_charge_derivative_mean`（D から平均を引く）があります。**`charge_transfer` モデルではどちらも走りません**。
- 電荷の損失 `(Σ charge − charge_ref)/Na` は `charge` 配列を使うため、`charge_transfer` モデルでは `charge` が q になり、損失は |charge_ref|/Na（中性のラベルなら 0）です。0 でない `charge=` ラベルは表せないので入力エラーにしています。`charge_*.out` は q（`gpumd` がダンプするもの）です。
- `nep.in` の `charge_transfer` と `nep.txt` の値は、予測モードで一致を検査します。
- 長距離は Ewald（α = π / rc_radial、`|k|² < (2πα)²`）で、`gpumd` の `kspace ewald` と同じ設定です（検証のため設定を合わせる必要はありません）。
- 薄い箱（L < 2 rc）は、学習側の内部の複製を使います（gpumd との一致で確認 [VER-0035]）。

## 3.4 `gpumd_replica`

[VER-0036]

- 追加：`ParaMB::{charge_transfer, rc_phi, num_phi_pairs, num_phi_per_pair}`、`ANN::phi_coeff`、`NEP_Charge_Data::{phi_parameters, ct_h, ct_Dt, ct_V}`。パーサーは `cutoff` の直後の `charge_transfer rc_phi`（`charge_mode` 2/3、`0 < rc_phi ≤ rc_radial`、`gpumd` と同じエラー文）と、q_scaler の後の φ ブロック（EOF までを要求）を読みます。バッファは構造体の構築時に確保し、`compute()` では確保しません。
- 力の経路：`zero_total_charge` を、h のコピー（`copy_from_device_async`、レプリカのストリーム）と `ct_find_charge_*` に置換。BEC カーネル（`find_bec_*`、`scale_bec`）は**省略**。`zero_mean_D_real` を `ct_find_Dt_and_pair_force_*` に置換し、`ct_Dt` を力カーネルの重みにします。
- バッチ積分（`replica_batch.cu`）は NEP（エネルギーモデル）のみで、qNEP は `Replica_Runtime::initialize_batches` で除外されます（"batched integration: off (not supported for qNEP)"）。`charge_transfer` モデルはバッチ経路に入りません。
- `charge_transfer` なしのモデルは旧経路（起動順序・起動数が同じ）を通ります。旧カーネルの SASS を比べると、22 本中 20 本が同一（定数バンクの正規化後）で、残る 2 本（`find_descriptor`、`find_descriptor_small_box`）は `ParaMB` の増大による定数ロードのオフセット（+0x10）だけが違います。

## 3.5 互換性（ADR-0006 の契約）

| 項目 | 内容 |
|---|---|
| 追加的 | 新しいキーワードと `nep.txt` の行・ブロックのみ。既存キーワードの変更なし |
| 旧モデル・旧入力 | 電荷・力・エネルギーは変更前とビット同一（`gpumd` の静的計算、学習側の学習・予測）。BEC とビリアルの下位桁だけ、既存の `atomicAdd` の順序の雑音がある（旧バイナリ同士でも同程度） |
| 旧カーネル | `cuobjdump -sass` で、既存の 30 本（`nep_charge.o`）が基準ビルドと同一（定数バンクのオフセットの正規化後）。新規は `ct_*` のみ |
| 旧バイナリ + 新 `nep.txt` | 明確なメッセージで停止 |
| 新モデルを旧モデルへ変換 | 不可（電荷の定義が違う） |
| 適用範囲外 | `vdw`、`charge_vdw`、spin、`charge_mode 1`、`model_type` 系、`fine_tune`、`USE_CJ` |
