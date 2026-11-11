# GPUMD 向け 電荷保存型 qNEP（`charge_transfer`）と厳密な Born 有効電荷（BEC）日本語マニュアル

本マニュアルは、GPUMD フォーク（mkatouda/GPUMD）で実装した **電荷保存型 qNEP（`nep.in` の `charge_transfer` キーワード）** と、その **BEC が分極 P の厳密な勾配になること** について、利用者と開発者の両方に向けて日本語でまとめたものです。フォーク固有の資料であり、upstream（brucefan1983/GPUMD）の公式マニュアルではありません。数値や主張は `specs/`（要求仕様 DR-005、設計 DES-0005、ADR-0006、検証記録 VER-0028〜VER-0036）、英語の公式マニュアルページ（`doc/nep/input_parameters/charge_transfer.rst`、`doc/theory/qnep_charge_transfer.rst`）、およびソースコードの読解に基づきます。出典は角括弧に示します。

## 版情報

| 項目 | 内容 |
|---|---|
| 作成日 | 2026-11-11 |
| 対象ブランチ・コミット | フォークの `develop`。作成時点の HEAD は `a74ee293`（実装の最終コミットは段階 3b の `338e7955`） |
| 対象とする要求 | DR-005（qNEP の BEC を分極 P の勾配と整合させる）。状態は **`partial`**（段階 1〜4 の大部分は完了。段階 3b の GPU 検証、全回帰、段階 5 の再学習は未完。05 章） |
| 動作確認環境 | NVIDIA RTX A6000（sm_86）、CUDA 12.3、float32 ビルド（`make CUDA_ARCH="-arch=sm_86" -j4`） |
| 検証に使ったデータ | **合成データのみ**（乱数の重みと構造）。物理的な妥当性は未確認（Tier P：試作） |
| 配布バイナリ | `/home/katouda/bin/GPUMD/bec_rev0/`（コミット `93d203d4` から。`gpumd_replica` は `charge_transfer` を受け付けない版） |

## 目次

| 章 | ファイル | 内容 |
|---|---|---|
| 0 | [00_introduction.md](00_introduction.md) | 背景（なぜ必要か）、結論、使い分け |
| 1 | [01_usage.md](01_usage.md) | `nep.in` の書き方、`nep.txt` の形式、エラー、運用手順、注意 |
| 2 | [02_theory.md](02_theory.md) | モデルの式、厳密な BEC の導出、力・ビリアル、電場中のエネルギー |
| 3 | [03_implementation.md](03_implementation.md) | ソース構成、カーネルの流れ、学習側・replica 側、互換性 |
| 4 | [04_verification.md](04_verification.md) | 検証の方法と結果（診断、参照実装、GPU、学習側、replica） |
| 5 | [05_status_limits.md](05_status_limits.md) | 現状、未完了の作業、限界、再学習の途中経過、再現手順 |

## 読者別の読み方

| 読者 | 推奨する章 |
|---|---|
| まず使ってみたい利用者 | 00 → 01 |
| 結果の信頼性を知りたい利用者・共同研究者 | 00 → 04 → 05 |
| 理論的な背景を知りたい方 | 02 |
| コードを読む・修正する開発者 | 03 → 04 → 05 |

## 表記上の約束

- 文体は「です・ます調」です。キーワード、変数名、ファイル名、関数名は原文のまま表記します。
- 仕様書番号（DR-/DES-/ADR-/VER-/UPSTREAM-）は、フォーク内の `specs/` を参照します。upstream の利用者はアクセスできません。
- BEC の配列の添字は `[3β + α] = ∂P_β/∂r_α`（GPUMD の既存の規約）です。BEC のテンソルは対称ではありません。
- 数値には出典を付けます。出典に無い測定値は書かず、「未測定」と明記します。
- 行番号の引用は使いません（関数名・ファイル名で探してください）。
