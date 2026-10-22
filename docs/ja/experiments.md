# 実験記録

[English](../experiments.md) | 日本語

2026-10-22時点。これは、方法を応用したpruning実験の初期記録であり、UltraBERTの再現完了報告ではありません。方法上の違いは[研究上の比較](research.md)に記載しています。

## 参照実験

参照モデルは`cl-nagoya/ruri-v3-310m`で、revisionを`18b60fb8c2b9df296fb4212bb7d23ef94e579cd3`に固定しています。全25層、768次元の残差ストリーム、既存の系列長上限を維持し、各層のFFN中間幅を3,072から2,304へ減らします。ゲート適用後の活性値の絶対値平均を使い、層ごとにニューロンを選択します。

校正と回復学習には、JSTSの学習splitの先頭128ペアから得た256件の文を使用します。データのrevisionは`5bac629e25799df4c9c80a6a5db983d6cba9e77d`です。意図的に小さくした参照実験であり、代表的な学習データの構成でも、ライブラリが定めるデータ量の上限でもありません。校正と回復学習には同じ学習コーパスを使いますが、最終評価は別のデータで行う必要があります。回復学習の教師は元モデルです。正規化前の文埋め込みのMSEを用い、1エポック、バッチサイズ4、学習率2e-5、seed 42で生徒を学習します。

総パラメータ数は314,611,968から270,375,168へ、約14.06%減少しました。FFN幅25%削減はモデル全体25%削減を意味しません。また、どちらの削減率もレイテンシー改善の証拠にはなりません。

## ソースに含まれるサンプルの実行

ソースのcheckoutから次のコマンドを実行すると、参照データの準備、ニューロン選択、生徒モデルの保存、埋め込みの回復学習、保存したエンコーダーの検証を行えます。

```bash
uv sync --locked --extra training
uv run python examples/prepare_data.py --output data/train.txt
uv run python examples/ruri_pilot.py --texts data/train.txt --target-width 2304 --policy activation --seed 0 --output artifacts/student
uv run python examples/distill.py --student artifacts/student --texts data/train.txt --epochs 1 --seed 42 --output artifacts/recovered
uv run python examples/verify_saved.py --model artifacts/recovered --output reports/recovered-verification.json
```

コーパスのファイル名には`.json`以外の拡張子を使ってください。`.json`は由来を記録するsidecar用に予約しており、衝突する出力パスは書き込み前に拒否します。

サンプルはrevisionを固定した元モデルと学習データセットをダウンロードし、CPUで実行します。生成物とともに由来や測定結果を保存します。選択方法の対照条件には`--policy random`または`--policy weight`を使い、別の出力ディレクトリを指定してください。各方法について結論を出す前に、同等の回復学習と評価が必要です。

この手順で再現するのは参照実験の各段階であり、下流タスクの評価全体ではありません。検証スクリプトが確認するのは少数テキストでの読み込み、次元、正規化です。JMTEBの実行や、8,192トークン全体を使う入力の検証は行いません。

## 品質に関する途中結果

以下は、同じ元モデルと回復済み生徒モデルについて、NVIDIA A100-SXM4-40GBでモデルごとに1回ずつ実行し、完了した比較ペアです。スコアはnDCG@10の100倍で、差分は回復後から元モデルを引いた値です。

| タスク | 元モデル | pruning＋回復後 | 差分（ポイント） |
| --- | ---: | ---: | ---: |
| JaCWIRRetrieval | 84.027 | 83.950 | -0.077 |
| MultiLongDocReranking | 93.495 | 90.961 | -2.534 |

どちらもfloat32でTF32を有効にし、SDPA、最大系列長8,192、バッチ上限64を使用しました。系列長に応じてバッチを調整し、attention-cellの予算は4,194,304です。長い入力ではバッチサイズ1になる場合もあります。環境はPyTorch 2.14.1+cu130、Transformers 4.57.6、Sentence Transformers 5.7.0、datasets 5.0.1、MTEB 2.26.3、JMTEB 2.1.0です。

両タスクとも日本語のtestデータを使用しました。JaCWIRのデータrevisionは`abf6d5fb6759ad516f998bd887b75420d595672e`で、4,000クエリ、513,107文書、コーパスのチャンクサイズ1,024、top-k 1,000です。MultiLongDocRerankingのrevisionは`ad09ce14c17bce6edae151b7f6ef12e15d91dbf3`で、200クエリ、1,581件の候補文書です。クエリと文書には、それぞれ`検索クエリ: `と`検索文書: `を接頭辞として使いました。

この観測では、長文の再順位付けで品質が低下しています。ただし、pruningと回復学習の影響を切り分けた結果ではありません。同条件のpruning直後の結果と、対照実験一式が必要です。また、一般的な高速化やJMTEB全体のスコアも実証しません。バッチサイズ16の実験を含め、推論条件が違う結果は集約していません。

評価用プログラム全体と実行時の生のアーカイブは、このソース公開には含まれていません。この表は実験の途中記録であり、現状では、このcheckoutだけから第三者が独立して再実行できるベンチマークではありません。再現可能な評価の実行手段と共有可能な結果の公開は、[ロードマップ](roadmap.md)に残る作業です。開発用のロックファイルはソースのサンプルを対象としており、JMTEB評価環境全体は含みません。
