# 埋め込みの回復学習サンプル

[English](../recovery.md) | 日本語

これは現在のプロトタイプ向けの、文埋め込みを蒸留するサンプルです。UltraBERTのtoken-logit蒸留や事前学習の目的関数は実装していません。[研究上の比較](research.md)を参照してください。回復学習はサンプルスクリプトとして提供しており、汎用的な学習APIではありません。

training extraをインストールし、1行に1つの学習用テキストを記載したUTF-8ファイルを用意します。用途に合うタスク接頭辞をテキストに含めてください。

```bash
uv sync --locked --extra training
uv run python examples/distill.py --student artifacts/student --texts data/train.txt --output artifacts/recovered
```

このサンプルは、教師の正規化前の埋め込みを事前計算して固定し、Sentence TransformersのMSELossで生徒を学習します。既定はCPU、バッチサイズ4、学習率2e-5、1エポックです。参照教師を変更する場合はrevisionを明示的に固定してください。この例は出発点であり、最適と検証済みの回復学習スケジュールや、汎用の学習フレームワークではありません。

学習・校正データと最終評価データは分けてください。学習時のMSEが下がっても、検索順位の回復が実証されたことにはなりません。回復後、pruning直後、元モデルを同じ条件で評価してください。[実験記録](experiments.md)には小規模な参照実験と観測した品質低下を、[ロードマップ](roadmap.md)には今後の回復学習への対応を記載しています。
