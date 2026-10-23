# 実験記録

[English](../experiments.md) | 日本語

2026-10-24時点。これは、方法を応用したpruning実験の初期記録であり、UltraBERTの再現完了報告ではありません。方法上の違いは[研究上の比較](research.md)に記載しています。

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

| タスク | split | 元モデル | pruning＋回復後 | 差分（ポイント） |
| --- | --- | ---: | ---: | ---: |
| JaCWIRRetrieval | test | 84.027 | 83.950 | -0.077 |
| MIRACLRetrieval | dev | 72.310 | 64.454 | -7.856 |
| MrTidyRetrieval | test | 61.638 | 56.602 | -5.036 |
| MultiLongDocReranking | test | 93.495 | 90.961 | -2.534 |
| MultiLongDocRetrieval | dev | 44.327 | 42.542 | -1.785 |
| MultiLongDocRetrieval | test（補足） | 44.019 | 40.276 | -3.743 |

いずれもfloat32でTF32を有効にし、SDPA、最大系列長8,192、バッチ上限64を使用しました。系列長に応じてバッチを調整し、attention-cellの予算は4,194,304です。長い入力ではバッチサイズ1になる場合もあります。環境はPyTorch 2.14.1+cu130、Transformers 4.57.6、Sentence Transformers 5.7.0、datasets 5.0.1、MTEB 2.26.3、JMTEB 2.1.0です。

すべて日本語のデータを使用しました。JaCWIRのデータrevisionは`abf6d5fb6759ad516f998bd887b75420d595672e`で、4,000クエリ、513,107文書、コーパスのチャンクサイズ1,024、top-k 1,000です。MultiLongDocRerankingのrevisionは`ad09ce14c17bce6edae151b7f6ef12e15d91dbf3`で、200クエリ、1,581件の候補文書です。MultiLongDocRetrievalのrevisionは`837028901907a7d419b4ab906f28e011ce1cc824`で、各splitは200クエリ、10,000文書、コーパスのチャンクサイズ1,024、top-k 1,000です。このタスクのJMTEB集計はdevを採用するため、testは補足として別に報告します。2つのsplitを独立した2タスクと数えたり、平均したりしません。クエリと文書には、それぞれ`検索クエリ: `と`検索文書: `を接頭辞として使いました。

MIRACLRetrievalはdev、860クエリ、6,953,614文書、revision `9c09abc13478308c27598f350e31d8f06b9b5481`を使用しました。MrTidyRetrievalは日本語のtest subset、720クエリ、7,000,027文書、revision `fc24a3ce8f09746410daee3d5cd823ff7a0675b7`です。両ペアとも上記の推論条件、コーパスのチャンクサイズ1,024、top-k 1,000を使用しました。完了した検索checkpointで全文書の処理を確認し、元モデルと回復後でコーパスのfingerprintとクエリ順序が一致しています。評価コードのhash、ライブラリのバージョン、回復後checkpointのhashも過去の実行と一致します。

元モデルと回復後の2条件の観測では、MIRACL、Mr.TyDi、長文の検索と再順位付けで品質が低下しています。下記のpruning直後を含む比較では、5タスクについてcheckpoint間の変化を分けて確認します。MIRACL、Mr.TyDi、長文retrievalのpruning直後の結果は未取得です。一般的な高速化やJMTEB全体のスコアは実証していません。バッチサイズ16の実験を含め、推論条件が違う結果は集約していません。

## 3条件の参照パネル

固定したパネルで、元モデル・activation pruning直後・回復後のcheckpointを、上記と同じA100の推論条件で評価しました。各checkpointを1回だけ読み込み、JSTS・JSICK・JaGovFaqsRetrievalで使い回しています。全9評価が完了し、評価controllerの所要時間は約298秒でした。環境構築・対照モデルの復元・exportは含まず、純粋なエンコードのレイテンシーでもありません。スコアは各指標の100倍で、回復差分は回復後からpruning直後を引いたポイント数です。

| タスク | split | 指標 | 元モデル | pruning直後 | 回復後 | 回復差分 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| JSTS | validation | cosine Spearman | 83.5880 | 83.5490 | 85.3377 | +1.7887 |
| JSICK | test | cosine Spearman | 77.9658 | 78.7879 | 79.6457 | +0.8578 |
| JaGovFaqsRetrieval | test | nDCG@10 | 77.3160 | 76.8680 | 76.9920 | +0.1240 |

データのrevisionはJSTSが`5bac629e25799df4c9c80a6a5db983d6cba9e77d`、JSICKが`729cfe4a16d3c2b61c6aa9f9f6c8a96bb5512868`、JaGovFaqsRetrievalが`0726b2af53c907628067871e2e7c84e0d8e099c2`です。検索では各条件で2,048クエリに対し全22,794文書を処理し、コーパスの同一性とクエリ順序を照合しました。復元したpruning直後のテンソルは過去のpruning直後checkpointと一致し、回復後checkpointも固定しています。tokenizerの規則・実際のencoding・モデルのpipeline設定の互換性も確認しました。過去の回復学習レポートには学習前重みの暗号学的hashがないため、この比較から回復後checkpointの学習系譜を遡及して証明することはできません。

このパネルでは、回復学習によって文類似度の2スコアが上昇しました。JaGovFaqsではpruningによる0.4480ポイントの低下のうち0.1240ポイントを取り戻しましたが、元モデルからは0.3240ポイント低いままです。各条件1回の実行から得た、タスクに依存する観測です。JSTS validationは学習splitとは別ですが、校正・回復に使ったデータセットと同じ領域の評価です。有意差やMIRACL・Mr.TyDi・長文retrievalの悪化原因は実証していません。別に実施したreranking診断は下記に記載します。random/weight対照、幅の比較、反復実行は今後の課題です。3タスクの異なる指標を平均しません。

## 3条件のreranking診断

別の診断実行で、参照パネルと同じ3つのcheckpoint・ソースhash・ライブラリ版・推論条件を使い、JQaRAとMultiLongDocRerankingを評価しました。各モデルを1回だけ読み込み、両タスクで使い回しています。全6評価が完了し、評価controllerの所要時間は約1,407秒でした。環境構築・対照モデルの復元・exportは含みません。スコアは各指標の100倍で、回復差分は回復後からpruning直後を引いたポイント数です。

| タスク | split | 指標 | 元モデル | pruning直後 | 回復後 | 回復差分 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| JQaRAReranking | test | MAP@1000 | 63.469 | 58.586 | 58.498 | -0.088 |
| MultiLongDocReranking | test | nDCG@10 | 93.495 | 90.888 | 90.961 | +0.073 |

JQaRAのデータrevisionは`63254fa512e133e6300e35b6f9175c32c6cc8455`、subsetは`default`です。MultiLongDocRerankingは上記のrevision、subset `ja`を使用しました。長文rerankingの元モデルと回復後のスコアは、報告した桁数で過去の2条件実行と一致しています。準備記録では、固定したpruning plan、復元テンソルのsignature、回復後checkpoint、tokenizerの互換性を確認しました。

この固定checkpoint間の比較では、JQaRAはpruningによって4.883ポイント、長文rerankingは2.607ポイント低下しています。回復による変化はそれぞれ-0.088、+0.073ポイントで、元モデルからの低下は4.971、2.534ポイント残りました。参照パネルの文類似度では改善しましたが、現在の小規模コーパスによる埋め込みMSEの回復recipeでは、rerankingの品質を取り戻せていません。この観測は今回のcheckpointとタスクに限られ、pruningや知識蒸留という手法全般の有効性を判定するものではありません。

この2タスクは既知の悪化を調べるために選んだ診断であり、独立したholdoutではありません。各条件1回の評価で、小さい回復差分について統計的な有意差を主張しません。2本のNotebookで完了したのは5タスクの計15評価であり、全JMTEBの3条件比較ではありません。MIRACL・Mr.TyDi・長文retrievalのpruning直後評価、選択方法の対照、反復、公開環境で再実行可能な評価は残課題です。このtest結果で幅や回復学習の条件を調整した後、同じtestを未使用の最終評価として扱わないでください。

評価用プログラム全体と実行時の生のアーカイブは、このソース公開には含まれていません。この表は実験の途中記録であり、現状では、このcheckoutだけから第三者が独立して再実行できるベンチマークではありません。再現可能な評価の実行手段と共有可能な結果の公開は、[ロードマップ](roadmap.md)に残る作業です。開発用のロックファイルはソースのサンプルを対象としており、JMTEB評価環境全体は含みません。
