# Deepvoice3の再現実装

[r9y9](https://github.com/r9y9/deepvoice3_pytorch) 様の実装したDeepvoice3を、より論文に近いネットワーク構造へと実装し直しました。具体的な変更点は  
- 言語処理部を、論文の形式へと変更  
- 1×1convをすべて全結合層（FC）へと変更  
- attention layerを全てのDecoder layerに適用  
- positional encodingはEmbeddingで特徴量次元に合わせるのではなく、特徴量方向にexpandしたものを使用  
- 各種ハイパーパラメータを論文に遵守
- `torch.nn.utils.weight_norm(dim=None)`に変更し，各層の出力を正規化するのではなく，ネットワーク全体を通して出力を正規化するように変更．詳しくは[こちら](https://pytorch.org/docs/master/generated/torch.nn.utils.weight_norm.html)

1. [arXiv:1710.07654](https://arxiv.org/abs/1710.07654): Deep Voice 3: Scaling Text-to-Speech with Convolutional Sequence Learning.

# use_waveglow Repository

[WaveGlow](https://github.com/NVIDIA/waveglow) を利用できるように修正．ただし，こちらのRepositoryをforkして，それをsubmoduleとして扱えるようにWaveGlowも修正している．
masterとの変更点として
- WaveGlowに合わせてメルスペクトログラムのfft_size等を調整
- メルスペクトログラムを正規化せずに学習
- このRepositoryでは，waveglowを利用する前提のため，Linear, WORLDの学習データは出力しない
- LJSpeechのみ対応


## Setup
### cloneする場合

```
git clone https://github.com/mitsu-h/deepvoice3
cd deepvoice3
git submodule init
git submodule update
```

### master Repositoryをforkしている場合

```
git remote add upstream git://github.com/mitsu-h/deepvoice3
git fetch upstream
git branch use_waveglow
git checkout use_waveglow
git merge upstream/use_waveglow
```

## 学習済みモデル
- [本家の学習済みデータ](https://drive.google.com/file/d/1rpK8CzAAirq9sWZhe9nlfvxMF1dRgFbF/view)
- [deepvoice3のメルスペクトログラムでファインチューニング](https://drive.google.com/file/d/1voxcNRVwMhaOKUAk6MhLkU5lbhdyONSP/view?usp=sharing)

download後，適当なフォルダに配置する．使用する学習済みデータはお好みの方を選択


## Requirements
[torch13.yml](torch13.yml)参照

また，コンソールで`python -c "import nltk; nltk.download('cmudict')"`を実行して音素辞書のダウンロードをする．
## Getting started
### 概要
- データの前処理：[preprocess.py](preprocess.py)
- 学習：：[train_seq2seq.py](train_seq2seq.py)
- 推論：[synthesis.py](synthesis.py)

また，ハイパーパラメータは`hparams.py`で主に設定を行う．ただし，メルスペクトログラムのパラメータは`ljspeech.py`で調整を行う．
### データの準備
このリポジトリでは，英語話者のみ学習を行ったため，他言語に関する動作は保証しない
- LJSpeech：https://keithito.com/LJ-Speech-Dataset/
- VCTK：http://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html
### データの前処理
使い方はr9y9様の実装と同じように使用可能
>Usage:
>```
>python preprocess.py ${dataset_name} ${dataset_path} ${out_dir} --preset=<json>
>```
>Supported `${dataset_name} s are:
>- `ljspeech`(en,single speaker)
>- `vctk`(en,multi-speaker)
>- `jsut`(jp, single speaker)
>- `nikl_m`(ko, multi-speaker)
>- `nikl_s`(ko, single speaker)
  
変更点として，メルスペクトログラム，スペクトログラム，WORLD Vocoderのパラメータ全てを出力するようにしている．

また，`hparams.py`の`key_position_rate`及び`world_upsample`は`python comute_timestamp_ratio.py <data-root>`を実行することで求める事ができる．

### 学習
使い方：
```
python train_${training_type}.py --data-root=${data-root} --log-event-path=${log_dir} --checkpoint=${checkpoint_path} --waveglow_path=${waveglow_path}
```
`--checkpoint`は学習済みのデータを再学習する場合のみ指定．

### 推論
学習済みデータを用いて，自己回帰で推論を行う．waveglowを利用する場合，
```
python synthesis.py --type='seq2seq' --waveglow_path=${waveglow_path} ${checkpoint_path} ${test_list.txt} ${output_dir}
```

）

