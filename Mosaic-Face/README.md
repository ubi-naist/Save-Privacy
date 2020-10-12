# Mosaic-Face
モザイクをかけるモジュール @Yohei Katayama

2020年秋卒業のDang Chenyuさんが作成したものを改変．

モデルには[LightDSFD](https://github.com/lijiannuist/lightDSFD)を利用．(動画を主な対象としているため，　精度低/速度高)

## Introduction
Mosaic pedestrians' faces for protecting their pravicy.

ubi-dl03サーバにコンテナとして作成済み．

## Installation

    $ git clone https://github.com/ubi-naist/Save-Privacy.git
    $ cd Save-Privacy/Mosaic-Face

ホスト側（左）のポート番号を変更 8888→XXXX

    $ vi docker-compose.yml

イメージのビルド

    $ docker-compose build
    
イメージの起動
    
    $ docker-compose up -d
    
イメージにアクセス
    
    $ docker exec -i -t facemask_notebook_1 bash
    $ もしくは
    $ ブラウザからxxxx番ポートにアクセスしてjupyterを起動．　下はdl03サーバのコンテナを利用の場合
    $ ex) ubi-dl03.naist.jp:7778/lab?




## Usage
Run test.py and change parameters of save_folder, video_folder, and video_output with your own configuration. 

### 1. notebooks/data/input以下にマスキング処理をしたい動画/画像の入ったフォルダを配置する．
 
    $ ex) notebooks/data/input/test/Sample01.mp4
    $ ex) notebooks/data/input/test/Sample02.jpg


### 2. マスキング処理を実行
 
    $ python3 test.py 
    $ ex) python3 test.py --input_folder data/input/test01/
     


## 注意

実行時に以下のエラーが出ることがあります．　動画が作成できてるか要確認．(ubi-dl03上では問題なし)

    $ FFMPEG: tag 0x34363248/'H264' is not supported with codec id 27 and format 'mp4 / MP4 (MPEG-4 Part 14)'
