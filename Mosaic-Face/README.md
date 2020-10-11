# Mosaic-Face
モザイクをかけるモジュール @Yohei Katayama

2020年秋卒業のDang Chenyuさんが作成したものを改変．
 

## Introduction
Mosaic pedestrians' faces for protecting their pravicy.

ubi-dl03サーバにコンテナとしておいてあります．

## Usage
Run test.py and change parameters of save_folder, video_folder, and video_output with your own configuration. 

### 1. notebooks/data/input以下にマスキング処理をしたい動画の入ったフォルダを配置する．
 
   例 notebooks/data/input/test/Sample01.mp4


### 2. マスキング処理を実行
 
    $ test.py 
    


## 注意

実行時に以下のエラーが出ることがあります．　動画が作成できてるか要確認．(ubi-dl03上では問題なし)

    $ FFMPEG: tag 0x34363248/'H264' is not supported with codec id 27 and format 'mp4 / MP4 (MPEG-4 Part 14)'
