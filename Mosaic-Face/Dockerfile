FROM continuumio/anaconda3

# python3.6環境を作成
RUN conda create -n mosaic-face python=3.6

# 以降のRUNでbashを適用させる。
SHELL ["/bin/bash", "-l", "-c"]
RUN echo "conda activate mosaic-face" >> ~/.bashrc

#RUN . /opt/conda/etc/profile.d/conda.sh && conda activate mosaic-face 

# 作成したconda環境をアクチベートとする
RUN conda activate mosaic-face


RUN conda update --all
RUN conda info -e

RUN conda install pytorch-gpu
RUN conda install torchvision

RUN conda install -c conda-forge jupyterlab
# versionを指定してx264をインストールする
## codec(h264)の確認方法：仮想環境内でconda list コマンドを実行
## conda-forgeでopencv, x264が入っていることを確認する。
RUN conda install -c conda-forge opencv=4.1.0
RUN conda install -c anaconda scipy
RUN conda install -c anaconda matplotlib
RUN apt-get install -y libgl1-mesa-dev


RUN jupyter notebook --generate-config
RUN echo "c.NotebookApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_notebook_config.py
RUN export TF_FORCE_GPU_ALLOW_GROWTH=true

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES utility,compute
RUN export TF_FORCE_GPU_ALLOW_GROWTH=true

#RUN conda install keras-gpu

RUN echo $PASSWORD

## jupyter-lab起動
WORKDIR /notebooks
CMD bash
CMD jupyter-lab --allow-root --NotebookApp.token=''
