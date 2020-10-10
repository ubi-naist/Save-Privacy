FROM tensorflow/tensorflow:1.12.0-gpu-py3

RUN apt-get update && apt-get install -y \
    git graphviz

RUN pip install --upgrade pip

RUN pip install https://github.com/ipython-contrib/jupyter_contrib_nbextensions/tarball/master \
plotly yapf keras pydot graphviz seaborn autopep8 jupyter_contrib_nbextensions

RUN jupyter contrib nbextension install --user && jupyter nbextensions_configurator enable --user

#以下katayama追記########
RUN pip install jupyterlab
RUN pip install jupyter-console
RUN pip install numpy
RUN pip install pandas

# ffmpeg
RUN apt-get update
RUN apt-get install -y ffmpeg

# OpenCV
RUN apt-get update && apt-get install -y libopencv-dev
RUN pip install --upgrade pip
RUN pip install opencv-python


RUN pip install torch===1.1.0 torchvision==0.3.0


# jupyter 拡張機能
#RUN jupyter labextension install @jupyterlab/toc

#########################

