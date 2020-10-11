FROM continuumio/anaconda3

RUN conda update --all
#RUN conda install keras-gpu
#RUN conda install opencv
RUN conda install -c conda-forge jupyterlab
RUN conda install -c conda-forge opencv

RUN jupyter notebook --generate-config
#RUN jupyter-lab --generate-config
RUN echo "c.NotebookApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_notebook_config.py
RUN export TF_FORCE_GPU_ALLOW_GROWTH=true

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES utility,compute
RUN export TF_FORCE_GPU_ALLOW_GROWTH=true

RUN conda install pytorch-gpu torchvision 

RUN conda install -c conda-forge ffmpeg
RUN apt-get install -y libgl1-mesa-dev
WORKDIR /notebooks
CMD bash
CMD jupyter-lab --allow-root --NotebookApp.token=''
