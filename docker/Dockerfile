FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

RUN apt update && apt install -y wget unzip curl bzip2 git
#RUN curl -LO http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
#RUN bash Miniconda3-latest-Linux-x86_64.sh -p /miniconda -b
#RUN rm Miniconda3-latest-Linux-x86_64.sh
#ENV PATH=/miniconda/bin:${PATH}
#RUN conda update -y conda
#RUN conda install -y pytorch torchvision -c pytorch

WORKDIR /app
RUN git clone https://github.com/bric-tb-softwares/rxpixp2pixcycle.git && cd rxpixp2pixcycle && pip install -r requirements.txt

RUN python -c "import cv2"