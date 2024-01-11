FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime


ENV LC_ALL C.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV TERM screen
ENV TZ=America/New_York
ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y wget unzip curl bzip2 git libgl1 libgtk2.0-dev
RUN apt-get update && apt-get install -y git
RUN apt-get install msttcorefonts -qq
RUN pip install virtualenv poetry

