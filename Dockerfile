FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime


RUN apt update && apt install -y wget unzip curl bzip2 git libgl1 libgtk2.0-dev
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
RUN ls
RUN echo "export PYTHONPATH=/app:$PYTHONPATH\nexport PATH=/app/scripts:$PATH:/app" > /setup_envs.sh