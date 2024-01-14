FROM python:3.8.10-buster

RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime

# パッケージの追加とタイムゾーンの設定
# 必要に応じてインストールするパッケージを追加してください
RUN apt-get update && apt-get install -y \ 
    tzdata \
    git \ 
    wget \
    libglib2.0-0 \
    libsm6  \ 
    libxrender1  \ 
    libxext6 \
    libgl1-mesa-dev \
&&  apt-get clean \
&&  rm -rf /var/lib/apt/lists/*

# 研究室用追加パッケージ
RUN pip install --no-cache-dir \
    pip install praat-parselmouth

#　追加パッケージのインストール
COPY requirements.txt /install/requirements.txt
RUN pip3 install --upgrade pip
RUN pip3 install -r /install/requirements.txt
RUN pip3 install --no-cache-dir \
    praat-parselmouth

RUN export TF_CPP_MIN_LOG_LEVEL=2
