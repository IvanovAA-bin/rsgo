FROM ubuntu:22.04

RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y --no-install-recommends git gcc g++ make python3 python3-dev python3-pip python3-venv python3-wheel espeak-ng libsndfile1-dev && rm -rf /var/lib/apt/lists/*
RUN pip3 install llvmlite --ignore-installed

RUN pip3 install torch torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
RUN rm -rf /root/.cache/pip && mkdir /mnt/data

WORKDIR /root
RUN git clone https://github.com/coqui-ai/TTS.git
WORKDIR /root/TTS
RUN git checkout tags/v0.22.0 && pip3 install --ignore-installed -e .[all,dev,notebooks] && make install

COPY ./tts_testing.py /root/TTS
ENTRYPOINT [ "python3" ]
CMD [ "tts_testing.py" ]
