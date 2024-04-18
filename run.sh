#!/bin/bash

docker build -t tts_testing_img .
docker run -it --mount type=bind,source="$(pwd)"/output,target="/mnt/data" tts_testing_img
