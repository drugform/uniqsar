#!/bin/sh

train_py_file=`realpath $1` 
devices=$2 #'4,5,6,7'

if test $# -ge 3 -a "$3" = "dev"
then
    echo "Interactive mode"
    iflag="-i"
    dflag=""
    docker compose -f ../../drugform/deploy/docker-compose.yml build uniqsar
else
    iflag=""
    dflag="-d"
fi


docker run $dflag \
     -v $(pwd)/../models:/app/models:rw \
     -v $(pwd)/../data:/app/data:rw \
     -v $train_py_file:/app/src/train.py:rw \
     -e DF_REGISTRY_ENDPOINT='tcp://0.0.0.0:40000' \
     -e MKL_NUM_THREADS=64 \
     --net host \
     -u $(id -u):$(id -g) \
     --gpus \"device=${devices}\" \
     -it deploy-uniqsar \
     python $iflag -u train.py
