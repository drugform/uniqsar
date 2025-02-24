#!/bin/sh
docker compose up -d && \
    echo 'Waiting 2 minutes to get basic models running' && \
    sleep 120 && \
    python3 gen_uniqsar.py && \
    docker compose build uniqsar && \
    docker compose -f docker-compose-uniqsar.yml up -d
