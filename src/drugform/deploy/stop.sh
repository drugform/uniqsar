#!/bin/sh
docker compose -f docker-compose-uniqsar.yml down --remove-orphans
docker compose down --remove-orphans
