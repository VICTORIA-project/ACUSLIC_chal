#!/bin/bash

CONTAINER_NAME="acouslic-container"
LOG_FILE="docker_stats.log"

echo "Timestamp, CPU %, Mem Usage / Limit, Mem %, Net I/O, Block I/O, PIDs" > $LOG_FILE

while true; do
  TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
  STATS=$(docker stats --no-stream --format "{{.CPUPerc}}, {{.MemUsage}}, {{.MemPerc}}, {{.NetIO}}, {{.BlockIO}}, {{.PIDs}}" $CONTAINER_NAME)
  echo "$TIMESTAMP, $STATS" >> $LOG_FILE
  sleep 5
done
