#!/bin/bash

CONTAINER_NAME="acouslic-container"
LOG_FILE="memory_usage.log"

echo "Timestamp, Memory Usage" > $LOG_FILE

while true; do
  TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
  MEMORY_USAGE=$(docker stats --no-stream --format "{{.MemUsage}}" $CONTAINER_NAME | awk '{print $1}')
  echo "$TIMESTAMP, $MEMORY_USAGE" >> $LOG_FILE
  sleep 5
done