#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

./build.sh

# Create an output directory if it doesn't exist
OUTPUT_DIR="${SCRIPT_DIR}/test/output"

mkdir -p "$OUTPUT_DIR"

# Run the Docker container, mounting the input directory and the output directory
docker run --cpus=4 --memory=32gb --shm-size=32gb --gpus='device=1' --rm \
        -v "$SCRIPT_DIR"/test/input:/input/ \
        -v "$OUTPUT_DIR:/output/" \
        acouslicai_baseline

# No need to remove a Docker volume since we're using a bind mount