#!/bin/sh
#
# Run vllm-freethreaded container with docker.

echo "Command to test: (cd /test && python simple_generate.py)"

docker run -it --rm \
    --gpus all \
    --security-opt seccomp=unconfined \
    --cap-add SYS_NICE \
    --shm-size=16g \
    -p 8000:8000 \
    -e HF_HOME=/vllm-cache \
    --mount=type=bind,src=`pwd`/cache,dst=/vllm-cache \
    --mount=type=bind,src=`pwd`/test,dst=/test \
    vllm-freethreaded bash
