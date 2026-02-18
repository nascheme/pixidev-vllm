#!/bin/sh
#
# Run vllm-freethreaded container with docker.

echo "Command to test: (cd /test && python simple_generate.py)"

docker run -it --rm \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --group-add render \
    --ipc=host \
    --shm-size=8g \
    --security-opt seccomp=unconfined \
    --ulimit memlock=-1:-1 \
    -p 8000:8000 \
    -e HF_HOME=/vllm-cache \
    --mount=type=bind,src=`pwd`/cache,dst=/vllm-cache \
    --mount=type=bind,src=`pwd`/test,dst=/test \
    --mount=type=bind,src=$HOME/src/vllm_threaded,dst=/ft \
    vllm-freethreaded-rocm bash
