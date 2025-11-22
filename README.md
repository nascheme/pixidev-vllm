# pixidev-vllm

*This is EXPERIMENTAL code, please use with caution!*

This repository contains a development setup for working on vLLM and some of
its dependencies in order to improve compatibility with free-threaded Python.
Only CPython 3.14t on Linux x86-64 is used at the moment, with CPU and CUDA
builds.

To get started, install Pixi and then:

```bash
$ pixi run clone-all

# Note: the below command doesn't work out of the box yet, you need to allow
# Python 3.14 in `pyproject.toml` and in `SUPPORTED_VERSIONS` in
# `CMakeLists.txt`. It also needs vllm-project/vllm#28762 and
# vllm-project/flash-attention#112.

$ pixi run build-cuda  # or `build-cpu`
```

