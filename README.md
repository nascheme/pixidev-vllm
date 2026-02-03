# pixidev-vllm

*This is EXPERIMENTAL code, please use with caution!*

## Introduction

This repository contains a development setup for working on vLLM and some of
its dependencies in order to improve compatibility with free-threaded Python.
Only Python 3.14t on Linux x86-64 is used at the moment, with CPU and CUDA
builds.


## Quickstart

To get started, install Pixi and then:

```bash
$ pixi run clone-all

$ pixi run build-cuda  # or `build-cpu`
```

Run a simple text inference test.  Note that `cache/token` will likely
need to contain a HuggingFace token in order to download the model.

```bash
$ ./run_simple.sh
```


## Details

If you are building for the CUDA backend, you will want to review the
`TORCH_CUDA_ARCH_LIST` and `MAX_JOBS` variables inside the `pixi.toml` file.
You likely want to set `TORCH_CUDA_ARCH_LIST` for the hardware being used.  Common
values are as follows:

* 7.5 — Turing (T4)
* 8.0 — Ampere (A100)
* 8.9 — Ada Lovelace (L4, L40, RTX 4090)
* 9.0a — Hopper (H100, H200)

The sources for the build come from a combination of pre-built wheels and from
packages compiled from source code, via a git checkout.  See the `pixi.toml`
for the details of the packages.

A number of dependency packages have been upgraded.  For example, "torch" is
2.10.0 rather than 2.9.1.

The `git-repos.txt` file contains a list of git repos to use and the commit IDs
which will be checked out (to ensure reproducibility).  When the git repo is
checked out, optional patches may be applied.  See the contents of
`patches/<repo>/*`.  These patches will hopefully be merged upstream and will
eventually not be needed.  If you want to include your own change, commit it to
git and then use `git format-patch <...>` to create the patch file.

The vllm package has a lot of dependencies and not all of them are compatible with
Python 3.14t (free-threaded) yet.  Specifically, the following packages have issues:

* ray: Ray Compiled Graph, required for pipeline parallelism in V1
* triton: A language and compiler for custom Deep Learning operations
* mistral_common: required for the `MistralTokenizer`
* numba: required for N-gram speculative decoding
* opencv-python-headless: required for video IO
* deepgemm: GEMM kernels
* EP kernels wheels (pplx-kernels and DeepEP)

