## memorymodels

### Overview

Code for work on text compression and memory augmentation via masked mixer autoencoders.

### Quickstart

The usual: spin up a venv, install `requirements.txt`, run the driver code in the `memorymodels` directory.

Driver code typically uses the `transformers.trainer` utility which is handy for its compatibility with distributed training algorithms for saving model checkpoints, scheduling lrs, etc and is really just a sophisticated PyTorch wrapper. Note that the `trainer` does have a few long-standing bugs such that one occasionally needs to write a custom evaluation script or schedule, which is usually most straightforward to achieve via injecting a function rather than subclassing or making a branch and rewriting the library.

All driver code is compatible with using a GPU-accelerated server either via Distributed Data Parallel as follows,

```bash
$ torchrun --nproc_per_node {n_gpus} {training_script.py}
```

or Fully Sharded Data Parallel,

```bash
$ accelerate launch --config_file "configs/fsdp_config.yaml" {training_script.py}
```

or Deepspeed ZeRO stage 3

```bash
$ accelerate launch --config_file "configs/zero_config.yaml" {training_script.py}
```

Note that these configs do not auto-detect compute capabilities and GPU numbers, so switching between 2 and 4 and 8 GPUs may require changing the `num_processes` variable in the corresponding YAML. Currently all drivers and configs are set to use fp16/fp32 mixed precision training for reproducability with V100s, but for experiments run only on A100s/H100s bf16 can be used instead.

### Experimental Records

This project is currently focused on training efficiency with respect to compression achieved (or equivalently NLL loss or CEL) per compute amount applied, and so right now the primary experimental results are training runs. Trainers will save model, optimizer, losses, and some hyperparameter states which can be plotted.

In the future functional benchmarks will likely be used to supplement the compression data.

### TODOs
- [ ] Refactor driver code to keep model implementations separate from trainers and initializers
- [ ] Refactor hardcoded paths into separate envs for each server node
