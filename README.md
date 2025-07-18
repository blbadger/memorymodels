## memorymodels

### Overview

Code for work on text compression and memory augmentation via masked mixer autoencoders.

### Quickstart

To run this code using a GPU-accelerated node, spin up a venv, install dependencies via `uv pip install requirements.txt`, run the driver code in the `memorymodels` directory. Note that the requirements expect a CUDA device capability of at least 7.0, Python >=3.10, CUDA runtime major version of 12 and driver of at least 535.xxx.xx

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

### Precomputation

The drivers are designed to collate and batch pretokenized (and pre-padded) inputs as an alterative to loading cached datasets, tokenizing, batching and collating. This design choice trades disk space for training speed: older CPUs (the V100s cluster has Intel E5 v4s) typically do throttle training slightly when using large batches if tokenization is performed during training, even though tokenization occurs during forward passes to attempt to avoid blocking by default. 

This means that if a new dataset is introduced, you must tokenize it first via `fineweb_packed_tokenizer.py` before loading the safetensors file containing the tokenized data. In that file, note that `packed` switches from a default of truncating all dataset documents that are too long and padding all that are too short to instead linearize the documents, tokenize all input data, chunk while avoid padding unless necessary, and reshape and save the chunks. 

If you want to train a tokenizer on a corpus, use `fineweb_tokenizer_trainer.py`. 

### Experimental Records

This project is currently focused on training efficiency with respect to compression achieved (or equivalently NLL loss or CEL) per compute amount applied, and so right now the primary experimental results are training runs. Trainers will save model, optimizer, losses, and some hyperparameter states which can be plotted. The driver code in the target checkpoint directory, but it is preferred to use an unambiguous name for this folder that describes all relevant detais of each experiment to avoid making results too difficult to find.

In the future functional benchmarks will likely be used to supplement the compression data.

### TODOs
- [ ] Refactor driver code to keep model implementations separate from trainers and initializers
- [ ] Refactor hardcoded paths into envs, one for each server node
