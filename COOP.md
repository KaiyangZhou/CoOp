## How to Run

We provide the running scripts in `scripts/coop`, which allow you to reproduce the results on the IJCV'22 paper.

Make sure you change the path in `DATA` and run the commands under the main directory `CoOp/`.

### Few-Shot Learning

All you need is `CoOp/scripts/coop/main.sh`, which contains six input arguments.

`DATASET` takes as input a dataset name, like `imagenet` or `caltech101`. The valid names are the files' names in `CoOp/configs/datasets/`.

`CFG` means which config file to use, such as `rn50`, `rn101` or `vit_b32` (see `CoOp/configs/trainers/CoOp/`). Note that for ImageNet, we use `CoOp/configs/trainers/CoOp/*_ep50.yaml` for all settings (please follow the implementation details shown in the paper).

Below we provide examples on how to run CoOp on Caltech101.

**CLIP + CoOp (M=16, end)**:
- 1 shot: `bash scripts/coop/main.sh caltech101 rn50_ep50 end 16 1 False`
- 2 shots: `bash scripts/coop/main.sh caltech101 rn50_ep100 end 16 2 False`
- 4 shots: `bash scripts/coop/main.sh caltech101 rn50_ep100 end 16 4 False`
- 8 shots: `bash scripts/coop/main.sh caltech101 rn50 end 16 8 False`
- 16 shots: `bash scripts/coop/main.sh caltech101 rn50 end 16 16 False`

**CLIP + CoOp (M=16, mid)**:
- 1 shot: `bash scripts/coop/main.sh caltech101 rn50_ep50 middle 16 1 False`
- 2 shots: `bash scripts/coop/main.sh caltech101 rn50_ep100 middle 16 2 False`
- 4 shots: `bash scripts/coop/main.sh caltech101 rn50_ep100 middle 16 4 False`
- 8 shots: `bash scripts/coop/main.sh caltech101 rn50 middle 16 8 False`
- 16 shots: `bash scripts/coop/main.sh caltech101 rn50 middle 16 16 False`

**CLIP + CoOp (M=16, end, CSC)**:
- 1 shot: `bash scripts/coop/main.sh caltech101 rn50_ep50 end 16 1 True`
- 2 shots: `bash scripts/coop/main.sh caltech101 rn50_ep100 end 16 2 True`
- 4 shots: `bash scripts/coop/main.sh caltech101 rn50_ep100 end 16 4 True`
- 8 shots: `bash scripts/coop/main.sh caltech101 rn50 end 16 8 True`
- 16 shots: `bash scripts/coop/main.sh caltech101 rn50 end 16 16 True`

**CLIP + CoOp (M=16, mid, CSC)**:
- 1 shot: `bash scripts/coop/main.sh caltech101 rn50_ep50 middle 16 1 True`
- 2 shots: `bash scripts/coop/main.sh caltech101 rn50_ep100 middle 16 2 True`
- 4 shots: `bash scripts/coop/main.sh caltech101 rn50_ep100 middle 16 4 True`
- 8 shots: `bash scripts/coop/main.sh caltech101 rn50 middle 16 8 True`
- 16 shots: `bash scripts/coop/main.sh caltech101 rn50 middle 16 16 True`

After the experiments are finished, you can use `parse_test_res.py` to calculate the average results instead of manually looking into the log files. Say the structure of `output/` is

```
output
|–– caltech101/
|   |–– CoOp/
|   |   |–– rn50_16shots/
|   |   |   |–– nctx16_cscFalse_ctpend/
|   |   |   |   |–– seed1/
|   |   |   |   |–– seed2/
|   |   |   |   |–– seed3/
|   |   |–– rn50_8shots/
|   |   |   |–– nctx16_cscFalse_ctpend/
|   |   |   |   |–– seed1/
|   |   |   |   |–– seed2/
|   |   |   |   |–– seed3/
```

To calculate the average results for the folder `rn50_16shots/nctx16_cscFalse_ctpend/`, you can run

```bash
python parse_test_res.py output/caltech101/CoOp/rn50_16shots/nctx16_cscFalse_ctpend
```

Then, you will see something like this in your terminal

```bash
Parsing files in output/caltech101/CoOp/rn50_16shots/nctx16_cscFalse_ctpend
file: output/caltech101/CoOp/rn50_16shots/nctx16_cscFalse_ctpend/seed1/log.txt. accuracy: 91.81%. error: 8.19%.
file: output/caltech101/CoOp/rn50_16shots/nctx16_cscFalse_ctpend/seed2/log.txt. accuracy: 92.01%. error: 7.99%.
file: output/caltech101/CoOp/rn50_16shots/nctx16_cscFalse_ctpend/seed3/log.txt. accuracy: 92.17%. error: 7.83%.
===
Summary of directory: output/caltech101/CoOp/rn50_16shots/nctx16_cscFalse_ctpend
* accuracy: 92.00% +- 0.15%
* error: 8.00% +- 0.15%
===
```

**How to initialize the context tokens with pre-trained word vectors?** Specify the words for the parameter `TRAINER.COOP.CTX_INIT` in your config file. In our paper, we use `configs/trainers/rn50_ctxv1.yaml` (give this file to `--config-file`, see `scripts/coop/main.sh`), which uses "a photo of a" as the initialization words.

**How to visualize nearest words for the learned context tokens?** All you need is `interpret_prompt.py`. Say the learned tokens are saved in `a/b/c/prompt_learner/model.pth.tar` and you would like to see the top-3 nearest words for each token. In this case, run `python interpret_prompt.py a/b/c/prompt_learner/model.pth.tar 3`

### Robustness to Distribution Shift
To reproduce the robustness experiments, you can simply load the models learned on ImageNet and evaluate them on the following datasets: `imagenetv2`, `imagenet-sketch`, `imagenet-a` and `imagenet-r`.

The command is provided in `CoOp/scripts/coop/eval.sh`. The key arguments are `--model-dir`, `--load-epoch` and `--eval-only`. `--model-dir` indicates the directory where the models are saved (i.e. the entire folder containing `log.txt`, the tensorboard file and `prompt_learner/`). `--load-epoch` tells the code to load the model saved at a specific epoch, like `--load-epoch 50` for ImageNet (see the [source code](https://github.com/KaiyangZhou/Dassl.pytorch/blob/master/dassl/engine/trainer.py#L169) for more details).

For example, to evaluate `CLIP + CoOp (M=16, end)` on ImageNetV2, you can do

```bash
# Don't need to use rn5_ep50 here as no training is performed
bash scripts/coop/eval.sh imagenetv2 rn50
```

The default setting is `SHOTS=16`. Feel free to modify the script.

Again, you can use `parse_test_res.py` to automate the calculation of average performance. This time you should append `--test-log`, e.g., `python parse_test_res.py directory --test-log`.

### Zero-Shot CLIP

See `CoOp/scripts/coop/zeroshot.sh`.

### Linear Probe CLIP

Please move to [lpclip/](lpclip/).