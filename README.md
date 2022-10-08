# Prompt Learning for Vision-Language Models

This repo contains the codebase of a series of research projects focused on adapting vision-language models like [CLIP](https://arxiv.org/abs/2103.00020) to downstream datasets via *prompt learning*:

* [Conditional Prompt Learning for Vision-Language Models](https://arxiv.org/abs/2203.05557), in CVPR, 2022.
* [Learning to Prompt for Vision-Language Models](https://arxiv.org/abs/2109.01134), IJCV, 2022.

## Updates

- **07.10.2022**: Just added to both [CoOp](https://arxiv.org/abs/2109.01134) and [CoCoOp](https://arxiv.org/abs/2203.05557) (in their appendices) the results on the newly proposed DOSCO (DOmain Shift in COntext) benchmark, which focuses on contextual domain shift and covers a diverse set of classification problems. (The paper about DOSCO is [here](https://arxiv.org/abs/2209.07521) and the code for running CoOp/CoCoOp on DOSCO is [here](https://github.com/KaiyangZhou/on-device-dg).)

- **17.09.2022**: [Call for Papers](https://kaiyangzhou.github.io/assets/cfp_ijcv_lvms.html): IJCV Special Issue on *The Promises and Dangers of Large Vision Models*.

- **16.07.2022**: CoOp has been accepted to IJCV for publication!

- **10.06.2022**: Our latest work, [Neural Prompt Search](https://arxiv.org/abs/2206.04673), has just been released on arxiv. It provides a novel perspective for fine-tuning large vision models like [ViT](https://arxiv.org/abs/2010.11929), so please check it out if you're interested in parameter-efficient fine-tuning/transfer learning. The code is also made public [here](https://github.com/Davidzhangyuanhan/NOAH).

- **08.06.2022**: If you're looking for the code to draw the few-shot performance curves (like the ones we show in the CoOp's paper), see `draw_curves.py`.

- **09.04.2022**: The pre-trained weights of CoOp on ImageNet are released [here](#pre-trained-models).

- **11.03.2022**: The code of our CVPR'22 paper, "[Conditional Prompt Learning for Vision-Language Models](https://arxiv.org/abs/2203.05557)," is released.

- **15.10.2021**: We find that the `best_val` model and the `last_step` model achieve similar performance, so we set `TEST.FINAL_MODEL = "last_step"` for all datasets to save training time. Why we used `best_val`: the ([tiny](https://github.com/KaiyangZhou/CoOp/blob/main/datasets/oxford_pets.py#L32)) validation set was designed for the linear probe approach, which requires extensive tuning for its hyperparameters, so we used the `best_val` model for CoOp as well for fair comparison (in this way, both approaches have access to the validation set).

- **09.10.2021**: Important changes are made to Dassl's transforms.py. Please pull the latest commits from https://github.com/KaiyangZhou/Dassl.pytorch and this repo to make sure the code works properly. In particular, 1) `center_crop` now becomes a default transform in testing (applied after resizing the smaller edge to a certain size to keep the image aspect ratio), and 2) for training, `Resize(cfg.INPUT.SIZE)` is deactivated when `random_crop` or `random_resized_crop` is used. Please read this [issue](https://github.com/KaiyangZhou/CoOp/issues/8) on how these changes might affect the performance.

- **18.09.2021**: We have fixed an error in Dassl which could cause a training data loader to have zero length (so no training will be performed) when the dataset size is smaller than the batch size (due to `drop_last=True`). Please pull the latest commit for Dassl (>= `8eecc3c`). This error led to lower results for CoOp in EuroSAT's 1- and 2-shot settings (others are all correct). We will update the paper on arxiv to fix this error.

## How to Install
This code is built on top of the awesome toolbox [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch) so you need to install the `dassl` environment first. Simply follow the instructions described [here](https://github.com/KaiyangZhou/Dassl.pytorch#installation) to install `dassl` as well as PyTorch. After that, run `pip install -r requirements.txt` under `CoOp/` to install a few more packages required by [CLIP](https://github.com/openai/CLIP) (this should be done when `dassl` is activated). Then, you are ready to go.

Follow [DATASETS.md](DATASETS.md) to install the datasets.

## How to Run

Click a paper below to see the detailed instructions on how to run the code to reproduce the results.

* [Learning to Prompt for Vision-Language Models](COOP.md)
* [Conditional Prompt Learning for Vision-Language Models](COCOOP.md)

## Models and Results

- The pre-trained weights of CoOp (both M=16 & M=4) on ImageNet based on RN50, RN101, ViT-B/16 and ViT-B/32 can be downloaded altogether via this [link](https://drive.google.com/file/d/18ypxfd82RR0pizc5MM1ZWDYDk4j0BtPF/view?usp=sharing). The weights can be used to reproduce the results in Table 1 of CoOp's paper (i.e., the results on ImageNet and its four variants with domain shift). To load the weights and run the evaluation code, you will need to specify `--model-dir` and `--load-epoch` (see this [script](https://github.com/KaiyangZhou/CoOp/blob/main/scripts/eval.sh) for example).
- The raw numerical results can be found at this [google drive link](https://docs.google.com/spreadsheets/d/12_kaFdD0nct9aUIrDoreY0qDunQ9q9tv/edit?usp=sharing&ouid=100312610418109826457&rtpof=true&sd=true).

## Citation
If you use this code in your research, please kindly cite the following papers

```bash
@inproceedings{zhou2022cocoop,
    title={Conditional Prompt Learning for Vision-Language Models},
    author={Zhou, Kaiyang and Yang, Jingkang and Loy, Chen Change and Liu, Ziwei},
    booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2022}
}

@article{zhou2022coop,
    title={Learning to Prompt for Vision-Language Models},
    author={Zhou, Kaiyang and Yang, Jingkang and Loy, Chen Change and Liu, Ziwei},
    journal={International Journal of Computer Vision (IJCV)},
    year={2022}
}
```
