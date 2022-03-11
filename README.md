# Prompt Learning Research for Vision-Language Models

This repo contains the codebase of a series of research projects focused on adapting vision-language models like [CLIP](https://arxiv.org/abs/2103.00020) to downstream datasets via *prompt learning*:

* [Zhou et al., "Conditional Prompt Learning for Vision-Language Models," in CVPR 2022.](https://arxiv.org/abs/2203.05557)
* [Zhou et al., "Learning to Prompt for Vision-Language Models," arXiv 2021.](https://arxiv.org/abs/2109.01134)

## Updates

- **11.03.2022**: The code of our CVPR'22 paper, "[Conditional Prompt Learning for Vision-Language Models](https://arxiv.org/abs/2203.05557)," is released.

- **15.10.2021**: We find that the `best_val` model and the `last_step` model achieve similar performance, so we set `TEST.FINAL_MODEL = "last_step"` for all datasets to save training time. Why we used `best_val`: the ([tiny](https://github.com/KaiyangZhou/CoOp/blob/main/datasets/oxford_pets.py#L32)) validation set was designed for the linear probe approach, which requires extensive tuning for its hyperparameters, so we used the `best_val` model for CoOp as well for fair comparison (in this way, both approaches have access to the validation set).

- **09.10.2021**: Important changes are made to Dassl's transforms.py. Please pull the latest commits from https://github.com/KaiyangZhou/Dassl.pytorch and this repo to make sure the code works properly. In particular, 1) `center_crop` now becomes a default transform in testing (applied after resizing the smaller edge to a certain size to keep the image aspect ratio), and 2) for training, `Resize(cfg.INPUT.SIZE)` is deactivated when `random_crop` or `random_resized_crop` is used. Please read this [issue](https://github.com/KaiyangZhou/CoOp/issues/8) on how these changes might affect the performance.

- **18.09.2021**: We have fixed an error in Dassl which could cause a training data loader to have zero length (so no training will be performed) when the dataset size is smaller than the batch size (due to `drop_last=True`). Please pull the latest commit for Dassl (>= `8eecc3c`). This error led to lower results for CoOp in EuroSAT's 1- and 2-shot settings (others are all correct). We will update the paper on arxiv to fix this error.

Please email [Kaiyang Zhou](https://kaiyangzhou.github.io/) if you need the results' raw numbers.

## How to Install
This code is built on top of the awesome toolbox [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch) so you need to install the `dassl` environment first. Simply follow the instructions described [here](https://github.com/KaiyangZhou/Dassl.pytorch#installation) to install `dassl` as well as PyTorch. After that, run `pip install -r requirements.txt` under `CoOp/` to install a few more packages required by [CLIP](https://github.com/openai/CLIP) (this should be done when `dassl` is activated). Then, you are ready to go.

Follow [DATASETS.md](DATASETS.md) to install the datasets.

## How to Run

Click a paper below to see the detailed instructions on how to run the code to reproduce the results.

* [Learning to Prompt for Vision-Language Models](COOP.md)
* [Conditional Prompt Learning for Vision-Language Models](COCOOP.md)

## Citation
If you use this code in your research, please kindly cite the following papers

```bash
@inproceedings{zhou2022cocoop,
    title={Conditional Prompt Learning for Vision-Language Models},
    author={Zhou, Kaiyang and Yang, Jingkang and Loy, Chen Change and Liu, Ziwei},
    booktitle={CVPR},
    year={2022}
}

@article{zhou2021coop,
    title={Learning to Prompt for Vision-Language Models},
    author={Zhou, Kaiyang and Yang, Jingkang and Loy, Chen Change and Liu, Ziwei},
    journal={arXiv preprint arXiv:2109.01134},
    year={2021}
}
```
