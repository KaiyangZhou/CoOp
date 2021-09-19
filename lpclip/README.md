# Linear Probe CLIP
To run linear probe baselines, make sure that your current working directory is `lpclip`.

Step 1: Extract Features from CLIP Image Encoder
```bash
sh feat_extractor.sh
```

Step 2: Train few-shot linear probe
```bash
sh linear_probe.sh
```
We follow the instruction stated in the Appendix A3 (pp.38) of [the original CLIP paper](https://arxiv.org/pdf/2103.00020.pdf), with a careful hyperparameter sweep.
We report mean/std results after 3 runs for every experiments.
