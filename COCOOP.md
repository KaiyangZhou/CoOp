## How to Run

The running scripts are provided in `scripts/cocoop/`, which allow you to reproduce the results on the CVPR'22 paper.

Make sure you change the path in `DATA` and run the commands under the main directory `CoOp/`.

### Generalization From Base to New Classes

This corresponds to the experiments in Section 4.1, i.e., Table 1.

You will need both `scripts/cocoop/base2new_train.sh` and `scripts/cocoop/base2new_test.sh`. The former trains a model on bash classes while the latter evaluates the trained model on new classes. Both scripts have two input arguments, i.e., `DATASET` and `SEED`.

`DATASET` takes as input a dataset name, like `imagenet` or `caltech101`. The valid names are the files' names in `CoOp/configs/datasets/`.

Below we provide an example on how to evaluate the model on ImageNet.

```bash
# seed=1
bash scripts/cocoop/base2new_train.sh imagenet 1
bash scripts/cocoop/base2new_test.sh imagenet 1

# seed=2
bash scripts/cocoop/base2new_train.sh imagenet 2
bash scripts/cocoop/base2new_test.sh imagenet 2

# seed=3
bash scripts/cocoop/base2new_train.sh imagenet 3
bash scripts/cocoop/base2new_test.sh imagenet 3
```

When the evaluation is done, you can use `parse_test_res.py` to automatically calculate the average results. For instance, after you finish the evaluation (including `base2new_train.sh` and `base2new_test.sh`) on ImageNet using the aforementioned commands, you would get

```
output
|–– base2new/
|   |–– test_new/
|   |   |–– imagenet/
|   |   |   |–– shots_16/
|   |   |   |   |–– CoCoOp/
|   |   |   |   |   |–– vit_b16_c4_ep10_batch1_ctxv1/
|   |   |   |   |   |   |–– seed1/
|   |   |   |   |   |   |–– seed2/
|   |   |   |   |   |   |–– seed3/
|   |–– train_base/
|   |   |–– imagenet/
|   |   |   |–– shots_16/
|   |   |   |   |–– CoCoOp/
|   |   |   |   |   |–– vit_b16_c4_ep10_batch1_ctxv1/
|   |   |   |   |   |   |–– seed1/
|   |   |   |   |   |   |–– seed2/
|   |   |   |   |   |   |–– seed3/
```

Then, to get the average performance on the base classes, run

```bash
python parse_test_res.py output/base2new/train_base/imagenet/shots_16/CoCoOp/vit_b16_c4_ep10_batch1_ctxv1
```

To get the average performance on the new classes, run

```bash
python parse_test_res.py output/base2new/test_new/imagenet/shots_16/CoCoOp/vit_b16_c4_ep10_batch1_ctxv1 --test-log
```

### Cross-Dataset Transfer

This corresponds to the experiments in Section 4.2, i.e., Table 2.

The relevant scripts are `scripts/cocoop/xd_train.sh` and `scripts/cocoop/xd_test.sh` where the `DATASET` variable is set to the default, namely `imagenet`. To train the model, run

```bash
# seed=1
bash scripts/cocoop/xd_train.sh 1

# seed=2
bash scripts/cocoop/xd_train.sh 2

# seed=3
bash scripts/cocoop/xd_train.sh 3
```

Then, you evaluate the model on other datasets, e.g.,

```bash
for SEED in 1 2 3
do
    bash scripts/cocoop/xd_test.sh caltech101 ${SEED}
    bash scripts/cocoop/xd_test.sh oxford_pets ${SEED}
    bash scripts/cocoop/xd_test.sh stanford_cars ${SEED}
done
```

### Domain Generalization

This corresponds to the experiments in Section 4.3, i.e., Table 3.

The steps are similar to those discussed in "Cross-Dataset Transfer" except you evaluate the model on the variants of ImageNet, i.e., `imagenetv2`, `imagenet_sketch`, `imagenet_a` and `imagenet_r`.