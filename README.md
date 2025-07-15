# DMICF
This repo is the official implementation for the paper "Dual-Perspective Disentangled Multi-Intent Alignment for Enhanced Collaborative Filtering"


## Requirements

```
scikit-learn==1.1.2
torch==2.1.0
numpy==1.22.3
pandas==1.5.0
scipy==1.9.3
```

## Datasets
We directly employ the three representative datasets reflecting typical interaction distributions provided by [LightGCL](https://github.com/HKUDS/LightGCL/tree/main/data).

## Preprocessing
For each dataset, we use the same hyperparameters, so only a minor modification to the dataset specification in the evaluation is required.

```shell
# Generate the trained model.
python train.py

# Evaluate on the test set.
python eval_results.py
```
