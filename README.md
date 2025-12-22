# DMICF
This repo is the official implementation for the paper **Dual-Perspective Disentangled Multi-Intent Alignment for Enhanced Collaborative Filtering**

### **For every dataset used in our experiments, we provide the corresponding model checkpoints saved at each training epoch**, facilitating detailed analysis and ensuring reproducibility.

### Moreover, **we offer the full implementation for user interaction group partitioning, along with tools to perform comprehensive quantitative and qualitative analyses of intent disentanglement**.

## Requirements

```
################### Complete environment setup can be found in the environment.yml file. ########################
scikit-learn==1.1.2
torch==2.1.0
numpy==1.22.3
pandas==1.5.0
scipy==1.9.3
```

## Datasets
We directly employ the three representative datasets reflecting typical interaction distributions provided by **LightGCL**.

## Preprocessing
For each dataset, we use the same hyperparameters, so only a minor modification to the dataset specification in the evaluation is required.

```shell
# Generate the trained model.
python train.py

# Evaluate on the test set.
python eval_results.py
```
