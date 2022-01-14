# DMLF: Enzyme Commission Number Predicting and Benchmarking with Multi-agent Dual-core Learning

This repo contains source codes for a EC prediction tool namely ECRECer, which is an implementation  of our paper: 「ECRECer: Enzyme Commission Number Recommendation and Benchmarking based on Multiagent Dual-core Learning」
Detailed information about the framework can be found in our paper xxx.

For simply use our tools to predict EC numbers, pls visit our web service at https://ecrecer.biodesign.ac.cn

To re-implement our experiments or offline use, pls use read the details below:


# Prerequisites
+ Python >= 3.6
+ Sklearn
+ Xgboost
+ conda
+ jupyter lab
+ ...

> Create conda env use [env.yaml](./env.yaml)

```python
conda env create -f env.yaml
```

# Preprocessing
Download and prepare the data set use the.

> [prepare_task_dataset.ipynb](./prepare_task_dataset.ipynb)

# Step by step benchmarking
### Task 1: Enzyme or None-Enzyme Prediction
> [./tasks/task1.ipynb](./task1.ipynb)

### Task 2: Polyfunctional Enzyme Prediction
> [./tasks/task2.ipynb](./task2.ipynb)

### Task 3: EC Number Prediction
> [./tasks/task3.ipynb](./task3.ipynb)

# High throughput benchmarking

# Train
```python
python benchmark_train.py
```

# Test
```python
python benchmark_test.py
```

# Evaluation
```python
python benchmark_evaluation.py
```

# Production

```python
python production.py -i input_fasta_file -o output_tsv_file -mode [p|r] -topk 5
```

# Other details are coming soon...