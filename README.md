# DMLF: Enzyme Commission Number Predicting and Benchmarking with Multi-agent Dual-core Learning

This repo contains source codes for a EC prediction tool namely ECRECer, which is an implementation  of our paper: 「Dual-core Multi-agent Learning Framework For EC Number Prediction」.
Detailed information about the framework can be found in our paper xxx.

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