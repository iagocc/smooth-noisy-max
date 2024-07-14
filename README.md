# Smooth Noisy Max
The source code for the paper called "Differentially Private Selection using Smooth Sensitivity".
The Smooth Noisy Max, formerly known as "report-locally-noisy-max," will be referred to as RLNM in the code.

This repository contains the three applications shown in the paper.
- Percentile Selection (`percentile_selection` folder)
- Greedy Decision Tree (`greedy_decision_tree` folder)
- Random Forest (`random_forest` folder)

## Depedencies
Each application folder has its dependencies. For the percentile selection, the file `requirements.txt` shows its dependencies to be installed with:
```sh
pip install -r requirements.txt
```
For the others, the dependencies are shown by the file `pyproject.toml` to be installed using the Poetry.

## Reproducibility
Each application has its execution method.

### Percentile Selection
The file `execute_experiment.sh` shows how to execute each experiment with its parameters.

### Greedy Decision Tree
The file `experiment.py` shows how to run the experiments.

### Random Forest
The file `experiment.py` shows how to run the experiments.
