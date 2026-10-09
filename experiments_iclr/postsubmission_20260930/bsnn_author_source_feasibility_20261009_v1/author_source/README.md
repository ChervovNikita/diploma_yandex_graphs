# Bayesian Sheaf Neural Networks

> **📍 Updated Version Available**: This work has been extended and improved. Please see the latest version at [https://github.com/Layal-Bou-Hamdan/BSNN](https://github.com/Layal-Bou-Hamdan/BSNN) for the most current implementation.

---

This repository contains the code for the paper 
**[Bayesian Sheaf Neural Networks](https://arxiv.org/abs/2410.09590)**.

## Getting started

To set up the environment, run the following command:

```bash
conda env create --file=environment_gpu.yml
conda activate nsd
```

### Run Hyperparameter Sweep

To run a hyperparameter sweep, you will need a `wandb` account. Create a [Weights & Biases account](https://wandb.ai/site) and run the following
commands to log in and follow the displayed instructions:
```bash
wandb online
wandb login
```

Once you have an account, you can run an example
sweep as follows:
```bash
export ENTITY=<WANDB_ACCOUNT_ID>
wandb sweep --project sheaf config/webkb_small/bayes_orth_sweep.yml
```
This will set up the sweep for a Bayesian sheaf neural network with orthogonal restriction maps on the WebKB datasets.
