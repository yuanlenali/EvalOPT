# EvalOPT - An end-to-end framework to evaluate deep learning optimizers

EvalOPT is an e2e framework that simplifies, automates the evaluation of deep learning optimizers.
The goal is to provide a standard criterion to evaluate different optimizers.

EvalOPT is implemented based on tensorflow, and it supports to
  - evaluate both tensorflow built-in optimizers as well as self-implemented (new) optimizers.
  - evaluate the optimizer on a test problem combining a dataset among different open-source datasets, e.g., MINIST, CIFAR10, etc.., and a model architecture among academic classic models, e.g., MLP, VGG, etc...
  - automatically decides the best setting of optimizer hyper-parameters.

## Usage
  - run prepare_data.sh to download a collection of open-source datasets. (Currently ImageNet is not supported).
  - Run example_momentum_runner.py as an example of evaluating tensorflow's momentum optimizer.