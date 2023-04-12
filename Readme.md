# FedWard: Flexible Federated Backdoor Defense Framework with NON-IID DATA

## Preparation
### Downloading dependencies

```
pip3 install -r requirements.txt
``` 

## Run on federated benchmarks
* `dataset` chosen from `[MNIST, FASHION, CIFAR10]`.
* `iid_rate` chosen from `[0, 0.25, 0.5, 0.75]`.
* `pr` chosen from `[0.46875, 0.3125, 0.15625]`.
* `defend` chosen from `[fedward, flame, fedcc, crfl, median, trimmed_mean, fedavg]`.