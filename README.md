##  (BWFT) Buffer with Feature Tokens: Combating Confusion in Medical Image Continual Learning
PyTorch code for BIBM paper: Buffer with Feature Tokens: Combating Confusion in Medical Image Continual Learning



## Setup
 * Install anaconda: https://www.anaconda.com/distribution/
 * set up conda environment w/ python 3.8, ex: `conda create --name coda python=3.8`
 * `conda activate coda`
 * `sh install_requirements.sh`
 * <b>NOTE: this framework was tested using `torch == 2.0.3` but should work for previous versions</b>
 
## Datasets
 * Create a folder `data/`
 * **cifar-100-python**: should automatically be downloaded
 * **imagenet-r**: retrieve from: https://github.com/hendrycks/imagenet-r
 * **imagenet-100**: download ImageNet1000(2012) in  https://image-net.org/ ,and split imagenet100 from it, We follow the division of icarl(https://github.com/arthurdouillard/incremental_learning.pytorch/tree/master/imagenet_split)
 * **medmnist**: https://medmnist.com/

```bash
sh experiments/run_all.sh

```

## Results
Results will be saved in a folder named `outputs/`. To get the final average accuracy, retrieve the final number in the file `outputs/**/results-acc/global.yaml`

## Note on setting
Our setting is class-incremental continual learning. Our method has not been tested for other settings such as domain-incremental continual learning.

This repo is based on aspects of https://github.com/GT-RIPL/CODA-Prompt
