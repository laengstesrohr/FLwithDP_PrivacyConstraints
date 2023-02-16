<h1 align="center">Privacy contrsint in FL with DP </h1>
<div align="center"> 
</div>
This repository aims at simulating adverse behaviour on Federated learning with differential privacy in terms of data poisoning and collects related papers and corresponding codes on DP-based FL.

## Code
Tip: the code of this repository is my personal implementation, if there is an inaccurate place please contact me, welcome to discuss with each other. The FL code of this repository is based on this [repository](https://github.com/wenzhu23333/Federated-Learning) .I hope you like it and support it. Welcome to submit PR to improve the  repository.

Note that in order to ensure that each client is selected a fixed number of times (to compute privacy budget each time the client is selected), this code uses round-robin client selection, which means that each client is selected sequentially.

Important note: The number of FL local update rounds used in this code is all 1, please do not change, once the number of local iteration rounds is changed, the sensitivity in DP needs to be recalculated, the upper bound of sensitivity will be a large value, and the privacy budget consumed in each round will become a lot, so please use the parameter setting of Local epoch = 1.

### Parameter List

**Datasets**: MNIST, MNIST_rotated

**Model**: CNN 

**DP Mechanism**: Laplace, Gaussian

**DP Parameter**: $\epsilon$ and $\delta$

**DP Clip**: In DP-based FL, we usually clip the gradients in training and the clip is an important parameter to calculate the sensitivity.

## Run the simulation
### No DP

You can run like this:

python main.py --dataset mnist --iid --model cnn --epochs 50 --dp_mechanism no_dp

### Laplace Mechanism

This code is based on Simple Composition in DP. In other words, if a client's privacy budget is $\epsilon$ and the client is selected $T$ times, the client's budget for each noising is $\epsilon / T$.

You can run like this:

python main.py --dataset mnist --iid --model cnn --epochs 50 --dp_mechanism Laplace --dp_epsilon 10 --dp_clip 10

### Gaussian Mechanism

The same as Laplace Mechanism.

You can run like this:

python main.py --dataset mnist --iid --model cnn --epochs 50 --dp_mechanism Gaussian --dp_epsilon 10 --dp_delta 1e-5 --dp_clip 10

## Remark

The new version uses [Opacus](https://opacus.ai/) for **Per Sample Gradient Clip**, which limits the norm of the gradient calculated by each sample.
