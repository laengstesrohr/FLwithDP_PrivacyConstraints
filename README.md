<h1 align="center">Privacy contraints in FL with DP </h1>
<div align="center"> 
</div>
This repository aims at simulating adverse behaviour on federated learning with differential privacy in terms of data poisoning and collects related papers and corresponding codes on DP-based FL. It is part of a research project for university to investigate data privacy in federated learning healthcare settings and DP as a "silver bullet".

## Code
Tip: the code of this repository is my personal adaption, so if you find issues just contact me, I am open to help and receive feedback. The FL code of this repository is based on this [repository](https://github.com/wenzhu23333/Federated-Learning). Enjoy the code!

FYI, this code uses round-robin client selection to select each client equally often, which means that each client is selected sequentially.

### Parameter List

**Datasets**: MNIST, MNIST_rotated

**Model**: CNN 

**DP Mechanism**: Laplace, Gaussian

**DP Parameter**: $\epsilon$ and $\delta$

**DP Clip**: In DP-based FL, we usually clip the gradients in training and the clip is an important parameter to calculate the sensitivity.

## Run the simulation through the following demands:
### No DP

python main.py --dataset mnist --iid --model cnn --epochs 50 --dp_mechanism no_dp

### MNIST rotated

python main.py --dataset mnistrotated --iid --model cnn --epochs 50 --dp_mechanism no_dp

### Laplace Mechanism

This code is a very simple DP mechanism. If a client's privacy budget is $\epsilon$ and the client is selected $T$ times, the client's budget for each noise injection is $\epsilon / T$. Run with the following command:

python main.py --dataset mnist --iid --model cnn --epochs 50 --dp_mechanism Laplace --dp_epsilon 10 --dp_clip 10

### Gaussian Mechanism

The same as Laplace Mechanism, run with this command:

python main.py --dataset mnist --iid --model cnn --epochs 50 --dp_mechanism Gaussian --dp_epsilon 10 --dp_delta 1e-5 --dp_clip 10

## Remark

The new version uses [Opacus](https://opacus.ai/) for **Per Sample Gradient Clip**, which limits the norm of the gradient calculated by each sample.

##Sources

The MNIST rotated dataset was cloned from this [repository](https://github.com/ChaitanyaBaweja/RotNIST.git). It is already downloaded and doesn't need to be converted anymore.
