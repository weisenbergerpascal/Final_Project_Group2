Background & Motivation

Implicit Neural Representations (INRs) use neural networks to represent continuous physical fields by mapping coordinates directly to quantities like intensity or potential. The SIREN model introduced sinusoidal activation functions, enabling smooth and accurate modeling of both signals and their derivatives 
1
1. This makes SIRENs especially suitable for solving and analyzing physical systems governed by partial differential equations (PDEs).

Our goal is to reproduce the main results of the SIREN paper and extend them to other types of PDEs. We want to test whether the periodic activation and initialization scheme that worked well for Poisson and Helmholtz equations can also handle equations of different character, such as diffusion (parabolic) or quantum mechanical (Schrödinger-type) systems. The broader motivation is to explore how neural networks can act as continuous, differentiable solvers for physics problems.

Methods

We will first aim to reproduce the architecture of the SIREN paper. This involves the following steps:

The network architecture creates an implicit representation of signals (audio, video, image) by learning a function

Φ
𝜃
(
𝑥
)
Φ
θ
	​

(x)

which minimizes a loss function computed using ground-truth data. The neural network is a fully connected multilayer perceptron (MLP) that uses sine functions as activation functions, with activations given by

𝜙
𝑖
(
𝑥
)
=
sin
⁡
(
𝑊
𝑖
𝑥
+
𝑏
𝑖
)
.
ϕ
i
	​

(x)=sin(W
i
	​

x+b
i
	​

).

In the paper, the authors used a 5–6 layer MLP with hidden dimensions ranging from 256 to 1024. All hidden-layer activations were replaced with sine functions, and a linear output layer was used.

A critical component of training is the initialization scheme, which significantly improves convergence. The first layer is initialized uniformly at random and scaled by a frequency parameter 
𝜔
0
ω
0
	​

. Subsequent layers are also initialized uniformly at random according to

𝑊
𝑖
∼
𝑈
(
−
6
fan
in
,
6
fan
in
)
,
W
i
	​

∼U(−
fan
in
	​

6
	​

	​

,
fan
in
	​

6
	​

	​

),

which ensures stable gradient propagation when using periodic activations.

Training proceeds by sampling coordinates using Monte Carlo sampling at each iteration (pixel coordinates for images, and interior/boundary points for PDEs). This provides an unbiased approximation of the continuous loss function defined over the domain. Using automatic differentiation, one can also supervise derivatives of the network output, minimizing losses of the form

𝐿
=
∥
∇
Φ
(
𝑥
)
−
∇
𝑓
(
𝑥
)
∥
2
.
L=∥∇Φ(x)−∇f(x)∥
2
.

For PDEs, SIREN is trained by minimizing PDE residuals at interior and boundary points (similar to Physics-Informed Neural Networks, or PINNs), with the key difference being the use of periodic activation functions.

We will implement all of the above in PyTorch, leveraging tools such as autograd for computing higher-order derivatives. Once a minimal working architecture is established, we will apply it to an image-fitting task and a PDE problem, benchmarking SIREN against standard ReLU- and Tanh-based MLPs.

Expected Outcomes / Deliverables

The project deliverables are divided into two parts.

First, we will construct implicit neural representation networks for at least one image-based task and one physics-based PDE, reproducing results from the original SIREN paper using our own test data. For each task, we will train two networks: one using ReLU or Tanh activations and one using sine activations, keeping all other architectural details identical. The results will be compared against ground truth data, either the original image or a solution obtained from a traditional grid-based numerical solver.

Second, we will extend the SIREN methodology to a previously untested physics PDE. This will involve defining a new loss function tailored to the chosen equation and generating synthetic data for evaluation. As before, we will train both a standard activation network and a sine-activated network, and visually and quantitatively compare their results to the ground truth.

In addition to these core goals, we will explore possible improvements to the original SIREN implementation, such as enhancements in training speed, numerical stability, or memory efficiency.

Project Schedule

Week 8:
Create GitHub repository (A); implement basic SIREN architecture and test on a sample image (B); generate or preprocess initial PDE datasets (C).

Week 9:
Train and evaluate the model on one PDE (A+B); analyze results and visualize reconstructions (C).

Week 10:
Extend the model to additional equations or modified scenarios (A+B); generate new synthetic data and compare against baseline models (C).

Finals Week:
Refine experiments; clean up code and documentation (A+B+C); prepare presentation slides and write the final 4-page report (A+B+C).

Final_Project_Group2

Our goal is to reproduce the main results of the SIREN paper and extend them to other types of PDEs.
