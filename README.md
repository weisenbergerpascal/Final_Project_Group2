\section*{Background \& Motivation}

Implicit Neural Representations (INRs) use neural networks to represent continuous physical fields by mapping coordinates directly to quantities like intensity or potential. The SIREN model introduced sinusoidal activation functions, enabling smooth and accurate modeling of both signals and their derivatives \cite{sitzmann2020implicit}. This makes SIRENs especially suitable for solving and analyzing physical systems governed by partial differential equations (PDEs) \cite{sitzmann2020implicit}.
\par \vspace{5mm}
Our goal is to reproduce the main results of the SIREN paper and extend them to other types of PDEs. We want to test whether the periodic activation and initialization scheme that worked well for Poisson and Helmholtz equations can also handle equations of different character, such as diffusion (parabolic) or quantum mechanical (Schrödinger-type) systems. The broader motivation is to explore how neural networks can act as continuous, differentiable solvers for physics problems.

\section*{Methods}

We will first aim to reproduce the architecture of the paper. This involves the following sequence of steps:

\begin{itemize}
    \item The network architecture creates an implicit representation of signals (audio, video, image) by creating a function $\Phi_{\theta}(x)$ which minimizes a loss function computed using the ground truth. This neural network is a fully connected perceptron that uses sine functions as their activation functions, with the activation results given by

    $$
    \phi_i(x) = sin(W_i x + b_i)
    $$

    In the paper, the authors took an architecture of 5-6 layer MLP with hidden dimensions 256-1024. The activation functions were changed to sin everywhere with a linear output layer. 

    \item Critical for training was the initialization (which sped up convergence) of the layers, where the first layer was initialized uniformly randomly and then scaled by a frequency $\omega_0$. Other layers are initialized uniformly randomly as well with the form

    \begin{equation}
        W_i \sim U(-\sqrt{\dfrac{6}{{fan}_{in}}},\sqrt{\dfrac{6}{{fan}_{in}}})
    \end{equation}
    which ensures stable gradient propagation for periodic activation.
    
    \item Finally, the training proceeds by sampling coordinates via Monte Carlo sampling at every iteration (pixel coordinates for images and interior/boundary points for PDEs). This ensures unbiased approximation of the continuous loss-function defined on the domain. Using autodiff, one can also implement other re-constructions where the gradient $\nabla\Phi_{\theta}$ can be supervised to minimizes loss functions like

    $$
    L = \| \nabla \Phi(x) - \nabla f(x)\|^2
    $$

    \item For PDEs, SIREN is trained by minimizing PDE residuals at interior/boundary points (same as in a PINN) with the exception of using periodic activation function.
\end{itemize}

We will implement all of the above in PyTorch, enabling us to use tools such as autograd for higher derivative computation. Once we get the minimal working architecture done, we will move on to implementing this for a image-fitting task and a PDE, benchmarking the effectiveness of SIREN against ReLU/Tanh MLPs. 



% \section*{Dataset}
% We will generate synthetic datasets by numerically solving target PDEs (e.g., Poisson, Helmholtz, heat, Schrödinger) on simple 1D–2D domains. From these, we’ll sample coordinate–value pairs and, when needed, derivative information to supervise the SIREN. For validation, we plan to reproduce image-fitting and wavefield experiments from the paper using publicly available data like CelebA and simulated wavefields.


\section*{Expected Outcomes / Deliverables}
The deliverables for our project will be broken down into two parts. 
First, we expect to create implicit neural representation networks for at least one image and at least one physics-based PDE to reproduce the work outlined in~\cite{sitzmann2020implicit} using our own test data. These will consist of two networks each, one trained using ReLu or Tanh activation functions and one trained using Sine activation functions, with the remainder of the network architecture kept the same. Each result will be compared to the ground truth (either the original image or a solution generated using a traditional grid-based solver). The second outcome of the project will be more exploratory, as we plan to extend the methodology to an untested physics PDE. This will require defining a new loss function tailored to the chosen equation and generating synthetic data for evaluation. Again we will train one ReLu or Tanh-based network and one Sine-based network before visually comparing the results to the ground truth. In addition to these core deliverables, we will also explore potential improvements to the original SIREN implementation, such as optimizing training speed, numerical stability, or memory usage. 


\section*{Project Schedule}

\begin{itemize}
    \item \textbf{Week 8:} Create GitHub repository (A), implement basic SIREN network architecture and test on a sample image (B), generate or preprocess initial PDE datasets (C).
    \item \textbf{Week 9:} Train and evaluate the simple model on one equation (A+B), analyze results and visualize reconstructions (C).
    \item \textbf{Week 10:} Extend the model to additional equations or modified scenarios (A+B), generate new synthetic data and compare to baseline results (C).
    \item \textbf{Finals Week:} Refine experiments, clean up code and documentation (A+B+C), prepare presentation slides and write the final 4-page report (A+B+C).
\end{itemize}

# Final_Project_Group2
Our goal is to reproduce the main results of the SIREN paper and extend them to other types of PDEs.
