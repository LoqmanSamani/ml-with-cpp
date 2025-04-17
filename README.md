# ML with C++ 🚀

Welcome to **ML with C++**, a repository dedicated to implementing machine learning models in C++ from scratch and with modern libraries. This project is designed to build proficiency in C++ for numerical computing, performance optimization, and ML algorithm development, with a long-term goal of creating a high-performance C++ diffusion model library (inspired by my Python [diffusion library](https://github.com/LoqmanSamani/DiffusionModels) built on PyTorch).


## 🎯 Goals

- **Master C++ for ML**: Learn C++ syntax, memory management, and optimization techniques for numerical computing.
- **Implement ML Models**: Build classic and advanced ML models, balancing from-scratch implementations with library-based approaches.
- **Prepare for Diffusion Models**: Develop skills for a C++ diffusion library by progressing from simple models to complex generative architectures.


## 📚 Projects

Below is a list of planned ML projects, each with a description, the approach (from scratch or library-based), and the libraries used. The projects start with classic ML models to build C++ proficiency and progress to deep learning.

### 1. Linear Regression
- **Description**: Predict continuous outputs using a linear model (`y = wX + b`), optimized via gradient descent or normal equations.
- **Approach**: From scratch, with Eigen for matrix operations.
- **Libraries**:
  - [Eigen](http://eigen.tuxfamily.org): For matrix and vector computations (NumPy-like).
  - Custom CSV parser for data loading.
- **Why**: Teaches matrix operations, gradient descent, and loss functions, foundational for deep learning.
- **Status**: Planned.

### 2. Logistic Regression
- **Description**: Binary classification using a sigmoid function and cross-entropy loss, optimized via gradient descent.
- **Approach**: From scratch, using Eigen for linear algebra.
- **Libraries**:
  - Eigen: For matrix operations and sigmoid implementation.
  - Custom data loader for CSV datasets (e.g., Iris).
- **Why**: Introduces non-linear activations and classification, relevant for neural networks.
- **Status**: Planned.

### 3. K-Means Clustering
- **Description**: Cluster data into `k` groups by iteratively assigning points to centroids and updating centroids.
- **Approach**: From scratch, using `std::vector` and Eigen for distance calculations.
- **Libraries**:
  - Eigen: For Euclidean distance and centroid updates.
  - Custom CSV I/O for 2D data.
- **Why**: Explores unsupervised learning and iterative algorithms, useful for generative model training loops.
- **Status**: Planned.

### 4. Decision Tree
- **Description**: Classification/regression tree with recursive splitting based on Gini impurity or entropy.
- **Approach**: From scratch, using `std::vector` and recursive `struct` for tree nodes.
- **Libraries**:
  - Standard C++ STL (`std::vector`, `std::map`) for data handling.
  - Custom CSV parser for datasets (e.g., UCI Wine).
- **Why**: Builds recursive data structures and optimization skills, applicable to complex model architectures.
- **Status**: Planned.

### 5. Multi-Layer Perceptron (MLP)
- **Description**: A feedforward neural network with hidden layers, backpropagation, and activation functions (e.g., ReLU, sigmoid).
- **Approach**: Hybrid—core backpropagation from scratch, with libtorch for tensor operations.
- **Libraries**:
  - [libtorch](https://pytorch.org/cppdocs/): PyTorch’s C++ API for tensors and autograd.
  - Eigen (optional): For initial experiments before libtorch.
- **Why**: Bridges classic ML to deep learning, preparing for diffusion model architectures (e.g., U-Net).
- **Status**: Planned.

### 6. Denoising Diffusion Probabilistic Model (DDPM)
- **Description**: A generative model that learns to denoise data through a forward and reverse process, inspired by my Python diffusion library.
- **Approach**: Library-based, using libtorch for tensor operations and CUDA for GPU acceleration.
- **Libraries**:
  - libtorch: For tensors, autograd, and neural network layers.
  - CUDA (optional): For GPU-accelerated training.
- **Why**: The ultimate goal—translating my Python diffusion library to C++ for high performance.
- **Status**: Long-term goal.



## License 

This project is licensed under the MIT License (LICENSE). See the [LICENSE]([https://github.com/LoqmanSamani/ml-with-cpp/license](https://github.com/LoqmanSamani/ml-with-cpp/blob/systembiology/LICENSE)) for details.
