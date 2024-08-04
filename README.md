# 🎨 MNIST Dataset Analysis and Visualization

Welcome to the MNIST Dataset project! This repository contains code and documentation to perform a series of tasks on the MNIST dataset. Follow along to visualize data, apply Quadratic Discriminant Analysis (QDA), and Principal Component Analysis (PCA). Let's get started! 🚀

## 📁 Dataset

Use the [MNIST dataset](https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz) for the following tasks. The dataset contains:

- **60,000** training samples from **10 classes** (digits 0-9).
- **10,000** testing samples with labels/classes.

## 🎨 Task 1: Visualize Samples

1. **Visualize** 5 samples from each class in the training set as images.
2. **Image Size**: 28×28 pixels.
3. **Vectorize** images to make them 784-dimensional.

## 📊 Task 2: Quadratic Discriminant Analysis (QDA)

1. Compute the **mean vector** and **covariance vector** for each of the 10 classes using the training set.
2. Use the **QDA expression** derived in the lecture. Include this expression clearly in your code.
3. **Classify** all samples in the test set and report:
   - Overall accuracy.
   - Class-wise accuracy.

## 🔍 Task 3: Principal Component Analysis (PCA)

1. Choose **100 samples** from each class and create a **784×1000** data matrix (X).
2. **Remove the mean** from X.
3. Apply **PCA**:
   - Compute covariance \( S = XX^T / 999 \).
   - Find eigenvectors and eigenvalues.
   - Sort them in descending order to create matrix **U**.
4. Perform **Y = U^T X** and reconstruct **X_recon = UY**. Compute **MSE**:
   - \( \text{MSE} = \sum_{i,j} (X(i, j) - X_{recon}(i, j))^2 \).

## 🖼️ Task 4: Visualize PCA Reconstruction

1. Choose **p = 5, 10, 20** eigenvectors from U.
2. For each p:
   - Obtain **U_p Y**.
   - Add the removed mean to **X**.
   - Reshape each column to **28×28** and plot the image.
   - Plot 5 images from each class.

## 🔬 Task 5: QDA on PCA Components

1. Let the test set be **X_test**.
2. Find **Y = U_p^T X_test**.
3. For each p, apply **QDA** from Task 2 on Y.
4. Obtain overall and class-wise accuracy on the test set. Observe how accuracy increases as p increases.


## 🛠️ Usage

1. **Clone** the repository:
   git clone https://github.com/yourusername/mnist-analysis.git

## 🤝 Contributing
Feel free to open issues or submit pull requests if you have any suggestions or improvements.

## 📄 License
This project is licensed under the MIT License.
