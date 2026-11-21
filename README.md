# tensor-autograd

C++ library for learning tensors, reverse-mode autograd, and a few classical models.

This is not a transformer, and nothing here is distributed. The C++ namespace is still `tiny_transformer` because that was the original goal. The code that exists is below.

## What is implemented

- N-dimensional tensors: broadcasting, matmul, reshape, transpose, sigmoid, relu, softmax
- Reverse-mode autograd
- Linear regression by gradient descent, with fit, refit, and predict
- A generalized linear model for Gaussian, Bernoulli, and multinomial families, using gradient descent or Newton

`make test` covers tensors, autograd, and linear regression. The GLM has no tests yet.

## Build and test

```bash
make
make test
make test-tensor
make test-autograd
make test-linear_regression
```
