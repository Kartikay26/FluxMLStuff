# FluxMLStuff

Basic ML projects in Julia using the Flux ML library.

## Introduction

Julia is a high-level, high-performance open source programming language for technical computing.

- It is fast, it has a JIT compiler that compiles julia code to native code via LLVM at runtime
- It is dynamically typed, feels like a scripting language and can be used interactively.
- It is a general purpose language well suited for Data Science, Machine Learning, Parellel Computing and Scientific Domains.

Flux is a julia library for machine learning. It features:

- Compiled eager code
- Intuitive way to define models just like in mathematics
- Automatic Differentiation (Differentiable Programming)
    - This means that for using gradient descent, Flux will compute the gradients
      by itself at runtime. We do not have to specify the details of the
      backpropagation step.
- Can be used with GPUs/TPUs
- Can be exported easily to web technologies

## 1. Curve Fitting using Linear/Polynomial Regression

In this one we have taken a nonlinear function, and tried to approximate it using a simple polynomial model. We can think of it either as simply curve futting using a polynomial, or Linear Regression with features `x, x^2, ...` etc.


The function to be aproximated:

```julia
y(x) = 5x + 2sin(5x) + 3 + 0.1 * randn()
```

The model:

```julia
p = param(rand(5))
model(x) = p' * (x .^ Array(0:4))
# model(x) = p1 + p2 x + p3 x^2 + p4 x^3 + p5 x^4
```

Training code:
```julia
for i in 1:1000
    ps = params(p)
    Flux.train!(loss, ps, zip(xs, ys), Momentum())
    # plot the model from 0 to 1
    scatter(xs, ys)
    display(plot!((x) -> (model(x).data), 0, 1))
    println("Total Loss: $(totalLoss()), p: $p")
end
```
