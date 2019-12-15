using Flux
using Flux: @epochs
println("Loaded Flux")

using Plots
println("Loaded Plots")


# Again, similar to the previous one
# Build some random data
y(x) = x^2 + sin(7x) + 0.1 * randn()
xs = rand(100)
ys = y.(xs)


# Build a model having 2 hidden layers with 3 neurons
# and using the sigmoid activation function
model = Chain(
    Dense(1, 3, σ),
    Dense(3, 3, σ),
    Dense(3, 1),
)

loss(xs, ys) = Flux.mse(model(xs), ys)

# each data point is a 1-element array
data = [ ([x], [y]) for (x,y) in zip(xs, ys) ]

display(scatter(xs, ys))

println("Starting Training.")

for i in 1:1000
    Flux.Optimise.train!(
        loss,
        params(model),
        data,
        Momentum()
    )
    println("Iteration $i")
    scatter(xs, ys)
    display(plot!(x -> model([x])[1].data, 0, 1))
end

println("""Weights & Biases:
        1: W1 $(model[1].W), B1 $(model[1].b),
        2: W2 $(model[2].W), B2 $(model[2].b),
        3: W3 $(model[3].W), B3 $(model[3].b).""")
