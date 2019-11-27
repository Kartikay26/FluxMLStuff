using Plots
using Flux

# the function to approximate
y(x) = 5x + 2sin(5x) + 3 + 0.1 * randn()

# build some random data
xs = rand(100)
ys = y.(xs)

# build a model for the data and try training it
p = param(rand(5))
model(x) = p' * (x .^ Array(0:4))
# model(x) = p1 + p2 x + p3 x^2 + p4 x^3 + p5 x^4
loss(x, y) = (y - model(x))^2
totalLoss() = sum(loss.(xs, ys))

for i in 1:1000
    ps = params(p)
    Flux.train!(loss, ps, zip(xs, ys), Momentum())
    # plot the model from 0 to 1
    scatter(xs, ys)
    display(plot!((x) -> (model(x).data), 0, 1))
    println("Total Loss: $(totalLoss()), p: $p")
end

print("Press Enter to exit.")
readline()
