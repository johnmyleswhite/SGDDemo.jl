require(joinpath("src", "utils.jl"))

using Distributions
using DataFrames

# Regression examples

mu = [0.0, 5.0, 10.0]
sigma = [1.0 0.3 0.0; 0.3 1.0 0.0; 0.0 0.0 1.0;]

d = MultivariateNormal(mu, sigma)

srand(1)
x = rand(d, 10_000)
beta = [1.1, 2.1, 3.1]
y = x * beta + randn(size(x, 1))

df = DataFrame(x)
df["y"] = y

write_table(joinpath("test", "data", "regression.csv"), df)

srand(2)
x = rand(d, 10_000)
beta = [1.1, 2.1, 3.1]
y = x * beta + randn(size(x, 1))

df = DataFrame(x)
df["y"] = y

write_table(joinpath("test", "data", "regression2.csv"), df)

# Classification examples

mu = [0.0, 1.0, -1.0]
sigma = [1.0 0.3 0.3; 0.3 1.0 0.3; 0.3 0.3 1.0;]

d = MultivariateNormal(mu, sigma)

srand(2)
x = rand(d, 10_000)
beta = [.11, .21, .31]
p = map(invlogit, x * beta)
y = [int(rand() < p_i) for p_i in p]

df = DataFrame(x)
df["y"] = y

write_table(joinpath("test", "data", "classification.csv"), df)

# SVM examples

mu = [0.0, 1.0, -1.0]
sigma = [1.0 0.3 0.3; 0.3 1.0 0.3; 0.3 0.3 1.0;]

d = MultivariateNormal(mu, sigma)

srand(2)
x = rand(d, 10_000)
beta = [.11, .21, .31]
z = x * beta + randn(10_000)
y = Array(Float64, 10_000)
for i in 1:10_000
	y[i] = sign(z[i])
end

df = DataFrame(x)
df["y"] = y

write_table(joinpath("test", "data", "svm.csv"), df)
