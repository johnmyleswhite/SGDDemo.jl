using SGD

df = read_table(joinpath("test", "data", "regression.csv"))
mm = model_matrix(Formula(:(y ~ x1 + x2 + x3)), df)
y, x = vec(mm.response), mm.model

m = OLS([1.1, 2.1, 3.1, 4.1])
preds = predict(m, x)
@assert isa(preds, Vector{Float64})
@assert length(preds) == size(x, 1)

r = residuals(m, y, x)
@assert isa(r, Vector{Float64})
@assert length(r) == size(x, 1)

g = gradient(m, y, x)
@assert isa(g, Vector{Float64})
@assert length(g) == size(x, 2)
