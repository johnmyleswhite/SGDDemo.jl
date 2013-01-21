using SGD

m = OLS(zeros(1))
@assert isa(m, OLS)

m = Lasso(zeros(1), 1.0)
@assert isa(m, Lasso)

m = Ridge(zeros(1), 1.0)
@assert isa(m, Ridge)

m = Logistic(zeros(1))
@assert isa(m, Logistic)

m = L2Logistic(zeros(1), 1.0)
@assert isa(m, L2Logistic)

m = L1Logistic(zeros(1), 1.0)
@assert isa(m, L1Logistic)

m = LinearSVM(zeros(1), 1.0)
@assert isa(m, LinearSVM)

ds = DataStream(Pkg.dir("SGD", "test", "data", "regression.csv"))

fit = SGDFit(m, Formula(:(y ~ x1 + x2 + x3)), ds, 0, 0)
@assert isa(fit, SGDFit)

df = start(fit.ds)
mm = model_matrix(fit.f, df)
