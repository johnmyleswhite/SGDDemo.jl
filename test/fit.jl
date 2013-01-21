using SGD

ds = DataStream(joinpath("test", "data", "regression.csv"))
ex = :(y ~ x1 + x2 + x3)
f = Formula(ex)
df = start(ds)
mm = model_matrix(f, df)
npreds = length(mm.model_colnames)
mname = OLS
m = mname(zeros(npreds))
fit = SGDFit(m, f, ds)
fit.old_dw = zeros(npreds)
SGD.update!(fit, df)

ds = DataStream(joinpath("test", "data", "regression.csv"))
fit = sgd(:(y ~ x1 + x2 + x3), OLS, ds)

ds = DataStream(joinpath("test", "data", "regression.csv"), 10)
fit = sgd(:(y ~ x1 + x2 + x3), OLS, ds)

ds = DataStream(joinpath("test", "data", "regression.csv"), 100)
fit = sgd(:(y ~ x1 + x2 + x3), OLS, ds)

ds = DataStream(joinpath("test", "data", "regression.csv"), 1000)
fit = sgd(:(y ~ x1 + x2 + x3), OLS, ds)

df = read_table(joinpath("test", "data", "regression.csv"))
df["p"] = predict(fit, df)

ds = DataStream(joinpath("test", "data", "regression.csv"), 1)
fit = sgd(:(y ~ x1 + x3), OLS, ds)

ds = DataStream(joinpath("test", "data", "regression.csv"), 100)
fit = sgd(:(y ~ x1 + x2 + x3), Ridge, ds)

ds = DataStream(joinpath("test", "data", "classification.csv"), 100)
fit = sgd(:(y ~ x1 + x2 + x3), Logistic, ds)
fit = sgd(:(y ~ x1 + x2 + x3), Logistic, ds, Options(:c, 0.0))

ds = DataStream(joinpath("test", "data", "classification.csv"), 100)
fit = sgd(:(y ~ x1 + x2 + x3), Logistic, ds)
fit = sgd(:(y ~ x1 + x2 + x3), Logistic, ds, Options(:c, 0.0))

ds = DataStream(joinpath("test", "data", "svm.csv"), 100)
fit = sgd(:(y ~ x1 + x2 + x3), LinearSVM, ds, Options(:c, 0.0))

df = read_table(joinpath("test", "data", "svm.csv"))
df["p"] = predict(fit, df)
