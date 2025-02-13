SGD.jl
======

Fit predictive models to large data sets using SGD in Julia.

_WARNING: This package is a only working rough draft. Making the SGD work
on new problems is non-trivial and often requires rescaling features and
manually adjusting the learning rate schedule. This code will change
substantially over the coming months._

# Regression Examples

    using SGD
    
    regression_data = Pkg.dir("SGD", "test", "data", "regression.csv")

    ds = DataStream(regression_data)
    fit = sgd(:(y ~ x1 + x2 + x3), OLS, ds)
    cost(fit, ds)

    ds = DataStream(regression_data, 1000)
    fit = sgd(:(y ~ x1 + x2 + x3), OLS, ds)
    cost(fit, ds)

    ds = DataStream(regression_data, 1000)
    fit = sgd(:(y ~ x1 + x2 + x3), OLS, ds,
              Options(:averaging, false,
                      :momentum, 0.0,
                      :logging, false,
                      :tracing, false,
                      :learning_rate, SGD.lr_power(0.01, 0.0, 0.0),
                      :total_epochs, 1))
    cost(fit, ds)

    ds = DataStream(regression_data, 10_000)
    fit = sgd(:(y ~ x1 + x2 + x3), OLS, ds,
              Options(:averaging, false,
                      :momentum, 0.05,
                      :logging, true,
                      :tracing, true,
                      :learning_rate, SGD.lr_power(0.01, 0.0, 0.0),
                      :total_epochs, 25))
    cost(fit, ds)

# Classification Examples

    classification_data = Pkg.dir("SGD", "test", "data", "classification.csv")

    ds = DataStream(classification_data, 100)

    fit = sgd(:(y ~ x1 + x2 + x3), Logistic, ds,
              Options(:total_epochs, 25,
                      :averaging, true))
    cost(fit, ds)

    df = read_table(classification_data)
    df["p"] = predict(fit, df)
    df["p"] = (sign(df["p"] - 0.5) + 1) / 2
    by(df, ["y", "p"], nrow)

# Run Individual Epochs by Hand

    cost(fit, ds)
    update!(fit, ds)
    cost(fit, ds)

# Cross-Validation

    regression_data = Pkg.dir("SGD", "test", "data", "regression.csv")
    regression_data2 = Pkg.dir("SGD", "test", "data", "regression2.csv")

    train = DataStream(regression_data, 500)
    test = DataStream(regression_data2, 500)

    fit = sgd(:(y ~ x1 + x2 + x3), OLS, train,
              Options(:total_epochs, 0))

    n_epochs = 50

    results = DataFrame({Int, Float64, Float64},
                       ["Epoch", "Train", "Test"],
                       n_epochs)

    for epoch in 1:n_epochs
      update!(fit, train)
      results[epoch, "Epoch"] = epoch
      results[epoch, "Train"] = cost(fit, train)
      results[epoch, "Test"] = cost(fit, test)
    end

    head(results)
    tail(results)
