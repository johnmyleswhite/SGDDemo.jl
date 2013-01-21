function report(fit::SGDFit, df::DataFrames.AbstractDataFrame)
  if fit.logging || fit.tracing
    @printf "Epoch: %d\nObservations: %d\n" fit.epochs fit.n
    if fit.logging && rem(fit.n, fit.logging_interval) == 0
      @printf "Weights: %s\n" join(fit.m.weights, " ")
    end
    if fit.tracing && rem(fit.n, fit.tracing_interval) == 0
      @printf "Cost: %f\n" cost(fit, df) # This might be broken
    end
    @printf "\n"
  end
end

function update!(fit::SGDFit,
                 df::DataFrames.AbstractDataFrame)
  # Evaluate the gradient on these examples
  mm = model_matrix(fit.f, df)
  dw = -gradient(fit.m, vec(mm.response), mm.model)

  # Set the learning rate
  eta = fit.learning_rate(fit.n)

  # Scale the weight change by the learning rate
  dw = eta * dw

  # Use momentum to average gradients over time
  alpha = fit.momentum
  dw = alpha * fit.old_dw + (1.0 - alpha) * dw
  fit.old_dw = dw

  # Precompute the new weights before averaging weights
  new_weights = fit.m.weights + dw

  # Use ASGD to average weights over time
  if fit.averaging && fit.epochs > 1
    # fit.m.weights = ((fit.n - 1) / fit.n) * fit.m.weights +
    #                 (1 / fit.n) * new_weights
    # Experiment with convex averaging instead of ASGD
    phi = 0.1
    fit.m.weights = phi * fit.m.weights + (1.0 - phi) * new_weights
  else
    fit.m.weights = new_weights
  end

  # Stop processing if weights become corrupted
  if any(isnan(fit.m.weights))
    error("NaN's produced as weights during an update step")
  end
end

function update!(fit::SGDFit,
                 ds::DataFrames.AbstractDataStream)
  for df in ds
    update!(fit, df)
    fit.n += nrow(df)
    report(fit, df)
  end
  fit.epochs += 1
end

function sgd{T <: SGDModel}(ex::Expr,
                            mtype::Type{T},
                            ds::DataFrames.AbstractDataStream,
                            opts::Options)
  @defaults opts lambda = 0.01
  @defaults opts total_epochs = 1
  # Constant learning rate
  @defaults opts learning_rate = lr_power(0.01, 0.0, 0.0)
  @defaults opts averaging = false
  @defaults opts logging = false
  @defaults opts logging_interval = 1_000
  @defaults opts tracing = false
  @defaults opts tracing_interval = 1_000
  @defaults opts momentum = 0.0

  f = Formula(ex)

  # TODO: Get npreds from f without opening the DataStream
  df = start(ds)
  mm = model_matrix(f, df)
  npreds = length(mm.model_colnames)

  m = mtype(zeros(npreds))
  if isa(m, RegularizedModel)
    m.lambda = lambda
  end

  fit = SGDFit(m, f, ds)

  fit.learning_rate = learning_rate
  fit.averaging = averaging
  fit.logging = logging
  fit.logging_interval = logging_interval
  fit.tracing = tracing
  fit.tracing_interval = tracing_interval
  fit.old_dw = zeros(npreds)
  fit.momentum = momentum

  while fit.epochs < total_epochs
    update!(fit, ds)
  end

  @check_used opts

  return fit
end

function sgd{T <: SGDModel}(ex::Expr,
                            mtype::Type{T},
                            ds::DataFrames.AbstractDataStream)
  sgd(ex, mtype, ds, Options())
end
