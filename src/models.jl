function predict{T <: Real}(m::LinearModel, x::Matrix{T})
  n, p = size(x)
  predictions = x * m.weights
  return predictions
end

function predict{T <: Real}(m::LogisticModel, x::Matrix{T})
  n, p = size(x)
  predictions = x * m.weights
  for i = 1:n
    predictions[i] = SGD.invlogit(predictions[i])
  end
  return predictions
end

function predict{T <: Real}(m::LinearSVM, x::Matrix{T})
  n, p = size(x)
  predictions = x * m.weights
  return predictions
end

function predict(fit::SGDFit, df::DataFrames.AbstractDataFrame)
  mm = model_matrix(fit.f, df)
  return predict(fit.m, mm.model)
end

function predict(fit::SGDFit, ds::DataFrames.AbstractDataStream)
  predictions = Array(Float64, 0)
  for df in ds
    mm = model_matrix(fit.f, df)
    predictions = vcat(predictions, predict(fit.m, mm.model))
  end
  return predictions
end

function residuals{S <: Real, T <: Real}(m::SGDModel, y::Vector{S}, x::Matrix{T})
  return y - predict(m, x)
end

function gradient{S <: Real, T <: Real}(m::OLS, y::Vector{S}, x::Matrix{T})
  n, p = size(x)
  r = residuals(m, y, x)
  g = -r' * x
  for j in 1:p
    g[j] /= n
  end
	return vec(g)
end

function gradient{S <: Real, T <: Real}(m::Ridge, y::Vector{S}, x::Matrix{T})
  n, p = size(x)
  r = residuals(m, y, x)
  g = -r' * x
  for j in 1:p
    g[j] /= n
  end
  for j in 2:p
    g[j] += m.lambda * m.weights[j]
  end
  return vec(g)
end

function gradient{S <: Real, T <: Real}(m::Lasso, y::Vector{S}, x::Matrix{T})
  error("Lasso gradient not defined")
end

function gradient{S <: Real, T <: Real}(m::Logistic, y::Vector{S}, x::Matrix{T})
  n, p = size(x)
  r = residuals(m, y, x)
  g = -r' * x
  for j in 1:p
    g[j] /= n
  end
  return vec(g)
end

function gradient{S <: Real, T <: Real}(m::L2Logistic, y::Vector{S}, x::Matrix{T})
  n, p = size(x)
  r = residuals(m, y, x)
  g = -r' * x
  for j in 1:p
    g[j] /= n
  end
  for j in 2:p
    g[j] += m.lambda * m.weights[j]
  end
  return vec(g)
end

function gradient{S <: Real, T <: Real}(m::L1Logistic, y::Vector{S}, x::Matrix{T})
  error("L1Logistic gradient not defined")
end

function gradient{S <: Real, T <: Real}(m::LinearSVM, y::Vector{S}, x::Matrix{T})
  n, p = size(x)
  predictions = predict(m, x)
  g = Array(Float64, 1, p)
  for i in 1:n
    if predictions[i] < 1
      g[1, :] = y[i] * x[i, :]
    end
  end
  for j in 1:p
    g[j] /= n
  end
  for j in 2:p
    g[j] += m.lambda * m.weights[j]
  end
  return vec(g)
end

function cost{S <: Real, T <: Real}(m::OLS, y::Vector{S}, x::Matrix{T})
  0.5 * sum(residuals(m, y, x).^2)
end

function cost{S <: Real, T <: Real}(m::Ridge, y::Vector{S}, x::Matrix{T})
  0.5 * sum(residuals(m, y, x).^2) + m.lambda * sum(m.weights[2:end].^2)
end

function cost{S <: Real, T <: Real}(m::Lasso, y::Vector{S}, x::Matrix{T})
  0.5 * sum(residuals(m, y, x).^2) + m.lambda * sum(abs(m.weights[2:end]))
end

function cost{S <: Real, T <: Real}(m::Logistic, y::Vector{S}, x::Matrix{T})
  n, p = size(x)
  p = predict(m, x)
  ll = 0.0
  for i in 1:n
    ll += log(y[i] * p[i] + (1.0 - y[i]) * (1.0 - p[i]))
  end
  return -ll
end

function cost(fit::SGDFit, df::DataFrames.AbstractDataFrame)
  mm = model_matrix(fit.f, df)
  return cost(fit.m, vec(mm.response), mm.model)
end

function cost(fit::SGDFit, ds::DataFrames.AbstractDataStream)
  c = 0.0
  for df in ds
    mm = model_matrix(fit.f, df)
    c += cost(fit.m, vec(mm.response), mm.model)
  end
  return c
end
