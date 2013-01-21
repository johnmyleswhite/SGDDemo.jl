function invlogit(z::Float64)
  if z < -100.0
    return 0.0
  elseif z > 100.0
    return 1.0
  else
    return 1.0 / (1.0 + exp(-z))
  end
end
