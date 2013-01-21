abstract SGDModel

type OLS <: SGDModel
	weights::Vector{Float64}
end

type Ridge <: SGDModel
	weights::Vector{Float64}
	lambda::Float64
end
Ridge{T <: Real}(w::Vector{T}) = Ridge(w, 0.0)

type Lasso <: SGDModel
	weights::Vector{Float64}
	lambda::Float64
end
Lasso{T <: Real}(w::Vector{T}) = Lasso(w, 0.0)

type Logistic <: SGDModel
	weights::Vector{Float64}
end

type L2Logistic <: SGDModel
	weights::Vector{Float64}
	lambda::Float64
end
L2Logistic{T <: Real}(w::Vector{T}) = L2Logistic(w, 0.0)

type L1Logistic <: SGDModel
	weights::Vector{Float64}
	lambda::Float64
end
L1Logistic{T <: Real}(w::Vector{T}) = L1Logistic(w, 0.0)

type LinearSVM <: SGDModel
	weights::Vector{Float64}
	lambda::Float64
end
LinearSVM{T <: Real}(w::Vector{T}) = LinearSVM(w, 0.0)

typealias LinearModel Union(OLS, Ridge, Lasso)
typealias LogisticModel Union(Logistic, L2Logistic, L1Logistic)
typealias SVMModel Union(LinearSVM)
typealias RegularizedModel Union(Ridge, Lasso, L2Logistic, L1Logistic, LinearSVM)

# return a
# return a / (b + t)
# return a / (b + sqrt(t))

function lr_power(a::Real, b::Real, c::Real)
	function lr(t::Integer)
		return a / (b + t^c)
	end
	return lr
end
lr_power(a::Real) = lr_power(a, 1.0, 0.5)

type SGDFit
	m::SGDModel
	f::DataFrames.Formula
	ds::DataFrames.AbstractDataStream
	n::Int
	epochs::Int
	learning_rate::Function
	averaging::Bool
	logging::Bool
	logging_interval::Int
	tracing::Bool
	tracing_interval::Int
	old_dw::Vector{Float64}
	momentum::Float64
end

function SGDFit(m::SGDModel,
	            f::DataFrames.Formula,
	            ds::DataFrames.AbstractDataStream,
	            n::Int,
				epochs::Int)
	return SGDFit(m,
		          f,
		          ds,
		          n,
		          epochs,
		          lr_power(0.01, 0.0, 0.0),
		          false,
		          false,
		          1,
		          false,
		          1,
		          Array(Float64, 0),
		          0.0)
end

function SGDFit(m::SGDModel,
	            f::DataFrames.Formula,
	            ds::DataFrames.AbstractDataStream)
	return SGDFit(m,
		          f,
		          ds,
		          0,
		          0,
		          lr_power(0.01, 0.0, 0.0),
		          false,
		          false,
		          1,
		          false,
		          1,
				  Array(Float64, 0),
		          0.0)
end
