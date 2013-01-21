using DataFrames
using OptionsMod

module SGD
	using DataFrames
	using OptionsMod

	import Base.show

	export SGDModel, Lasso, Ridge, OLS, LinearModel
	export Logistic, L2Logistic, L1Logistic, LogisticModel
	export LinearSVM, SVMModel
	export SGDFit

	export predict, residuals, gradient, cost, sgd, update!

	include("utils.jl")
	include("types.jl")
	include("models.jl")
	include("fit.jl")
	include("show.jl")
end
