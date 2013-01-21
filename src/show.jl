function show(io::IO, m::SGDModel)
	print(io, "Model: $(typeof(m))\n")
	print(io, "Weights: ")
	show(io, m.weights)
end

function show(io::IO, fit::SGDFit)
	show(io, fit.m)
	print(io, "\n")
	print(io, "Minibatch Size: $(fit.ds.minibatch_size)\n")
	print(io, "Number of Observations: $(fit.n)\n")
	print(io, "Epochs: $(fit.epochs)\n")
	print(io, "Learning Rate Function: $(fit.learning_rate)\n")
	print(io, "Polyak Averaging: $(fit.averaging)\n")
	print(io, "Momentum: $(fit.momentum)\n")
	print(io, "Logging: $(fit.logging)\n")
	print(io, "Tracing: $(fit.tracing)\n")
end
