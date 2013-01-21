using SGD

my_tests = ["test/types.jl",
            "test/models.jl",
            "test/fit.jl"]

println("Running tests:")

for my_test in my_tests
    println(" * $(my_test)")
    include(my_test)
end
