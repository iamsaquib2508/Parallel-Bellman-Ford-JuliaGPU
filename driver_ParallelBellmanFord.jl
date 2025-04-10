using BenchmarkTools, CUDA

include("file_to_edges.jl")
include("parallelBellmanFord_VertexRelax.jl")
include("sequentialBellmanFord.jl")
include("csr_representation.jl")


# edges = [(1, 2, 1.5), (1, 3, 2.0), (2, 4, 3.5), (2, 3, 8.0), (3, 4, 4.0)]
# edges = [(Int32(u), Int32(v), Float32(w)) for (u, v, w) in edges]
# 1 -> 2 -> 4
#   \  |  /
#      3
edges = process_btc_file("input/soc-sign-bitcoinotc.csv")
# edges = process_usa_road_file("input/USA-road-d.E.gr")


offsetArray, endVertices, weights = graph_to_csr(edges)
offsetArray_gpu = CuArray(offsetArray); endVertices_gpu = CuArray(endVertices); weights_gpu = CuArray(weights)
n = length(offsetArray) - 1
m = length(weights)

# Print CSR representation
# println("offsetArray: ", offsetArray)
# println("endVertices: ", endVertices)
# println("weights: ", weights)

c = 8
for i in 1:1
    println("parallel + vertex relax\nn, m, chunk : ", n, ", ", m, ", ", c)
    D = @btime parallelBellmanFord_VertexRelax(1, offsetArray_gpu, endVertices_gpu, weights_gpu, debug = 0, chunkSize = c)
    # println("sequential\nn, m : ", n, " ", m)
    # D = @btime sequentialBellmanFord(1, offsetArray, endVertices, weights)
    # println(D)
end