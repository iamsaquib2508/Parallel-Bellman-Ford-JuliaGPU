
# Function to convert (u, v, w) edges to CSR format
function graph_to_csr(edges::Vector{Tuple{Int32, Int32, Float32}})
    n = maximum(max(e[1], e[2]) for e in edges)
    m = length(edges) 
    endVertices = Vector{Int32}(undef, m)
    weights = Vector{Float32}(undef, m)
    offsetArray = zeros(Int32, n + 1)

    for (u, _, _) in edges
        offsetArray[u] += 1
    end

    # Convert counts to offsets (exclusive prefix sum)
    sum = 1
    for i in 1:(n + 1)
        temp = offsetArray[i]
        offsetArray[i] = sum
        sum += temp
    end

    # Fill endVertices and weights arrays
    temp_offset = copy(offsetArray)  # Track where to insert edges
    for (u, v, w) in edges
        index = temp_offset[u]
        endVertices[index] = v
        weights[index] = w
        temp_offset[u] += 1
    end

    return offsetArray, endVertices, weights
end

# Example Graph
# 1 -> 2 -> 4
#   \  |  /
#      3

#     1    2     3     4
# 1  0.0  1.5   2.0   inf
# 2  inf  0.0   inf   3.5
# 3  inf  inf   0.0   4.0
# 4  inf  inf   inf   0.0

# needs to be
# offsetArray: [1, 3, 5, 6, 6]
# endVertices: [2, 3, 4, 3, 4]
# weights: [1.5, 2.0, 3.5, 8.0, 4.0]



# n = 4  # Number of vertices
# edges = [(1, 2, 1.5), (1, 3, 2.0), (2, 4, 3.5), (3, 4, 4.0), (2, 3, 8.0)]

# Convert to CSR format
# offsetArray, endVertices, weights = graph_to_csr(n, edges)

# Print CSR representation
# println("offsetArray: ", offsetArray)
# println("endVertices: ", endVertices)
# println("weights: ", weights)
