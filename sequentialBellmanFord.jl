

function sequentialBellmanFord(s::Int32, offsetArray::Vector{Int32}, endVertices::Vector{Int32}, weights::Vector{Float32}; debug = 0)
    n = length(offsetArray) - 1
    m = length(weights)

    D = fill(Inf32, n)
    D[s] = 0.0
    F1 = fill(s, 1)
    # push!(F1, s)
    n_cur = 1
    while(n_cur > 0)
        FN = similar(F1); FN .= offsetArray[F1]
        if debug == 2
        println("FN ", FN)
        end

        S = similar(F1); S .= offsetArray[F1 .+ 1] .- offsetArray[F1]
        if debug == 2
        println("S ", S)
        end

        SO = similar(S)
        sum = 0
        @inbounds for i in 1:n_cur
            SO[i] = sum
            sum += S[i]
        end

        e_cur = (SO[n_cur] + S[n_cur])
        if debug == 1 || debug == 2
        println("e_cur ", e_cur)
        end

        if(e_cur == 0)
            break
        end

        FO = Vector{Int32}(undef, e_cur)
        WO = Vector{Float32}(undef, e_cur)
        FI = Vector{Int32}(undef, e_cur)

        @inbounds for id in 1:e_cur
            i = find(SO, id - 1)
            j = id - SO[i]
            FO[id] = endVertices[FN[i] + j - 1]
            WO[id] = weights[FN[i] + j - 1]
            FI[id] = F1[i]
        end
        if debug == 2
        println("FO ", FO)
        println("WO ", WO)
        println("FI ", FI)
        end


        B = similar(FO)
        @inbounds for i in 1:e_cur
            if(D[FO[i]] > D[FI[i]] + WO[i])
                B[i] = 1
                D[FO[i]] = D[FI[i]] + WO[i]
            else
                B[i] = 0
            end
            # println(D[FO[i]], " ", D[FI[i]], " ", WO[i], " ", B[i])
        end
        if debug == 2
        println("D ", D)
        println("B ", B)
        end

        BS = similar(B)
        # CUDA.scan!(+, BS, B, dims = 1, init = 0, neutral = 0)
        sum = 1
        @inbounds for i in 1:e_cur
            BS[i] = sum
            sum += B[i]
            # println(" sum ", sum)
        end

        n_next = BS[e_cur]
        if(B[e_cur] == 0) 
            n_next = n_next - 1
        end
        F2 = fill(0, n_next)
        @inbounds for i in 1:e_cur
            if(B[i] > 0)
                F2[BS[i]] = FO[i]
            end
        end
        if debug == 1 || debug == 2
        println("F2 ", F2)
        end

        F1, F2 = F2, F1

        n_cur = length(F1)
        # n_cur = 0
    end
    return D
end



# function sequentialBellmanFordDummy(s::Int32, offsetArray::Vector{Int32}, endVertices::Vector{Int32}, weights::Vector{Float32}; debug = 0)
#     n = length(offsetArray) - 1
#     m = length(weights)

#     D = fill(Inf32, n)
#     D[s] = 0; F1 = [s]
#     while(F1 not empty)
#         FN .= offsetArray[F1]                          # where does current node start in the TRUE edges list
#         S .= offsetArray[F1 .+ 1] .- offsetArray[F1]   # how many outgoing edges current node has
#         SO = exclusive prefix sum of S           # where does current node start in the ACTIVE edges list
#         e_cur = (SO[n_cur] + S[n_cur])           # length of the ACTIVE edges list
#         for i in 1:n_cur
#             for j in 1:S[i]
#             FO[SO[i] + j] = endVertices[FN[i] + j - 1]    # end vertex of current edge
#             WO[SO[i] + j] = weights[FN[i] + j - 1]        # weight of current edge
#             FI[SO[i] + j] = F1[i]                         # start vertex of current edge i.e. current node

#         for i in 1:e_cur
#             if(D[FO[i]] > D[FI[i]] + WO[i])             # for all edges, try to relax
#                 R[FO[i]] = 1                                # and store if current edge has been relaxed
#                 D[FO[i]] = D[FI[i]] + WO[i]
#             else B[i] = 0 

#         BS = exclusive prefix sum of R      # where does current edge start in the NEXT vertices list
#         n_next = BS[n]                  # length of the NEXT vertices list

#         for i in 1:n
#             if(R[i] > 0)         
#                 F2[BS[i]] = i          # store the current vertex in the NEXT vertices list

#         F1, F2 = F2, F1     # swap
#     end
#     return D
# end