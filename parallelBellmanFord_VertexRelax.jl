using CUDA

function find(A, val)
    lo = 1
    hi = length(A) + 1
    while(hi - lo > 1)
        mid = fld(lo + hi, 2)
        if(A[mid] <= val)
            lo = mid
        else
            hi = mid
        end
    end
    return lo
end

function edge_breakdown_kernel!(E, W, FO, FI, WO, F1, FN, SO, chunk)
    id = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    idEnd = id * chunk
    idStart = idEnd - chunk + 1
    idEnd = min(length(FO), idEnd)

    @inbounds for id in idStart:idEnd
        i = find(SO, id - 1) # binary search

        # lo = 1; hi = length(SO) + 1; target = id - 1
        # while(hi - lo > 1)
        #     mid = (lo + hi) ÷ 2
        #     if(SO[mid] <= target)
        #         lo = mid
        #     else
        #         hi = mid
        #     end
        # end
        # i = lo
        
        j = id - SO[i]
        FO[id] = E[FN[i] + j - 1]
        WO[id] = W[FN[i] + j - 1]
        FI[id] = F1[i]
    end
    return nothing
end

function relaxation_kernel!(FO, FI, WO, R, D, chunk)
    id = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    idEnd = id * chunk
    idStart = idEnd - chunk + 1
    idEnd = min(length(FO), idEnd)

    @inbounds for id in idStart:idEnd
        fi = FI[id]
        fo = FO[id]
        new_dist = D[fi] + WO[id]
        
        # Atomic min using CAS
        old = D[fo]
        while old > new_dist
            assumed = old
            old = CUDA.atomic_cas!(pointer(D, fo), assumed, new_dist)
            old == assumed && (R[fo] = 1)
        end
    end
end


function frontier_generation_kernel2!(F2, B, BS, chunk)
    id = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    idEnd = id * chunk
    idStart = idEnd - chunk + 1
    idEnd = min(length(B), idEnd)

    @inbounds for id in idStart:idEnd
        if B[id] > 0
            F2[BS[id]] = id
        end
    end
    return nothing
end


function parallelBellmanFord_VertexRelax(s, offsetArray_gpu, endVertices_gpu, weights_gpu; debug = 0, chunkSize = 128)
    n = length(offsetArray) - 1
    m = length(weights)

    # offsetArray_gpu = CuArray(offsetArray); endVertices_gpu = CuArray(endVertices); weights_gpu = CuArray(weights)

    D = fill(Inf32, n) |> CuArray
    CUDA.@allowscalar D[s] = 0
    F1 = CUDA.fill(s, 1)
    n_cur = 1
    iter = Int32(0)
    while(n_cur > 0)
        FN = similar(F1); FN .= offsetArray_gpu[F1]
        if debug == 2
        println("FN ", FN)
        end

        S = similar(F1); 
        S .= offsetArray_gpu[F1 .+ 1] .- offsetArray_gpu[F1]
        if debug == 2
        println("S ", S)
        end

        SO = similar(S); 
        CUDA.scan!(+, SO, S, dims = 1, init = 0, neutral = 0)
        # right shift and initialize with 0
        SO .= circshift(SO, 1); 
        CUDA.@allowscalar SO[1] = CUDA.zero(Int32)

        CUDA.@allowscalar e_cur = SO[end] + S[end]
        if debug == 1 || debug == 2
        println("e_cur ", e_cur)
        end

        if(e_cur == 0)
            break
        end

        FO = CuArray{Int32}(undef, e_cur); WO = CuArray{Float32}(undef, e_cur); FI = CuArray{Int32}(undef, e_cur)

        nthreads = 16
        nblocks = cld(cld(e_cur, chunkSize), nthreads)
        @cuda threads=nthreads blocks=nblocks edge_breakdown_kernel!(endVertices_gpu, weights_gpu, FO, FI, WO, F1, FN, SO, chunkSize)
        
        if debug == 2
        println("FO ", FO); println("WO ", WO); println("FI ", FI)
        end

        B = CUDA.fill(Int32(0), n)
        @cuda threads=nthreads blocks=nblocks relaxation_kernel!(FO, FI, WO, B, D, chunkSize)
        if debug == 2
        println("R ", B)
        println("D ", D)
        end

        BS = similar(B)
        CUDA.scan!(+, BS, B, dims = 1, init = 0, neutral = 0)
        # right shift and initialize with 1 (for future indexing)
        BS .= circshift(BS, 1); 
        CUDA.@allowscalar BS[1] = CUDA.zero(Int32)
        BS .= BS .+ 1

        n_next = CUDA.@allowscalar(BS[end])
        if(CUDA.@allowscalar B[end] == 0) 
            n_next = n_next - 1
        end
        
        F2 = CuArray{Int32}(undef, n_next)
        nblocks = cld(cld(n, chunkSize), nthreads)
        @cuda threads=nthreads blocks=nblocks frontier_generation_kernel2!(F2, B, BS, chunkSize)
        if debug == 1 || debug == 2
        println("F2 ", F2)
        end

        F1, F2 = F2, F1
        n_cur = n_next
        iter += 1
    end
    return D
end