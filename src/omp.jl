struct OMPTracer{T}
    iteration::Vector{Int}
    lambda::Vector{T}
    solution::Vector{SparseVector{T}} 
end

Base.length(t::OMPTracer) = length(t.iteration)

function Base.getindex(t::OMPTracer, i::Integer)
    return t.solution[i], t.lambda[i]
end

Base.lastindex(t::OMPTracer) = lastindex(t.iteration)

@doc raw"""
    ```julia
    asp_omp(A,B)```
    Orthogonal matching pursuit for sparse 
    ```math 
    Ax=b
    ```
    Applies the orthogonal matching pursuit (OMP) algorithm to
    estimate a sparse solution of the underdetermined system `Ax=b`.

    (BP)   
    ```math 
    \min_x  \|x\|_1  
    ```
    subject to  
    ```math 
    Ax = b.
    ```
"""
function asp_omp(
    A::Union{AbstractMatrix, AbstractLinearOperator},
    b::Vector,
    λin::Real;
    active::Union{Nothing, Vector{Int}} = nothing,  
    state::Union{Nothing, Vector{Int}} = nothing,
    S::Matrix{Float64} = Matrix{Float64}(undef, size(A, 1), 0),
    R::Union{Nothing, Matrix{Float64}} = nothing,
    loglevel::Int = 1,
    λmin::Real = sqrt(eps(1.0)),
    itnMax::Int = 10 * maximum(size(A)),
    feaTol::Real = 5e-05,
    optTol::Real = 1e-05,
    gapTol::Real = 1e-06,
    pivTol::Real = 1e-12,
    actMax::Union{Real, Nothing} = nothing) 
    
    time0 = time()

    z = A' * b
    
    m = length(b)
    n = length(z)

    work = rand(size(A,2))
    work2 = rand(size(A,2))
    work3 = rand(size(A,2))
    work4 = rand(size(A,2))
    work5 = rand(size(A,1))

    nprodA = 0
    nprodAt = 1

    tracer = OMPTracer(
        Int[],                 
        Float64[],              
        Vector{SparseVector{Float64}}() 
    )

    if loglevel > 0
        @info "-"^124
        @info @sprintf("%-30s : %-10d    %-30s : %-10.4e", "No. rows", m, "λ", λin)
        @info @sprintf("%-30s : %-10d    %-30s : %-10.1e", "No. columns", n, "Optimality tol", optTol)
        @info @sprintf("%-30s : %-10d    %-30s : %-10.1e", "Maximum iterations", itnMax, "Duality tol", gapTol)
        @info "-"^124
    end


    # Initialize local variables.
    EXIT_INFO = Dict(
        :EXIT_OPTIMAL => "Optimal solution found -- full Newton step",
        :EXIT_TOO_MANY_ITNS =>  "Too many iterations",
        :EXIT_SINGULAR_LS => "Singular least-squares subproblem",
        :EXIT_LAMBDA => "Reached minimum value of lambda",
        :EXIT_RHS_ZERO => "b = 0. The solution is x = 0",
        :EXIT_UNCONSTRAINED => "Unconstrained solution r = b is optimal",
        :EXIT_ACTMAX => "Max no. of active constraints reached",
        :EXIT_UNKNOWN => "unknown exit"
    )

    itn = 0
    eFlag = :EXIT_UNKNOWN
    x = zeros(Float64, 0)
    zerovec = zeros(Float64, n)
    p = 0
    cur_r_size = 0

    if norm(b, Inf) == 0
        r = zeros(m)
        eFlag = :EXIT_RHS_ZERO
    end

    # Solution is unconstrained for lambda large.
    zmax = norm(z, Inf)
    if eFlag == :EXIT_UNKNOWN && zmax < λin
        r = b
        eFlag = :EXIT_UNCONSTRAINED
    end

    if eFlag != :EXIT_UNKNOWN || active === nothing
        active = Vector{Int}([])
    end
    if state === nothing
        state = zeros(Int, n)
    end
    if R === nothing
        R = Matrix{Float64}(undef, size(A,2), size(A,2))
        S = Matrix{Float64}(undef, size(A,1), size(A,2))
    end

    if actMax === nothing
        actMax = size(A, 2)
    end

    if loglevel>0
        @info @sprintf("%4s  %8s %12s %12s %12s", "Itn", "Var", "λ", "rNorm", "xNorm")
    end

    # Main loop.
    while true
        # Compute dual obj gradient g, search direction dy, and residual r.
        if itn == 0
            x = Float64[]
            r = b
            z = A' * r
            nprodAt += 1
            zmax = norm(z, Inf)
        else
            x,y = csne(R[1:cur_r_size, 1:cur_r_size], S[:,1:cur_r_size], vec(b))
            if norm(x, Inf) > 1e12
                eFlag = :EXIT_SINGULAR_LS
                break
            end

            Sx = S[:,1:cur_r_size] * x
            r = b - Sx
        end

        rNorm = norm(r, 2)
        xNorm = norm(x, 1)

        if loglevel>0
            @info @sprintf("%4i  %8i %12.5e %12.5e %12.5e", itn, p, zmax, rNorm, xNorm)
        end

        # Check exit conditions.
        if eFlag != :EXIT_UNKNOWN
            # Already set. Don't test the other exits.
        elseif zmax <= λin
            eFlag = :EXIT_LAMBDA
        elseif rNorm <= optTol
            eFlag = :EXIT_OPTIMAL
        elseif itn >= itnMax
            eFlag = :EXIT_TOO_MANY_ITNS
        elseif itn == actMax
            eFlag = :EXIT_ACTMAX
        end

        if eFlag != :EXIT_UNKNOWN
            break
        end

        # New iteration starts here.
        itn += 1

        # Find step to the nearest inactive constraint
        z = A' * r

        nprodAt += 1
        zmax, p = findmax(abs.(z))

        if z[p] < 0
            state[p] = -1
        else
            state[p] = 1
        end

        zerovec[p] = 1   # Extract a = A[:, p]
        a = A * zerovec

        nprodA += 1
        zerovec[p] = 0

        qraddcol!(S, R, a, cur_r_size, work, work2, work3, work4, work5)  # Update R
        # S = hcat(S, a)  # Expand S, active
        cur_r_size +=1 
        push!(tracer.iteration, itn)
        push!(tracer.lambda, zmax)
        sparse_x_full = SparseVector(n, copy(active), copy(x))
        push!(tracer.solution, copy(sparse_x_full))
        push!(active, p)

    end #while true

    push!(tracer.iteration, itn)
    push!(tracer.lambda, zmax)
    sparse_x_full = SparseVector(n, copy(active), copy(x))
    push!(tracer.solution, copy(sparse_x_full))

    tottime = time() - time0
    if loglevel > 0
        @info @sprintf("\nEXIT BPdual -- %s\n", EXIT_INFO[eFlag])
        @info @sprintf("%-20s: %8i", "Products with A", nprodA)
        @info @sprintf("%-20s: %8i", "Products with At", nprodAt)
        @info @sprintf("%-20s: %8.1e", "Solution time (sec)", tottime)
        @info "\n"
    end
    return tracer
end