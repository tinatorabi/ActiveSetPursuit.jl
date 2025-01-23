struct OMPTracer{T}
    iteration::Vector{Int}
    lambda::Vector{T}
    active::Vector{Vector{Int}}  # Store active indices instead of full sparse vectors
    values::Vector{Vector{T}}    # Store corresponding values for active indices
end

Base.length(t::OMPTracer) = length(t.iteration)

function Base.getindex(t::OMPTracer, i::Integer)
    return t.active[i], t.values[i], t.lambda[i]
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
    actMax::Union{Real, Nothing} = nothing,
) 
    time0 = time()

    m = length(b)
    n = size(A, 2)
    T = eltype(A)

    # Pre-allocate work vectors
    work = Vector{T}(undef, n)
    work2 = Vector{T}(undef, n)
    work3 = Vector{T}(undef, n)
    work4 = Vector{T}(undef, n)
    work5 = Vector{T}(undef, m)

    nprodA = 0
    nprodAt = 0

    # Pre-allocate tracer fields
    tracer = OMPTracer(
        iteration=Vector{Int}(undef, itnMax),
        lambda=Vector{T}(undef, itnMax),
        active=Vector{Vector{Int}}(undef, itnMax),
        values=Vector{Vector{T}}(undef, itnMax),
    )

    # Pre-allocate residuals and solution variables
    r = copy(b)
    x = zeros(T, n)
    z = A' * b
    nprodAt += 1

    # Exit condition flags
    eFlag = :EXIT_UNKNOWN

    # Initialize active set
    if active === nothing
        active = Vector{Int}(undef, itnMax)
        active .= 0  # Ensure initialized
    end
    if state === nothing
        state = zeros(Int, n)
    end
    if R === nothing
        R = Matrix{Float64}(undef, n, n)
        S = Matrix{Float64}(undef, m, n)
    end
    if actMax === nothing
        actMax = n
    end

    # Logging information
    if loglevel > 0
        @info "-"^124
        @info @sprintf("%-30s : %-10d    %-30s : %-10.4e", "No. rows", m, "λ", λin)
        @info @sprintf("%-30s : %-10d    %-30s : %-10.1e", "No. columns", n, "Optimality tol", optTol)
        @info "-"^124
    end

    cur_r_size = 0
    zmax, p = findmax(abs.(z))

    while eFlag == :EXIT_UNKNOWN
        # Check for convergence or stopping criteria
        if zmax <= λin
            eFlag = :EXIT_LAMBDA
            break
        elseif norm(r) <= optTol
            eFlag = :EXIT_OPTIMAL
            break
        elseif cur_r_size >= itnMax
            eFlag = :EXIT_TOO_MANY_ITNS
            break
        end

        # Update active set and QR factorization
        zerovec = zeros(T, n)
        zerovec[p] = 1.0
        a = A * zerovec
        nprodA += 1

        qraddcol!(S, R, a, cur_r_size, work, work2, work3, work4, work5)
        cur_r_size += 1
        active[cur_r_size] = p

        # Solve the reduced system using the QR factorization
        x[1:cur_r_size], y = csne(
            @view R[1:cur_r_size, 1:cur_r_size],
            @view S[:, 1:cur_r_size],
            vec(b),
        )
        r .= b .- @view S[:, 1:cur_r_size] * x[1:cur_r_size]

        # Update dual variables
        z .= A' * r
        nprodAt += 1
        zmax, p = findmax(abs.(z))

        # Store iteration in the tracer
        tracer.iteration[cur_r_size] = cur_r_size
        tracer.lambda[cur_r_size] = zmax
        tracer.active[cur_r_size] = copy(active[1:cur_r_size])
        tracer.values[cur_r_size] = copy(x[1:cur_r_size])

        if loglevel > 0
            @info @sprintf("%4i  %8i %12.5e %12.5e %12.5e", cur_r_size, p, zmax, norm(r), norm(x))
        end
    end

    # Finalize tracer
    tracer.iteration .= tracer.iteration[1:cur_r_size]
    tracer.lambda .= tracer.lambda[1:cur_r_size]
    tracer.active .= tracer.active[1:cur_r_size]
    tracer.values .= tracer.values[1:cur_r_size]

    if loglevel > 0
        @info "-"^124
        @info @sprintf("Exit reason: %s", EXIT_INFO[eFlag])
    end

    return tracer
end
