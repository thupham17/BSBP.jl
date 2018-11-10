using MathProgBase, Gurobi
using Distributions, DataFrames
using Compat

InputVector{T<:Union{Real,Char}} = Union{Vector{T},Real,Char}
const SymbolInputVector = Union{Vector{Symbol},Symbol}

"""
Builds a mixed integer programming problem as defined in MathProgBase.
Args:
     c:      is the objective vector, always in the sense of minimization
     A:      is the constraint matrix
     sense:  is a vector of constraint sense characters '<', '=', and '>'
     b:      is the right-hand side vector
     l:      is the vector of lower bounds on the variables
     u:      is the vector of upper bounds on the variables, and
     solver: specifies the desired solver, see :ref:`choosing solvers <choosing-solvers>`.

A scalar is accepted for the b, sense, l, and u arguments, in which case its value is replicated.
The values -Inf and Inf are interpreted to mean that there is no corresponding lower or upper bound.
"""
function buildmio(c::InputVector, A::AbstractMatrix, rowlb::InputVector, rowub::InputVector, vartypes::SymbolInputVector, lb::InputVector, ub::InputVector, solver)
    m = MathProgBase.LinearQuadraticModel(solver)
    nrow,ncol = size(A)

    c = MathProgBase.HighLevelInterface.expandvec(c, ncol)
    rowlbtmp = MathProgBase.HighLevelInterface.expandvec(rowlb, nrow)
    rowubtmp = MathProgBase.HighLevelInterface.expandvec(rowub, nrow)
    lb = MathProgBase.HighLevelInterface.expandvec(lb, ncol)
    ub = MathProgBase.HighLevelInterface.expandvec(ub, ncol)
    vartypes = MathProgBase.HighLevelInterface.expandvec(vartypes, ncol)

    # rowlb is allowed to be vector of senses
    if eltype(rowlbtmp) == Char
        realtype = eltype(rowubtmp)
        sense = rowlbtmp
        rhs = rowubtmp
        @assert realtype <: Real
        rowlb = Array{realtype}(undef, nrow)
        rowub = Array{realtype}(undef, nrow)
        for i in 1:nrow
            if sense[i] == '<'
                rowlb[i] = typemin(realtype)
                rowub[i] = rhs[i]
            elseif sense[i] == '>'
                rowlb[i] = rhs[i]
                rowub[i] = typemax(realtype)
            elseif sense[i] == '='
                rowlb[i] = rhs[i]
                rowub[i] = rhs[i]
            else
                error("Unrecognized sense '$(sense[i])'")
            end
        end
    else
        rowlb = rowlbtmp
        rowub = rowubtmp
    end

    MathProgBase.loadproblem!(m, A, lb, ub, c, rowlb, rowub, :Min)
    MathProgBase.setvartype!(m, vartypes)
    return m
end

"""
Solves the maximization problem max |beta0*x(i,1)+x(i,:)*t| over t subject to bnd.
Args:
    x (array):      (n by k) matrix of covariate data.
    beta0 (int):    coefficient for the first covariate in x.
    bnd (array):    ((k-1) by 2) matrix where the 1st and 2nd columns store the lower and
                    upper bounds of the unknown coefficients.
"""
function MIObnd(x::AbstractArray,beta0::Integer,bnd::AbstractArray)
    println("Start mio bound")
    n=size(x)[1]
    k=size(x)[2]-1

    v = zeros(2,1)
    value=zeros(n,1)

    #Set Params
    tol=0.000001
    solver = GurobiSolver(OutputFlag=0,OptimalityTol=tol,FeasibilityTol=tol,IntFeasTol=tol)

    for i = 1:n
        println("i = ",i)
        alpha = beta0*x[i,1]
        obj = x[i,2:k+1]

        try
            A = x[i,2:k+1][:,:]'
            rhs = -alpha
            result= linprog(-obj, A, '>', rhs, bnd[:,1], bnd[:,2], solver)
            v[1]=-result.objval+alpha
            catch
                println("Error Reported")
        end

        obj = -x[i,2:k+1]
        try
            A = -x[i,2:k+1][:,:]'
            rhs = alpha
            result = linprog(-obj, A, '>', rhs, bnd[:,1], bnd[:,2], solver)
            v[2] = -result.objval-alpha
            catch
                println("Error Reported")
        end
        value[i]=maximum(v)
    end
    return value
end

"""
Calculates the maximum score of the objective function.
Args:
    y (array):      vector of binary outcomes.
    x_foc (array):  (n by k) matrix of focus covariates where the first column should
                    contain data of the continous regressor with respect to the scale normalization.
    x_aux (array):  (n by d) matrix of auxiliary covariates which will be selected based on best
                    subset covariate selection procedure.
    beta0 (int):    Coefficient taking value either 1 or -1 to normalize the scale for the first
                    first covariate in x_foc.
    q (int):        Cardinality constraint for the covariate selection.
    abgap (int):    Absolute gap specified for early termination of the MIO solver.
    bnd (array):    (((k-1)+d) by 2) matrix where the first and second columns respectively
                    store the lower and upper bounds of the unknown coefficients. First (k-1)
                    rows correspond to the bounds of the focused covariates excluding the first one.
                    Remaining d rows correspond to the bounds of the auxiliary covariates.
    mio (int) :     MIO formulation of the best subset maximum score problem
                    set mio = 1 for Method 1 of the paper
                    set mio = 2 for Method 2 of the paper
Yields:
    bhat (array):   Maximum score estimates for the unknown coefficients.
    score (array):  Value of maximum score objective function.
    gap   (array):  MIO optimization gap value in case of early termination
                    gap = 0 ==> optimal solution is found within the time limit.
    rtime (array):  Time used by the MIO solver in the estimation procedure.
"""
function MaxScore(y::AbstractArray,x_foc::AbstractArray,x_aux::AbstractArray,beta0::Integer,q::Integer,T::Real,abgap::Real,bnd::AbstractArray,mio::Integer)
    N=length(y)
    k=size(x_foc)[2]-1
    d=size(x_aux)[2]
    bhat=zeros(k+d)
    score = 0
    gap = 0
    rtime = 0
    ncount = 0

    miobnd=BSBP.MIObnd([x_foc x_aux],beta0,bnd)

    #Set Params
    tol=0.000001
    env = Gurobi.Env()
    setparams!(env; OutputFlag=0,OptimalityTol=tol,FeasibilityTol=tol,IntFeasTol=tol)
    solver = GurobiSolver(env)

    if T > 0
    setparams!(env; TimeLimit = T)
    solver = GurobiSolver(env)
    end

    if abgap > 0
    setparams!(env; MIPGapAbs = abgap)
    solver = GurobiSolver(env)
    end

    #Bounds
    lb = [zeros(N,1); bnd[:,1]; zeros(d,1)]
    ub = [ones(N,1); bnd[:,2]; ones(d,1)]

    #Variable Type
    vtype = [repmat([:Bin],N); repmat([:Cont],k+d); repmat([:Bin],d)]

    if T > 0
        MathProgBase.setparameters!(solver, TimeLimit=T)
    end

    if abgap > 0
        setparams!(env;MIPGapAbs=abgap)
        solver = GurobiSolver(env)
    end

    ztemp1=zeros(N,d)
    ztemp2=zeros(2*d+1,N)
    htemp=[eye(d); -eye(d); zeros(1,d)]
    etemp=[-Diagonal(bnd[k+1:k+d,2]); Diagonal(bnd[k+1:k+d,1]); ones(1,d)]
    mtemp1=[ztemp2 zeros(2*d+1,k) htemp etemp]
    mtemp2=[zeros(2*d,1);[q]]

    if mio == 1 # Method 1 formulation
        obj = vcat((2*y-1), zeros(k+2*d,1))
        objcon = sum(1-y)
        miobnd_bar = miobnd+tol
        mtemp3= [-Diagonal(vec(miobnd_bar)) x_foc[:,2:k+1] x_aux ztemp1]
        A = [[Diagonal(vec(miobnd)) -x_foc[:,2:k+1] -x_aux ztemp1]; mtemp3; mtemp1]
        rhs = [miobnd*(1-tol)+beta0*x_foc[:,1];-tol*miobnd_bar-beta0*x_foc[:,1]; mtemp2]
    else # Method 2 formulation
        obj = [ones(1,N) zeros(1,k+2*d)]
        objcon = 0
        temp2=(1-2*y)
        A = [[Diagonal(vec(miobnd)) repmat(temp2,1,k).*x_foc[:,2:k+1] repmat(temp2,1,d).*x_aux ztemp1]; mtemp1]
        rhs = [miobnd*(1-tol)-(beta0*temp2.*x_foc[:,1]);mtemp2]
    end

    try
        m = buildmio(-vec(obj),A,'<',vec(rhs),vtype,vec(lb),vec(ub),solver)
        MathProgBase.optimize!(m)
        bhat = MathProgBase.getsolution(m)[N+1:N+k+d]
        score = -MathProgBase.getobjval(m) + objcon
        gap = (-MathProgBase.getobjbound(m)-score+objcon)
        rtime= MathProgBase.getsolvetime(m)
        ncount = MathProgBase.getnodecount(m)
        println("Optimization returned status: ", MathProgBase.status(m))

    catch
        println("Error reported")
    end
    return bhat[:,:], score, gap, rtime, ncount
end

export MaxScore
