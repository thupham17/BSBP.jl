using MathProgBase

"""
Computes the estimates of the logit regression of y on x.
Args:
    y (array): (n by 1) array.
    x (array): (n by k) array.
"""
function logit(y::AbstractArray,x::AbstractArray)
    println("Start logit")
    cnv = 0
    b0 = zeros(size(x)[2],1)
    b1 = zeros(size(x)[2],1)
    while cnv==0
        ind = x*b0
        P = (exp.(ind))./(1+exp.(ind))
        grd = sum(repmat(y-P,1,size(x)[2]).*x,1)
        hes = -x'*(repmat(P.*(1-P),1,size(x)[2]).*x)
        b1 = b0 - inv(hes)*grd'
        dev = maximum(abs.(b1.-b0))
        if dev < 1e-8
            println("true")
            cnv = 1
        end
        b0 = b1
    end
    return b1
end

"""
Refines the bounds of the unknown coefficients for the parameter space in warm start approach.
Args:
    x (array): (n by k) matrix of covariate data.
    beta0 (int): Coefficient for the first covariate in x.
    bnd (array): ((k-1) by 2) matrix where the first and second columns respectively
                 store the lower and upper bounds of the unknown coefficients
Yields:
    bound (array): ((k-1) by 2) matrix for the refined lower and upper bounds
                of the unknown coefficients for the parameter space used
                in the warm-start approach.
"""
function getBnd(y::AbstractArray,x::AbstractArray,beta0::Integer,bnd::AbstractArray)
    println("Start get bound")
    k=size(x)[2]-1

    p_hat = 1./(1+exp.(-x*logit(y,x)))

    constr=repmat(p_hat-0.5,1,size(x)[2]).*x
    bound= copy(bnd)

    A = constr[:,2:size(x)[2]]
    rhs = -constr[:,1]*beta0

    #Set Params
    tol=0.000001
    solver = GurobiSolver(OutputFlag=0,OptimalityTol=tol,FeasibilityTol=tol,IntFeasTol=tol)

    lb = bound[:,1]
    ub = bound[:,2]

    for i=1:k
        objcoef=zeros(1,k)
        objcoef[i]=1
        obj = objcoef

        try
            result= linprog(vec(obj), A, '>', vec(rhs), vec(lb), vec(ub), solver)
            bound[i,1]=result.objval
            catch
                println("Error reported")
        end
        lb = bound[:,1]
        try
            result= linprog(-vec(obj), A, '>', vec(rhs), vec(lb), vec(ub), solver)
            bound[i,2]=-result.objval
            catch
                println("Error reported")
        end
        ub = bound[:,2]
    end
    return bound
end

"""
Implements the warm start approach.
Args:
    y (array):      vector of binary outcomes.
    x_foc (array):  (n by k) matrix of focus covariates where the first column should
                    contain data of the continous regressor with respect to the scale normalization.
    x_aux (array):  (n by d) matrix of auxiliary covariates which will be selected based on best
                    subset covariate selection procedure.
    beta0 (int):    Coefficient taking value either 1 or -1 to normalize the scale for the first
                    first covariate in x_foc.
    q (int):        Cardinality constraint for the covariate selection.
    T (int):        Time limit specified for the MIO solver.
    bnd (array):    (((k-1)+d) by 2) matrix where the first and second columns respectively
                    store the lower and upper bounds of the unknown coefficients. First (k-1)
                    rows correspond to the bounds of the focused covariates excluding the first one.
                    Remaining d rows correspond to the bounds of the auxiliary covariates.
    mio (int) :     MIO formulation of the best subset maximum score problem
                    set mio = 1 for Method 1 of the paper
                    set mio = 2 for Method 2 of the paper
    tau (int):      Tuning parameter for enlarging the estimated bounds.
Yields:
    bhat (array):   Maximum score estimates for the unknown coefficients.
    score (array):  Value of maximum score objective function.
    gap   (array):  MIO optimization gap value in case of early termination
                    gap = 0 ==> optimal solution is found within the time limit.
    rtime (array):  Time used by the MIO solver in the estimation procedure.
"""
function WarmStartMaxScore(y::AbstractArray,x_foc::AbstractArray,x_aux::AbstractArray,beta0::Integer,q::Integer,T::Real,tol::Real,bnd::AbstractArray,mio::Integer,tau::Real)
    bnd_h = getBnd(y,[x_foc x_aux],beta0,bnd)
    bnd_abs = tau*maximum(abs.(bnd_h),2)
    bnd0 = [maximum([-bnd_abs bnd[:,1]],2) minimum([bnd_abs bnd[:,2]],2)] # this is the refind bound used for warm start MIO
    bhat,score,gap,rtime,ncount = MaxScore(y,x_foc,x_aux,beta0,q,T,tol,bnd0,mio)
    return bhat,score,gap,rtime,ncount
end

export WarmStartMaxScore
