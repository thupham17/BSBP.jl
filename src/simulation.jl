module simulation
using Distributions, DataFrames, Compat
using MathProgBase, Gurobi
using ..BSBP

""" Generates data from a normal distribution.
Args:
    n (int):        Number of observations.
    beta (float):   The upper limit of the range to generate, from 0 to `n` - 1.
    sigma (float):  The upper limit of the range to generate, from 0 to `n` - 1.
    typ (int):      1 for heteroskedasticity
Yields:
    y:              outcome
    datax:          covariates
"""
function simdata(n::Integer,theta::AbstractArray,sigma::AbstractArray,typ::Integer,seed::Integer)

    #Set seed
    if seed != nothing
        rng = MersenneTwister(seed)
        reg = Distributions.rand(rng,MvNormal(sigma),n)'
    else
        reg = Distributions.rand(MvNormal(sigma),n)'
    end

    k = length(theta)
    datax = [reg[:,1] ones(n,1) reg[:,2:k-1]]
    if typ == 1
        sigmaW = reg[:,1] .+ reg[:,2]
        e = 0.25*(ones(n) .+ 2*sigmaW.^2 .+ sigmaW.^4).*randn(n)
    else
        e = 0.25*randn(n)
    end
    y = Int.((datax*theta).>=e)
    return y,datax
end

""" Simulates best subset binary prediction on simulated data.
Args:
    warm_start (int,optional):  Set to 1 for warm start strategy. Default = 0.
    tau (float,optional):       Tuning paramater to construct the refined bound used in the warm start approach. Default = 1.5.
    mio (int,optional):         Set to 1 for using Method 1 for the MIO formulation. Set to 2 for Method 2. Default to 1.
    q (int, optional):          Value of the variable selection bound. Default = 1.
    N (int, optional):          Size of training sample. Default = 100.
    N_val (int, optional):      Size of validation sample. Default = 5000.
    R (int, optional):          Simulation repetitions. Default = 10.
    typ (int, optional):        Set to 1 for heteroskedastic error design. Set to 0 for homoskedastic error design. Default = 1.
"""
function Simulation(;warm_start=0,tau=1.5,mio=1,q=1,p=10,N=10,N_val=50,R=1,typ=1,beta0=1,rho=0.25,maxT=0,seed=1)

    # theta
    if typ==1
        beta_s = -1.5
    else
        beta_s = -0.35
    end

    beta = [beta0; [0]; beta_s; zeros(p-1,1)]

    K=length(beta)

    bhat=zeros(K-1,R)
    # sigma matrix
    row = zeros(p+1)
    for i in 1:p+1
        row[i]= rho^(i-1)
    end
    sigma = zeros(p+1, p+1)
        for i in 1:p+1
            sigma[i,:] = [row[i:-1:1];row[2:p-i+2]]
        end

    y, datax = simdata(N,beta,sigma,typ,seed)

    gap=zeros(R) # MIO gap
    rtime=zeros(R) # MIO running time
    ncount=zeros(R) # MIO node count
    score=zeros(R) # MIO score

    DGP_score=zeros(R) # in-sample score at the DGP parameter vector
    val_score=zeros(R) # in-sample score at the estimated parameter vector
    DGP_score_test=zeros(R) # out-of-sample score at the DGP parameter vector
    val_score_test=zeros(R) # out-of-sample score at the estimated parameter vector

    bnd=[-10*ones(K-1,1) 10*ones(K-1,1)]
    bnd_h = zeros(size(bhat)[1],2,R)

    if q>=1 && p>N
        tol=minimum([0.5*sqrt((1+q)*log(p)*N),0.05*N])  # early stopping rule
    else
        tol=0
    end

    println("Start")
    for i = 1:R
        println(i)

        y,datax = simdata(N,beta,sigma,typ,seed)

        try
            if warm_start == 1 # warm start MIO
                println("Start")
                bhat[:,i],score[i],gap[i],rtime[i],ncount[i]  = WarmStartMaxScore(y,datax[:,1:2],datax[:,3:size(datax)[2]],beta0,q,maxT,tol,bnd,mio,tau)
            else # cold start MIO
                bhat[:,i],score[i],gap[i],rtime[i],ncount[i]  = MaxScore(y,datax[:,1:2],datax[:,3:size(datax)[2]],beta0,q,maxT,tol,bnd,mio)
                println("Done")
            end
            catch y
                println(y.msg)
                throw(ErrorException("Error"))
        end

        DGP_score[i] = mean(Int.(y .== Int.(datax*beta.>=0)))
        val_score[i] = mean(Int.(y .== Int.(datax*[beta0;bhat[:,i]].>=0)))
        if N_val>0
            y_val,datax_val = simdata(N_val,beta,sigma,typ,seed)
            DGP_score_test[i] = mean(Int.(y_val .== Int.(datax_val*beta.>=0)))
            val_score_test[i] = mean(Int.(y_val .== Int.(datax_val*[beta0;bhat[:,i]].>=0)))
         end

    end

    println(mean([val_score[:,:] DGP_score[:,:] val_score_test[:,:] DGP_score_test[:,:]],2))
    println(mean([(val_score./DGP_score)[:,:] (val_score_test./DGP_score_test)[:,:]],2))
    return score,gap,rtime,ncount
end

export Simulation
end
