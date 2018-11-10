"""
Reproduction of the empirical results concerning an empirical application
in the prediction of transportation mode choice using the best subset
maximum score cross validation approach of Chen and Lee (2017).
The work-trip mode choice dataset of Horowitz (1993) is included in the package.
"""
module transportationModeCV

using DataFrames, Compat
using MathProgBase, Gurobi
using ..BSBP

""" Best subset maximum score approach with cross validation on the work-trip mode choice dataset of Horowitz.
Args:
    warm_start (int,optional):  Set to 1 for warm start strategy. Default = 1.
    tau (float,optional):       Tuning paramater to construct the refined bound used in the warm start approach. Default = 1.5.
    mio (int,optional):         Set to 1 for using Method 1 for the MIO formulation. Set to 2 for Method 2. Default to 1.
    q (int, optional):          Value of the variable selection bound. Default = 1.
    series_exp (int, optional): Set to 1 to use quadratic expansion terms as covariates. Default = 1.
    b (int, optional):          Bound value. Default = 1.
    time_limit (int, optional): MIO solver time limit. Default = 8640000.
"""
function TransportationModeCV(;warm_start=1,tau=1.5,mio=1,series_exp=0,beta0=1,b=10,q=1,time_limit=86400)
    data = readdlm("./data_horowitz.csv",',')
    tr_ind = readdlm("./tr_ind.csv",',') .== "TRUE"
    test_ind = readdlm("./test_ind.csv",',') .== "TRUE"

    # create variables from training and validation samples
    # data columins : [Y DCOST CARS DOVTT DIVTT]

    fold=size(tr_ind)[2]

    if series_exp==1
        bhat=zeros(10,fold)
    else
        bhat=zeros(4,fold)
    end

    score=zeros(fold)
    gap=zeros(fold)
    in_score=zeros(fold)
    rtime=zeros(fold)
    ncount=zeros(fold)
    p_ratio=zeros(fold)

    for i=1:fold
        data_tr=data[tr_ind[:,i],:]
        data_v=data[test_ind[:,i],:]

        println("estimation based on training sample at fold: ",i)
        Y_tr=data_tr[:,1]
        temp=data_tr[:,2:end]

        n_tr=length(Y_tr)
        # [DCOST CARS DOVTT DIVTT]
        x_std=(temp.-repmat(mean(temp,1),n_tr,1))./repmat(std(temp,1),n_tr,1)
        x_foc = [x_std[:,1] ones(n_tr,1)]  # [DCOST Intercept]

        if series_exp == 1
            z2 = x_std[:,2]
            z3 = x_std[:,3]
            z4 = x_std[:,4]
            x_aux1 = [z2 z3 z4] # linear terms
            x_aux2 = [z2.*z3 z3.*z4 z2.*z4]
            x_aux3 = [z2.*z2 z3.*z3 z4.*z4]
            x_aux = [x_aux1 x_aux2 x_aux3]
        else
            x_aux = x_std[:,2:4] # [CARS DOVTT DIVTT]
        end
        k=size(x_foc)[2]
        d=size(x_aux)[2]

        bnd= [-b*ones(k-1+d,1) b*ones(k-1+d,1)] # set the initial parameter bounds

        tol = floor(sqrt(log(n_tr)*n_tr)/2)
        println("tolerance level: ", tol)

        if warm_start == 1 # warm start MIO
            bhat[:,i],score[i],gap[i],rtime[i],ncount[i]  = WarmStartMaxScore(Y_tr,x_foc,x_aux,beta0,q,time_limit,tol,bnd,mio,tau)
        else # cold start MIO
            bhat[:,i],score[i],gap[i],rtime[i],ncount[i]  = MaxScore(Y_tr,x_foc,x_aux,beta0,q,time_limit,tol,bnd,mio)
        end

        println("coefficient values: ")
        println("gurobi score: ",score[i])
        println("gurobi absolute gap: ",gap[i])
        println("gurobi running time: ",rtime[i])
        println("gurobi node count: ",ncount[i])
        in_score[i]=sum(Int.(Y_tr.==(([x_foc x_aux]*[1;bhat[:,i]]).>0)))
        println("in-sample score: ",in_score[i])

        # validation sample

        Y_val=data_v[:,1]
        n_val=length(Y_val)
        temp=data_v[:,2:size(data_v)[2]]
        x_std=(temp.-repmat(mean(temp,1),n_val,1))./repmat(std(temp,1),n_val,1)

        if series_exp == 1
            z2 = x_std[:,2]
            z3 = x_std[:,3]
            z4 = x_std[:,4]
            x_aux1 = [z2 z3 z4] # linear terms
            x_aux2 = [z2.*z3 z3.*z4 z2.*z4]
            x_aux3 = [z2.*z2 z3.*z3 z4.*z4]
            x_aux = [x_aux1 x_aux2 x_aux3]
            x_v = [x_std[:,1] ones(n_val,1) x_aux]
        else
            x_v = [x_std[:,1] ones(n_val,1) x_std[:,2:4]]

        end

        y_hat= Int.((x_v*[1;bhat[:,i]]).>0)
        p_ratio[i]=mean(Int.(Y_val.==y_hat))
        println("out-of-sample performance: ", p_ratio[i])
    end

    println("Average coefficient vector: ",mean(bhat,2))
    println("Average score: ",mean(score))
    println("Average gap: ",mean(gap))
    println("Average running time: ",mean(rtime))
    println("Average node count: ",mean(ncount))
    println("Average in-sample score: ",mean(in_score))
    println("average out-of-sample performance: ",mean(p_ratio))
    return score,gap,rtime,ncount
end

export TransportationModeCV
end
