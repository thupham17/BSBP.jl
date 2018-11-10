using MathProgBase, Gurobi
using Distributions, DataFrames

"""
Computes the optimal q value via the cross validation procedure.
Args:
    data (array):       vector of binary outcomes.
    tr_ind (array):     Indices for training sample.
    test_ind (array):   Indices for validation sample.
    focus_ind (array):  Indices for focus covariates.
    aux_ind (array):    Indices for auxiliary covariates.
    beta0 (int):        Coefficient taking value either 1 or -1 to normalize the scale for the first
                        first covariate in x_foc.
    q (int):            Cardinality constraint for the covariate selection.
    T (int):            Time limit specified for the MIO solver.
    bnd (array):        (((k-1)+d) by 2) matrix where the first and second columns respectively
                        store the lower and upper bounds of the unknown coefficients. First (k-1)
                        rows correspond to the bounds of the focused covariates excluding the first one.
                        Remaining d rows correspond to the bounds of the auxiliary covariates.
    mio (int) :         MIO formulation of the best subset maximum score problem
                        set mio = 1 for Method 1 of the paper
                        set mio = 2 for Method 2 of the paper
    tau (int):          Tuning parameter for enlarging the estimated bounds.
Yields:
    best_q:             Best q value.
    bhat (array):       Maximum score estimates for the unknown coefficients.
    score (array):      Value of maximum score objective function.
    gap   (array):      MIO optimization gap value in case of early termination
                        gap = 0 ==> optimal solution is found within the time limit.
    rtime (array):      Time used by the MIO solver in the estimation procedure.
    ncount (array):     Node count.
"""
function CVMaxScore(tr_ind::AbstractArray,test_ind::AbstractArray,data::AbstractArray,focus_ind::AbstractArray,aux_ind::AbstractArray,beta0::Integer,q_range::AbstractMatrix,T::Real,tol::Real,bnd::AbstractArray,mio::Integer)
    q_num = len(q_range)
    fold=size(tr_ind)[1]
    score=zeros(fold,q_num)
    gap=zeros(fold,q_num)
    rtime=zeros(fold,q_num)
    ncount=zeros(fold,q_num)
    bhat=zeros(len(focus_ind)+len(aux_ind)-1,fold,q_num)
    val_score=zeros(q_num,1)

    for q=1:q_num
        for i=1:fold
            println("(q,fold) : ",q_range[q]," ",i+1)
            y=data[tr_ind[:,i],1]
            datax=data[tr_ind[:,i],2:size(data)[2]]

            bhat[:,i,q],score[i,q],gap[i,q],rtime[i,q],ncount[i,q]  = MaxScore(y,datax[:,focus_ind],datax[:,aux_ind],beta0,q_range[q],T,tol[q],bnd,mio)

            y_v=data[test_ind[:,i],1]
            datax_v=data[test_ind[:,i],2:size(data)[2]]
            val_score[q] = val_score[q] + mean(Int.(y_v .== (datax_v*[beta0;bhat[:,i,q]].>=0)))
        end
        val_score[q]=val_score[q]/fold
    end
    best_q = maximum(val_score)
    return best_q, bhat,score,gap,rtime,ncount
end

export CVMaxScore
