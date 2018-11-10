"""
Reproduction of the empirical results concerning an empirical application
in the prediction of transportation mode choice using the best subset
maximum score approach of Chen and Lee (2017).
The work-trip mode choice dataset of Horowitz (1993) is included in the package.
"""
module transportationMode

using DataFrames, Compat
using MathProgBase, Gurobi
using ..BSBP

""" Best subset maximum score approach on the work-trip mode choice dataset of Horowitz.
Args:
    warm_start (int,optional):  Set to 1 for warm start strategy. Default = 1.
    tau (float,optional):       Tuning paramater to construct the refined bound used in the warm start approach. Default = 1.5.
    mio (int,optional):         Set to 1 for using Method 1 for the MIO formulation. Set to 2 for Method 2. Default to 1.
    q (int, optional):          Value of the variable selection bound. Default = 1.
    series_exp (int, optional): Set to 1 to use quadratic expansion terms as covariates. Default = 0.
    b (int, optional):          Bound value. Default = 1.
"""
function TransportationMode(;warm_start=0,tau=1.5,mio=2,series_exp=0,beta0=1,b=10,q=1,time_limit=86400)
  # load data
  data = readdlm("./data_horowitz.csv",',')

  bhat = []
  score = 0
  gap = 0
  rtime = 0
  ncount = 0

  println("estimation based on full sample")
  Y_tr=data[:,1]
  temp=data[:,2:end]


  n_tr=length(Y_tr)
  # [DCOST CARS DOVTT DIVTT]
  x_std=(temp.-repmat(mean(temp,1),n_tr,1))./repmat(std(temp,1),n_tr,1)
  x_foc = [x_std[:,1] ones(n_tr,1)] # [DCOST Intercept]

  if series_exp == 1
      z2 = x_std[:,2]
      z3 = x_std[:,3]
      z4 = x_std[:,4]
      x_aux1 = [z2, z3, z4] # linear terms
      x_aux2 = [z2.*z3, z3.*z4, z2.*z4]
      x_aux3 = [z2.*z2,z3.*z3, z4.*z4]
      x_aux = [x_aux1, x_aux2, x_aux3]
  else
      x_aux = x_std[:,2:4] # [CARS DOVTT DIVTT]
  end

  k=size(x_foc)[2]
  d=size(x_aux)[2]

  bnd= hcat(-b*ones(k-1+d,1),b*ones(k-1+d,1)) # set the initial parameter bounds

  tol = floor(sqrt(log(n_tr)*n_tr)/2) # set the tolerance level value
  println("tolerance level: ", tol/n_tr)

  if warm_start == 1 # warm start MIO
      bhat,score,gap,rtime,ncount  = WarmStartMaxScore(Y_tr,x_foc,x_aux,beta0,q,time_limit,tol,bnd,mio,tau)
  else # cold start MIO
      bhat,score,gap,rtime,ncount  = MaxScore(Y_tr,x_foc,x_aux,beta0,q,time_limit,tol,bnd,mio)
  end

  println("parameter estimates: ", bhat)
  println("avg_score gap time node_count:", score, " , ", gap, " , ", rtime, " , ", ncount)
  return score,gap,rtime,ncount
end

export TransportationMode
end
