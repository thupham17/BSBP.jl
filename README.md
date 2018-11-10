# Best-Subset-Binary-Prediction

Julia package implementation of the best subset maximum score binary prediction method proposed by Chen and Lee (2017). Description of this prediction method and its computation details can be found in the paper:

Chen, Le-Yu and Lee, Sokbae (November 2017), ["Best Subset Binary Prediction"](https://arxiv.org/pdf/1610.02738.pdf).

## Installation
1. Install via Pkg:
```
Pkg.add("BSBP")
```

2. Import the package to call the main functions:

## Main functions
- MaxScore:
  Used to compute the the best subset maximum score prediction rule via the mixed integer optimization (MIO) approach.
- WarmStartMaxScore:
  Implements warm-start strategy by refining the input parameter space to improve the MIO computational performance.
- WarmStartMaxScoreCV:
  Implements cross validation best subset binary prediction and computes the optimal q value.

### Support functions
- MIObnd: Solves the maximization problem max |beta0*x(i,1)+x(i,:)\*t| over t subject to given bounds.
- logit: logit regression.
- getBnd: gets bound for warm start approach.


Using main functions:
```
using BSBP
MaxScore()
```

Using support functions:
```
using BSBP
MIObnd()
```

Use ? to see parameters and outputs (ie. `?MaxScore`).

## Examples
Included in the package are 2 examples from the original paper.

###Simulation
Implements best subset approach to simulated data.
```
using BSBP
simulation(args)
```
Arguments are optional. Use option `?simulation` to list arguments.

###Horowitz
Implements best subset approach to Horowitz (1993) data.
```
using BSBP
transportationMode(args)
```
Arguments are optional. Use option `?transportationMode` to list arguments.

Implement best subset approach with cross validation to Horowitz (1993) data.
```
using BSBP
transportationModeCV(args)
```
Arguments are optional. Use option `?transportationModeCV` to list arguments.

## Requirements
Requires Julia 0.6.4 and the Gurobi solver (available free for academic purposes).
