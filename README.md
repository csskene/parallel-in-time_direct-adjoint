# A parallel-in-time approach for accelerating direct-adjoint studies (companion code)

This repository contains companion code for the article ["A parallel-in-time approach for accelerating direct-adjoint studies"](https://doi.org/10.1016/j.jcp.2020.110033) by C. S. Skene, M. F. Eggl and P. J. Schmid (JCP, 2020). The code is not the code used for the article but is a new version written for educational purposes. We recommend that the user independently validates the code for their own purposes.

The algorithms are based on ["PARAEXP: A Parallel Integrator for Linear Initial-Value Problems"](https://doi.org/10.1137/110856137) by M. J. Gander and S. Güttel (SIAM J. Sci. Comput., 2013) as well as ["A block Krylov subspace implementation of the time-parallel Paraexp method and its extension for nonlinear partial differential equations"](https://doi.org/10.1016/j.cam.2016.09.036) by G. L. Kooij, M. A. Botchev and B. J. Geurts (J. Comput. Appl. Math, 2017).

The code is written in python and utilises the following libraries

* numpy
* scipy
* mpi4py
* matplotlib

The algorithms are contained in the files _paraExpLin.py_, _paraExpNL.py_ and _paraExpHyb.py_. A simple exponential Euler timestepper is provided in _exponentialIntegrators.py_ but it is straightforward for the user to implement their own exponential solvers. Also provided are the following examples

## 1 - Linear algorithm
The linear algorithm from section 3.1 is demonstrated by _advectionDiffusion.py_. An optimisation is conducted to determine the true forcing that generates a specific solution for a 1D advection diffusion equation. It can be run as

``
mpiexec -n np python advectionDiffusion.py
``

## 2 - Non-linear algorithm (parallel)
The non-linear parallel algorithm from section 3.2 is demonstrated by _viscousBurgersNL.py_. An optimisation is conducted to determine the true forcing that generates a specific solution for a 1D viscous Burgers equation, with the non-linear direct equation being solved in parallel. It can be run as

``
mpiexec -n np python viscousBurgersNL.py
``

## 3 - Non-linear algorithm (hybrid)
The non-linear hybrid algorithm from section 3.3 is demonstrated by _viscousBurgersHyb.py_. An optimisation is conducted to determine the true forcing that generates a specific solution for a 1D viscous Burgers equation, with the non-linear direct equation being solved in series.  It can be run as

``
mpiexec -n np python viscousBurgersHyb.py
``

The authors gratefully acknowledge the EPSRC and Roth PhD scholarships on which this research was conducted.
