# `scocp`: **Sequential Convexified Optimal Control Problem solver in Python**

[![PyPI version](https://badge.fury.io/py/scocp.svg)](https://badge.fury.io/py/scocp)
![test-scocp workflow](https://github.com/Yuricst/scocp/actions/workflows/test_scocp.yml/badge.svg)
[![documentation workflow](https://github.com/Yuricst/scocp/actions/workflows/documentation.yml/badge.svg)](https://yuricst.github.io/scocp/)

`scocp` is a pythononic framework for solving general optimal control problems (OCPs) of the form:

```math
\begin{align}
\min_{u(t), t_f, y} \quad& \phi(x(t_f),u(t_f),t_f,y) + \int_{t_0}^{t_f} \mathcal{L}(x(t),u(t),t) \mathrm{d}t
\\ \mathrm{s.t.} \quad&     \dot{x}(t) = f(x(t),u(t),t)
\\&     g(x(t),u(t),t,y) = 0
\\&     h(x(t),u(t),t,y) \leq 0
\\&     x(t_0) \in \mathcal{X}(t_0) ,\,\, x(t_f) \in \mathcal{X}(t_f)
\\&     x(t) \in \mathcal{X}(t),\,\, u(t) \in \mathcal{U}(t)
\end{align}
```
with either fixed or free $t_f$ via sequential convex programming (SCP).

Installing is as easy as 

```
pip install scocp
```

and to uninstall

```
pip uninstall scocp
```

Read the full documentation [here](https://yuricst.github.io/scocp/)!


### Overview

with either fixed or free $t_f$ via sequential convex programming (SCP).
The SCP is solved with the `SCvx*` algorithm, an augmented Lagrangian framework to handle non-convex constraints [1].

The dynamics in the OCP are handled by defining an integrator class, which requires a `solve(tspan, x0, u=None,stm=False)` method.
`scocp` provides wrappers to be used with either `scipy`'s [`solve_ivp()` method](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html) or [`heyoka`](https://bluescarni.github.io/heyoka.py/index.html), but a user-defined integrator class can be used instead as well.
A custom integrator class should look like:

```python
class MyIntegrator:
    def __init__(self, nx, nu, impulsive: bool, nv, *args):
        self.nx = nx                  # dimension of states
        self.nu = nu                  # dimension of controls
        self.impulsive = impulsive    # whether to consider impulsive or continuous control
        self.nv = nv        # dimension of control magnitudes to be augmented in the linearized map
        # (whatever other stuff you want to do with the integrator)

    def solve(self, tspan, x0, u=None, stm=False):
        """Solve IVP
        If `u` is provided, solve IVP with control
        If `stm = True`, also propagate sensitivities
        """
        # (solve initial value problem)
        return t_eval, states
```

To solve an OCP, the user needs to define a problem class, for example:

```python
class MyOptimalControlProblem(scocp.ContinuousControlSCOCP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs):
        return

    def evaluate_objective(self, xs, us, gs, ys):
        """Evaluate the objective function"""
        # (compute objective)
        return objective
    
    def solve_convex_problem(self, xbar, ubar, vbar, ybar):
        N,nx = xbar.shape
        xs = cp.Variable((N, nx), name='state')
        us = cp.Variable((N, nu), name='control')
        gs = cp.Variable((N, 1),  name='Gamma')
        ts = cp.Variable((N, 1),  name='ys')
        xis_dyn = cp.Variable((N-1,nx), name='xi_dyn')   # slacks for dynamics constraints
        xis     = cp.Variable(self.ng, name='xi')        # slacks for non-dynamics equality constraints
        zetas   = cp.Variable(self.nh, name='xi')        # slacks for inequality constraints
        # (formulate & solve OCP)
        return xs.value, us.value, gs.value, ys.value, xis.value, xis.value, zetas.value

    def evaluate_nonlinear_constraints(self, xs, us, gs, ys=None):
        g_eval = ...        # array of nonlinear equality constraints evaluated along `xs`, `us`, `gs`
        h_eval = ...        # array of nonlinear inequality constraints evaluated along `xs`, `us`, `gs`
        return g_eval, h_eval
```

In addition, we provide problem classes that can be readily used for typical OCPs in astrodynamics: 

- Fixed final time continuous rendezvous's: `FixedTimeContinuousRdv`, `FixedTimeContinuousRdvLogMass`
- Fixed final time impulsive rendezvous's: `FixedTimeImpulsiveRdv`
- Free final time continuous rendezvous's: `FreeTimeContinuousRdv`, `FreeTimeContinuousRdvLogMass`


### References

[1] K. Oguri, “Successive Convexification with Feasibility Guarantee via Augmented Lagrangian for Non-Convex Optimal Control Problems,” in 2023 62nd IEEE Conference on Decision and Control (CDC), IEEE, Dec. 2023, pp. 3296–3302. doi: 10.1109/CDC49753.2023.10383462.

## Setup

1. `git clone` this repository

2. Setup virtual environment (requirements: `python 3.11`, `cvxpy`, `heyoka`, `numba`, `numpy`, `matplotlib`, `scipy`)

3. Run test from the root of the repository (requires `pytest`)


> [!NOTE]  
> Additional methods compatible with the [`pykep`](https://esa.github.io/pykep/index.html) infrastructure are provided in `scocp_pykep`.

```
pytest tests
```

Or, to also get coverage report

```
coverage run -m pytest  -v -s tests
coverage report -m
```


## Examples

See example notebooks in `./examples`.

### Continuous Control

#### `FixedTimeContinuousRdv`:Fixed TOF Continuous control rendez-vous

- State: Cartesian position, velocity
- Controls: acceleration
- Fixed TOF
- Fixed boundary conditions

<img src="tests/plots/scp_scipy_continuous_transfer.png" width="70%">


#### `FixedTimeContinuousRdvLogMass`: Fixed TOF Continuous control rendez-vous with mass dynamics

- State: Cartesian position, velocity + log(mass)
- Controls: acceleration
- Fixed TOF
- Fixed boundary conditions

<img src="tests/plots/scp_scipy_logmass_transfer.png" width="100%">


#### `FreeTimeContinuousRdvLogMass`: Free TOF Continuous control rendez-vous with mass dynamics

- State: Cartesian position, velocity + log(mass) + dilated time
- Controls: acceleration + time dilation factor
- Free TOF
- Fixed boundary conditions

<img src="tests/plots/scp_scipy_logmass_freetf_transfer.png" width="100%">


#### `FreeTimeContinuousMovingTargetRdvLogMass`: Free TOF Continuous rendez-vous with mass dynamics & moving target

- State: Cartesian position, velocity + log(mass) + dilated time
- Controls: acceleration + time dilation factor
- Free TOF
- Fixed initial conditions, moving terminal conditions

<img src="tests/plots/twobody_logmass_freetf_rdv.png" width="100%">


#### `FreeTimeContinuousMovingTargetRdvMass`: Free TOF Continuous rendez-vous with mass dynamics & moving target

- State: Cartesian position, velocity + mass + dilated time
- Controls: acceleration + time dilation factor
- Free TOF
- Fixed initial conditions, moving terminal conditions

<img src="tests/plots/twobody_mass_freetf_rdv.png" width="100%">


### Impulsive Control

#### `FixedTimeImpulsiveRdv`:Fixed TOF impulsive control rendez-vous

- State: Cartesian position, velocity
- Controls: impulsive delta-V's
- Fixed TOF
- Fixed boundary conditions

<img src="tests/plots/scp_scipy_impulsive_transfer.png" width="70%">
