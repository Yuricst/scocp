# scocp
Sequential convexified optimal control problem (scocp) solver in python

![pytest workflow](https://github.com/Yuricst/scocp/actions/workflows/pytest.yml/badge.svg)

## Setup

1. `git clone` this repository

2. Setup virtual environment (requirements: `python 3.11`, `cvxpy`, `heyoka`, `numba`, `numpy`, `matplotlib`, `scipy`)

3. Run test from the root of the repository (requires `pytest`)

```
pytest tests
```


## Examples

**Impulsive control rendez-vous between libration point orbits**

<img src="tests/plots/scp_scipy_impulsive_transfer.png" width="70%">

**Continuous control rendez-vous between libration point orbits**

<img src="tests/plots/scp_scipy_continuous_transfer.png" width="70%">

**Continuous control rendez-vous between libration point orbits with mass dynamics**

<img src="tests/plots/scp_scipy_logmass_transfer.png" width="100%">