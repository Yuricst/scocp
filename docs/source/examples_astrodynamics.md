# Astrodynamics Problems

There are several problem classes within `scocp` that are ready-made for common astrodynamics problems.
These all require an integrator - in other words, by swapping the integrator, the same class can be used to design e.g. a rendez-vous trajectory in a restricted two-body dynamics, CR3BP, Clohessy-Wiltshire model, high-fidelity ephemeris model, etc.

Below is a list of implemented classes:


## Continuous control problems

| Class |  Dynamics | Final time | Initial conditions | Final conditions |
|-------|-----------|------------|--------------------|------------------|
| [`FixedTimeContinuousRdv`](scocp.FixedTimeContinuousRdv)                   | Translational dynamics            | Fixed | Fixed | Fixed |
| [`FixedTimeContinuousRdvLogMass`](scocp.FixedTimeContinuousRdvLogMass)            | Translational dynamics + log-mass | Fixed | Fixed | Fixed |
| [`FreeTimeContinuousRdv`](scocp.FreeTimeContinuousRdv)                    | Translational dynamics            | Free  | Fixed | Fixed |
| [`FreeTimeContinuousRdvLogMass`](scocp.FreeTimeContinuousRdvLogMass)             | Translational dynamics + log-mass | Free  | Fixed | Fixed |
| [`FreeTimeContinuousMovingTargetRdvLogMass`](scocp.FreeTimeContinuousMovingTargetRdvLogMass) | Translational dynamics + log-mass | Free  | Fixed | Moving target |
| [`FreeTimeContinuousMovingTargetRdvMass`](scocp.FreeTimeContinuousMovingTargetRdvMass)    | Translational dynamics + mass     | Free  | Fixed | Moving target |

## Log-mass vs. mass dynamics?

As the name suggests, there are classes that include a log-mass dynamics and mass dynamics, i.e. the state includes either {math}```z = \log{m}``` or {math}```m```.
This makes a difference in the way control comes into the dynamics. With {math}```z```, the control {math}```\boldsymbol{u}``` is the acceleration, and we get

```{math}
\dot{\boldsymbol{x}} =
\begin{bmatrix}
    \dot{\boldsymbol{r}} \\ \dot{\boldsymbol{v}} \\ \dot{z}
\end{bmatrix}
 = 
\begin{bmatrix}
    \boldsymbol{f}_{\mathrm{natural}} (\boldsymbol{x}(t), t) \\ 0
\end{bmatrix}
+ \begin{bmatrix}
    \boldsymbol{0}_{3\times3} & \boldsymbol{0}_{3\times1} \\
    \boldsymbol{I}_3 & \boldsymbol{0}_{3\times1} \\
    \boldsymbol{0}_{1\times3} & -1/c_{\mathrm{ex}}
\end{bmatrix}
\begin{bmatrix}
    \boldsymbol{u} \\ \| \boldsymbol{u} \|_2
\end{bmatrix}
```

with the additional nonconvex constraint

```{math}
0 \leq \| \boldsymbol{u} \|_2 \leq T_{\max} e^{-z}
```

In contrast, with {math}```m```, the control {math}```\boldsymbol{u}``` is the thrust throttle, and we get


```{math}
\dot{\boldsymbol{x}} =
\begin{bmatrix}
    \dot{\boldsymbol{r}} \\ \dot{\boldsymbol{v}} \\ \dot{m}
\end{bmatrix}
= 
\begin{bmatrix}
    \boldsymbol{f}_{\mathrm{natural}} (\boldsymbol{x}(t), t) \\ 0
\end{bmatrix}
+ \begin{bmatrix}
    \boldsymbol{0}_{3\times3} & \boldsymbol{0}_{3\times1} \\
    \dfrac{T_{\max}}{m(t)} \boldsymbol{I}_3 & \boldsymbol{0}_{3\times1} \\
    \boldsymbol{0}_{1\times3} & -\dfrac{T_{\max}}{I_{\mathrm{sp}}g_0}
\end{bmatrix}
\begin{bmatrix}
    \boldsymbol{u} \\ \| \boldsymbol{u} \|_2
\end{bmatrix}
```

with the additional second-order cone constraint

```{math}
 \| \boldsymbol{u} \|_2 \leq 1 
```

Here are some pros and cons to either model:

| Property | Model with {math}```\log{m}``` | Model with {math}```z``` |
| -------- | ------------------------------ | ------------------------ |
| Control type | Acceleration               | Throttle                 |
| Affinity | Control-affine                 | General nonconvex        |
| Additional constraint | Nonconvex         | Second-order cone        |