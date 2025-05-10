# Astrodynamics Problems

There are several problem classes within `scocp` that are ready-made for common astrodynamics problems.
These all require an integrator - in other words, by swapping the integrator, the same class can be used to design e.g. a rendez-vous trajectory in a restricted two-body dynamics, CR3BP, Clohessy-Wiltshire model, high-fidelity ephemeris model, etc.

Below is a list of implemented classes:


## Continuous control problems

| Class |  Dynamics | Final time | Initial conditions | Final conditions |
|-------|-----------|------------|--------------------|------------------|
| `FixedTimeContinuousRdv`                   | Translational dynamics            | Fixed | Fixed | Fixed |
| `FixedTimeContinuousRdvLogMass`            | Translational dynamics + log-mass | Fixed | Fixed | Fixed |
| `FreeTimeContinuousRdv`                    | Translational dynamics            | Free  | Fixed | Fixed |
| `FreeTimeContinuousRdvLogMass`             | Translational dynamics + log-mass | Free  | Fixed | Fixed |
| `FreeTimeContinuousMovingTargetRdvLogMass` | Translational dynamics + log-mass | Free  | Fixed | Moving target |
