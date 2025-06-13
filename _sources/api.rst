API
===


SCP algorithms and utility functions
--------------------------------------

.. autoclass:: scocp.SCvxStar
   :members:

.. autofunction:: scocp.get_augmented_lagrangian_penalty

.. autoclass:: scocp.MovingTarget
   :members:

Continuous control problems
-----------------------------

Base class for continuous control problems
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: scocp.ContinuousControlSCOCP
   :members:

Example continuous control problems
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: scocp.FixedTimeContinuousRdv
   :members:

.. autoclass:: scocp.FixedTimeContinuousRdvLogMass
   :members:

.. autoclass:: scocp.FreeTimeContinuousRdv
   :members:

.. autoclass:: scocp.FreeTimeContinuousRdvLogMass
   :members:

.. autoclass:: scocp.FreeTimeContinuousMovingTargetRdvLogMass
   :members:

.. autoclass:: scocp.FreeTimeContinuousMovingTargetRdvMass
   :members:


Impulsive control problems
-----------------------------

Base class for impulsive control problems
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: scocp.ImpulsiveControlSCOCP
   :members:

Example impulsive control problems
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: scocp.FixedTimeImpulsiveRdv
   :members:


Integrators 
-------------

.. autoclass:: scocp.ScipyIntegrator
   :members:
