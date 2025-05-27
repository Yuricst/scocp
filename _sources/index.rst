.. scocp documentation master file, created by
   sphinx-quickstart on Thu May  8 12:59:41 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

scocp: Sequential Convexified Optimal Control Problem solver in Python
============================================================================

`scocp` is a pythononic framework for solving general optimal control problems (OCPs) of the form:

.. math::
   \begin{align}
   \min_{u(t), t_f, y} \quad & \phi(x(t_f),u(t_f),t_f,y) + \int_{t_0}^{t_f} L(x(t),u(t),t) \mathrm{d}t
   \\ \mathrm{s.t.} \quad&     \dot{x}(t) = f(x(t),u(t),t)
   \\&     g(x(t),u(t),t,y) = 0
   \\&     h(x(t),u(t),t,y) \leq 0
   \\&     x(t_0) \in X(t_0) ,\,\, x(t_f) \in X(t_f)
   \\&     x(t) \in X(t),\,\, u(t) \in U(t)
   \end{align}



.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   tutorials
   examples_astrodynamics
   examples/ex_pl2pl.ipynb
   api


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`