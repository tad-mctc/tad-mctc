.. automodule:: tad_mctc.storch.linsolve

Krylov methods
--------------

.. automodule:: tad_mctc.storch.linsolve.krylov
   :members:

Implicit linear solve
---------------------

.. automodule:: tad_mctc.storch.linsolve.implicit
   :members:

Equality constraints
--------------------

.. automodule:: tad_mctc.storch.linsolve.constrained
   :members:

Preconditioners
---------------

.. automodule:: tad_mctc.storch.linsolve.preconditioners
   :members:

Dense solve
-----------

.. automodule:: tad_mctc.storch.linsolve.dense
   :members:

Results and status
------------------

.. automodule:: tad_mctc.storch.linsolve.result
   :members:

Application convergence masks
----------------------------

Applications with their own physical residual checks can use the public
mask below without constructing a linear-solver result. The caller chooses
when to raise eagerly; the mask itself replaces failed values and their
active derivatives with NaNs.

Select accepted systems before applying nonlinear downstream operations.
Even a first derivative of ``masked.square()[0]`` can become NaN if another
entry failed: its square's backward can produce ``0 * NaN`` before the mask
receives the cotangent. Use ``masked[0].square()`` for an accepted entry, or
compute finite intermediate values and mask only the final output.

.. autofunction:: tad_mctc.storch.linsolve.mask_converged
