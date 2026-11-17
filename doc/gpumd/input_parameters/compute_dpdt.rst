.. _kw_compute_dpdt:
.. index::
   single: compute_dpdt (keyword in run.in)

:attr:`compute_dpdt`
====================

This keyword calculates the time derivative of the total dipole moment of the simulation cell by contracting each atomic Born effective charge (:term:`BEC`) tensor with the atomic velocity and summing over atoms.
The keyword is only meaningful for a simulation with the qNEP model trained with target :term:`BEC` values.

.. note::

   For qNEP models trained with :ref:`charge_transfer <kw_charge_transfer>`, the BEC is the exact gradient of the polarization-like quantity :math:`\mathbf{P}` of the model (see :ref:`qnep_charge_transfer_theory`) and the total charge is zero by construction.
   The BEC array of each atom has nine components and is NOT symmetric: component ``3*beta+alpha`` is :math:`\partial P_\beta/\partial r_\alpha` (up to the factor :math:`\sqrt{\epsilon_\infty}` of the model).

   The time derivative is :math:`\mathrm{d}P_\beta/\mathrm{d}t=\sum_{k,\alpha}Z_{k,\alpha\beta}v_{k,\alpha}`.
   The polarization :math:`\mathbf{P}` itself is not output; a direct :math:`\mathbf{P}` output is not implemented.

Syntax
------

This keyword has 1 parameter::

  compute_dpdt <sampling_interval>

Parameters
----------

Here,

* :attr:`sampling_interval` is a positive integer giving the number of time steps between samples.

Output
------

The results will be written to the file :ref:`dpdt.out <dpdt_out>`.

Examples
--------

Example 1
^^^^^^^^^

.. code::

   compute_dpdt 5

This means to calculate the time derivative of the total dipole moment every 5 time steps.

Notes
-----

None.
