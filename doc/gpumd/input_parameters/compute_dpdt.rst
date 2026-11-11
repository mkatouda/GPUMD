.. _kw_compute_dpdt:
.. index::
   single: compute_dpdt (keyword in run.in)

:attr:`compute_dpdt`
====================

This keyword can be used to calculate the time derivative of the polarization via the dot product between the Born effective charge (:term:`BEC`) and the velocity. 
The keyword is only meaningful for a simulation with the qNEP model trained with target :term:`BEC` values.
The results will be written to the file :ref:`dpdt.out <dpdt_out>`.

.. note::

   For qNEP models trained with :ref:`charge_transfer <kw_charge_transfer>`, the BEC is the exact gradient of the polarization-like quantity :math:`\mathbf{P}` of the model (see :ref:`qnep_charge_transfer_theory`) and the total charge is zero by construction.
   The BEC array of each atom has nine components and is NOT symmetric: component ``3*beta+alpha`` is :math:`\partial P_\beta/\partial r_\alpha` (up to the factor :math:`\sqrt{\epsilon_\infty}` of the model).

   The time derivative is :math:`\mathrm{d}P_\beta/\mathrm{d}t=\sum_{k,\alpha}Z_{k,\alpha\beta}v_{k,\alpha}`.
   The polarization :math:`\mathbf{P}` itself is not output; a direct :math:`\mathbf{P}` output is not implemented.

Syntax
------
This keyword has 1 parameter::

  compute_dpdt <sampling_interval>

Here,

* :attr:`sampling_interval` is the sampling interval for calculating the time derivative of the polarization.

Examples
--------

Example 1
^^^^^^^^^

.. code::

   compute_dpdt 5

This means to calculate the time derivative of the polarization every 5 time steps.