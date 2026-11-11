.. _qnep_charge_transfer_theory:
.. index::
   single: charge-transfer qNEP
   single: Born effective charge; charge-transfer qNEP

Charge-transfer qNEP
********************

This section describes the charge definition selected by the :ref:`charge_transfer <kw_charge_transfer>` keyword of ``nep.in`` (``qnep2`` and ``qnep3`` only).

Charges
=======

Let :math:`h_a` be the output of the charge node of the per-atom network of atom :math:`a`.
The charge is the graph-Laplacian transfer

.. math::

   q_a = \sum_{b} \phi_{ab}(r_{ab}) \, (h_a - h_b),

where the sum runs over the atoms :math:`b` within the cutoff :math:`r_\mathrm{c}^\phi` (minimum image convention), and :math:`\phi_{ab}=\phi_{ba}` is a symmetric function of the distance with one set of radial-basis coefficients per type pair.
Because the transfer from :math:`b` to :math:`a` is the negative of the transfer from :math:`a` to :math:`b`,

.. math::

   \sum_a q_a = 0

holds exactly for every configuration, without subtracting a mean charge.

Born effective charges
======================

The polarization-like quantity

.. math::

   \mathbf{P} = \sum_a q_a \mathbf{r}_a = \sum_{a<b} \phi_{ab}(r_{ab}) \, (h_a - h_b) \, \mathbf{r}_{ab}

depends on the pair vectors :math:`\mathbf{r}_{ab}` only, and is therefore invariant under a rigid translation of the system.
The Born effective charge reported by ``gpumd`` and ``nep`` is its exact gradient (up to the factor :math:`\sqrt{\epsilon_\infty}` of the qNEP model, which is part of the reported BEC and of :math:`\mathbf{P}` below),

.. math::

   Z_{k,\alpha\beta} = \frac{\partial P_\beta}{\partial r_{k,\alpha}}\quad(\text{array index } 3\beta+\alpha),

including the dependence of :math:`h_a` on the atomic positions through the descriptors.
The tensor :math:`Z_k` is not symmetric.

Forces and virial
=================

The forces and the virial are computed from the same energy as for other qNEP models, with the derivative of the charge definition included: a weight on the descriptor derivatives obtained from the transposed transfer operator, plus an explicit pair term containing :math:`\phi'(r_{ab})(h_a-h_b)`.
They are consistent with the energy; this has been checked against finite differences of the energy of the model itself.

Guarantees and limits
=====================

* Guaranteed: exact charge neutrality (up to floating-point rounding), translation invariance of :math:`\mathbf{P}`, the BEC as the gradient of :math:`\mathbf{P}`, and the consistency of forces and virial with the energy.
* Not guaranteed: any physical accuracy of a trained model. In particular, the charges are not trained against target charges, and the expressiveness of the model for real data is not established here.
* For periodic cells, the quantity :math:`\sum_a q_a \mathbf{r}_a` with wrapped positions is multivalued (it depends on the choice of periodic images).
  The pair form of :math:`\mathbf{P}` above is single valued, but ``gpumd`` and ``nep`` do not output :math:`\mathbf{P}` itself (a direct polarization output is not implemented); only the BEC and, through :ref:`compute_dpdt <kw_compute_dpdt>`, :math:`\mathrm{d}\mathbf{P}/\mathrm{d}t` are available.
* Because :math:`\mathbf{P}` is a function of the positions, the work done by a constant electric field applied with :ref:`add_efield <kw_add_efield>` in ``bec`` mode is consistent with the energy :math:`-\mathbf{E}\cdot\mathbf{P}`.
  The thermodynamic energy (``thermo.out``) does not contain :math:`-\mathbf{E}\cdot\mathbf{P}`, so :math:`K+U` alone is not conserved in a field; :math:`K+U-\mathbf{E}\cdot\mathbf{P}` is.
