.. _kw_charge_transfer:
.. index::
   single: charge_transfer (keyword in nep.in)

:attr:`charge_transfer`
=======================

This keyword switches a qNEP model to a charge-transfer (charge-conserving) charge definition::

  charge_transfer <rc_phi>

Here, :attr:`rc_phi` is the cutoff radius (in units of Å) of the pair functions :math:`\phi`.
It is required, there is no default, and it must satisfy :math:`0 < r_\mathrm{c}^\phi \leq` the radial cutoff (the first :ref:`cutoff <kw_cutoff>` value) and :math:`r_\mathrm{c}^\phi \leq 100` Å.
The keyword can be given only once.

Meaning
-------

The per-atom network output :math:`h_a` is not used as the charge directly.
Instead, the charge is formed as :math:`q_a = \sum_b \phi_{ab}(r_{ab}) (h_a - h_b)`, where the sum runs over the neighbors within :math:`r_\mathrm{c}^\phi`.
The pair functions :math:`\phi_{ab}` are expanded in the radial basis of the model, with one set of coefficients per (unordered) pair of atom types.
The total charge of every structure is therefore zero by construction, and the :term:`BEC` is the exact gradient of the polarization-like quantity described in :ref:`the theory section <qnep_charge_transfer_theory>`.
See :ref:`nep_txt` for the file format.

Requirements
------------

* The model must be ``qnep2`` or ``qnep3`` (equivalently, the legacy ``charge_mode 2`` or ``3``).
* The number of added trainable parameters is ``N_pair * (basis_size_radial + 1)``, where ``N_pair = N_type * (N_type + 1) / 2`` and ``basis_size_radial`` is the first value of :ref:`basis_size <kw_basis_size>`.
  They are appended at the end of the parameter vector, are initialized uniformly in :math:`[0, 0.1]`, and are regularized like the other parameters (:ref:`lambda_1 <kw_lambda_1>`, :ref:`lambda_2 <kw_lambda_2>`).
* The meaning of the network output changes from :math:`q` to :math:`h`.
  A network trained without this keyword cannot be reused, and the model has to be trained from scratch.
  Models and inputs without the keyword are not affected.
* Target charges are not needed. Because the total charge is always zero, a structure with a nonzero ``charge=`` label in the training data is an error.

Errors
------

The following cases stop ``nep`` with an error message:

* ``charge_transfer`` with ``qnep1`` (``charge_mode 1``), plain ``nep``, ``qnep3_vdw`` (``charge_vdw``), ``nep_vdw``, spin models, or the ``model_type`` models.
* ``charge_transfer`` with :ref:`fine_tune <kw_fine_tune>`, and in the ``USE_CJ`` build.
* A missing, non-numeric, non-positive, or too large :attr:`rc_phi`, or :attr:`rc_phi` larger than the radial cutoff.
* A repeated ``charge_transfer`` keyword.
* A nonzero ``charge=`` label in ``train.xyz`` or ``test.xyz``.
* A mismatch between the ``charge_transfer`` setting of ``nep.in`` and the model read from an existing ``nep.txt`` (for example in prediction mode).

Example
-------

.. code::

  type 2 Na Cl
  model qnep2
  cutoff 6 5
  charge_transfer 6

Notes
-----

* For this model type, ``gpumd`` reads the resulting ``nep.txt`` with ``potential nep.txt`` as usual.
  ``gpumd_replica`` does not support these models.
* No claim is made here about the physical accuracy of the trained model; the construction guarantees charge neutrality and the consistency between the BEC and the energy-related quantities, not the quality of the fit.
