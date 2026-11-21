.. _kw_zbl:
.. index::
   single: zbl (keyword in nep.in)

:attr:`zbl`
===========

This keyword can be used to spline the pair potential to the universal :term:`ZBL` potential [Ziegler1985]_ at short distances.
This is useful, for example, for models to be used in simulations of ion irradiation or extreme compression, when the interatomic distances can become very short.

Syntax
------

::

  zbl <cutoff>

Parameters
----------

Here, :attr:`<cutoff>` is a real number that specifies the "outer" cutoff :math:`r_\mathrm{c}^\mathrm{ZBL-outer}`, beyond which the :term:`ZBL` potential is zero.
The "inner" cutoff of the :term:`ZBL` potential, below which value the pair interaction is completely given by the :term:`ZBL` potential, is fixed to half of the outer cutoff, :math:`r_\mathrm{c}^\mathrm{ZBL-inner} = r_\mathrm{c}^\mathrm{ZBL-outer} /2`, which we have empirically found to be a reasonable choice.

When this keyword is absent, the :term:`ZBL` potential will not be enabled and the value of :math:`r_\mathrm{c}^\mathrm{ZBL-outer}` is irrelevant.

The ``nep_dipole`` and ``nep_polarizability`` models reject this keyword and
:ref:`use_typewise_cutoff_zbl <kw_use_typewise_cutoff_zbl>`.

NEP-spin supports universal ZBL but not flexible ZBL. With ZBL enabled,
remove ``zbl.in`` for :ref:`spin_version 1 <kw_spin_version>` training.
ZBL contributes to energy, force, and virial, but not magnetic force or torque.

Permissible values are 1 Å :math:`\leq r_\mathrm{c}^\mathrm{ZBL-outer} \leq` 4 Å.

Flexible ZBL parameters are supported by ordinary NEP (``model nep``, including its default selection)
and by the charge models ``qnep1``, ``qnep2``, and ``qnep3``, with or without :ref:`charge_transfer <kw_charge_transfer>`.
They can be used in ``nep``, in ``gpumd``, and with :ref:`multi_replica <kw_multi_replica>`.
To use them, enable this keyword
and provide a ``zbl.in`` file in the working directory; the :attr:`<cutoff>`
parameter is still needed but will not be used. The models ``nep_dipole``, ``nep_polarizability``,
NEP-spin, ``nep_vdw``, ``qnep3_vdw`` (``nep_charge_vdw``), and ``tnep`` reject flexible
ZBL. Flexible ZBL cannot be combined with :ref:`use_typewise_cutoff_zbl <kw_use_typewise_cutoff_zbl>`.
For the charge models, ``gpumd`` supports at most 10 elements that are present in the simulated
structure (550 flexible ZBL values) and stops with an error otherwise.
Without this keyword, a ``zbl.in`` file is ignored. In prediction mode,
ZBL parameters are read from ``nep.txt`` rather than ``zbl.in``.
Old-format ``nep.txt`` files ``nep4_zbl_charge1/2/3`` with ``zbl 0 0`` (also together with ``charge_transfer``)
can be converted with ``potentials/nep/nep_model_converter.py``; restart files of models with ``charge_transfer`` are not converted.
For a :math:`n`-species system, there should be :math:`n(n+1)/2` lines in the `zbl.in` files.
Each line represents a unique pair of species.
Counting from 1, the order of the lines is 1-1, 1-2, ..., 1-:math:`n`, 2-2, 2-3, ..., 2-:math:`n`, :math:`n`-:math:`n`.
For each pair of species, there are 10 parameters to be listed from left to right: the first two are the inner and outer cutoff radii, respectively; the next 8 are the parameters :math:`a_1` to :math:`a_8` in the ZBL function :math:`\phi(x)=a_1e^{-a_2x}+a_3e^{-a_4x}+a_5e^{-a_6x}+a_7e^{-a_8x}`.

The ZBL term uses the angular neighbor list, so a pair whose outer cutoff exceeds :math:`r_\mathrm{c}^\mathrm{A}` (the angular cutoff) is truncated at the angular cutoff, which causes an energy discontinuity; keep all outer cutoffs at most the angular cutoff (the same holds for ordinary NEP).
For the charge models ``gpumd`` prints a warning when a flexible-ZBL outer cutoff exceeds the angular cutoff.

.. note::

  In upstream GPUMD, flexible ZBL is accepted only for ``model nep``.
  Its use with the charge models is specific to this fork.

Output
------

None.

Examples
--------

::

  zbl 2.5

Notes
-----

None.
