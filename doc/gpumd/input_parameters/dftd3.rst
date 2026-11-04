.. _kw_dftd3:
.. index::
   single: dftd3 (keyword in run.in)

:attr:`dftd3`
=============

This keyword is used to add the DFT-D3 dispersion correction to NEP.
It has no effect when any other potential is used.
For theoretical background on DFT-D3, see [Grimme2010]_ and [Grimme2011]_.

Syntax
------

  dftd3 <functional> <potential_cutoff> <coordination_number_cutoff>

where :attr:`functional` is the exchange-correlation functional used in generating the reference data for training the NEP model, :attr:`potential_cutoff` is the cutoff radius (in units of Angstrom) for the D3 potential, and :attr:`coordination_number_cutoff` is the cutoff radius (in units of Angstrom) for calculating the coordination numbers. 

This keyword can be put anywhere in the ``run.in`` input file.

We have only implemented the Becke-Johnson (BJ) damping, and a full list of supported functionals can be found from the following link: https://www.chemiebn.uni-bonn.de/pctc/mulliken-center/software/dft-d3/functionalsbj

Example
-------

.. code::

  dftd3 pbe 12 6
  
This will add the DFT-D3 dispersion correction to NEP with the PBE functional and a cutoff radius of 12 Angstrom for the D3 potential and a cutoff radius of 6 Angstrom for the coordination number.

.. _dftd3_cost:

Cost
----

The cost of the D3 correction is the cost of visiting all atom pairs within the potential cutoff :math:`r_\mathrm{c}` (:attr:`potential_cutoff`).
The number of neighbors of an atom is proportional to :math:`r_\mathrm{c}^3` (cubic law); for liquid water (0.10 atoms/Å\ :sup:`3`) it is about 220, 730, 1700 and 3400 for :math:`r_\mathrm{c}` = 8, 12, 16 and 20 Å.
Doubling the cutoff therefore multiplies the number of pair terms by 8.
The time per step follows this law only roughly (see the measurements below): for small systems, or small cutoffs, the GPU is not filled and fixed per-step costs dominate, so the time grows more slowly.
The coordination numbers use the pairs within :attr:`coordination_number_cutoff`, which also grows with the cube of that cutoff; with the usual 6 Å it is a minor part of the cost.

Two code paths are used, chosen automatically every step:

* **Small-box path** (any periodic direction with a box thickness of at most 2.5 :math:`r_\mathrm{c}`): the box is replicated to cover the cutoff and a neighbor list is built for each atom. The neighbor list has room for 10000 neighbors per atom; this limit is not checked, so keep the number of neighbors within :math:`r_\mathrm{c}` below it (for water about :math:`r_\mathrm{c}` < 28 Å). A periodic box with a thickness of less than 2.5 cutoffs in one direction and more than 10 cutoffs in another direction is rejected with an error. Several GPU threads share one atom when the number of atoms is small.
* **Large-box path** (all periodic directions thicker than 2.5 :math:`r_\mathrm{c}`, and non-periodic systems): a cell list with cells of about half the cutoff is used, and no neighbor list is stored. Several GPU threads share one atom when the number of atoms is small.

The following extra times per MD step were measured on an RTX A6000 (CUDA 12.3, NEP89, NVE, time step 0.5 fs).
The extra time is the time per step with the correction minus the time per step without it; each time per step is the slope between runs of 100 and 500 steps, which removes the start-up cost; the values are medians of three repetitions.
They are measurements on one GPU and are accurate to about 0.1-0.5 ms (the scatter of the repetitions is at most about 0.2 ms for the small systems and 0.1 ms for the largest); other GPUs and other systems will differ.

Cutoffs 12 Å (potential) and 6 Å (coordination number), the example above (``dftd3 pbe 12 6``):

.. list-table::
   :header-rows: 1

   * - System
     - Atoms
     - Path
     - D3 extra time (ms/step)
   * - water
     - 81
     - small box
     - 0.15
   * - water
     - 648
     - small box
     - 0.46
   * - water
     - 2187
     - small box
     - 0.75
   * - water
     - 5184
     - large box
     - 0.85
   * - water
     - 8748
     - small box
     - 8.3
   * - water
     - 41472
     - large box
     - 6.2
   * - fcc Cu (4000 atoms, :math:`a` = 3.61 Å)
     - 4000
     - large box
     - 0.55
   * - water cluster (non-periodic)
     - 972
     - large box
     - 1.2

The small-box path is fast for few atoms (0.15-0.75 ms for 81-2187 atoms) but its cost grows faster than linearly with the number of atoms (the time of building the explicit neighbor list grows roughly with the square of the number of atoms): the 8748-atom box (thickness 27.9 Å, below 2.5 :math:`r_\mathrm{c}` = 30 Å in one direction) costs 8.3 ms, more than the 5184-atom large-box system.
The large-box path scales with the number of atoms at fixed density: 0.85 ms for 5184 atoms and 6.2 ms for 41472 atoms.

Dependence on the potential cutoff (coordination number cutoff 6 Å):

.. list-table::
   :header-rows: 1

   * - System
     - :math:`r_\mathrm{c}` (Å)
     - Path
     - D3 extra time (ms/step)
   * - water, 648 atoms
     - 8 / 12 / 16 / 20
     - small box
     - 0.11 / 0.46 / 0.55 / 1.6
   * - water, 5184 atoms
     - 8 / 12
     - large box
     - 0.45 / 0.85
   * - water, 5184 atoms
     - 16 / 20
     - small box
     - 3.3 / 23

For 648 atoms the time grows by 3.4 when going from 12 to 20 Å (the cubic law gives 4.6).
For 5184 atoms the time from 16 to 20 Å grows by 6.9, which is more than the cubic law (1.95); it also increases by a factor of 3.9 from 12 to 16 Å, where the path changes from large box to small box (the box thickness is 37.2 Å, which is below 2.5 :math:`r_\mathrm{c}` = 40 Å at 16 Å).
The cause of the faster growth was not investigated; do not extrapolate the table to other cutoffs or box sizes by the cubic law alone, and measure the cost for your system.

Dependence on the coordination number cutoff (potential cutoff 12 Å):

.. list-table::
   :header-rows: 1

   * - System
     - Coordination number cutoff (Å)
     - D3 extra time (ms/step)
   * - water, 648 atoms
     - 3 / 6 / 12
     - 0.44 / 0.46 / 0.87
   * - water, 5184 atoms
     - 3 / 6 / 12
     - 0.83 / 0.85 / 1.4

The NEP evaluation itself takes about 1.6 ms/step for 648 atoms, 1.8 ms/step for 5184 atoms, 6.8 ms/step for 41472 atoms and 1.3 ms/step for 4000 Cu atoms in the same measurement.
For the cost of the :ref:`dftd2 <kw_dftd2>` correction, see the section "Cost" on its page.

Tips
----

* It usually requires to test the convergence with respect to the cutoff radii, and a balance between accuracy and efficiency must be achieved. The cost grows with the cube of the potential cutoff, see :ref:`Cost <dftd3_cost>`.

* The user is responsible for not double counting the dispersion correction, i.e., it is not a good idea to add the DFT-D3 correction to a NEP model that has been trained against a dataset containing dispersion correction (no matter what flavor it is).

* Currently cannot use DFT-D3 for the multi-GPU version of NEP.

* :ref:`dftd3 <kw_dftd3>` and :ref:`dftd2 <kw_dftd2>` are mutually exclusive: specifying both is an input error.
