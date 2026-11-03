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
The extra time is the time per step with the correction minus the time per step without it; each time per step is the slope between runs of 100 and 500 steps (the minimum of two runs each), which removes the start-up cost.
They are single measurements on one GPU and are accurate to about 0.1-0.5 ms; other GPUs and other systems will differ.

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
     - 1.5
   * - water
     - 648
     - small box
     - 1.9
   * - water
     - 5184
     - large box
     - 1.1
   * - water
     - 41472
     - large box
     - 7.5
   * - fcc Cu (4000 atoms, :math:`a` = 3.615 Å)
     - 4000
     - large box
     - 0.53

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
     - 0.63 / 1.9 / 3.9 / 7.8
   * - water, 5184 atoms
     - 8 / 12
     - large box
     - 0.56 / 1.1
   * - water, 5184 atoms
     - 16 / 20
     - small box
     - 7.2 / 31

For 648 atoms the time grows by 4.0 when going from 12 to 20 Å (the cubic law gives 4.6).
For 5184 atoms the time from 16 to 20 Å grows by 4.2, which is more than the cubic law (1.95); it also increases by a factor of 6.6 from 12 to 16 Å, where the path changes from large box to small box (the box thickness is 37.2 Å, which is below 2.5 :math:`r_\mathrm{c}` = 40 Å at 16 Å).
The cause of the faster growth was not investigated; do not extrapolate the table to other cutoffs or box sizes by the cubic law alone, and measure the cost for your system.

Dependence on the coordination number cutoff (potential cutoff 12 Å):

.. list-table::
   :header-rows: 1

   * - System
     - Coordination number cutoff (Å)
     - D3 extra time (ms/step)
   * - water, 648 atoms
     - 3 / 6 / 12
     - 1.8 / 1.9 / 2.9
   * - water, 5184 atoms
     - 3 / 6 / 12
     - 0.98 / 1.1 / 1.9

The NEP evaluation itself takes about 1.6 ms/step for 648 atoms, 1.8 ms/step for 5184 atoms, 6.8 ms/step for 41472 atoms and 1.4 ms/step for 4000 Cu atoms in the same measurement.
For the cost of the :ref:`dftd2 <kw_dftd2>` correction, see the section "Cost" on its page.

Tips
----

* It usually requires to test the convergence with respect to the cutoff radii, and a balance between accuracy and efficiency must be achieved. The cost grows with the cube of the potential cutoff, see :ref:`Cost <dftd3_cost>`.

* The user is responsible for not double counting the dispersion correction, i.e., it is not a good idea to add the DFT-D3 correction to a NEP model that has been trained against a dataset containing dispersion correction (no matter what flavor it is).

* Currently cannot use DFT-D3 for the multi-GPU version of NEP.

* :ref:`dftd3 <kw_dftd3>` and :ref:`dftd2 <kw_dftd2>` are mutually exclusive: specifying both is an input error.
