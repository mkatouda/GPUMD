.. _kw_dftd2:
.. index::
   single: dftd2 (keyword in run.in)

:attr:`dftd2`
=============

This keyword adds the DFT-D2 pairwise dispersion correction of Grimme [Grimme2006]_ to a NEP model.
It is an *external* correction: it is not assumed to be part of the data the NEP model was trained on.

Syntax
------

  dftd2 <s6> [<cutoff>] [table grimme2006|qe] [trunc sf|plain]

where :attr:`s6` is the dimensionless global scaling factor (must be positive; there is no functional table, the value is given by the user) and :attr:`cutoff` is the cutoff radius in units of Angstrom (must be positive).
If :attr:`cutoff` is omitted, a default of 30 Angstrom is used.
The default is the smallest cutoff in a 12-30 Angstrom scan (grid 12, 16, 20, 25, 30; reference 40 Angstrom) for which the D2 energy per atom (within 0.5 meV), the forces and the pressure (within 0.02 GPa) agree with a 40 Angstrom cutoff for water, graphite and MgO (light elements only; see Truncation effects). A warning is printed for cutoffs above 200 Angstrom. The cost grows quickly with the cutoff (on an RTX A6000, the D2 correction at 30 Angstrom adds about 1.1 times the NEP evaluation time for 648 water atoms and about 10 times for 41,472 atoms); a smaller cutoff can be specified explicitly if a lower accuracy is acceptable.

The optional keywords :attr:`table` and :attr:`trunc` select the parameter table and the truncation; they are keyword/value pairs that may be given in any order, each at most once, after the (optional) cutoff. The values are case sensitive (lower case).
The cutoff is recognised only as the second parameter, directly after :attr:`s6`; a number after an option is an error.
The defaults are ``table grimme2006`` and ``trunc sf``, which is the behavior described in the rest of this page: an input without options gives results and screen output identical to those of versions without these options.
``table qe`` and ``trunc plain`` require an explicit cutoff (an input without a cutoff is an error), so that the cutoff matches the one that was used to generate the reference data (for example the labels of the training set): the default of 30 Angstrom belongs to the shifted-force convergence study described below.
For example, ``dftd2 0.75 12 table qe trunc plain`` gives the D2 form of Quantum ESPRESSO 7.2 with ``dftd3_version = 2`` and a hard cutoff of 12 Angstrom (the own cutoff of Quantum ESPRESSO is 50.2 Å, so give the value that was used for the reference data). When the energy, forces or virial of the NEP training data include such a correction, the table and the truncation of the correction used in the simulation should be the same as those used to generate the data: record them with the data and set them identically in ``run.in``.

This keyword can be put anywhere in the ``run.in`` input file.
It may appear only once and cannot be combined with :ref:`dftd3 <kw_dftd3>`.

Model
-----

For every pair of atoms :math:`i,j` (including periodic images) at distance :math:`r<r_\mathrm{c}`,

.. math::

   \phi(r) = - s_6\, C_{6,ij}\, \frac{f(r)}{r^{6}}, \qquad
   f(r) = \frac{1}{1+\exp\left[-d\left(r/R_{0,ij}-1\right)\right]},

with :math:`C_{6,ij}=\sqrt{C_{6,i}C_{6,j}}`, :math:`R_{0,ij}=R_{0,i}+R_{0,j}` and :math:`d=20` (fixed).
The energy is :math:`E=\sum_{i<j}\phi_\mathrm{sf}(r_{ij})`.

Units: positions and :math:`r_\mathrm{c}` in Angstrom, :math:`C_6` in eV Å\ :sup:`6` (converted from the tabulated J nm\ :sup:`6` mol\ :sup:`-1` with 1 E\ :sub:`h` = 2625.4999 kJ/mol and 1 E\ :sub:`h` = 27.21138505 eV; the value of the bohr radius enters the conversion twice and cancels, so only these two energy constants matter), :math:`R_0` in Angstrom (1.1 times the tabulated :math:`r_0`), energy in eV, force in eV/Å, virial in eV.

Options :attr:`table` and :attr:`trunc`
---------------------------------------

``table grimme2006`` (default) is the DFT-D2 parameter table of [Grimme2006]_ described above.

``table qe`` uses the pair parameters of the ``dftd3_version = 2`` option of Quantum ESPRESSO 7.2 (``dft-d3`` module, ``pars.f90`` and ``core.f90`` of tag ``qe-7.2``, commit ``b4c5e8de``). The QE input documentation (``INPUT_PW``) describes this option as "Original Grimme-D2 parametrization"; this is inaccurate for ``dftd3_version = 2``: the energy is the D2 functional form evaluated with DFT-D3 tables, namely

.. math::

   C_{6,ij} = C_6^{\mathrm{D3}}(Z_i,Z_j;\ \text{first reference point of both atoms}), \qquad R_{0,ij} = 1.1\, r_{0,ij}^{\mathrm{D3}},

with the damping function and :math:`d=20` unchanged. The :math:`C_6` is the coefficient of the first stored DFT-D3 reference point of the two elements (not a geometric mean of atomic values, and no coordination-number dependence). It is a fixed pair value and is not the free-atom :math:`C_6` for every element: the first reference point of H has the coordination number 0.9118 (the H\ :sub:`2` value) and that of Ce has 2.7991; :math:`r_{0,ij}^{\mathrm{D3}}` is the D3 cutoff radius (``r0ab``, in Angstrom). :attr:`s6` is still given by the user (there is no functional table; ``dftd3_version = 2`` with ``input_dft = 'pbe'`` corresponds to :math:`s_6=0.75`).
The tables cover **Z ≤ 86 (H-Rn)**; an atom type with Z ≥ 87 in the structure is an input error naming the element(s), detected when the correction is first evaluated.
The :math:`C_6` values are tabulated in :math:`E_\mathrm{h}\,a_0^6` and converted on the host in double precision with the CODATA 2018 constants of Quantum ESPRESSO 7.2 (:math:`a_0` = 0.529177210903 Å, :math:`E_\mathrm{h}` = 27.211386245988 eV). The tables are generated by ``tools/Miscellaneous/dftd2/generate_dftd2qe_table.py`` from the Quantum ESPRESSO sources into ``src/force/dftd2qe_table.cuh``.
Quantum ESPRESSO divides ``r0ab`` by a Bohr radius of 0.52917720859 Å and multiplies it back with its own constant; this changes :math:`R_0` by a relative 4.4e-9 and is not reproduced (on purpose). In addition, QE 7.2 stores ``r0ab`` (and the values of :math:`s_6` other than 0.75) as single-precision literals: the relative differences to the decimal values are up to 5.9e-8 in :math:`r_0` and up to 1.2e-6 in the damping exponent :math:`d (r/R_0-1)`. Agreement with QE 7.2 is therefore expected to about 1e-6 relative, i.e. the results are consistent with QE 7.2, not identical to it. This was verified against the 7.2 tables and data supplied by the requester and against QE 7.6-dev (verification of the implementation against the source tables and a numpy reference, not validation against experiment); QE 7.2 itself was never run by the fork.
``table qe`` reproduces the ``dftd3_version = 2`` energy only. It is **not** the real DFT-D2 of QE (``vdw_corr = 'grimme-d2'``: its own table and ``london_rcut`` = 200 bohr); ``table grimme2006 trunc plain`` has not been validated against that option. ``dftd3_version = 2`` with ``input_dft = 'dsd-blyp'`` uses a damping steepness of 60, which ``table qe`` cannot reproduce (the steepness is fixed to 20).
Quantum ESPRESSO is described in P. Giannozzi et al., J. Phys.: Condens. Matter 21, 395502 (2009), 29, 465901 (2017) and J. Chem. Phys. 152, 154105 (2020).

``trunc sf`` (default) is the shifted-force truncation described in the next section. ``trunc plain`` is a hard cutoff, :math:`\phi_\mathrm{plain}(r)=\phi(r)` for :math:`r<r_\mathrm{c}` and 0 otherwise (as in Quantum ESPRESSO, which also does not shift); the shift terms are not used. Both truncations can be combined with both tables.
With ``trunc plain`` the energy and the force are discontinuous at the cutoff. A pair crossing :math:`r_\mathrm{c}` changes the energy by a step of :math:`|\phi(r_\mathrm{c})|` (about 7e-6 eV for a C-C pair at 12 Å with ``table qe`` and :math:`s_6=0.75`) and the force by a step of about :math:`6|\phi(r_\mathrm{c})|/r_\mathrm{c}` (about 3.7e-6 eV/Å); the sizes decrease as :math:`r_\mathrm{c}^{-6}` and :math:`r_\mathrm{c}^{-7}` with the cutoff. In an NVE simulation the total energy therefore shows small steps. A systematic drift is not expected in equilibrium, because pairs cross the cutoff in both directions and the steps cancel on average; a short NVE measurement found no visible drift: in a 20 ps NVE run of a 180-atom PTFE crystal at 300 K (NEP89 + ``dftd2 0.75 12 table qe``, 0.5 fs, 3 seeds per variant) the fitted slope of the total energy is |slope| <= 2e-4 meV/atom/ps for ``trunc plain`` and <= 1e-4 meV/atom/ps for ``trunc sf``, and the standard deviation of the total energy is 5.1-5.7e-4 eV for ``plain`` and 4.9-5.1e-4 eV for ``sf``; the single steps of 7e-6 eV are far below the step-to-step noise of the integration (median 9e-5 eV), so they are not resolved. The pair convention is :math:`r<r_\mathrm{c}`: a pair at exactly :math:`r_\mathrm{c}` is excluded, while QE excludes only :math:`r>r_\mathrm{c}` (the difference is negligible). The force and the virial are the exact derivatives of the smooth part inside the cutoff; the delta-function contribution of the cutoff itself is not included.

Shifted-force truncation
------------------------

This section applies to ``trunc sf`` (default). The sum is truncated with a shifted-force scheme so that both energy and force go to zero continuously at the cutoff:

.. math::

   \phi_\mathrm{sf}(r) = \phi(r) - \phi(r_\mathrm{c}) - \phi'(r_\mathrm{c})\,(r-r_\mathrm{c}) \quad (r<r_\mathrm{c}), \qquad \phi_\mathrm{sf}(r)=0 \quad (r\ge r_\mathrm{c}).

The term :math:`-\phi'(r_\mathrm{c})(r-r_\mathrm{c})` is linear in :math:`r`. The energy shift therefore depends on the number of pairs within the cutoff *and* on their distribution of distances (it is not a constant), and every pair force is changed by the same constant :math:`\phi'(r_\mathrm{c})`, directed along the pair vector.
Force and virial are the exact derivatives of this shifted energy.

Truncation effects
^^^^^^^^^^^^^^^^^^

The dispersion energy has a long :math:`r^{-6}` tail, and the shifted-force scheme changes the energy and the pressure relative to the untruncated sum.
For a system of number density :math:`\rho` in which the pair distribution is uniform beyond a few Angstrom (continuum estimate, neglecting the damping function, which is 1 there), the energy per atom of the shifted-force sum minus that of the untruncated sum is

.. math::

   \Delta E = \frac{7\pi}{3}\,\frac{\rho\,\langle s_6 C_6\rangle}{r_\mathrm{c}^{3}}, \qquad
   \Delta P = \rho\,\Delta E,

where :math:`\langle C_6\rangle=\left(\sum_a x_a\sqrt{C_{6,a}}\right)^2` for atomic fractions :math:`x_a` (the energy shift gives :math:`2\pi/3` from the missing tail and :math:`5\pi/3` from the shift terms).
Consequences:

* The difference decreases as :math:`r_\mathrm{c}^{-3}`: going from 30 to 25 Å increases it by :math:`(30/25)^3 = 1.73`, and from 30 to 40 Å decreases it by a factor 2.4.
* The sign is positive: the shifted-force energy is *less negative* than the untruncated sum, i.e. the dispersion binding is underestimated, and the pressure is shifted towards positive values.
  In constant-pressure (NPT) simulations this is a systematic error that makes the volume slightly too large; it decreases as the cutoff increases.
* The error is proportional to density times :math:`C_6`. **Dense systems and heavy elements (large** :math:`C_6` **) need a larger cutoff or at least a check of the convergence with respect to the cutoff**; the default is not a guarantee for them.
* Both quantities are linear in :math:`s_6`.

The default of 30 Å is based on measurements for light-element systems only: liquid water (648 atoms), graphite (972 atoms) and MgO (512 atoms), with NEP89 and :math:`s_6` = 0.75.
Cutoffs of 12, 16, 20, 25 and 30 Å were compared with a 40 Å reference for the energy per atom (limit 0.5 meV), the forces, and the pressure (limit 0.02 GPa).
At 30 Å the energy per atom differs from the 40 Å result by about 0.03 (water), 0.24 (graphite) and 0.34 (MgO) meV and the pressure by 0.6 to 6 MPa; relative to the untruncated sum (extrapolated with the :math:`r_\mathrm{c}^{-3}` law) the errors are about 0.06 (water), 0.4 (graphite) and 0.6 (MgO) meV per atom and 1 to 10 MPa.
At 25 Å the energy criterion was not met for graphite and MgO.
These numbers are of the size given by the formula above.
For heavy, dense elements the error is much larger: for fcc Au (4.078 Å lattice constant, :math:`s_6` = 0.75) the energy per atom of the shifted-force sum is 10.1 meV above the untruncated sum at 30 Å and 0.5 meV above it at 80 Å, and the pressure is 96 MPa too positive at 30 Å (fcc Pt: 11.3 meV and 120 MPa at 30 Å). The formula reproduces these measured differences to better than 0.1 %. Check the convergence with respect to the cutoff for such systems and consider cutoffs much larger than 30 Å (at a cost growing as :math:`r_\mathrm{c}^3`).

Parameters and element coverage
--------------------------------

This section describes ``table grimme2006`` (default); the limits of ``table qe`` are given above. The built-in :math:`C_6` and :math:`R_0` table covers the elements **H to Rn (Z ≤ 86)** only.
It is generated by script from the DFT-D2 parameters stored in ``dftd3-lib`` (``prg/extras.f90``, subroutine ``loadoldpar``, commit ``3c99ee9``), following S. Grimme, J. Comput. Chem. 27, 1787 (2006).
The values for H–Xe agree with the table of Grimme (2006) as reprinted in a secondary source (the Turbomole manual); they were not compared with the original paper directly.
The published table of Grimme (2006) gives H to Xe (Z = 1-54). The values for **Z = 55-86 and the group values for Sc-Zn and Y-Cd come from a single source, the** ``dftd3-lib`` **table**, and some of them are group-filled (for example, Ce-Lu share one value); they should not be regarded as independently verified.

License note: the GPUMD source code under ``src/`` is distributed under LGPL-3.0-or-later. The tabulated :math:`C_6` and :math:`R_0` values were derived from ``dftd3-lib`` (distributed under GPL version 1 or later) and from the published H-Xe values; this is stated as a fact about the origin of the data and is not a legal assessment.

If any atom of the system has an element with Z > 86 (for the NEP89 model: Ac, Th, Pa, U, Np, Pu), the run stops with an error naming the element(s).
There is no silent zero, extrapolation, or fallback.
The check uses the types of the atoms actually present in the structure (not the full element list of the model, so NEP89 can be used with ``dftd2`` as long as the structure contains no element with Z > 86) and is performed once, when the correction is first evaluated. A later change of atom types during the run (for example by Monte Carlo moves) is not checked.

Supported potentials
--------------------

NEP (``nep4``, ``nep4_zbl``, and temperature variants), the charge variants (``nep4_charge*``, ``nep4_zbl_charge*``), and the vdW models ``nep_vdw`` and ``qnep3_vdw``.
Using ``dftd2`` with any other potential, or with the multi-GPU version of NEP, is an input error. The ``gpumd_replica`` executable does not implement ``dftd2`` (the keyword is not read there and the correction is not applied).
(``dftd3`` remains unsupported for ``nep_vdw`` and ``qnep3_vdw``.)

Cost
----

The cost per step is proportional to the number of atoms times the number of neighbors within the cutoff, i.e. :math:`O(N r_\mathrm{c}^3)`. The same cubic dependence on the cutoff holds for :ref:`dftd3 <kw_dftd3>`, whose cost and measured times are given in the section :ref:`Cost <dftd3_cost>` of its page; the cutoffs used for D3 (for example 12 and 6 Å) are much smaller than the D2 default (30 Å). A cell list is used, so there is no :math:`O(N^2)` term, and no coordination numbers are needed.
Cells of any size relative to the cutoff (including cells smaller than the cutoff) are handled by the same kernel. The number of cells in each direction is chosen every step from an estimate of the work (candidate atoms visited plus a cost per neighbor cell visited), not from a fixed ratio of the cell size to the cutoff.
For small systems several threads share one atom (up to 32, chosen from the number of atoms and the number of SMs so that the GPU is filled); each thread handles a fixed subset of the neighbors and the partial sums are added in a fixed order in shared memory, so the result stays deterministic.
The pair terms are evaluated in single precision and accumulated in single precision with a compensation term (error-free TwoSum, so the accumulation error does not grow with the number of pairs); the sums of the threads that share an atom and the per-atom results are in double precision. The accumulation is deterministic (no atomic operations). The candidate positions are stored in single precision relative to the center of their cell and the displacement between two cells is formed in double precision, so the absolute error of each component of a pair vector is about :math:`6\times10^{-8}\,(r + w)`, where :math:`r` is the pair distance and :math:`w` the cell width (the cell-to-cell displacement, up to about the cutoff, is rounded to single precision and added to the difference of the cell-relative positions). It does not depend on the distance from the coordinate origin or on the box size; the cell width is of the order of the cutoff divided by a small integer (at most 6). For close pairs this is larger than the rounding error of a pair vector computed in double precision (about :math:`6\times10^{-8}\,r`) by a factor of up to :math:`1 + w/r`; for example, about 5 times for :math:`r` = 3 Å and :math:`w` = 12 Å (2e-7 Å compared with 1e-6 Å). The resulting energy, force and virial errors were checked against an independent double-precision reference and stay well inside the tolerances of the test suite (relative 2e-6 for energy and virial, 1e-5 for the largest force component), including dense water with the largest neighbor range of 6 cells.
For systems that are not periodic in all three directions, the extent of the atoms is determined every step with a small reduction kernel and a device-to-host copy, which adds a synchronization per step.
The atoms inside each cell are kept in order of increasing atom index, and every thread sums its neighbors, and the threads of one atom add their partial sums, in a fixed order, so repeated runs on the same GPU with the same binary give bit-identical D2 energy, forces and virial (checked for a dense 2187-atom water system with a 30 Å cutoff, which uses several threads per atom). Results can still differ between different GPU architectures or compilers.

Limitations
-----------

* **Parameters are functional dependent.** :math:`s_6`, :math:`R_0` and :math:`d` were fitted by Grimme for specific GGA functionals; :math:`s_6` = 0.75 is the value for PBE. The correction is meaningful only when the functional used to generate the training data of the NEP model matches the functional for which :math:`s_6` was chosen (and the training data must not already contain dispersion, see Tips). Other values of :math:`s_6` are the responsibility of the user.
* **Fixed neutral-atom** :math:`C_6`. The coefficients do not depend on the coordination number or the oxidation state. Ionic and metallic systems (for example oxides, halides, metals), where the effective polarizability differs from that of the neutral atom, may be overbound.
* **Pairwise only.** There are no many-body (three-body, Axilrod-Teller-Muto) terms and no screening or collective effects (for example in metals or at surfaces).
* **Slabs and vacuum.** For periodic directions the sum includes all periodic images within the cutoff. For a slab with a vacuum layer thinner than the cutoff, a slab interacts with its own periodic images across the vacuum; make the vacuum larger than the cutoff or accept this interaction.
* **Charge variants.** The correction is independent of the (predicted) charges and does not interact with the electrostatics of the ``nep4_charge*`` and ``qnep3_vdw`` models.
* **Heat transport.** Heat-current (``compute_hac``, ``compute_hnemd``, ``compute_shc``) calculations with ``dftd2`` have not been validated at the default cutoff.
* **Cost with several NEP potentials.** If several ``potential`` lines (NEP) are present, the D2 correction is evaluated for each of them, multiplying its cost.
* **Coincident atoms are unsupported.** Two atoms at the same position give a diverging :math:`r^{-6}` term and non-finite values (the NEP models themselves also return non-finite values in this case); this is not checked during the run.
* **Cutoff above 200 Å.** A warning is printed; the cost grows as :math:`r_\mathrm{c}^3`.
* At least two atoms are required (a general requirement of ``gpumd``).

Example
-------

.. code::

  potential nep89.txt
  dftd2 0.75 15

This adds the DFT-D2 correction with :math:`s_6=0.75` and a cutoff of 15 Angstrom to the NEP model.

.. code::

  potential nep89.txt
  dftd2 0.75 12 table qe trunc plain

This adds the D2 form with the DFT-D3 tables and a hard cutoff of 12 Angstrom, matching Quantum ESPRESSO ``dftd3_version = 2`` data generated with a 12 Angstrom cutoff.

Tips
----

* The correction is external. The user is responsible for not double counting dispersion: do not add DFT-D2 to a NEP model trained on reference data that already contain dispersion, for example data generated with PBE-D3, PBE-D2, or a non-local functional such as vdW-DF or rVV10. Check the functional used for the training data of any published model before using ``dftd2`` with it.

* ``nep_vdw`` and ``qnep3_vdw`` models already contain their own van der Waals term. Adding ``dftd2`` on top of them counts dispersion twice unless that is what you intend.

* The energy depends on the cutoff (and on the shift). Test the convergence with respect to the cutoff.
