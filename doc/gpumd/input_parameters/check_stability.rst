.. _kw_check_stability:
.. index::
   single: check_stability (keyword in run.in)

:attr:`check_stability`
=======================

Stop a run, with a diagnostic, as soon as it leaves the physical regime.
Without this keyword nothing changes; with it, a run that finishes without a violation is bit-for-bit identical to the same run without the keyword (the guard only reads the state of the simulation).

A run that has entered an unphysical state (for example after a collapse of the structure) either ends far from the cause with a CUDA error, or, worse, continues in a collapsed state under a thermostat that has already cooled it back to the target temperature.
The guard evaluates a few simple criteria at a regular interval and stops at the first violation.

Syntax
------

::

  check_stability <interval> [max_temperature <T>] [max_potential_change <dU>] [min_distance <d>] [max_force <F>] [max_charge <q>] [keep_frames <n>]

Parameters
----------

* ``interval`` is a positive integer: the criteria are evaluated every ``interval`` steps of the run (at the end of the step, after the forces, velocities and positions of that step are known).
* At least one criterion must be given. There are no default thresholds, because any default would silently change the behavior for some systems. All thresholds are positive real numbers. Each criterion may be given once.

  * ``max_temperature <T>``: the instantaneous temperature, :math:`2E_\mathrm{kin}/(3Nk_\mathrm{B})`, in K, is above ``T``. :math:`E_\mathrm{kin}` and :math:`N` are those of **all** atoms, with 3N degrees of freedom: there is no correction for fixed atoms or for the centre-of-mass motion, so with ``fix`` (or ``move``) the value differs from the temperature that the thermostat and ``thermo.out`` use.
  * ``max_potential_change <dU>``: :math:`|U-U_0|/N` is above ``dU`` (eV/atom). :math:`U_0` is the potential energy at the **first evaluation of the current** ``run`` (that is, after ``interval`` steps), not the energy at step 0 of the whole script. This is the only criterion that detects a collapse that the thermostat has already cooled. Because :math:`U_0` is re-based at the first evaluation of each ``run``, a slow drift across two ``run`` commands, or within the first ``interval`` steps of a run, is not seen.
  * ``min_distance <d>``: the shortest interatomic distance is below ``d`` (Å). The distance is found with a dedicated cell-list kernel that does not depend on the potential, with the periodic boundary conditions (any box supported by GPUMD, including triclinic boxes; periodic images of an atom count if the box is thin) and open directions. It is evaluated only at the check interval.
  * ``max_force <F>``: the largest force component magnitude is above ``F`` (eV/Å). The forces include external forces added by ``add_force``, ``add_efield`` and similar keywords.
  * ``max_charge <q>``: the largest :math:`|q|` is above ``q`` (elementary charge). Only for charge models (qNEP); an error for other potentials. The first potential of the run must be the charge model.
  * A non-finite value (NaN or infinity) of any evaluated quantity counts as a violation.
* ``keep_frames <n>`` is a non-negative integer (default 3): the last ``n`` evaluated frames, including the violating one, are written on a violation. With ``keep_frames 0`` only the violating frame is written.

Behavior on a violation
-----------------------

The run stops at the first evaluation that violates a criterion (so within one ``interval`` of the violation) with a non-zero exit status.
All violated criteria of that evaluation are printed; the final error message names the first one in the order temperature, potential change, distance, force, charge.
The message gives the step of the run, the criterion, the observed value, the threshold and, for distance, force and charge, the atom indices (counted from 0, in the order of ``model.xyz``).
Before stopping, the frames are written to ``stability_frames.xyz`` in the current directory.
No restart file is written, because the state is unphysical.

Output
------

``stability_frames.xyz`` (extended xyz) is written only on a violation. Each frame contains the species, positions (Å), velocities (Å/fs, as in :ref:`dump_xyz <kw_dump_xyz>`), forces (eV/Å) and, for charge models, the charges (``charge:R:1``), with the lattice, ``pbc`` and the run ``step`` in the comment line.

Example
-------

::

  ensemble nvt_bdp 300 300 100
  check_stability 100 max_potential_change 0.5 min_distance 0.9 max_force 30 max_charge 3
  run 1000000

Notes
-----

* The guard acts only on the ``run`` it precedes, like other measurement keywords; repeat the line before every ``run`` that should be guarded. It can be given only once per ``run``.
* Not supported, and refused with a message (the guard is never silently inactive): ``multi_replica`` runs, multi-GPU NEP runs, path-integral (PIMD) ensembles, and a ``minimize`` command in an input that already contains ``check_stability`` before it.
* Cost: each evaluation is a few reductions over the atoms, the copy of the last frames within GPU memory, and, only if ``min_distance`` is given, one cell-list kernel; no host-device transfer except small scalars and the per-atom nearest-pair result of the distance kernel. With ``interval`` of 100 or more the cost is small compared with the force evaluation; see the verification record for measured values.
* The thresholds depend on the system; choose them from the values seen in a stable run (``thermo.out``, ``dump_xyz``).
