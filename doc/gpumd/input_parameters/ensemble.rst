.. _kw_ensemble:
.. index::
   single: ensemble (keyword in run.in)

:attr:`ensemble` (overview)
===========================

This keyword selects the integration method for the next :ref:`run <kw_run>`.

Syntax
------

::

  ensemble <method> <parameters>

Parameters
----------

For conventional classical MD, the main choices are:

* ``nve``: dynamics without a thermostat or barostat.
* ``nvt_*``: temperature control at fixed cell.
* ``npt_*``: temperature and pressure control, allowing the cell to change.
* ``nph_mttk``: pressure control without a thermostat.

The available methods, including specialized integrators, are described on the following pages:

* :ref:`"standard" ensembles <kw_ensemble_standard>`
* :ref:`MTTK integrators <kw_ensemble_mttk>`
* :ref:`TSPIN integrators for Spin potentials <kw_ensemble_tspin>`
* :ref:`SIB integrators for Spin potentials <kw_ensemble_sib>`
* :ref:`GLSD variable-magnitude dynamics <kw_ensemble_glsd>`
* :ref:`local heat-source and heat-sink thermostats <kw_ensemble_heat>`
* :ref:`two-temperature-model integrators <kw_ensemble_ttm>`
* :ref:`quantum thermal bath integrators <kw_ensemble_qtb>`
* :ref:`integrators for path integral molecular dynamics simulations <kw_ensemble_pimd>`
* :ref:`MSST integrator for simulating compressive shock wave <kw_ensemble_msst>`
* :ref:`NPHug integrator for simulating compressive shock wave <kw_ensemble_nphug>`
* :ref:`NEMD for simulating compressive shock wave <kw_ensemble_shock_nemd>`
* Thermodynamic integration and switching: :ref:`TI <kw_ensemble_ti>`, :ref:`TI_Spring <kw_ensemble_ti_spring>`, :ref:`TI_AS <kw_ensemble_ti_as>`, :ref:`TI_RS <kw_ensemble_ti_rs>`, :ref:`TI_Liquid <kw_ensemble_ti_liquid>`, and :ref:`TI_NEP <kw_ensemble_ti_nep>`

.. _choice_of_parameters:

Units and suggested parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The units of temperature and pressure for this keyword are K and GPa, respectively.
These are thermostat and barostat targets; they do not set the initial atomic velocities, which are determined by ``model.xyz`` or the :ref:`velocity keyword <kw_velocity>`.
Use equal initial and final targets to keep the target constant during a run.

The coupling parameters below are used by the :ref:`standard integrators <kw_ensemble_standard>` and :ref:`local heat baths <kw_ensemble_heat>` where applicable.
For methods with different parameter names or requirements, follow their individual pages.

The temperature coupling constant :attr:`<T_coup>` means :math:`\tau_T/\Delta t`, where :math:`\tau_T` is the relaxation time of the thermostat and :math:`\Delta t` is the time step for integration.
We require :math:`\tau_T/\Delta t \geq 1` and a good choice is :math:`\tau_T/\Delta t \approx 100`.

When :math:`\tau_T/\Delta t > 100000`, the Berendsen thermostat in :attr:`npt_ber` will be completely ignored, leaving only the Berendsen barostat. In this case, the NPT ensemble reduces to the NPH ensemble that can be useful for melting point calculations using the two-phase method. This thermostat-free mode is not available with :ref:`stochastic cell rescaling <stochastic_cell_rescaling>`.

The pressure coupling constant :attr:`<p_coup>` means :math:`\tau_p/\Delta t`, where :math:`\tau_p` is the relaxation time of the barostat and :math:`\Delta t` is the time step for integration.
We require :math:`\tau_p/\Delta t \geq 1` and a good choice is :math:`\tau_p/\Delta t \approx 1000`.

The elastic constants are in units of GPa.

.. _ensemble_thermostat_seed:

Random seed of the stochastic thermostats
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The stochastic thermostats ``nvt_bdp`` and ``npt_scr`` accept an optional trailing ``seed`` option::

  ensemble nvt_bdp <T_1> <T_2> <T_coup> seed <n>
  ensemble npt_scr <T_1> <T_2> <T_coup> <pressure_control_parameters> seed <n>

:attr:`<n>` is a positive integer that initializes the random-number generator of the thermostat (and, for ``npt_scr``, of the cell rescaling) deterministically.
A zero, negative, or non-integer value, a missing value, or a repeated ``seed`` is an input error.
An ``ensemble`` line applies to a single ``run``, so each ``ensemble ... seed <n>`` line starts its own stream from its seed: the generator and its cached Gaussian number are reset at the start of that run, independently of earlier runs in the same input (also of earlier unseeded runs).
Two consecutive runs that use the same seed therefore draw the same random numbers; give different seeds to get different streams.
Without ``seed``, the generator is initialized from the system clock as before (a fixed seed in builds compiled with ``-DDEBUG``); an explicit seed overrides the fixed seed of the ``-DDEBUG`` builds.
The option is accepted only by ``nvt_bdp`` and ``npt_scr`` of ``gpumd``: ``heat_bdp`` (a parameter-count error), the Langevin-type, PIMD and other ensembles, and the ``nvt_bdp`` of ``multi_replica`` runs (which have their own seed handling) do not accept it.

Reproducibility scope: the seed fixes only the random stream of the thermostat.
With the same input script, in a fresh process, with the same binary, the thermostat random numbers are identical.
It does not make the whole trajectory reproducible: the forces are accumulated on the GPU with atomic additions, so the last digits of the forces, and hence the trajectories of an interacting system, can differ from run to run even with the same seed.
The initial velocities are controlled separately by ``velocity <T> seed <n>`` (see the :ref:`velocity keyword <kw_velocity>`).
Record the seeds used in a study together with the input files.

Output
------

None.

Examples
--------

::

  ensemble nvt_bdp 300 300 100

Notes
-----

Specify exactly one ``ensemble`` command before each :ref:`run <kw_run>`.
The integrator is initialized for that run, and any temperature or pressure ramp is defined over that run.
