.. _replica_kw_potential:
.. index::
   single: potential (keyword in gpumd_replica run.in)

:attr:`!potential`
==================

Syntax
------

Specify one potential file::

  potential <filename>

Supported potential headers are ``nep4``, ``nep4_zbl``, ``nep5``, ``nep5_zbl``, ``nep4_charge1``, ``nep4_zbl_charge1``, ``nep4_charge2``, and ``nep4_zbl_charge2``.
Charge NEP models require periodic boundaries in all three directions.
qNEP models trained with the :ref:`charge_transfer <kw_charge_transfer>` option (a ``charge_transfer`` line in ``nep.txt``) are not supported by ``gpumd_replica``; the run stops with an error message.

Example
-------

::

  potential nep.txt
