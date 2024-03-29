.. pynomial documentation master file, created by
   sphinx-quickstart on Mon Jan 24 09:59:55 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pynomial's documentation!
====================================

**Pynomial** is a lightweight python library for implementing the many condifence intervals for the binomial risk parameter. Pynomial is more or less a python port of the R library *{binom}* by Sundar Dorai-Raj.  

The classic confidence interval for a binomial risk parameter (implemented by the `pynomial.intervals.asymptotic`) is not bounded by (0, 1).  This means that for small samples, or for risks close to 0 or 1, the confidence interval may exceed 0 or 1.  To ameliorate this, may alternative confidence intervals have been proposed.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   quickstart
   interval_info
   apidocs

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
