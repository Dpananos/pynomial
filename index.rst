.. pynomial documentation master file, created by
   sphinx-quickstart on Wed Aug 27 15:33:40 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

pynomial documentation
======================

Welcome to pynomial's documentation! This package provides various confidence interval 
methods for binomial proportions.

Rationale and Background
========================

**pynomial** is a Python library intended to re-implement R's `{binom}` package.  It offers most of the same functionality.

Why pynomial?
-------------

To screw around with package development.

Comparison with R's binom Package
----------------------------------

The R `binom` package provides functions like `binom.confint()` that compute confidence 
intervals using various methods. **pynomial** provides equivalent functionality with 
these methods:

* **wald()** ↔ `method="wald"` - Wald interval (normal approximation)
* **wilson()** ↔ `method="wilson"` - Wilson score interval  
* **agresti_coull()** ↔ `method="ac"` - Agresti-Coull interval
* **clopper_pearson()** ↔ `method="exact"` - Exact (Clopper-Pearson) interval
* **bayesian_beta()** ↔ `method="bayes"` - Bayesian interval with Beta prior
* **logit()** ↔ `method="logit"` - Logit transformation interval
* **cloglog()** ↔ `method="cloglog"` - Complementary log-log interval
* **arcsine()** ↔ `method="asymptotic"` - Arcsine transformation interval

Getting Started
---------------

Install pynomial via pip::

    pip install pynomial

Basic usage::

    import pynomial
    
    # Calculate Wilson confidence interval
    result = pynomial.wilson(successes=85, trials=100, confidence_level=0.95)
    print(f"95% CI: [{result.lower:.3f}, {result.upper:.3f}]")
    
    # Compare multiple methods
    methods = [pynomial.wilson, pynomial.wald, pynomial.clopper_pearson]
    for method in methods:
        ci = method(85, 100)
        print(f"{method.__name__}: [{ci.lower:.3f}, {ci.upper:.3f}]")

API Reference
=============

.. automodule:: pynomial.intervals
   :members:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

