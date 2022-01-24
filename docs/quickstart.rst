Quickstart
==========

Installation
------------

You can install pynomail using pip

.. code::
   
     pip install git+https://github.com/Dpananos/pynomial

Usage
-----

To use the many interval functions, simply import pynomial and start generating confidence intervals.  Suppose we flip a coin 20 times and observe 10 heads.  A confidence Wilson score interval for the probability of getting a heads can be obtained using

.. code::
   
   >>> import pynomial
   >>> ci = pynomial.wilson(x=10, n=20)
   >>> print(ci)

            estimate     lower     upper
    Wilson       0.5  0.299298  0.700702


The variable ``ci`` is a pandas DataFrame containing as columns the estimate of the risk (the probability we get a heads in this example) and the upper and lower bounds for the confidence interval.  All the interval functions in pynomial are vectorized so we can get multiple confidecne intervals for multiple experiments at the same times

.. code::
   
   >>> import pynomial
   >>> ci = pynomial.wilson(x=[9, 10, 11], n=[19, 20, 21])
   >>> print(ci)

            estimate     lower     upper
    Wilson  0.473684  0.273298  0.682922
    Wilson  0.500000  0.299298  0.700702
    Wilson  0.523810  0.323695  0.716560


The default confidence level for each interval is 95%.  To get a different confidence level, pass a float between 0 and 1 to the ``conf`` argument


.. code::
   
   >>> import pynomial
   >>> ci = pynomial.wilson(x=[9, 10, 11], n=[19, 20, 21], conf = 0.99)
   >>> print(ci)

            estimate     lower     upper
    Wilson  0.473684  0.226383  0.734607
    Wilson  0.500000  0.250448  0.749552
    Wilson  0.523810  0.273309  0.762877