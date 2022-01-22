# Pynomial

Pynomial (pronounced like "binomial") is a lightweight python library for implementing the many confidence intervals for the risk parameter of a binomial model.  Pynomial is more or less a python port of the R library [`{binom}`](https://cran.r-project.org/web/packages/binom/binom.pdf) by Sundar Dorai-Raj.  As a point of philosophy and until otherwise stated, if `{binom}` does a thing then so should pynomial (e.g. error throwing or handling cases when the number of successes is the same as the number of trials).

# Features

The following confidence intervals are implemented:

* [The Agresti Coull Interval](https://www.jstor.org/stable/2685469?origin=crossref)

* The asymptotic interval based on the central limit theorem (this is the interval you probably see in most statistics textbooks)

* An equal tailed posterior credible interval using a conjugate Beta prior

* The complimentary log-log interval

* [The Wilson score interval](https://www.tandfonline.com/doi/abs/10.1080/01621459.1927.10502953)

* The exact interval based on the incomplete beta function.

# Installation

You can install pynomial from github using

```
 pip install git+https://github.com/Dpananos/pynomial
```

# Getting Started

## Information on Binomial Random Variables

Many textbooks have their own treatment of binomial random variables and confidence intervals. Recommended resources to familliarize one's self with the methods in this library are:

* Lachin, John M. *Biostatistical methods: the assessment of relative risks*. Vol. 509. John Wiley & Sons, 2009.

* Brown, Lawrence D., T. Tony Cai, and Anirban DasGupta. *Interval estimation for a binomial proportion.* Statistical science 16.2 (2001): 101-133.

* Brown, Lawrence D., T. Tony Cai, and Anirban DasGupta. *Confidence intervals for a binomial proportion and asymptotic expansions.* The Annals of Statistics 30.1 (2002): 160-201.

