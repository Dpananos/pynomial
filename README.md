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

## Usage

Using pynomial is very straight forward. Each interval function has three common arguments: `x` -- the number of success, `n` -- the number of trials, and `conf` -- the desired confidence level.  Both `x` and `n` can be either integers or arrays of integers and conf must be a float between 0 and 1 (the default is 0.95 for a 95% confidence interval).  After calling an interval function with the propper arguments, a dataframe will be returned yeilding an estimate of the risk as well as the lower and upper confidence limits.  As an example, suppose I flipped a coin 20 times and observed 12 heads.  Using the `wilson` function to compute a Wilson score confidence interval, the output would be

```python
from pynomial import wilson
x = 12
n = 20
wilson(x=x, n=n)
```

```
        estimate     lower     upper
Wilson       0.6  0.386582  0.781193
```

Each interval function is vectorized, so we can compute confidence intervals for many experiments at once.

```python
from pynomial import wilson
x = np.array([11, 12, 13])
n = 20
wilson(x=x, n=n)
```

```
        estimate     lower     upper
Wilson      0.55  0.342085  0.741802
Wilson      0.60  0.386582  0.781193
Wilson      0.65  0.432854  0.818808
```

The output of each interval function is a pandas dataframe, making plotting the confidence intervals straightforward.

## Information on Binomial Random Variables

Many textbooks have their own treatment of binomial random variables and confidence intervals. Recommended resources to familliarize one's self with the methods in this library are:

* Lachin, John M. *Biostatistical methods: the assessment of relative risks*. Vol. 509. John Wiley & Sons, 2009.

* Brown, Lawrence D., T. Tony Cai, and Anirban DasGupta. *Interval estimation for a binomial proportion.* Statistical science 16.2 (2001): 101-133.

* Brown, Lawrence D., T. Tony Cai, and Anirban DasGupta. *Confidence intervals for a binomial proportion and asymptotic expansions.* The Annals of Statistics 30.1 (2002): 160-201.

