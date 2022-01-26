Binomial Confidence intervals
=============================

Many confidence intervals exist for the binomial risk parameter.  Below is information on each interval implemented.  For all scenarios, let :math:`x` be the number of observed successes, :math:`n` be the number of trials, :math:`p=x/n` be the estimated risk, and :math:`z` be the :math:`1-\alpha` quantile of a standard normal corresponding to the desired level of confidence.


Agresti-Coull
~~~~~~~~~~~~~~

The Agresti-Coull interval defines

    .. math::
        \tilde{n}=n+z^{2}

and

    .. math::
        \tilde{p}=\frac{1}{\widetilde{n}}\left(x+\frac{z^{2}}{2}\right)

The Agresti-Coull interval is then

    .. math::
        \tilde{p} \pm z \sqrt{\tilde{p}(1-\tilde{p}) \over \tilde{n}}

References:

1. Agresti, Alan, and Brent A. Coull. “Approximate Is Better than ‘Exact’ for Interval Estimation of Binomial Proportions.” The American Statistician, vol. 52, no. 2, [American Statistical Association, Taylor & Francis, Ltd.], 1998, pp. 119–26, https://doi.org/10.2307/2685469.


Asymptotic
~~~~~~~~~~

Large sample theory says that as :math:`n \to \infty` then

    .. math::
        p \overset{d}{\sim} \mathcal{N}\left(\pi, {\pi(1-\pi)\over n}\right)

Here, :math:`\pi` is the true value of the risk.  From this fact, together with use of Slutsky's convergence theorem, the asymptotic confidence interval can be derived as

    .. math::
        p \pm z \sqrt{p(1-p) \over n}

References:

1. Lachin, John M. Biostatistical methods: the assessment of relative risks. Vol. 509. John Wiley & Sons, 2009.


Bayes
~~~~~

A beta distirbution is a conjugate prior for the binomial likelihood, allowing for the posterior distirbution to be written analytically.  Using the prior

    .. math::
        P(p) = \operatorname{Beta}(a, b)

for the binomial risk parameter, the posterior of :math:`p` given :math:`x` successes and :math:`n` heads  is

    .. math::
        P(p \vert x, n ) = \operatorname{Beta}(a + x, b + n - x)


Due to conjugacy, :math:`a` can be considered as the pseudocount of successes and :math:`b` the pseudocount of failures. The posterior mean is estimated as :math:`(a+x)/(a+b+n)`. The confidence interval is obtained by computing the :math:`\alpha` and :math:`1-\alpha` quantiles of the posterior.

References:

1. Svensén, Markus, and Christopher M. Bishop. "Pattern recognition and machine learning." (2007).


Complimentary Log-Log
~~~~~~~~~~~~~~~~~~~~~~

Applying the function

     .. math::
         \theta = g(p) = \log(-\log(p))

to the risk, a confidence interval on the log-log scale is

     .. math::
         \left(\theta_{L}, \theta_{U} \right) = \theta \pm z \sqrt{\frac{(1-p)}{n p(\log p)^{2}}}

where the delta method has been used to obtain the variance of :math:`\theta`. Applying :math:`g^{-1}` to each endpoint yields a confidence interval on the original scale.

    .. math::
        \left(p_L , p_R \right)= \left(\exp \left[ -\exp \left(\theta_{U} \right) \right], \exp \left[ -\exp \left(\theta_{L} \right) \right] \right)


References:

1. Lachin, John M. Biostatistical methods: the assessment of relative risks. Vol. 509. John Wiley & Sons, 2009.


Exact (Clopper-Pearson)
~~~~~~~~~~~~~~~~~~~~~~~

One approach to deriving a confidence interval for the binomial risk parameter is to solve the equation

    .. math::
        \sum_{k=0}^{k=x} \operatorname{Binomial}(k; \pi, n) = \alpha/2

for :math:`\pi` to obtain a lower interval, and the equation

    .. math::
        \sum_{k=a}^{k=n} \operatorname{Binomial}(k; \pi, n) = \alpha/2

for :math:`\pi` to obtain an upper interval.  It can be shown that the soluition to these equations can be written interms of the beta distribution, namely

    .. math::
        \operatorname{Beta}\left(\frac{\alpha}{2} ; x, n-x+1\right)<p<\operatorname{Beta}\left(1-\frac{\alpha}{2} ; x+1, n-x\right)


References:

1. Lachin, John M. Biostatistical methods: the assessment of relative risks. Vol. 509. John Wiley & Sons, 2009.


Logit
~~~~~

The logit interval uses the function

    .. math::
        g(p) = \log\left( 1 \ \over 1-p \right)

to constrain interval boundaries to be within (0, 1).  The confidence interval on the logit scale is 

    .. math::
        (\theta_L, \theta_U) = \log\left(1  \ \over 1-p \right) \pm z\sqrt{1 \over np(1-p)}

where the delta method has been used to obtain the variance of :math:`theta`. Appling :math:`g^{-1}` to each limit yields a confidence interval on the original scale

    .. math::
        (p_L, p_U) = \left( {1 \over 1 + e^{-\theta_L}}, {1 \over 1 + e^{-\theta_U}} \right)

    
References:

1. Lachin, John M. Biostatistical methods: the assessment of relative risks. Vol. 509. John Wiley & Sons, 2009.


LRT
~~~

 The LRT for a binomial risk parameter is

    .. math::
        \ell( \hat{\theta}, \theta^\star) =  -2 \left( x\log\left( \hat{\theta} \over \theta^{\star} \right) + (n-x)\log\left( (1-\hat{\theta}) \over (1-{\theta}^\star)  \right) \right)

Here, :math:`\hat{\theta}` is the estimated risk and :math:`\theta^{\star}` is the risk under the null hypothesis.  To create confidence intervals, this test is inverted to solve for the two roots of the equation

    .. math::
        \ell( \hat{\theta}, \theta^\star) - \chi^2_{1-\alpha} = 0

Where  :math:`\chi^2_{1-\alpha}` is the critical value for the LRT.

Wilson
~~~~~~

The asymptotic z test for the risk parameter is

    .. math::

        Z = \dfrac{p - \pi_0}{\sqrt{{p(1-p) \over n}}} 


Squaring both sides and solving for :math:`\pi_0` yields two solutions which are the endpoints of the Wilson interval

    .. math::

        (p_L, p_U) = \dfrac{{z \over 2n} + p \pm \sqrt{{z \over 4n}\left( {{z \over n} + 4p(1-p)} \right)}}{{z\over n }+1}

References:

1. Lachin, John M. Biostatistical methods: the assessment of relative risks. Vol. 509. John Wiley & Sons, 2009.