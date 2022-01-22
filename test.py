import matplotlib.pyplot as plt
import numpy as np
from pynomial.intervals import agresti_coull, asymptotic, bayes, wilson, loglog, exact
from scipy.stats import binom
import pandas as pd
from scipy.special import expit, logit

x=10
n = 20
iv = exact(x, n)

print(iv)