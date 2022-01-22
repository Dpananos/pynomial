import pandas as pd
from pynomial.intervals import agresti_coull, bayes, exact
import pandas as pd
import numpy as np


binom_results = pd.read_csv("pynomial/tests/binom_output.csv")
binom_methods = binom_results.groupby(['method'])

results = binom_methods.get_group('exact')
x = results.x.values
n = results.n.values
conf = results.conf.values

target_output = results.loc[:, ['mean', 'lower','upper']].rename(columns={'mean':'estimate'}).reset_index(drop=True)
pynomial_output = bayes(x=x, n=n, conf=conf).reset_index(drop=True)


diff = target_output.values - pynomial_output.values
print(diff.max())