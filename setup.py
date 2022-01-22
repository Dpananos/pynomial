from setuptools import setup

setup(
    name='pynomial',
    version='0.0.1',    
    description='A python package for binomial confidence intervals',
    url='https://github.com/Dpananos/pynomial',
    author='Demetri Panano',
    author_email='dpananos@gmail.com',
    license='MIT',
    packages=['pynomial'],
    install_requires=['scipy>=1.0','numpy', 'pandas']
)
