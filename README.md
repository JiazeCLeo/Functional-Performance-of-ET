# Functional-Performance-of-ET

This project aims to apply four causal discovery methods to time series data in order to analyse the functional performance of an evapotranspiration system. The file [Four Causal Discovery Methods Demo.py](./Four%20Causal%20Discovery%20Methods%20Demo.py) involves an artificial time series dataset. The mathematical equations are shown here:
```
x1(t) = x1(t-1) * (3.9 - 3.9 x1(t-1)) + ξ1
x2(t) = x2(t-1) * (3.6 - 0.4 x1(t-1) - 3.6 x2(t-1)) + ξ2
x3(t) = x3(t-1) * (3.6 - 0.4 x2(t-1) - 3.6 x3(t-1)) + ξ3
x4(t) = x4(t-1) * (3.8 - 0.35 x3(t-1) - 3.8 x4(t-1)) + ξ4
x5(t) = x5(t-1) * (3.2 - 0.4 x2(t-1) - 3.6 x5(t-1)) + ξ5
```
We provided 
