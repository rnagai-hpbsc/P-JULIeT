# P-JULIeT

A generator of the lepton interaction matrices for JULIeT UHE Neutrino Simulator (the Java-based Ultra-high energy Lepton Integral Transporter). 
While the original matrix generator, which is available in JULIeT package, is of course written in Java, the package provides the Python implementation. 
The advantage of using this package can be the computation time. 
We can use the batch job system like "condor" and ~100 times faster than the original single core JULIeT calculation. 

To run the calculation, we provide two python scripts:
- `makeIntMtx.py`
- `makeIntMtxDiv.py`

The first one is for the local jobs, the latter is for the condor jobs. 
