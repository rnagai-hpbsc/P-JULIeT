#!/bin/sh

mkdir -p condor/log/ver${3}

sed -e "s/PARTICLE/${1}/" -e "s/MASS/${2}/" -e "s/TODAYSDATE/${3}/" -e "s/INTERACTION/${4}/" condor/makeSigmaDiff.sdf > condor/makeSigmaDiff_tmp.sdf
condor_submit condor/makeSigmaDiff_tmp.sdf
rm -v condor/makeSigmaDiff_tmp.sdf
