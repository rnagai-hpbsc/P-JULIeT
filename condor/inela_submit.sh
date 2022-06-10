#!/bin/sh

mkdir -p condor/log/ver${3}

sed -e "s/PARTICLE/${1}/" -e "s/MASS/${2}/" -e "s/TODAYSDATE/${3}/" -e "s/INTERACTION/${4}/" condor/makeInela.sdf > condor/makeInela_tmp.sdf
condor_submit condor/makeInela_tmp.sdf
rm -v condor/makeInela_tmp.sdf
