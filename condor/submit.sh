#!/bin/sh

mkdir -p condor/log/ver${3}

sed -e "s/PARTICLE/${1}/" -e "s/MASS/${2}/" -e "s/TODAYSDATE/${3}/" condor/makePairSurviv.sdf > condor/makePairSurviv_tmp.sdf
condor_submit condor/makePairSurviv_tmp.sdf
rm -v condor/makePairSurviv_tmp.sdf

sed -e "s/PARTICLE/${1}/" -e "s/MASS/${2}/" -e "s/TODAYSDATE/${3}/" condor/makePairTrans.sdf > condor/makePairTrans_tmp.sdf
condor_submit condor/makePairTrans_tmp.sdf
rm -v condor/makePairTrans_tmp.sdf
