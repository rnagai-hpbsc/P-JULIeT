#!/bin/env python3

import click
import numpy as np
import Xsec_libs as xs
from tqdm import tqdm
import sys, os
from multiprocessing import Pool
import itertools

@click.command()
@click.option('--mass',type=int,default=150)
@click.option('--out',type=str,default='e')
@click.option('--material',type=str,default='rock')
@click.option('--interaction',type=str,required=True)
@click.option('--sdir',type=str,default='test')
@click.option('--nprocess',type=int,default=10)
@click.option('--verify',is_flag=True)
def main(mass,out,material,interaction,sdir,nprocess,verify):
    stauObj = xs.XsecCalculator(m_lep=mass,material=material)

    if out=='e':
        stauObj.setMout('e')
    elif out=='mu':
        stauObj.setMout('mu')
    elif out=='tau':
        stauObj.setMout('tau')

    if interaction=='pairc':
        interactionName = f'To{out}PairCreation'
    elif interaction=='brems':
        interactionName = 'Bremsstrahlung'
    elif interaction=='phnuc':
        interactionName = 'PhotoNuclear'
    else:
        print("Invalid interaction. Set pairc, brems, or phnuc.")
        sys.exit()

    filename = f'data/{sdir}/stau{interactionName}Mtx{mass}GeV'
    os.makedirs(f'data/{sdir}',exist_ok=True)
    
    transMtx, survivMtx = getTransferArrayMultiprocess(stauObj,nprocess,interaction)
    sigmaMtx = getSigmaArray(stauObj,interaction)
    inelaMtx = getInelasticityArray(stauObj,interaction)


    if verify:
        for i in range(700):
            print(f'{np.sum(transMtx[i]):.5e}, {sigmaMtx[i]:.5e}, {np.sum(transMtx[i])<sigmaMtx[i]}')
        print(len(transMtx))
        return 

    with open(f'{filename}_sigma.txt','w') as f:
        for i,sigma in enumerate(sigmaMtx):
            f.write(str(sigma))
            if i+1 < len(sigmaMtx):
                f.write(',')
    with open(f'{filename}_inela.txt','w') as f:
        for i,inela in enumerate(inelaMtx):
            f.write(str(inela))
            if i+1 < len(inelaMtx):
                f.write(',')
    with open(f'{filename}_trans.txt','w') as f:
        for i,transs in enumerate(transMtx):
            for j,trans in enumerate(transs):
                f.write(str(trans))
                if j+1 < len(transs):
                    f.write(',')
                elif i+1 < len(transMtx):
                    f.write('\n')
    with open(f'{filename}_surviv.txt','w') as f:
        for i,survivs in enumerate(survivMtx):
            for j,surviv in enumerate(survivs):
                f.write(str(surviv))
                if j+1 < len(survivs):
                    f.write(',')
                elif i+1 < len(survivMtx):
                    f.write('\n')

def getSigmaArray(stauObj,interaction):
    sigmaMtx = []
    for logE in tqdm(range(700)):
        E = 10**(5+0.01*logE)
        method = 'quad' if interaction == 'phnuc' else 'quadLog'
        sigma = stauObj.getSigma(E,interaction,method)
        sigmaMtx.append(sigma)
    print(sigmaMtx[::10])
    return sigmaMtx

def getInelasticityArray(stauObj,interaction):
    inelaMtx = []
    for logE in tqdm(range(700)):
        E = 10**(5+0.01*logE)
        method = 'quad' if interaction == 'phnuc' else 'quadLog'
        inela = stauObj.getEnergyLossRaw(E,interaction,method)
        inelaMtx.append(inela)
    print(inelaMtx[::10])
    return inelaMtx

def ijCalc(args):
    i,j,stauObj,YZ,interactionName = args
    E = 10**(5+0.01*i)
    if j > i:
        return i, j, 0
    logY = 0.01*(j-i)
    logYLow = logY - 0.5*0.01
    logYUp  = logY + 0.5*0.01
    result = 0
    if YZ == 'y':
        result = stauObj.getPartialSigma(logYLow,logYUp,E,interactionName,method='quad')
    elif YZ == 'z':
        result = stauObj.getSurviveProb(10**logYLow,10**logYUp,E,interactionName,method='quad')
    else:
        pass
    return i,j,result

def getTransferArrayMultiprocess(stauObj,nprocess,interactionName):
    transferMtx = np.zeros((700,700))
    transferAMtx = np.zeros((700,700))
    ii = range(700)
    jj = range(700)
    ijtaple = list(itertools.product(ii,jj))
    yargstaples = []
    zargstaples = []
    for t in ijtaple:
        yargstaples.append(t+(stauObj,'y',interactionName))
        zargstaples.append(t+(stauObj,'z',interactionName))
    with Pool(nprocess) as p:
        with tqdm(total=len(yargstaples)) as pbar:
            for result in p.imap_unordered(ijCalc,yargstaples):
                (i, j, value) = result
                transferMtx[i][j] = value
                pbar.update()
    with Pool(nprocess) as p:
        with tqdm(total=len(zargstaples)) as pbar:
            for result in p.imap_unordered(ijCalc,zargstaples):
                (i, j, value) = result
                transferAMtx[i][j] = value
                pbar.update()
    print(transferMtx[350], transferAMtx[350])
    return transferMtx, transferAMtx

if __name__ == '__main__':
    main()
