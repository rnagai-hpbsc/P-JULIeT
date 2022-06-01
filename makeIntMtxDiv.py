#!/bin/env python3

import click
import numpy as np
import Xsec_libs as xs
from tqdm import tqdm
import sys, os
from multiprocessing import Pool
import itertools

@click.group()
@click.option('--mass',type=int,default=150)
@click.option('--out',type=str,default='e')
@click.option('--material',type=str,default='rock')
@click.option('--interaction',type=str,required=True)
@click.option('--sdir',type=str,default='test')
@click.pass_context
def cli(ctx,mass,out,material,interaction,sdir):
    stauObj, interactionName = init(mass,material,out,interaction)
    ctx.ensure_object(dict)
    ctx.obj['mass'] = mass
    ctx.obj['material'] = material
    ctx.obj['stauObj'] = stauObj
    ctx.obj['interactionName'] = interactionName
    ctx.obj['interaction'] = interaction
    ctx.obj['sdir'] = sdir
    filename = f'data/{sdir}/stau{interactionName}Mtx{mass}GeV'
    os.makedirs(f'data/{sdir}',exist_ok=True)
    ctx.obj['filename'] = filename
    pass

@cli.command()
@click.pass_context
@click.option('-z',is_flag=True)
def calcDiagonalTrans(ctx,z):
    calctype = 'z' if z else 'y'
    stauObj = ctx.obj['stauObj']
    interaction = ctx.obj['interaction']
    for iLogE in range(700):
        element = getElement(iLogE,iLogE,stauObj,calctype,interaction,method='quad')
        print(f'{iLogE}, {element:.10e}')

@cli.command()
@click.option('-i',type=int,required=True)
@click.option('-z',is_flag=True)
@click.option('--verify',is_flag=True)
@click.option('--method',default=None)
@click.pass_context
def transMtxElement(ctx,i,z,verify,method):
    SERIAL_JOBS = 25
    calctype = 'z' if z else 'y'
    filetype = 'surviv' if z else 'trans'
    iLogEs = []
    jLogEs = []
    elements = []
    intmethod = 'quad'
    if i>=int(350*701/SERIAL_JOBS):
        print(f"Invalid iteraction number: {i}")
        sys.exit()
    for n in range(SERIAL_JOBS*i,SERIAL_JOBS*i+SERIAL_JOBS):
        iLogE, jLogE = getij(n)
        stauObj = ctx.obj['stauObj']
        interaction = ctx.obj['interaction']
        if method is not None:
            intmethod = method
        element = getElement(iLogE,jLogE,stauObj,calctype,interaction,method=intmethod)
        print(i, iLogE, jLogE, element)
        iLogEs.append(iLogE)
        jLogEs.append(jLogE)
        elements.append(element)

    if verify:
        print('Close without storing txt.')
        return 

    with open(f"{ctx.obj['filename']}_{filetype}_tmp.txt",'a') as f:
        for n in range(SERIAL_JOBS):
            f.write(f'{iLogEs[n]},{jLogEs[n]},{elements[n]}\n')
    return 

def getij(n):
    N = n+1
    i = 1
    j = N
    while (j > int(.5*i*(i+1))): 
        i += 1
    j = int(N - .5*(i-1)*i)
    return i-1, j-1

@cli.command()
@click.pass_context
@click.option('-z',is_flag=True)
def formTransMtx(ctx,z):
    filename = ctx.obj['filename']
    transMtx = np.zeros((700,700))
    filetype = 'surviv' if z else 'trans'
    with open(f'{filename}_{filetype}_tmp.txt','r') as f:
        linenumber = 0
        for line in f:
            linenumber+=1
            words = line.split(',')
            if len(words)<3: 
                continue
            else:
                transMtx[int(words[0])][int(words[1])] = float(words[2])
        print(f'Processed #line: {linenumber}')
    with open(f'{filename}_{filetype}.txt','w') as f:
        for i,transs in enumerate(transMtx):
            for j,trans in enumerate(transs):
                f.write(str(trans))
                if j+1 < len(transs):
                    f.write(',')
                elif i+1 < len(transMtx):
                    f.write('\n')
    return 

@cli.command()
@click.pass_context
def sigmaMtx(ctx):
    filename = ctx.obj['filename']
    stauObj = ctx.obj['stauObj']
    interaction = ctx.obj['interaction']
    sigmaMtx = getSigmaArray(stauObj,interaction)
    with open(f'{filename}_sigma.txt','w') as f:
        for i,sigma in enumerate(sigmaMtx):
            f.write(str(sigma))
            if i+1 < len(sigmaMtx):
                f.write(',')
    return 

@cli.command()
@click.pass_context
def inelaMtx(ctx):
    filename = ctx.obj['filename']
    stauObj = ctx.obj['stauObj']
    interaction = ctx.obj['interaction']
    inelaMtx = getInelasticityArray(stauObj,interaction)
    with open(f'{filename}_inela.txt','w') as f:
        for i,inela in enumerate(inelaMtx):
            f.write(str(inela))
            if i+1 < len(inelaMtx):
                f.write(',')
    return 

def init(mass, material, out, interaction):
    stauObj = xs.XsecCalculator(m_lep=mass,material=material)
    if out=='e':
        stauObj.setMout('e')
    elif out=='mu':
        stauObj.setMout('mu')
    elif out=='tau':
        stauObj.setMout('tau')

    if interaction == 'pairc':
        interactionName = f'To{out}PairCreation'
    elif interaction == 'brems':
        interactionName = f'Bremsstrahlung'
    elif interaction == 'phnuc':
        interactionName = f'PhotoNuclear'
    else:
        print('Invalid interaction. Set pairc, brems, or phnuc.')
        sys.exit()
    return stauObj, interactionName

def getSigmaArray(stauObj,interaction):
    sigmaMtx = []
    for logE in tqdm(range(700)):
        E = 10**(5+0.01*logE)
        sigma = stauObj.getSigma(E,interaction,'quadLog')
        sigmaMtx.append(sigma)
    print(sigmaMtx[::10])
    return sigmaMtx

def getInelasticityArray(stauObj,interaction):
    inelaMtx = []
    for logE in tqdm(range(700)):
        E = 10**(5+0.01*logE)
        inela = stauObj.getEnergyLossRaw(E,interaction,'quadLog')
        inelaMtx.append(inela)
    print(inelaMtx[::10])
    return inelaMtx

def getElement(i,j,stauObj,YZ,interactionName,method='quad'):
    E = 10**(5.+0.01*i)
    if j > i:
        return 0.
    logY = 0.01*(j-i)
    logYLow = logY - 0.5*0.01
    logYUp  = logY + 0.5*0.01
    if YZ == 'y':
        result = stauObj.getPartialSigma(logYLow,logYUp,E,interactionName,method=method)
    elif YZ == 'z':
        result = stauObj.getSurviveProb(10**logYLow,10**logYUp,E,interactionName,method=method)
    else:
        print('invalid')
        result = 0.
    return result

if __name__ == '__main__':
    cli()
