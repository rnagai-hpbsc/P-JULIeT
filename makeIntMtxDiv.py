#!/bin/env python3

import click
import numpy as np
import Xsec_libs as xs
from tqdm import tqdm
import sys, os
import matplotlib.pyplot as plt

SERIAL_JOBS = 701
EPSNUM = 3.e-26

@click.group()
@click.option('--mass',type=int,default=150)
@click.option('--smin',default=None)
@click.option('--out',type=str,default='e')
@click.option('--material',type=str,default='rock')
@click.option('--interaction',type=str,required=True)
@click.option('--sdir',type=str,default='test')
@click.option('--eps',default=None)
@click.option('--quadlimit',default=None)
@click.pass_context
def cli(ctx,mass,out,material,interaction,sdir,eps,smin,quadlimit):
    if smin is not None:
        xsobj = xs.XsecCalculator(m_lep=mass,material=material)
        if smin == 'e':
            mass = xsobj.getME()
        elif smin == 'mu':
            mass = xsobj.getMMU()
        elif smin == 'tau':
            mass = xsobj.getMTAU()
    stauObj, interactionName = init(mass,material,out,interaction)
    if eps is not None:
        stauObj.setEps(float(eps))
    if quadlimit is not None:
        stauObj.setQuadlimit(int(quadlimit))
    ctx.ensure_object(dict)
    ctx.obj['mass'] = mass
    ctx.obj['material'] = material
    ctx.obj['stauObj'] = stauObj
    ctx.obj['interactionName'] = interactionName
    ctx.obj['interaction'] = interaction
    ctx.obj['sdir'] = sdir
    filename = f'data/{sdir}/stau{interactionName}Mtx{mass}GeV' if smin is None else f'data/{sdir}/{smin}{interactionName}Mtx'
    os.makedirs(f'data/{sdir}',exist_ok=True)
    ctx.obj['filename'] = filename
    pass

@cli.command()
@click.pass_context
@click.option('-z',is_flag=True)
@click.option('--short',is_flag=True)
def calcDiagonalTrans(ctx,z,short):
    calctype = 'z' if z else 'y'
    stauObj = ctx.obj['stauObj']
    interaction = ctx.obj['interaction']
    loop = range(700) if not short else range(7)
    factor = 1 if not short else 100
    for i in range(700):
        iLogE = int(i*factor)
        element = getElement(iLogE,iLogE,stauObj,calctype,interaction,method='quad')
        print(f'{iLogE}, {element:.10e}')

@cli.command()
@click.option('-i',type=int,required=True)
@click.option('-z',is_flag=True)
@click.option('--verify',is_flag=True)
@click.option('--method',default=None)
@click.pass_context
def transMtxElement(ctx,i,z,verify,method):
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
            if len(words)!=3:
                print(f"WARNING: Wrong value -- {linenumber}:{line}")
                continue
            else:
                transMtx[int(words[0])][int(words[1])] = float(words[2])
        print(f'Processed #line: {linenumber}')
    if os.path.exists(f'{filename}_{filetype}.txt'):
        print('Warning! The output file already exists. Exit.')
        return
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
@click.option('-z',is_flag=True)
@click.option('--verify',is_flag=True, default=False)
@click.option('--wointegral',is_flag=True, default=False)
def fixTransMtx(ctx,z,verify,wointegral):
    filename = ctx.obj['filename']
    filetype = 'surviv' if z else 'trans'
    if not os.path.exists(f'{filename}_{filetype}.txt'):
        print('File not found. Do formTransMtx or make transMtx first. Exit.')
        return
    else:
        if not verify:
            if os.path.exists(f'{filename}_{filetype}.txt_BAK'):
                os.system(f'rsync -av {filename}_{filetype}.txt {filename}_{filetype}.txt_BAK2')
            else:
                os.system(f'rsync -av {filename}_{filetype}.txt {filename}_{filetype}.txt_BAK')
                os.system(f'chmod -w {filename}_{filetype}.txt_BAK')
    stauObj = ctx.obj['stauObj']
    interaction = ctx.obj['interaction']
    calctype = 'z' if z else 'y'
    reconMtx = []
    sumdatalist = []
    elementlist = []
    flag = False
    with open(f'{filename}_{filetype}.txt','r') as f:
        for line in f:
            reconMtx.append(np.array([float(x) for x in line.split(',')]))
    for N in tqdm(range(350*701)):
        i, j = getij(N)
        value = reconMtx[i][j]
        if i>0:
            if verify & (i==j) & (i>500):
                if wointegral:
                    element = value
                else:
                    element = getElement(i,j,stauObj,calctype,interaction,method='quad')
                if calctype=='z':
                    xdata, data = getSimpleZintegration(stauObj,i,j,interaction)
                else:
                    xdata = []
                    data = []
                sumdata = 0
                for k in range(len(xdata)-1):
                    sumdata += (data[k+1]+data[k])*(xdata[k+1]-xdata[k])/2
                tqdm.write(f'Verification Notice: ({i},{j}): {value:.15e}, {element:.15e}, {sumdata:.15e}')
                sumdatalist.append(sumdata)
                elementlist.append(element)
            elif reconMtx[i-1][j-1]*0.95 > reconMtx[i][j]:
                element = getElement(i,j,stauObj,calctype,interaction,method='quad')
                if calctype=='z':
                    xdata, data = getSimpleZintegration(stauObj,i,j,interaction)
                else:
                    xdata = []
                    data = []
                sumdata = 0
                for k in range(len(xdata)-1):
                    sumdata += (data[k+1]+data[k])*(xdata[k+1]-xdata[k])/2
                tqdm.write(f'WARNING: Trans should be constantly increasing... ({i},{j}): {value}, {element}, {sumdata}')
                if (element > reconMtx[i][j-1]*0.95) & (element > reconMtx[i-1][j-1]*0.95):
                    reconMtx[i][j] = element
                    flag = True
                elif sumdata > reconMtx[i][j-1]*0.95:
                    reconMtx[i][j] = sumdata
                    flag = True                
        elif value <= 0.0:
            element = getElement(i,j,stauObj,calctype,interaction,method='quad')
            if element:
                reconMtx[i][j] = element
                tqdm.write(f"WARNING: Wrong value for (i,j) == ({i},{j}), value == {value}, truth == {element}")
                flag = True
            elif calctype=='z':
                xdata, data = getSimpleZintegration(stauObj,i,j,interaction)
                sumdata = 0
                for k in range(len(xdata)-1):
                    sumdata += (data[k+1]+data[k])*(xdata[k+1]-xdata[k])/2
                tqdm.write(f"WARNING: Wrong value for (i,j) == ({i},{j}), value == {value}, truth == {element}, simple == {sumdata}")
                if sumdata > 0:
                    reconMtx[i][j] = sumdata
                    flag = True
    if verify:
        import matplotlib.pyplot as plt
        plt.plot(range(len(sumdatalist)),sumdatalist)
        plt.plot(range(len(elementlist)),elementlist)
        plt.show()
        if flag:
            print('Need to fix matrix elements.\nVerification Done. Exit.')
        return
    
    if flag:
        with open(f'{filename}_{filetype}.txt','w') as f:
            for i,transs in enumerate(reconMtx):
                for j,trans in enumerate(transs):
                    f.write(str(trans))
                    if j+1 < len(transs):
                        f.write(',')
                    elif i+1 < len(reconMtx):
                        f.write('\n')
    return

@cli.command()
@click.pass_context
@click.option('--method',type=str,default='quadLog')
@click.option('--verify',is_flag=True)
def sigmaMtx(ctx,method,verify):
    filename = ctx.obj['filename']
    stauObj = ctx.obj['stauObj']
    interaction = ctx.obj['interaction']
    sigmaMtx = getSigmaArray(stauObj,interaction,method,verify)
    if verify:
        sys.exit()
    with open(f'{filename}_sigma_direct.txt','w') as f:
        for i,sigma in enumerate(sigmaMtx):
            f.write(str(sigma))
            if i+1 < len(sigmaMtx):
                f.write(',')
    return 

@cli.command()
@click.pass_context
@click.option('--method',type=str,default='quad')
@click.option('--verify',is_flag=True)
def sigmaMtxfromtrans(ctx,method,verify):
    filename = ctx.obj['filename']
    stauObj = ctx.obj['stauObj']
    interaction = ctx.obj['interaction']
    transMtx = []
    with open(f'{filename}_surviv.txt','r') as f:
        for line in f:
            transMtx.append([float(x) for x in line.split(',')])
    SumtransMtx = np.sum(np.array(transMtx),axis=1)
    print(SumtransMtx)
    transCutoff = []
    for i in tqdm(range(700)):
        result = getCutoff(i,stauObj,interaction)
        transCutoff.append(result)
    sigmaMtx = np.array(transCutoff) + SumtransMtx
    with open(f'{filename}_sigma_diff.txt','w') as f:
        for i,sigma in enumerate(sigmaMtx):
            f.write(str(sigma))
            if i+1 < len(sigmaMtx):
                f.write(',')
    return 

@cli.command()
@click.pass_context
def cutoffMtx(ctx):
    filename = ctx.obj['filename']
    stauObj = ctx.obj['stauObj']
    interaction = ctx.obj['interaction']
    transCutoff = []
    for i in tqdm(range(700)):
        result = getCutoff(i,stauObj,interaction)
        transCutoff.append(result)
    with open(f'{filename}_sigma_diff_tmp.txt','w') as f:
        for i,sigma in enumerate(transCutoff):
            f.write(str(sigma))
            if i+1 < len(transCutoff):
                f.write(',')

@cli.command()
@click.pass_context
@click.option('--verify',is_flag=True,default=False)
def formsigmaMtx(ctx,verify):
    filename = ctx.obj['filename']
    stauObj = ctx.obj['stauObj']
    interaction = ctx.obj['interaction']
    transMtx = []
    with open(f'{filename}_surviv.txt','r') as f:
        for line in f:
            transMtx.append([float(x) for x in line.split(',')])
    SumtransMtx = np.sum(np.array(transMtx),axis=1)
    transCutoff = []
    with open(f'{filename}_sigma_diff_tmp.txt','r') as f:
        for line in f:
            transCutoff.append([float(x) for x in line.split(',')])
    SumtransCutoff = np.array(transCutoff[0])#np.sum(np.array(transCutoff),axis=1)
    sigmaMtx = SumtransCutoff + SumtransMtx
    if verify:
        print(SumtransMtx, SumtransCutoff, sigmaMtx)
        import matplotlib.pyplot as plt
        plt.plot(np.arange(700),SumtransMtx)
        plt.plot(np.arange(700),SumtransCutoff)
        plt.plot(np.arange(700),sigmaMtx)
        plt.show()
        return
    with open(f'{filename}_sigma_diff.txt','w') as f:
        for i,sigma in enumerate(sigmaMtx):
            f.write(str(sigma))
            if i+1 < len(sigmaMtx):
                f.write(',')

@cli.command()
@click.pass_context
@click.option('--method',type=str,default='quadLog')
@click.option('--verify',is_flag=True)
def inelaMtx(ctx,method,verify):
    filename = ctx.obj['filename']
    stauObj = ctx.obj['stauObj']
    interaction = ctx.obj['interaction']
    inelaMtx = getInelasticityArray(stauObj,interaction,method,verify)
    if verify:
        sys.exit()
    with open(f'{filename}_inela.txt','w') as f:
        for i,inela in enumerate(inelaMtx):
            f.write(str(inela))
            if i+1 < len(inelaMtx):
                f.write(',')
    return 

@cli.command()
@click.pass_context
@click.option('-i',type=int,default=0)
@click.option('-j',type=int,default=0)
@click.option('-z',is_flag=True)
@click.option('--show',is_flag=True)
def verifyTransElement(ctx,i,j,z,show):
    value = -1
    filename = ctx.obj['filename']
    filetype = 'surviv' if z else 'trans'
    with open(f'{filename}_{filetype}.txt','r') as f:
        linenum = 0
        for line in f:
            if linenum==i:
                wordnum = 0
                for word in line.split(','):
                    if wordnum==j:
                        value = float(word)
                        break
                    else:
                        wordnum += 1
            else:
                linenum += 1
    stauObj = ctx.obj['stauObj']
    interaction = ctx.obj['interaction']
    calctype = 'z' if z else 'y'
    element = getElement(i,j,stauObj,calctype,interaction,method='quad',show=show)
    print(value, element)
    return

@cli.command()
@click.pass_context
@click.option('-z',is_flag=True)
def verifyTransMtx(ctx,z):
    filename = ctx.obj['filename']
    filetype = 'surviv' if z else 'trans'
    stauObj = ctx.obj['stauObj']
    interaction = ctx.obj['interaction']
    calctype = 'z' if z else 'y'
    reconMtx = []
    with open(f'{filename}_{filetype}.txt','r') as f:
        for line in f:
            reconMtx.append(np.array([float(x) for x in line.split(',')]))
    for N in range(350*701):
        i, j = getij(N)
        value = reconMtx[i][j]
        if value <= 0.0:
            element = getElement(i,j,stauObj,calctype,interaction,method='quad')
            if value != element:
                print(f"WARNING: Wrong value for (i,j) == ({i},{j}), value == {value}, truth == {element}")
                with open(f"{ctx.obj['filename']}_{filetype}_tmp.txt",'a') as f:
                    f.write(f'{i},{j},{element}\n')
    return

@cli.command()
@click.pass_context
@click.option('-z',is_flag=True)
@click.option('-d',is_flag=True)
@click.option('-j',is_flag=True)
def verifyAll(ctx,z,d,j):
    filename = ctx.obj['filename']
    interaction = ctx.obj['interaction']
    filetype = 'surviv' if z else 'trans'
    ii = 0
    reconMtx = []
    with open(f'{filename}_{filetype}.txt','r') as f:
        for line in f:
            reconMtx.append(np.array([float(x) for x in line.split(',')]))
    sumreconMtx = np.sum(reconMtx,axis=1)

    if j:
        calctype = ''
    elif d:
        calctype = '_direct'
    else:
        calctype = '_diff'

    with open(f'{filename}_sigma{calctype}.txt','r') as f:
        sigma = np.array([float(x) for x in f.read().split(',')])

    for i in range(len(sigma)):
        diff = sigma[i]-sumreconMtx[i]
        if diff<0:
            color = '\033[31m' # red
        else:
            color = '\033[32m' # green
        print(f'{sigma[i]:.10e}, {sumreconMtx[i]:.10e}, diff: {color}{diff:.10e}\033[0m')

    plt.figure()
    xdata = np.arange(len(sigma))
    #plt.plot(xdata,sigma,label='sigma')
    #plt.plot(xdata,sumreconMtx,label='sum dif',ls=':')
    plt.plot(xdata,sigma-sumreconMtx)
    plt.show()
    return

@cli.command()
@click.pass_context
@click.option('-i',type=int,default=0)
@click.option('-j',type=int,default=0)
@click.option('-z',is_flag=True)
@click.option('--method',type=str,default='quad')
@click.option('--show',is_flag=True)
@click.option('--div',default=100)
def singleTransElement(ctx,i,j,z,method,show,div):
    stauObj = ctx.obj['stauObj']
    calctype = 'z' if z else 'y'
    interaction = ctx.obj['interaction']
    element = getElement(i,j,stauObj,calctype,interaction,method,show)
    if not show:
        print(element)
        import matplotlib.pyplot as plt
        E = 10**(5.+0.01*i)
        logY = 0.01*(j-i)
        logYLow = logY - 0.5*0.01
        logYUp  = logY + 0.5*0.01
        ymin, ymax = stauObj.lim_y(E, interaction)
        ZminBound = 10**logYLow if 10**logYLow > 1.-ymax else 1.-ymax
        ZmaxBound = 10**logYUp  if 10**logYUp  < 1.-ymin else 1.-ymin
        print(ZminBound, ZmaxBound, ymin)
        dx = ZmaxBound - ZminBound
        #xdata = [ZminBound + dx*(1-2**(-N)) for N in range(div)]
        #data = np.array([stauObj.getDSigmaDy(1.-z,E,interaction,method='quad') for z in xdata])
        xdata = 10**np.linspace(np.log10((1-ZmaxBound) if (1-ZmaxBound)!=0 else ymin),np.log10(1-ZminBound),div)
        data = np.array([stauObj.getDSigmaDy(z,E,interaction,method='quad') for z in xdata])
        plt.plot(xdata,data)
        #print(xdata, data)
        sumdata = 0
        for k in range(len(xdata)-1):
            #if data[i+1]==data[i]:
            #    continue
            sumdata += (data[k+1]+data[k])*(xdata[k+1]-xdata[k])/2
        print(sumdata)
        plt.yscale('log')
        #plt.xscale('log')
        plt.xlabel('inelasticity y')
        plt.ylabel('Differential Cross Section $\mathrm{d}\sigma/\mathrm{d}y$')
        plt.tight_layout()
        plt.savefig('plot__.pdf')
        plt.show()
        
    return

def getSimpleZintegration(stauObj,i,j,interaction,div=10000):
    E = 10**(5.+0.01*i)
    logY = 0.01*(j-i)
    logYLow = logY - 0.5*0.01
    logYUp  = logY + 0.5*0.01
    ymin, ymax = stauObj.lim_y(E, interaction)
    ZminBound = 10**logYLow if 10**logYLow > 1.-ymax else 1.-ymax
    ZmaxBound = 10**logYUp  if 10**logYUp  < 1.-ymin else 1.-ymin
    #print(ZminBound, ZmaxBound, ymin)
    dx = ZmaxBound - ZminBound
    xdata = 10**np.linspace(np.log10((1-ZmaxBound) if (1-ZmaxBound)!=0 else ymin),np.log10(1-ZminBound),div)
    data = np.array([stauObj.getDSigmaDy(z,E,interaction,method='quad') for z in xdata])
    return xdata, data

@cli.command()
@click.pass_context
@click.option('-i',type=int,default=0)
@click.option('-j',type=int,default=0)
@click.option('--plot',is_flag=True,default=False)
def scanDivforSimpleSum(ctx,i,j,plot):
    stauObj = ctx.obj['stauObj']
    interaction = ctx.obj['interaction']
    element = getElement(i,j,stauObj,'z',interaction,'quad',False)
    diff = []
    step = 200
    divs = (np.arange(100)+1)*step
    for div in divs:
        xdata, data = getSimpleZintegration(stauObj,i,j,interaction,div)
        sumdata = 0
        for k in range(len(xdata)-1):
            sumdata += (data[k+1]+data[k])*(xdata[k+1]-xdata[k])/2
        print(f'{div}, {sumdata:.15e}, {sumdata-element:.15e}')
        diff.append(np.abs(sumdata-element))
    diff = np.array(diff)
    if plot:
        import matplotlib.pyplot as plt
        plt.plot(divs,diff/element*100)
        plt.yscale('log')
        plt.show()
        
@cli.command()
@click.pass_context
@click.option('-i',type=int,default=0)
def showybounds(ctx,i):
    stauObj = ctx.obj['stauObj']
    E = 10**(5.+0.01*i)
    interaction = ctx.obj['interaction']
    print(stauObj.lim_y(E,interaction))

@cli.command()
@click.pass_context
@click.option('-i',type=int,default=350)
@click.option('-j',type=int,default=350)
def verifyElementCalc(ctx,i,j):
    stauObj = ctx.obj['stauObj']
    calctype = 'z' if z else 'y'
    interaction = ctx.obj['interaction']
    element = getElement(i,j,stauObj,calctype,interaction,method='quad')

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

def getSigmaArray(stauObj,interaction,method='quadLog',verify=False):
    logeArray = []
    sigmaMtx = []
    totalnumber = 700 if not verify else 7
    for i in tqdm(range(totalnumber)):
        logE = i if not verify else 100*i
        E = 10**(5+0.01*logE)
        logeArray.append(logE)
        sigma = stauObj.getSigma(E,interaction,method)
        sigmaMtx.append(sigma)
    print(logeArray[::10] if not verify else logeArray)
    print(sigmaMtx[::10] if not verify else sigmaMtx)
    return sigmaMtx

def getInelasticityArray(stauObj,interaction,method='quadLog',verify=False):
    logeArray = []
    inelaMtx = []
    massnumber = np.sum(stauObj.A * stauObj.natoms)
    totalnumber = 700 if not verify else 7
    for i in tqdm(range(totalnumber)):
        logE = i if not verify else 100*i
        E = 10**(5+0.01*logE)
        logeArray.append(logE)
        inela = stauObj.getEnergyLossRaw(E,interaction,method)
        print(inela)
        inelaMtx.append(inela)
    print(logeArray[::10] if not verify else logeArray)
    print(inelaMtx[::10] if not verify else stauObj.NA/massnumber*np.array(inelaMtx))
    return inelaMtx

def getElement(i,j,stauObj,YZ,interactionName,method='quad',show=False):
    E = 10**(5.+0.01*i)
    if j > i:
        return 0.
    logY = 0.01*(j-i)
    logYLow = logY - 0.5*0.01
    logYUp  = logY + 0.5*0.01
    if YZ == 'y':
        result = stauObj.getPartialSigma(logYLow,logYUp,E,interactionName,method=method,show=show)
    elif YZ == 'z':
        result = stauObj.getSurviveProb(10**logYLow,10**logYUp,E,interactionName,method=method,show=show)
    else:
        print('invalid')
        result = 0.
    return result

@cli.command()
@click.pass_context
@click.option('-i',type=int,default=350)
def sigmacutoff(ctx,i):
    stauObj = ctx.obj['stauObj']
    interaction = ctx.obj['interaction']
    result = getCutoff(i,stauObj,interaction,show=True)
    return

def getCutoff(i,stauObj,interactionName,method='quad',show=False):
    E = 10**(5.+0.01*i)
    logYLow = 0.01*(-i) - 0.5*0.01
    YLow = 10**logYLow
    Ymin = 0.0
    result = stauObj.getSurviveProb(Ymin,YLow,E,interactionName,method=method,show=show)
    return result

if __name__ == '__main__':
    cli()
