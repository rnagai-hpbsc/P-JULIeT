import os
import click

@click.group()
def cli():
    pass

@cli.command()
@click.option('-p',type=str,required=True,help='Out-particle: e, mu, tau')
@click.option('-m',required=True,help='Mass setting, <int> or <str> (e, mu, tau)')
@click.option('-d',type=str,required=True,help="Directory setting, usually put today's date")
#@click.option('-i',type=str,required=True,help='Interaction name: pairc, brems, phnuc')
def maketrans(p,m,d):
    os.makedirs(f'condor/log/ver{d}',exist_ok=True)
    condorsubmit('PairSurviv',p,m,d,'pairc')
    condorsubmit('PairTrans',p,m,d,'pairc')

@cli.command()
@click.option('-p',type=str,required=True,help='Out-particle: e, mu, tau')
@click.option('-m',required=True,help='Mass setting, <int> or <str> (e, mu, tau)')
@click.option('-d',type=str,required=True,help="Directory setting, usually put today's date")
@click.option('-i',type=str,required=True,help='Interaction name: pairc, brems, and phnuc')
def makesigma(p,m,d,i):
    os.makedirs(f'condor/log/ver{d}',exist_ok=True)
    condorsubmit('SigmaDiff',p,m,d,i)
    condorsubmit('Inela',p,m,d,i)
    condorsubmit('SigmaDirect',p,m,d,i)

def condorsubmit(filetype,p,m,d,i):
    with open(f'condor/make{filetype}.sdf','r') as f:
        data = f.readlines()
    if isinstance(m,str):
        mass = 'smin'
    else:
        mass = 'mass'
    data_replace = [line.replace('PARTICLE',p).replace('MASS',m).replace('TODAYSDATE',d).replace('mass',mass).replace('INTERACTION',i) for line in data]
    with open(f'condor/make{filetype}_tmp.sdf','w') as f:
        f.writelines(data_replace)
    os.system(f'condor_submit condor/make{filetype}_tmp.sdf')
    os.system(f'rm -v condor/make{filetype}_tmp.sdf')

if __name__ == '__main__':
    cli()
