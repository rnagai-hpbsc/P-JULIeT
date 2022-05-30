import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import quad

###########################
#                         #
# Xsec Calculator ver 0.1 #
#                         #
###########################

#
# This provides cross-section calculation package 
# for the lepton pair-propagation, bremsstrahlung, 
# and photo-nuclear interactions. 
#
# This can be a substitute to the JULIeT interaction 
# cross-section calculation codes. 
#

class XsecCalculator:
    
    M_E   = 0.511e-3 #GeV
    M_MU  = 105.658e-3 #GeV
    M_TAU = 1.7768 #GeV

    sqrtE = np.sqrt(np.e) # sqrt of Napier's Constant
    NA = 6.02214e23 # Avogadro Constant 
    alpha = 1./137. # Fine Structure Constant 
    lambda_e = 2.42431023867e-12 / 2 / np.pi * 1e2 #3.8616e-11 #cm : reduced compton wavelength of the electron 

    def __init__(self, m_lep=M_MU, m_ele=M_E, material='rock', n=1): 
        self.m_lep = m_lep
        self.m_ele = m_ele
        self.n = n
        self.accrho = False
        self.quadlimit = 200
        if material=='rock':
            self.Z = np.array([11.])
            self.A = np.array([22.])
            self.R = np.array([189.])
            self.natoms = np.array([1])
        elif material=='ice':
            self.Z = np.array([8.,1.])
            self.A = np.array([15.9994,1.00794])
            self.R = np.array([173.4,202.4])
            self.natoms = np.array([1,2])
        else: 
            print('Warning: Undefined material seting... We set all to 1.')
            self.Z = np.array([1.])
            self.A = np.array([1.])
            self.R = np.array([1.])
            self.natoms = np.array([1])

    def setMout(self, value):
        if value=="mu":
            self.m_ele = self.M_MU
        elif value=="tau":
            self.m_ele = self.M_TAU
        elif type(value)=="int":
            self.m_ele = value
        else:
            self.m_ele = self.M_E
    
    def getME(self):
        return self.M_E

    def getMMU(self):
        return self.M_MU

    def getMTAU(self):
        return self.M_TAU

    def getMout(self):
        return self.m_ele

    def setQuadlimit(self, value):
        self.quadlimit = value

    def setAccRho(self): 
        self.accrho = True

    def getZ3(self):
        return self.Z**(1./3.)

    def getMassRatio(self):
        return self.m_lep / self.m_ele
    
    def lim_rho(self, y, E_lep): 
        if self.accrho is False: 
            return self.lim_rho_raw(y, E_lep)
        else: 
            return self.lim_rho_Ey(y*E_lep, E_lep)
    
    def lim_rho_raw(self, y, E_lep):
        rho_max = 0.
        if 1.-4.*self.m_ele/E_lep/y > 0:
            rho_max = (1.-6.*self.m_lep**2./E_lep**2./(1.-y))*np.sqrt(1.-4.*self.m_ele/E_lep/y)
        return rho_max

    def lim_rho_Ey(self, Ey, E_lep):
        rho_max = 0.
        if 1.-4.*self.m_ele/Ey > 0:
            rho_max = (1.-6.*self.m_lep**2/E_lep/(E_lep-Ey))*np.sqrt(1.-4.*self.m_ele/Ey)
        return rho_max
    
    def lim_y(self, E_lep, interaction): 
        if interaction=='pairc':
            return self.limy_pc(E_lep)
        elif interaction=='brems':
            return self.limy_br(E_lep)
        elif interaction=='phnuc':
            return self.limy_pn(E_lep)
        else:
            print("Warning: invalid interaction. Set [0,1].")
            return 0., 1.

    def limy_pc(self, E_lep):
        ymin = 4. * self.m_ele / E_lep
        ymax = np.min(1. - .75 * self.m_lep / E_lep * self.sqrtE * self.getZ3())
        return ymin, ymax

    def limy_br(self, E_lep):
        ymin = 1e-100
        ymax = np.min(1. - .75 * self.m_lep / E_lep * self.sqrtE * self.getZ3())
        return ymin, ymax 

    def limy_pn(self, E_lep):
        ymin = 1e-100
        ymax = 1.
        return ymin, ymax

    #------------------------
    # Lepton Pair Production 
    #------------------------
    def getBeta(self, y): 
        return y**2 / (2.*(1.-y))
    
    def getXi(self, rho, y): 
        return 0.5*self.getMassRatio()**2 * self.getBeta(y) * (1-rho**2)
    
    def getGamma(self, rho, y, E_lep):
        return self.R * 2 * self.m_ele * (1 + self.getXi(rho, y)) / E_lep / y / (1 - rho**2) / self.getZ3()

    def getYe(self, rho, y): 
        return (5. - rho**2 + 4.*self.getBeta(y)*(1.+rho**2) ) / (2.*(1.+3.*self.getBeta(y))*np.log(3.+1./self.getXi(rho,y)) - rho**2 - 2.*self.getBeta(y)*(2.-rho**2) ) 
    
    def getYl(self, rho, y): 
        return (4. + rho**2 + 3.*self.getBeta(y)*(1.+rho**2) ) / ((1.+rho**2)*(1.5+2.*self.getBeta(y))*np.log(3+self.getXi(rho,y))+1.-1.5*rho**2)

    def getLe(self, rho, y, E_lep): 
        return np.log(self.R/self.getZ3()*np.sqrt((1.+self.getXi(rho,y))*(1.+self.getYe(rho,y))) \
                / (1.+self.getGamma(rho,y,E_lep)*self.sqrtE*(1.+self.getYe(rho,y))) ) \
                - 0.5*np.log1p(2.25 * self.getZ3()**2 / self.getMassRatio()**2 * (1.+self.getXi(rho,y)) * (1.+self.getYe(rho,y)))
    
    def getLeDiv(self, rho, y, E_lep):
        return np.log(self.R/self.getZ3()*np.sqrt((1.+self.getXi(rho,y))*(1.+self.getYe(rho,y)))) \
                - np.log1p(2.*self.m_ele*self.sqrtE/self.getZ3()*(1.+self.getXi(rho,y))*(1.+self.getYe(rho,y)) / (E_lep*y*(1.-rho**2))) \
                - 0.5 * np.log1p(2.25 * self.getZ3()**2 / self.getMassRatio()**2 * (1.+self.getXi(rho,y)) * (1.+self.getYe(rho,y))) 

    def getPhie(self, rho, y, E_lep): 
        return (((2.+rho**2)*(1.+self.getBeta(y))+self.getXi(rho,y)*(3.+rho**2))*np.log1p(1./self.getXi(rho,y)) + (1.-rho**2-self.getBeta(y)) / (1.+self.getXi(rho,y)) - (3.+rho**2)) \
                *self.getLe(rho,y,E_lep)

    def getLl(self, rho, y, E_lep):
        return np.log(self.R / self.getZ3()**2 / 1.5 * self.getMassRatio() / (1.+self.getGamma(rho,y,E_lep)*self.sqrtE*(1.+self.getYl(rho,y)) ))

    def getPhil(self, rho, y, E_lep):
        return (((1.+rho**2)*(1.+1.5*self.getBeta(y)) - (1.+2.*self.getBeta(y)) * (1.-rho**2)/self.getXi(rho,y)) * np.log1p(self.getXi(rho,y)) \
                + self.getXi(rho,y) * (1.-rho**2-self.getBeta(y))/(1.+self.getXi(rho,y)) + (1.+2.*self.getBeta(y)) * (1.-rho**2)) * self.getLl(rho,y,E_lep)

    def getFactor(self, E_lep): 
        return self.alpha**4 / 1.5 / np.pi * self.Z * (self.Z+self.getScreenFactor(E_lep)) * self.natoms * (self.lambda_e*self.M_E/self.m_ele)**2

    def getScreenFactor(self, E_lep):
        factor = []
        for z in self.Z:
            if E_lep <= 35.*self.m_lep:
                factor.append(0.)
                continue
            g1 = 4.4e-5 if z==1. else 1.95e-5
            g2 = 4.8e-5 if z==1. else 5.30e-5
            screenFactor = ((0.073*(np.log(E_lep/self.m_lep)-np.log1p(g1*z**(2./3.)*E_lep/self.m_lep))-0.26)/\
                            (0.058*(np.log(E_lep/self.m_lep)-np.log1p(g2*z**(1./3.)*E_lep/self.m_lep))-0.14))
            factor.append(screenFactor if screenFactor>0 else 0.)
        return np.array(factor)

    def getAsymTerm(self, rho, y, E_lep, ith):
        return (self.getPhie(rho,y,E_lep) + (self.m_ele/self.m_lep)**2 * self.getPhil(rho,y,E_lep))[ith]

    def getIntAsymTerm(self, y, E_lep, ith, method='quad'):
        rho_max = self.lim_rho(y,E_lep)
        if method == 'quad':
            result = quad(self.getAsymTerm, -rho_max, rho_max, args=(y,E_lep,ith),limit=self.quadlimit)[0]
            return result if result>0. else 0.
        else: 
            print('Invalid Integration method. Return 0.')
            return 0

    def getDSigmaDyPairC(self, y, E_lep, method='quad'):
        return np.sum(self.getFactor(E_lep) * (1-y)/y * np.array([self.getIntAsymTerm(y, E_lep, ith, method) for ith in range(len(self.Z))]))

    #------------------
    # Bremsstrahlung 
    #------------------
    def getDSigmaDyBrems(self,y,E_lep):
        factor = 4.*self.alpha**3*self.lambda_e**2/self.getMassRatio()**2/y
        chargeFactor = np.sum(self.Z*(self.Z+1)*(self.natoms*self.getChargeFactor(y,E_lep)))
        return factor*chargeFactor

    def getChargeFactor(self,y,E_lep):
        y1 = 2.-2.*y+y**2
        y2 = 2./3.*(1.-y)
        qmin = self.m_lep**2*y/(2.*E_lep*(1.-y))
        a1 = 111.7/self.getZ3()/self.m_ele
        a2 = 724.2/self.getZ3()**2/self.m_ele
        x1 = a1*qmin
        x2 = a2*qmin
        xi = np.sqrt(1.+4./(1.9**2)*self.getZ3()**2)
        d1 = []
        d2 = []
        for i, z in enumerate(self.Z):
            if z == 1.:
                d1.append(0.)
                d2.append(0.)
            else:
                z3 = z**(1./3.)
                d1.append(np.log(z3/1.9)+xi[i]/2.*np.log((xi[i]+1.)/(xi[i]-1.)))
                d2.append(np.log(z3/1.9)+xi[i]*(3-xi[i]**2)/4.*np.log((xi[i]+1.)/(xi[i]-1.))+ 2.*z3**2/1.9**2)
        d1 = np.array(d1)
        d2 = np.array(d2)
        atan1 = np.arctan(1./x1)
        atan2 = np.arctan(1./x2)
        psi1 = .5*(1.+2.*np.log(self.m_lep*a1)-np.log1p(x1**2))-x1*atan1+(.5*(1.+2.*np.log(self.m_ele*a2)-np.log1p(x2**2))-x2*atan2)/self.Z - d1
        psi2 = .5*(2./3.+2.*np.log(self.m_lep*a1)-np.log1p(x1**2))+2.*x1**2*(1.-x1*atan1+.75*(2.*np.log(x1)-np.log1p(x1**2)))+2.*x2**2*(1.-x2*atan2+.75*(2.*np.log(x2)-np.log1p(x2**2)))/self.Z - d2

        term = y1*psi1 - y2*psi2
        return term if term>0. else 0.


    #-------------------
    # Photo-Nuclear 
    #-------------------
    def getDSigmaDyPhNuc(self,y,E_lep):
        factor = self.alpha/(8.*np.pi)
        atomicWeight = np.sum(self.A*self.natoms)
        softterm = np.sum(self.A*self.natoms*self.getSoftTerm(y,E_lep))
        return factor*softterm*self.getAbsorptionTerm(E_lep)*1.e-30*y + atomicWeight*self.getHardTerm(y,E_lep)

    def getSoftTerm(self,y,E_lep):
        h = 1. - 2./y + 2./y**2
        z = 0.00282*self.A**(1./3.)*self.getAbsorptionTerm(E_lep)
        t = self.m_lep**2*y**2/(1.-y)
        mSq = self.m_lep**2
        m1Sq = 0.54
        m2Sq = 1.80
        term = (h+2.*mSq/m2Sq)*np.log1p(m2Sq/t) - 2.*mSq/t*(1.-.25*m2Sq/t*np.log1p(t/m2Sq)) \
                + self.getG(z, self.Z)*(h*(np.log1p(m1Sq/t)-m1Sq/(m1Sq + t)) \
                + 4.*mSq/m1Sq*np.log1p(m1Sq/t) \
                - 2.*mSq/t*(1.-(.25*m1Sq-t)/(m1Sq + t)))
        return term if term>0. else 0.

    def getAbsorptionTerm(self,E_lep):
        return 114.3+1.647*(np.log(.0213*E_lep))**2

    def getG(self,z,arrayZ):
        gz = []
        for Z in arrayZ:
            if Z == 1.:
                gz.append(3.)
            else:
                gz.append(9./z*(.5+((1.+z)*np.exp(-z)-1.)/z**2))
        return np.array(gz)

    def getHardTerm(self,y,E_lep):
        return 0

    #-----------------
    # Get Results
    #-----------------
    def getDSigmaDyLogY(self, logy, E_lep, interactionName, method='quad'):
        return self.getDSigmaDy(10**logy, E_lep, interactionName, method)*np.log(10)*10**logy

    def getDSigmaDz(self, y, E_lep, interactionName, method='quad'):
        return getDSigmaDy(self, 1-y, E_lep, interactionName, method)

    def getDSigmaDy(self, y, E_lep, interactionName, method='quad'):
        if interactionName=='pairc':
            return self.getDSigmaDyPairC(y,E_lep,method)
        elif interactionName=='brems':
            return self.getDSigmaDyBrems(y,E_lep)
        elif interactionName=='phnuc':
            return self.getDSigmaDyPhNuc(y,E_lep)
        else:
            print("Invalid interaction. Return 0.")
            return 0

    def getSigma(self, E_lep, interactionName, method='quad'): 
        ymin, ymax = self.lim_y(E_lep, interactionName)
        if method == 'quad':
            return quad(self.getDSigmaDy, ymin, ymax, args=(E_lep, interactionName, method), limit=self.quadlimit)[0]
        elif method == 'quadLog':
            return quad(self.getDSigmaDyLogY, np.log10(ymin), np.log10(ymax), args=(E_lep, interactionName, 'quad'), limit=self.quadlimit)[0]
        else: 
            print("Invalid Integration method. Return 0.")
            return 0

    def getyDSigmaDy(self, y, E_lep, interactionName, method='quad'): 
        return y*self.getDSigmaDy(y, E_lep, interactionName, method)

    def getEnergyLoss(self, E_lep, interactionName, method='quad'): 
        return self.NA/self.A * self.getEnergyLossRaw(E_lep, interactionName, method)
    
    def getEnergyLossRaw(self, E_lep, interactionName, method='quad'):
        ymin, ymax = self.lim_y(E_lep, interactionName)
        if method == 'quad': 
            return quad(self.getyDSigmaDy, ymin, ymax, args=(E_lep, interactionName, method))[0]
        elif method == 'quadLog': # best  
            def ydsigmadylogfunc(logy):
                return self.getDSigmaDy(10**logy,E_lep,interactionName,method='quad')*10**(2*logy)*np.log(10)
            return quad(ydsigmadylogfunc,np.log10(ymin),np.log10(ymax))[0]
        else:
            print("Invalid Integration method. Return 0.")
            return 0

    def getPartialSigma(self, logYLow, logYUp, E_lep, interactionName, method='quadLog'):
        ymin, ymax = self.lim_y(E_lep, interactionName)
        LogYminBound = logYLow if logYLow > np.log10(ymin) else np.log10(ymin)
        LogYmaxBound = logYUp  if logYUp  < np.log10(ymax) else np.log10(ymax)
        if method == 'quadLog':
            def dsigmadylogfunc(logy):
                return self.getDSigmaDy(10**logy,E_lep,interactionName,method='quad')*10**logy*np.log(10)
            return quad(dsigmadylogfunc,LogYminBound,LogYmaxBound)[0]
        else:
            print("Invalid Integration method. Return 0.")
            return 0

    def getSurviveProb(self, ZLow, ZUp, E_lep, interactionName, method='quadLog'):
        ymin, ymax = self.lim_y(E_lep, interactionName)
        ZminBound = ZLow if ZLow > 1.-ymax else 1.-ymax
        ZmaxBound = ZUp  if ZUp  < 1.-ymin else 1.-ymin
        if method == 'quadLog':
            def dsigmadzlogfunc(logz):
                return self.getDSigmaDy(1.-10**logz,E_lep,interactionName,method='quad')*(1.-10**logz)*np.log(10)
            return quad(dsigmadzlogfunc,np.log10(ZminBound),np.log10(ZmaxBound))[0]
        else:
            print("Invalid Integration method. Return 0.")
            return 0


