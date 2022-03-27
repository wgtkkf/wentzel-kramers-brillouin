# Electron tunneling transmission by Non-equilibrium Green's function
# T. L. Westover and T. S. Fisher Phys. Rev. B. 77 115426 (2008)
# Coded by Takuro TOKUNAGA
# Last modified: January 25 2020

import numpy as np
import cmath
import time
import sys
from scipy.integrate import quad
from scipy import special, optimize
from numpy import linalg as LA
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

start = time.time()

# Unit conversion:
ucev_to_j = 1.602176620898*np.power(10.,-19) # electron volt to joule
ucjto_ev = 1/ucev_to_j # joule to electron volt
ucnano = 1.0*np.power(10.,-9)           # nm to m
ucangs = 1.0*np.power(10.,-10)          # angstrom to m

# physical constants
kb = 1.38064852*np.power(10.,-23)           # Boltzmann constant
rh = 6.62607004*np.power(10.,-34)/(2*np.pi) # reduced Planck constant
c = 299792458                               # light speed
eps_0 = 8.854187817*np.power(10.,-12)      # [F/m]
sq = 1.602176620898*np.power(10.,-19) # [C] elementary cahrge, small q
me = 9.10938356*np.power(10.,-31) # electron mass, [kg]
ac = 6.02214076*np.power(10.,23) # Avogadro constant, [mol-1]

# work function & voltage
phi_e = 5.8 # [eV], emitter work function 5.8
phi_c = 4.7 # [eV], collector work function 4.7
volt = 0.80 # [V], load (vias) voltage
se = 1 # [eV/V], single electron
volt = se*volt # [eV]
# fermi energy
Ef = 5.51 # [eV], gold

# gap
small = 0.01*ucnano # zero [m]
gapmin = small # [m]
gap = gapmin # [m]
gapmax = 1*ucnano-small # [m]
number = 10 # [-]
dgap = (gapmax-gapmin)/number # [m]

# parameters: Green's function matrix related
mass = me # [kg], electron mass
lc = 4.065*ucangs # [m]
st = np.power(rh/lc,2.0)/(2*mass) # small t, hopping
st = st*ucjto_ev # [eV]
print(str(1/(2*st))) # [1/eV]

# Green's function matrix related
Gamma1=np.zeros((number+1,number+1), dtype=np.complex) # eq. 10, size:number*number
Gamma2=np.zeros((number+1,number+1), dtype=np.complex) # eq. 10, size:number*number
Gd=np.zeros((number+1,number+1), dtype=np.complex)     # eq. 11, size:number*number
imatrix=np.identity(number+1, dtype=np.complex)  # size:number*number
# Hl, Sigma1, 2: Defined in the energy loop
delta = 1.0*np.power(10.,-6.0)

# tables for graphing
gap_table = np.zeros(number+1, dtype='float64')
Ez_table = np.zeros(number+1, dtype='float64')
trans_table = np.zeros(number+1, dtype='float64')
counter = 0

# function: begin
def begin():
    print ("begin")

# function: end
def end():
    print ("end")

# total profile
def wz_total(arg_x):
    # ideal barrier profile
    ibp = phi_e+volt-(phi_e+volt-phi_c)*(arg_x/gapmax) # [eV]

    # space charge
    scbp = 0 # in case of nanosacle, Devon's Ph.D. thesis

    # image charge barrier profile
    term1 = np.power(sq,2.0)/(16*np.pi*eps_0*gapmax) # [J]
    term1 = term1/ucev_to_j # [eV]
    term2 = -2*special.digamma(1) # [-]
    term3 = special.digamma(arg_x/gapmax) # [-]
    term4 = special.digamma(1-arg_x/gapmax) # [-]
    icbp = term1*(term2+term3+term4) # [eV]

    total = ibp+scbp+icbp

    # Fermi energy
    total = total + Ef

    return total # [eV]

# main start
begin()

# file open
f1 = open('bprofile.txt', 'w')
f2 = open('transmission.txt', 'w')

# initialization
old = 0

# find wmax(Ezmax)
while gap < gapmax: # [m]
    wz = wz_total(gap) # [eV]

    # find wmax(Ezmax)
    new = wz      # [eV]
    dif = new-old # [eV], difference

    if dif > 0:
        wmax = new # [eV], wmax

    old = new

    # output
    f1.write(str(gap/ucnano)) # [nm]
    f1.write(str(' '))
    f1.write(str(wz)) # [eV]
    f1.write('\n')

    # for graph & green's function
    gap_table[counter] = gap/ucnano # [nm]
    Ez_table[counter] = wz # [eV]

    # gap update
    gap = gap + dgap
    counter = counter + 1

# Ez discretization
Ezmin = 0 # [eV]
Ez = Ezmin # [eV]
Ezmax = wmax # [eV]
dEz = (Ezmax-Ezmin)/number # [eV]
counter = 0 # [-]

#print(str(Ezmax))
# first & last node
Ez_table[0] = Ef
Ez_table[number] = Ef

# Ez integral calculation
while Ez < Ezmax: # [eV]
    # matrix initialization
    Hlongitudinal=np.zeros((number+1,number+1), dtype=np.complex)  # eq. 11, size:number*number
    Sigma1=np.zeros((number+1,number+1), dtype=np.complex) # eq. 11, size:number*number
    Sigma2=np.zeros((number+1,number+1), dtype=np.complex) # eq. 11, size:number*number

    # Hlongitudinal construction
    for i in range(0,number):
        # diagonal components
        #Hlongitudinal[i][i] = 2*st + Ez_table[i]*ucev_to_j # [J]
        Hlongitudinal[i][i] = 2*st + Ez_table[i] # [eV]
        # tridiagonal components
        Hlongitudinal[i][i+1] = -st # [eV]
        Hlongitudinal[i+1][i] = -st # [eV]


    # Sigma1, only non zero: (0,0)
    argument1 = 1-((Ez-Ez_table[0])/(2*st)) # [eV/eV]
    if argument1 >=1 or argument1 <=-1:
        component1 = 0
    else:
        term1 = np.arccos(argument1) # k1*a
        component1 = -st*np.exp(1j*term1) # k1*a

    Sigma1[0][0] = component1

    # output
    f2.write(str(counter)) #
    f2.write(str(' '))
    f2.write(str(Ez_table[0])) #
    f2.write(str(' '))
    f2.write(str(Ez)) #
    f2.write(str(' '))
    f2.write(str(Ez_table[0]-Ez)) #
    f2.write(str(' '))
    f2.write(str((Ez-Ez_table[0])/(2*st)))
    f2.write(str(' '))
    f2.write(str(argument1)) #
    f2.write(str(' '))
    f2.write(str(Sigma1[0][0])) #
    f2.write('\n')

    # Ez update
    Ez = Ez + dEz # [eV]
    counter = counter + 1

f1.close
f2.close

# plot
csfont = {'fontname':'Times New Roman'} # define font
plt.plot(gap_table, Ez_table, 'red', label="potential")
plt.xlabel('Axial Position, x[nm]', fontdict=None, labelpad=None, **csfont)
plt.ylabel('Potential, V [eV]', fontdict=None, labelpad=None, **csfont)
# font for legend
font = font_manager.FontProperties(family='Times New Roman',
                                   weight='bold',
                                   style='normal', size=10)
plt.legend(loc='upper left', prop=font) # legend
# plot options
#plt.xticks([0.0, 20, 40, 60, 80, 100], **csfont)
plt.yticks([0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0], **csfont)
#plt.xlim(0.0,100.0) # x limit
#plt.ylim(0.0,15.0) # y limit
plt.xlim(0.0,1.0) # x limit
plt.ylim(0.0,12.0) # y limit
plt.savefig("potential.png") # 1. file saving (1. should be before 2.)
plt.show()                  # 2. file showing (2. should be after 1.)

# end
end()

# time display
elapsed_time = time.time()-start
print("elapsed_time:{:.2f}".format(elapsed_time) + "[sec]")
