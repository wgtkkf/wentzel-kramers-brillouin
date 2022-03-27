# Electron tunneling by Devon's formula
# Coded by Takuro TOKUNAGA
# Last modified: April 12 2019
# in-progress

import math
import numpy as np
import cmath
import time
import sys
from scipy.integrate import quad
from scipy import special, optimize
# for graph
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

start = time.time()

# Unit conversion:
ucev_to_j = 1.602176620898*np.power(10.,-19) # electron volt to joule
ucnano = 1.0*np.power(10.,-9)           # nm to m
ucangs = 1.0*np.power(10.,-10)          # angstrom to m

# physical constants
kb = 1.38064852*np.power(10.,-23)           # Boltzmann constant
rh = 6.62607004*np.power(10.,-34)/(2*np.pi) # reduced Planck constant
c = 299792458                               # light speed
eps_0 = 8.854187817*np.power(10.,-12)      # [F/m]
sq = 1.602176620898*np.power(10.,-19) # [C] elementary cahrge, small q
me = 9.10938356*np.power(10.,-31) # electron mass, [kg]

## parameters ##
# work function
phi_e = 2.10 # [eV], emitter work function
phi_c = 1.70 # [eV], collector work function
volt = 0.20 # [V], load (vias) voltage
se = 1 # [eV/V], single electron
volt = se*volt # [eV]
# fermi energy
Ef = 0  #5.51 # [eV], gold

# Integral
nzmax = 1000 # [-], integral parameter, initial: 1000

# gap
small = 0.01*ucnano # zero [m]
gapmin = small # [m]
gap = gapmin # [m]
gapmax = 100.01*ucnano-small # [m]
number = 100 # [-]
dgap = (gapmax-gapmin)/number # [m]
## parameters ##

## table for graphing
# barrier
gap_table = np.zeros(number+1, dtype='float64')
Ez_barrier_table = np.zeros(number+1, dtype='float64')
counter = 0 # [-]

# transmission
Ez_table = np.zeros(number+1, dtype='float64')
trans_table = np.zeros(number+1, dtype='float64')

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

    total = ibp+scbp+icbp # [eV]

    # fermi energy

    return total # [eV]

# main start
begin()

# file open
f1 = open('bprofile.txt', 'w')
f2 = open('z1z2.txt', 'w')
f3 = open('transmission.txt', 'w')

# initialization
old = 0

# barrier profile
while gap < gapmax: # [m]

  wz = wz_total(gap)

  # find wmax(Ezmax)
  new = wz
  dif = new-old # difference

  if dif>0:
      wmax = new # wmax

  old = new

  # output
  f1.write(str(gap/ucnano)) # [nm]
  f1.write(str(' '))
  f1.write(str(wz)) # [eV]
  f1.write('\n')

  # for graph & green's function
  gap_table[counter] = gap/ucnano # [nm]
  Ez_barrier_table[counter] = wz # [eV]

  # gap update
  gap = gap + dgap

  counter = counter + 1

# last component gap & energy
gap_table[number] = gap/ucnano # [nm]
Ez_barrier_table[number] = wz # [eV]

# z integral calculation
# Ez discretization
Ezmin = 0
Ez = Ezmin
Ezmax = wmax
dEz = (Ezmax-Ezmin)/number
Ez_counter = 0
z_counter = 0 # [-]

# z1 initialization
z1 = gapmin
z2 = gapmax

# Ez integral calculation
while Ez < Ezmax:

    # z1, z2 initialization
    z1 = gapmin
    z2 = gapmax

    # z1 & z2 finding routine for individual Ez
    z1 = z1 + dgap
    wz1 = wz_total(z1)
    while wz1 < Ez:
        z1 = z1 + dgap
        wz1 = wz_total(z1)
        if wz1 > Ez:
            break

    z2 = z2 - dgap
    wz2 = wz_total(z2)

    while wz2 < Ez:
        z2 = z2 - dgap
        wz2 = wz_total(z2)
        if wz2 > Ez:
            break

    # output
    f2.write(str(z1/ucnano)) # [nm]
    f2.write(str(' '))
    f2.write(str(z2/ucnano)) # [nm]
    f2.write(str(' '))
    f2.write(str(Ez)) # [eV]
    f2.write('\n')

    # define integrand
    def integrand_z(arg_z):
        integrand = wz_total(arg_z)-Ez # [eV]
        integrand = integrand*ucev_to_j # [J]
        #print(str(Ez))
        integrand = np.sqrt(integrand) # [sqrt[J]]

        return integrand

    #print(str(z2))

    # transmission for each Ez
    term1 = -np.sqrt(8*me)/rh # [sqrt(kg)/(J*s)]

    # Integral by QUADPACK
    #term2 = quad(integrand_z, z1, z2)[0] # [sqrt[J]*m]: m -> by dz

    # Trapezoidal
    sz = z1
    dz = (z2-z1)/nzmax
    term2 = 0

    while z_counter <= nzmax:
        if z_counter==0 or z_counter==nzmax:
            term2 = term2 + integrand_z(sz)*dz*0.5
        else:
            term2 = term2 + integrand_z(sz)*dz
        #print(str(z_counter))

        sz = sz + dz
        z_counter = z_counter + 1

    term3 = term1*term2 # [-]
    #print(str(term3))
    trans = np.exp(term3) # transmission, [-]

    # output
    f3.write(str(Ez)) # [eV]
    f3.write(str(' '))
    f3.write(str(trans)) # [-]
    f3.write('\n')

    # for graphing
    Ez_table[Ez_counter] = Ez # [eV]
    trans_table[Ez_counter] = trans # [-]

    # Ez update
    Ez = Ez + dEz
    Ez_counter = Ez_counter + 1

    # counter reset (transmission z1, z2 integral)
    z_counter = 0
    sz = 0
    dz = 0

# last component gap & energy
Ez_table[number] = Ez # [eV]
trans_table[number] = trans # [-]

# file close
f1.close
f2.close
f3.close

# plot, potential
csfont = {'fontname':'Times New Roman'} # define font
plt.plot(gap_table, Ez_barrier_table, 'red', marker="o", markersize=4,label="potential")
plt.xlabel('Axial Position, x[nm]', fontdict=None, labelpad=None, **csfont)
plt.ylabel('Potential, V [eV]', fontdict=None, labelpad=None, **csfont)
# font for legend
font = font_manager.FontProperties(family='Times New Roman',
                                   weight='bold',
                                   style='normal', size=10)
plt.legend(loc='upper left', prop=font) # legend
# plot options
plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], **csfont)
plt.yticks([0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0], **csfont)
plt.xlim(0.0,1.0) # x limit
plt.ylim(0.0,12.0) # y limit
plt.savefig("potential.png") # 1. file saving (1. should be before 2.)
plt.show()                  # 2. file showing (2. should be after 1.)

# plot, transmission
csfont = {'fontname':'Times New Roman'} # define font
plt.plot(Ez_table, trans_table, 'blue', marker="o", markersize=4, label="WKB transmission")
plt.xlabel('Energy [eV]', fontdict=None, labelpad=None, **csfont)
plt.ylabel('Transmission [-]', fontdict=None, labelpad=None, **csfont)
# font for legend
font = font_manager.FontProperties(family='Times New Roman',
                                   weight='bold',
                                   style='normal', size=10)
plt.legend(loc='upper left', prop=font) # legend
# plot options
plt.xticks([0.0, 2.0, 4.0, 6.0], **csfont)
plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], **csfont)
plt.xlim(0.0,6.0) # x limit
plt.ylim(0.0,1.0) # y limit
plt.savefig("transmission.png") # 1. file saving (1. should be before 2.)
plt.show()                  # 2. file showing (2. should be after 1.)


# end
end()

# time display
elapsed_time = time.time()-start
print("elapsed_time:{:.2f}".format(elapsed_time) + "[sec]")
