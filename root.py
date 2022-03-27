# Electron tunneling by Devon's formula
# Coded by Takuro TOKUNAGA
# Last modified: January 15 2020
# Until root finding

import numpy as np
import cmath
import time
import sys
from scipy import special, optimize
# for graph
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

start = time.time()

# Unit conversion:
ucev = 1.602176620898*np.power(10.,-19) # electron volt to joule
ucnano = 1.0*np.power(10.,-9)           # nm to m
ucangs = 1.0*np.power(10.,-10)          # angstrom to m

# physical constants
kb = 1.38064852*np.power(10.,-23)           # Boltzmann constant
rh = 6.62607004*np.power(10.,-34)/(2*np.pi) # reduced Planck constant
c = 299792458                               # light speed
eps_0 = 8.854187817*np.power(10.,-12)      # [F/m]
sq = 1.602176620898*np.power(10.,-19) # [C] elementary cahrge, small q

# work function & bias voltage
#phi_e = 5.80 # [eV], emitter work function
#phi_c = 4.70 # [eV], collector work function
phi_gold = 5.2 # [eV]
phi_e = phi_gold # [eV], emitter work function, gold
phi_c = phi_gold # [eV], collector work function, gold
volt = -0.60 # [V], load (vias) voltage

se = 1 # [eV/V], single electron
volt = se*volt # [eV]

# gap
small = 0.001*ucnano # zero [m]
gapmin = small # [m]
gap = gapmin # [m]
#gapmax = 100.001*ucnano-small # [m]
gapmax = 0.901*ucnano-small # [m]
number = 1000 # [-]
dgap = (gapmax-gapmin)/number # [m]

# table
gap_table = np.zeros(number, dtype='float64')
energy_table = np.zeros(number, dtype='float64')
root1_table = np.zeros(number, dtype='float64')
root2_table = np.zeros(number, dtype='float64')
Ez_table = np.zeros(number, dtype='float64')

# function: begin
def begin():
    print ("begin")

# function: end
def end():
    print ("end")

# total profile
def wz_total(arg_x): # [m]
    # ideal barrier profile
    #ibp = phi_e+volt-(phi_e+volt-phi_c)*(arg_x/gapmax) # [eV]
    ibp = phi_e-(phi_e-phi_c-volt)*(arg_x/gapmax) # [eV]

    # space charge
    scbp = 0 # in case of nanosacle, Devon's Ph.D. thesis

    # image charge barrier profile
    term1 = np.power(sq,2.0)/(16*np.pi*eps_0*gapmax) # [J]
    term1 = term1/ucev # [eV]
    term2 = -2*special.digamma(1) # [-]
    term3 = special.digamma(arg_x/gapmax) # [-]
    term4 = special.digamma(1-arg_x/gapmax) # [-]
    icbp = term1*(term2+term3+term4) # [eV]

    total = ibp+scbp+icbp

    return total # [eV]

# main start
begin()

# file open
f1 = open('bprofile.txt', 'w')
f2 = open('z1z2.txt', 'w')

# initialization
old = 0
counter = 0

# barrier profile
f1.write('d(nm) Wz(eV)')
f1.write('\n')
while gap < gapmax: # [m]

  wz = wz_total(gap)  # [eV], barrier profile

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

  # for graphing
  if counter < number:
      gap_table[counter] = gap/ucnano # [nm]
      energy_table[counter] = wz # [eV]

  # counter update
  counter = counter + 1

  # gap update
  gap = gap + dgap

  #print(str(counter))

# last one component
gap_table[number-1] = gap/ucnano-dgap # [nm]
eneryg_last = wz_total(gap_table[number-1])
if eneryg_last > 0:
    energy_table[number-1] = -eneryg_last # [eV]
else:
    energy_table[number-1] = eneryg_last # [eV]

#print(str(gap_table))
#print(str(energy_table))

# z integral calculation
# Ez discretization
Ezmin1 = wz_total(gapmin)
Ezmin2 = wz_total(100*ucnano-small)
Ezmin = max(Ezmin1,Ezmin2)

Ezmin = -3
Ez = Ezmin
Ezmax = wmax
factor = 1.000 # [-]
dEz = factor*(Ezmax-Ezmin)/number
counter = 0 # counter reset

f2.write('z1 z2 Wz')
f2.write('\n')
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

    # for graphing
    if counter < number:
        root1_table[counter] = z1/ucnano # [nm]
        root2_table[counter] = z2/ucnano # [nm]
        Ez_table[counter] = Ez

    # output
    f2.write(str(z1/ucnano)) # [nm]
    f2.write(str(' '))
    f2.write(str(z2/ucnano)) # [nm]
    f2.write(str(' '))
    f2.write(str(Ez)) # [eV]
    f2.write('\n')

    # Ez update
    Ez = Ez + dEz

    # counter update
    counter = counter + 1
    #print(str(counter))

    # transmission

#print(str(root1_table))
#print(str(root2_table))

z1max = max(root1_table)
z2max = max(root2_table)

f1.close
f2.close

# plot
csfont = {'fontname':'Times New Roman'} # define font
plt.plot(gap_table, energy_table, 'red', label="potential")
plt.plot(root1_table, Ez_table, 'blue', alpha=0.3, marker="o", markersize=4, label="z1")
plt.plot(root2_table, Ez_table, 'green', alpha=0.3, marker="o", markersize=4, label="z2")
plt.xlabel('Axial Position, x[nm]', fontdict=None, labelpad=None, **csfont)
plt.ylabel('Electron Potential, V [eV]', fontdict=None, labelpad=None, **csfont)
# font for legend
font = font_manager.FontProperties(family='Times New Roman',
                                   weight='bold',
                                   style='normal', size=10)
plt.legend(loc='upper left', prop=font) # legend
# plot options
#plt.xticks([0.0, 20, 40, 60, 80, 100], **csfont)
plt.yticks([-3.0,-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], **csfont)
#plt.xlim(0.0,101.0) # x limit
plt.xlim(0.0,1.0) # x limit
plt.ylim(-3.0,6.0) # y limit
plt.savefig("potential.png") # 1. file saving (1. should be before 2.)
plt.show()                  # 2. file showing (2. should be after 1.)

if z1max > z2max:
    print('# WARNING: z1 is larger z2')
else:
    # end
    end()

# time display
elapsed_time = time.time()-start
print("elapsed_time:{:.2f}".format(elapsed_time) + "[sec]")
