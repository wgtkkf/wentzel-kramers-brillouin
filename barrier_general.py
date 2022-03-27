# Electron tunneling by Devon's formula
# Coded by Takuro TOKUNAGA
# Last modified: July 21 2021

import numpy as np
import cmath
import time
import sys
from scipy import special, optimize
# graph
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
# regression
from sklearn import linear_model
clf = linear_model.LinearRegression()

start = time.time()

# Unit conversion:
ucev = 1.602176620898*np.power(10.,-19) # electron volt to joule
ucmicron = 1.0*np.power(10.,-6)           # micron to m
ucnano = 1.0*np.power(10.,-9)           # nm to m
ucangs = 1.0*np.power(10.,-10)          # angstrom to m

# physical constants
kb = 1.38064852*np.power(10.,-23)           # Boltzmann constant
rh = 6.62607004*np.power(10.,-34)/(2*np.pi) # reduced Planck constant
c = 299792458                               # light speed
eps_0 = 8.854187817*np.power(10.,-12)      # [F/m]
sq = 1.602176620898*np.power(10.,-19) # [C] elementary cahrge, small q

# work function & voltage related
phi_e = 5.10 # [eV], emitter work function
phi_c = 5.10 # [eV], collector work function
volt = 0.6 # [V], load (vias) voltage
se = 1 # [eV/V], single electron
volt = se*volt # [eV]
felmi = 5.53 # [eV]

# gap
number = 100 # [-], integral parameter, initial: 500
small = 0.01*ucnano # zero [m]
gapmin = small # [m]
gap = gapmin # [m]
gapmax = 5*ucnano # [m]
gapmax_fix = 4000*ucnano # [m]
dgap = (gapmax-gapmin)/number # [m]

# graph
gap_table = np.zeros(number, dtype='float64')
potential_table = np.zeros(number, dtype='float64')
liner_table = np.zeros(number, dtype='float64')
counter = 0

# function: begin
def begin():
    print ("begin")

# function: end
def end():
    print ("end")

# ideal barrier profile
def wid(arg_phi_e, arg_phi_c, arg_volt, arg_d, arg_x):
    #ibp = arg_phi_e+arg_volt-(arg_phi_e+arg_volt-arg_phi_c)*(arg_x/arg_d) # [eV]
    ibp = arg_phi_e-(arg_phi_e+arg_volt-arg_phi_c)*(arg_x/arg_d) # [eV]
    return ibp # [eV]

# space charge barrier profile
def wsc():
    scbp = 0 # in case of nanosacle, Devon's Ph.D. thesis
    return scbp # [eV]

# image charge barrier profile
def wic(arg_d, arg_x):
    term1 = np.power(sq,2.0)/(16*np.pi*eps_0*arg_d) # [J]
    term1 = term1/ucev # [eV]
    term2 = -2*special.digamma(1) # [-]
    term3 = special.digamma(arg_x/arg_d) # [-]
    term4 = special.digamma(1-arg_x/arg_d) # [-]

    icbp = term1*(term2+term3+term4) # [eV]

    return icbp # [eV]

# main start
begin()

# file open
f1 = open('bprofile.txt', 'w')
f1.write('nm [-] [eV]') # [nm]
f1.write('\n')
while gap < gapmax: # [m]

    # Devon's equations
    wz = wid(phi_e,phi_c,volt,gapmax,gap) + wsc() + wic(gapmax,gap)  # [eV], barrier profile

    # output
    f1.write(str(gap/ucnano)) # [nm]
    f1.write(str(' '))
    f1.write(str(gap/gapmax)) # [-], relative position
    f1.write(str(' '))
    f1.write(str(wz)) # [eV]
    f1.write('\n')

    # for graphing
    if counter < number:
        gap_table[counter] = gap/ucnano # [nm]
        potential_table[counter] = wz # [eV]

    # gap update
    gap = gap + dgap
    counter = counter + 1
# loop end

# file close
f1.close

# plot
# graph display
csfont = {'fontname':'Times New Roman'} # define font
plt.plot(gap_table, potential_table, 'blue', label="Potential")
plt.xlabel('Axial Position, x[nm]', fontdict=None, labelpad=None, **csfont)
plt.ylabel('Electron Potential, V [eV]', fontdict=None, labelpad=None, **csfont)
# font for legend
font = font_manager.FontProperties(family='Times New Roman',
                                   weight='bold',
                                   style='normal', size=10)
plt.legend(loc='upper right', prop=font) # legend
# plot options
plt.xticks([0.0, 2, 4, 6, 8, 10], **csfont)
plt.yticks([0.0, 2.0, 4.0, 6.0], **csfont)
plt.xlim(0.0,10.0) # x limit
#plt.xscale("log")
plt.ylim(0.0,6.0) # y limit
plt.savefig("potential.png") # 1. file saving (1. should be before 2.)
plt.show()                  # 2. file showing (2. should be after 1.)

# end
end()

# time display
elapsed_time = time.time()-start
print("elapsed_time:{:.2f}".format(elapsed_time) + "[sec]")
