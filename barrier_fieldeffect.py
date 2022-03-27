# Electron tunneling by Devon's formula
# Coded by Takuro TOKUNAGA
# Last modified: Nov 20 2019
# in-progress
# T.S. Fisher and D. G. Walker, Transactions of the ASME 124 954 (2002)

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
phi_e = 1.7 # [eV], emitter work function
phi_c = 1.7 # [eV], collector work function
volt = 4 # [V], load (vias) voltage
se = 1 # [eV/V], single electron
volt = se*volt # [eV]
felmi = 4.50 # [eV]

# gap
number = 100 # [-], integral parameter, initial: 500
small = 0.01*ucnano # zero [m]
gapmin = small # [m]
gap = gapmin # [m]
gapmax = 10*ucnano # [m]
gapmax_fix = 4000*ucnano # [m]
dgap = (gapmax-gapmin)/number # [m]

# tip parameters
radius = 5*ucnano # [m]
df = 5.5 # [-], real part of the dielectric function
Field = 1.08*np.power(10.,6) # V/m
#Field = 614*np.power(10.,6) # V/m
beta = 1

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
    ibp = arg_phi_e+arg_volt-(arg_phi_e+arg_volt-arg_phi_c)*(arg_x/arg_d) # [eV]
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

# Fisher's barrier
# Fisher, 1st term
def wid_tip(arg_F, arg_d, arg_R, arg_x):
    term1 = -sq*arg_F*(arg_d+arg_R)
    term2 = 1-(arg_R/(arg_x+arg_R))

    ibp = term1*term2 # [J]
    ibp = ibp/ucev # [eV]

    return ibp # [eV]

# Fisher, 2nd term, image charge barrier profile, tip
def wic_tip(arg_K, arg_R, arg_x):
    term1 = -np.power(sq,2.0)/(4*np.pi*eps_0)
    term2 = (arg_K-1)/(arg_K+1)
    term3 = arg_R/((2*arg_R+arg_x)*(2*arg_x))

    icbp = term1*term2*term3 # [J]
    icbp = icbp/ucev # [eV]

    return icbp # [eV]

def liner(arg_F, arg_beta, arg_x):
    y = -sq*arg_beta*arg_F*arg_x # J, [C*[-]*V/m*m]
    y = y/ucev # eV

    return y # [eV]

# eq.5, M.S. Chung Current Applied Physics 15 (2015), 57-63
def w_tip_plate(arg_W, arg_F, arg_d, arg_R, arg_x):
    # term1
    term1 = arg_W # [eV]

    # term2
    coefficient = 4*np.pi*eps_0 # probably, missing
    term2 = -np.power(sq,2.0)/(4*arg_x)
    term3 = 1/(1+0.5*(arg_x/arg_R))
    term4 = (term2*term3)/coefficient # [J]
    term4 = term4/ucev # [eV]

    # term3
    term5 = -sq*arg_F*arg_x
    term6 = (1+(arg_d/arg_R))/(1+(arg_x/arg_R))
    term7 = term5*term6 # [J]
    term7 = term7/ucev # [eV]

    ibp = term1 + term4 + term7

    return ibp # [eV]

# main start
begin()

# file open
f1 = open('bprofile.txt', 'w')
f1.write('nm [-] [eV]') # [nm]
f1.write('\n')
while gap < gapmax: # [m]

    # Devon's equations
    #wz = wid(phi_e,phi_c,volt,gapmax,gap) + wsc() + wic(gapmax,gap)  # [eV], barrier profile

    # Fisher's formula
    wz = wid_tip(Field,gapmax_fix,radius,gap) + wic_tip(df,radius,gap)  # [eV], barrier profile
    y = liner(Field,beta,gap)

    # M.S. Chung
    #wz = w_tip_plate(phi_e,Field,gapmax,radius,gap) + felmi

    # output
    f1.write(str(gap/ucnano)) # [nm]
    f1.write(str(' '))
    f1.write(str(gap/gapmax)) # [-], relative position
    f1.write(str(' '))
    f1.write(str(wz)) # [eV]
    f1.write(str(' '))
    f1.write(str(y)) #
    f1.write('\n')

    # for graphing
    if counter < number:
        gap_table[counter] = gap/ucnano # [nm]
        potential_table[counter] = wz # [eV]
        liner_table[counter] = y #

    # gap update
    gap = gap + dgap
    counter = counter + 1
# loop end

# regression
potential_max = max(potential_table)

for i in range(0, number):
    if potential_table[i] >= potential_max:
        temp_i_min = i

for i in range(0, number):
    if gap_table[i] > 2.0: # [nm], change depends on profile shape
        temp_i_max = i
        break

size_dif = abs(temp_i_max - temp_i_min)

# regression
x_regression = np.zeros(size_dif, dtype='float64')
y_regression = np.zeros(size_dif, dtype='float64')
counter=0
if temp_i_min >= temp_i_max:
    print('check i')
else:
    for i in range(temp_i_min,temp_i_max,1):
        x_regression[counter] = gap_table[i] # [nm]
        y_regression[counter] = potential_table[i] # [eV]
        counter = counter+1

_x = np.reshape(x_regression,(-1,1))
_y = np.reshape(y_regression,(-1,1))
clf.fit(_x, _y)
# regression, a
inclination = clf.coef_ # [eV/nm]
inclination_beta = (clf.coef_/(sq*(Field*ucnano)*(1/ucev))) # [-]
print(inclination)
print(inclination_beta)

# file close
f1.close

# plot
# graph display
csfont = {'fontname':'Times New Roman'} # define font
plt.plot(gap_table, potential_table, 'blue', label="Potential")
plt.plot(np.reshape(gap_table,(-1,1)), clf.predict(np.reshape(gap_table,(-1,1))), 'red', label="Linear Regression")
plt.plot(gap_table, liner_table, 'green', label="Liner Analytical")
plt.xlabel('Axial Position, x[nm]', fontdict=None, labelpad=None, **csfont)
plt.ylabel('Electron Potential, V [eV]', fontdict=None, labelpad=None, **csfont)
# font for legend
font = font_manager.FontProperties(family='Times New Roman',
                                   weight='bold',
                                   style='normal', size=10)
plt.legend(loc='upper right', prop=font) # legend
# plot options
plt.xticks([0.0, 2, 4, 6, 8, 10], **csfont)
plt.yticks([-6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0], **csfont)
plt.xlim(0.0,10.0) # x limit
#plt.xscale("log")
plt.ylim(-6.0,0.0) # y limit
plt.savefig("potential.png") # 1. file saving (1. should be before 2.)
plt.show()                  # 2. file showing (2. should be after 1.)

# end
end()

# time display
elapsed_time = time.time()-start
print("elapsed_time:{:.2f}".format(elapsed_time) + "[sec]")
