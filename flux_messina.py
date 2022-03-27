# Electron tunneling by Messina's formula
# Coded by Takuro TOKUNAGA
# Last modified: November 25 2019

import numpy as np
import time
import sys
from scipy.integrate import quad

start = time.time()

# Unit conversion:
ucev_to_j = 1.602176620898*np.power(10.,-19) # electron volt to joule
ucnano = 1.0*np.power(10.,-9)                # nm to m
ucangs = 1.0*np.power(10.,-10)               # angstrom to m
uccm = 1.0*np.power(10.,-2)                  # cm to m

# physical constants
kb = 1.38064852*np.power(10.,-23)           # Boltzmann constant, [J/K]
ch = 6.62607004*np.power(10.,-34)           # Planck constant
rh = ch/(2*np.pi)                           # reduced Planck constant
c = 299792458                               # light speed, [m]
eps_0 = 8.854187817*np.power(10.,-12)       # [F/m]
sq = 1.602176620898*np.power(10.,-19)       # elementary cahrge [C], small q
me = 9.10938356*np.power(10.,-31)           # electron mass, [kg]
se = 1                                      # [eV/V], single electron

# gap
gapmax = 0.7*ucnano # [m]

## Parameters (start) ##
# fermi energy
Ef = 5.53 # [eV]
mu_L = 0 # [eV], chemical potential
mu_R = 0.6 # [eV], chemical potential

# voltage
v0 = 1.25 # [eV]

# temperature
temperature_L = 280 # [K], emitter
temperature_R = 120 # [K], collector
delta_temperature = temperature_L-temperature_R # [K]

# area
radius = 15*ucnano # [m], 170~250 [nm]
Atip = np.pi*np.power(radius,2.0) # [m2]
#print(str(Atip))
## Parameters (end) ##

# Energy (integral range)
num_Ez = 100 # [-], integral parameter, initial: 500
Ezmin = 0 # [eV]
Ez = Ezmin # [eV]
Ezmax = 10 # [eV]
dEz = (Ezmax-Ezmin)/num_Ez

# function: begin
def begin():
    print ("begin")

def end():
    print ("end")

def voltage(arg_gap): # [m]
    arg_gap = arg_gap/(1.0*np.power(10.,-10)) # [m] to [angs]
    bracket = (1+arg_gap/1) # [-]

    if bracket < 0:
        print('negative value')
        sys.exit()
    else:
        y = v0*np.log(bracket)+Ef

    return y

def k2x(arg_Ez, arg_V):
    temp = 2*me*(arg_Ez-arg_V)
    if temp < 0:
        y = 0
    else:
        y = np.sqrt(temp)/rh

    return y

def transmission(arg_Ez, arg_V, arg_k2x, arg_gap):

    numerator = 4*arg_Ez*(arg_Ez-arg_V)
    term2 = 4*arg_Ez*(arg_Ez-arg_V)
    temp = arg_k2x*arg_gap
    term3 = np.power(arg_V,2.0)*np.power(np.sin(temp),2.0)
    denominator = term2*term3

    if denominator==0:
        y = 0
    else:
        y = numerator/denominator

    return y

def normal_energy(arg_Ez,arg_mu,arg_T): # [eV, eV, & K]
    term1 = me*kb*arg_T/(2*np.power(np.pi,2.0)*np.power(rh,3.0))
    term2 = -(arg_Ez-Ef-arg_mu)*ucev_to_j/(kb*arg_T) # [-] eV to J
    y = term1*np.log(1+np.exp(term2)) # P.35, Devon's Ph.D. thesis

    return y # [J]

# function: end

# main start
begin()


f1 = open('integrand.txt', 'w')
while Ez < Ezmax:
    v_temp = voltage(gapmax) # [eV]
    k2x_temp = k2x(Ez, v_temp) # ([eV], [eV])
    nl = normal_energy(Ez,mu_L,temperature_L) # [J]
    nr = normal_energy(Ez,mu_L,temperature_R) # [J]
    ts = transmission(Ez, v_temp, k2x_temp, gapmax) # transmission, [-]
    integrand = (Ez*ucev_to_j)*(nl-nr)*ts # [J], [J]*[J]*[-]

    # energy update
    Ez = Ez + dEz

    #print(str(nl-nr))

    # output
    f1.write(str(Ez)) # [nm]
    #f1.write(str(' '))
    #f1.write(str(k2x_temp))
    #f1.write(str(' '))
    #f1.write(str(nl))
    #f1.write(str(' '))
    #f1.write(str(nr))
    f1.write(str(' '))
    f1.write(str(nl-nr))
    f1.write(str(' '))
    f1.write(str(ts))
    f1.write(str(' '))
    f1.write(str(integrand)) # [J]
    f1.write('\n')

f1.close

# end
end()

# time display
elapsed_time = time.time()-start
print("elapsed_time:{:.2f}".format(elapsed_time) + "[sec]")
