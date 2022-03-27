# Electron tunneling by Devon's formula
# Gap: fixed value
# Coded by Takuro TOKUNAGA
# Last modified: July 27 2019
# on-going: Wmax needs to be found for every barrier profile

import numpy as np
import cmath
import time
import sys
from scipy.integrate import quad
from scipy import special, optimize

start = time.time()

# Unit conversion:
ucev_to_j = 1.602176620898*np.power(10.,-19) # electron volt to joule
ucnano = 1.0*np.power(10.,-9)                # nm to m
ucangs = 1.0*np.power(10.,-10)               # angstrom to m
uccm = 1.0*np.power(10.,-2)                  # cm to m

# physical constants
kb = 1.38064852*np.power(10.,-23)           # Boltzmann constant
ch = 6.62607004*np.power(10.,-34)           # Planck constant
rh = ch/(2*np.pi)                           # reduced Planck constant
c = 299792458                               # light speed, [m]
eps_0 = 8.854187817*np.power(10.,-12)       # [F/m]
sq = 1.602176620898*np.power(10.,-19)       # elementary cahrge [C], small q
me = 9.10938356*np.power(10.,-31)           # electron mass, [kg]

## Parameters (start) ##
# work function
phi_e = 2.10 # [eV], emitter work function
phi_c = 1.80 # [eV], collector work function
se = 1 # [eV/V], single electron
nmax = 100
Vmin = 0 # [V]
volt = Vmin # [V]
Vmax = 1.00 # [V]
dV = (Vmax-Vmin)/nmax
#Wmax = 2.161729 # [eV]
Wmax = 2.23 # [eV]


# temperature
temperature_L = 1575 # [K], emitter
temperature_R = 1000 # [K], collector

# function: begin
def begin():
    print ("begin")

# function: end
def end():
    print ("end")

def JL(arg_Wmax): # emitter, Eq (3.3), arg_Wmax: [eV]
    cA = 4*np.pi*me*np.power(kb,2.0)*sq/np.power(ch,3.0) # capital A
    term1 = np.power(temperature_L,2.0) # [K^2]
    term2 = np.exp(-arg_Wmax*ucev_to_j/(kb*temperature_L)) #[J/J]
    term3 = cA*term1*term2

    return term3 # [C/(m^2*s)]

def JR(arg_Wmax, arg_volt): # collector, Eq (3.3), arg_Wmax: [eV] arg_volt: [V]
    cA = 4*np.pi*me*np.power(kb,2.0)*sq/np.power(ch,3.0) # capital A
    term1 = np.power(temperature_R,2.0)
    term2 = np.exp((-(arg_Wmax-se*arg_volt)*ucev_to_j)/(kb*temperature_R)) # [J/J]
    term3 = cA*term1*term2

    return term3 # [C/(m^2*s)]

# main start
begin()

# file open
f1 = open('current.txt', 'w')

while volt < Vmax: # [V]

  JTE = JL(Wmax)-JR(Wmax,volt) # [A/m2]

  # output
  f1.write(str(volt)) # [V]
  f1.write(str(' '))
  f1.write(str(JTE*np.power(uccm,2.0))) # [A/cm2]
  f1.write('\n')

  # energy update
  volt = volt + dV

# file close
f1.close

# end
end()

# time display
elapsed_time = time.time()-start
print("elapsed_time:{:.2f}".format(elapsed_time) + "[sec]")
