# Current density by Fisher 2002
# Coded by Takuro TOKUNAGA
# Last modified: Nov 22 2019
# in-progress
# T.S. Fisher and D. G. Walker, Transactions of the ASME 124 954 (2002)
# F.A.M. Koeck and R.J. Nemanich, Diamond & Related Materials 14 2051-2054 (2005)

import numpy as np
import cmath
import time

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

# tip parameters
phi = 1.7 # [eV]
Field = 1.08*np.power(10,4) # V/cm
beta = 620

# function: begin
def begin():
    print ("begin")

# function: end
def end():
    print ("end")

# ideal barrier profile
def current(arg_beta, arg_F, arg_phi):
    term1 = 1.5*np.power(10.,-6)*np.power((arg_beta*arg_F),2.0)/arg_phi
    term2 = np.exp(10.4/arg_phi)
    temp = -6.44*np.power(10.,7)*np.power(arg_phi,1.5)
    term3 = np.exp(temp/(arg_beta*arg_F))

    y = term1*term2*term3

    return y # [A/cm2]

# main start
begin()
current_density = current(beta, Field, phi)

print(str(current_density))


# end
end()

# time display
elapsed_time = time.time()-start
print("elapsed_time:{:.2f}".format(elapsed_time) + "[sec]")
