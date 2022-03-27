# Electron tunneling by Devon's formula
# Gap: fixed value
# Coded by Takuro TOKUNAGA
# Last modified: July 26 2019
# Steably working (QUADPAC)

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

# gap
small = 0.01*ucnano # zero [m]
gapmin = small # [m]
gap = gapmin # [m]
gapmax = 100.01*ucnano-small # [m]
number = 500 # [-]
dgap = (gapmax-gapmin)/number # [m]

## Parameters (start) ##
# work function
phi_e = 2.10 # [eV], emitter work function
phi_c = 1.80 # [eV], collector work function
#phi_e = 4.74 # [eV], emitter work function
#phi_c = 4.74 # [eV], collector work function

volt = 0.4 # [V], load (vias) voltage
#volt = 0.6 # [V], load (vias) voltage

se = 1 # [eV/V], single electron
volt = se*volt # [eV]

# temperature
temperature_L = 1575 # [K], emitter
temperature_R = 1000 # [K], collector
#temperature_L = 295 # [K], emitter
#temperature_R = 195 # [K], collector

delta_temperature = temperature_L-temperature_R # [K]

# fermi energy
Ef_L = 0 # [eV], emitter (Devon's Ph.D. thesis)
Ef_R = volt # [eV], collector: correct??

# area
tip_diagonal = 210*ucnano # [m], 170~250 [nm]
lx = tip_diagonal/np.sqrt(2) # [m], 120.2~176.7 [nm]
ly = lx # [m]
Atip = lx*ly # [m2]
#print(str(Atip))

## Parameters (end) ##

# function: begin
def begin():
    print ("begin")

# function: end
def end():
    print ("end")

# total profile
def wz_total(arg_x):
    # ideal barrier profile
    #ibp = phi_e+volt-(phi_e+volt-phi_c)*(arg_x/gapmax) # [eV]
    ibp = phi_e-(phi_e-phi_c-volt)*(arg_x/gapmax) # [eV]

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

    return total # [eV]

# main start
begin()

# file open
f1 = open('bprofile.txt', 'w')
f2 = open('root_z1z2.txt', 'w')
f3 = open('transmission.txt', 'w')
f4 = open('results.txt', 'w')

# initialization
old = 0

# barrier profile
f1.write('nm - eV')
f1.write('\n')
while gap < gapmax: # [m]

  wz = wz_total(gap) # [eV]

  # find Wmax(Ezmax)
  new = wz # [eV]
  dif = new-old # [eV], difference

  if dif>0:
      Wmax = new # [eV]

  old = new

  # output
  f1.write(str(gap/ucnano)) # [nm]
  f1.write(str(' '))
  f1.write(str(gap/gapmax)) # [-], relative position
  f1.write(str(' '))
  f1.write(str(wz)) # [eV]
  f1.write('\n')

  # gap update
  gap = gap + dgap # [m]

# z integral calculation
# Ez discretization
Ezmin = 0 # [eV]
Ez = Ezmin # [eV]
Ezmax = Wmax #[eV]
dEz = (Ezmax-Ezmin)/number # [eV]
Ez_counter = 0 # [-]

# z1 initialization
z1 = gapmin # [m]
z2 = gapmax # [m]

# flux
flux_qe = 0 # [W/m2], initialization
Jqe = 0 # [A/m2], initialization

# Ez integral calculation
f2.write('nm nm - - eV')
f2.write('\n')
while Ez < Ezmax: # [eV]

    # z1, z2 initialization
    z1 = gapmin # [m]
    z2 = gapmax # [m]

    # z1 & z2 finding routine for individual Ez
    # z1
    z1 = z1 + dgap # [m]
    wz1 = wz_total(z1) # [eV]
    while wz1 < Ez: # [eV]
        z1 = z1 + dgap # [m]
        wz1 = wz_total(z1) # [eV]
        if wz1 > Ez: # [eV]
            break

    # z2
    z2 = z2 - dgap # [m]
    wz2 = wz_total(z2) # [eV]
    while wz2 < Ez: # [eV]
        z2 = z2 - dgap # [m]
        wz2 = wz_total(z2) # [eV]
        if wz2 > Ez: # [eV]
            break

    # output
    f2.write(str(z1/ucnano)) # [nm]
    f2.write(str(' '))
    f2.write(str(z2/ucnano)) # [nm]
    f2.write(str(' '))
    f2.write(str(z1/gapmax)) # [-], relative position
    f2.write(str(' '))
    f2.write(str(z2/gapmax)) # [-], relative position
    f2.write(str(' '))
    f2.write(str(Ez)) # [eV]
    f2.write('\n')

    # define integrand
    def integrand_z(arg_z):
        integrand = wz_total(arg_z)-Ez # [eV]
        integrand = (wz_total(arg_z)-Ez)*ucev_to_j # [J]
        #print(str(Ez))
        integrand = np.sqrt(integrand) # [sqrt(J)]
        return integrand

    #print(str(z2))

    # transmission for each Ez
    term1 = -np.sqrt(8*me)/rh
    term2 = quad(integrand_z, z1, z2)[0] # [sqrt(J)*m]
    term3 = term1*term2 # [-]
    #print(str(term3))
    trans = np.exp(term3) # transmission, [-]

    # output
    f3.write(str(Ez)) # [eV]
    f3.write(str(' '))
    f3.write(str(trans)) # [-]
    f3.write('\n')

    # define Ez left integrand
    # coefficient
    cA = 4*np.pi*me*np.power(kb,2.0)*sq/np.power(rh*(2*np.pi),3.0) # capital A

    def JL(arg_Wmax): # emitter, Eq (3.3), arg_Wmax: [eV]
        term1 = np.power(temperature_L,2.0) # [K^2]
        term2 = np.exp(-arg_Wmax*ucev_to_j/(kb*temperature_L)) #[J/J]
        term3 = cA*term1*term2

        return term3 # [C/(m^2*s)]

    def JR(arg_Wmax): # collector, Eq (3.3), arg_Wmax: [eV]
        term1 = np.power(temperature_R,2.0)
        term2 = np.exp((-(arg_Wmax-volt)*ucev_to_j)/(kb*temperature_R)) # [J/J]
        term3 = cA*term1*term2

        return term3 # [C/(m^2*s)]

    # define Ez left integrand
    def NL(arg_Ez): # [eV] emitter
        #term1 = me*kb*temperature_L/(2*np.power(np.pi,2.0)*np.power(rh,3.0))
        term1 = 4*np.pi*me*kb*temperature_L/np.power(ch,3.0)
        #term2 = -(arg_Ez-volt-Ef_L)*ucev_to_j/(kb*temperature_L) # [-] eV to J
        term2 = -(arg_Ez-Ef_L)*ucev_to_j/(kb*temperature_L) # [-] eV to J
        integrand = term1*np.log(1+np.exp(term2)) # P.35, Devon's Ph.D. thesis

        return integrand

    # define Ez right integrand
    def NR(arg_Ez): # [eV] collector
        #term1 = me*kb*temperature_R/(2*np.power(np.pi,2.0)*np.power(rh,3.0))
        term1 = 4*np.pi*me*kb*temperature_R/np.power(ch,3.0)
        term2 = -(arg_Ez-Ef_R)*ucev_to_j/(kb*temperature_R) # [-] eV to J
        integrand = term1*np.log(1+np.exp(term2)) # P.35, Devon's Ph.D. thesis

        return integrand

    # define total integrand, electron tunneling flux
    def total(arg_Ez): # [eV]
        term1 = (arg_Ez*ucev_to_j+kb*temperature_L)*NL(arg_Ez)
        term2 = (arg_Ez*ucev_to_j+kb*temperature_R)*NR(arg_Ez-volt)
        term3 = term1 - term2
        integrand = trans*term3 # Eq (3.10)

        return integrand

    def total_current(arg_Ez): # [eV]
        term1 = NL(arg_Ez)
        term2 = NR(arg_Ez-volt)
        term3 = term1 - term2
        integrand_i = sq*trans*term3 # Eq (3.8)

        return integrand_i

    # integration for Ez, quantum tunneling flux, Eq (3.10)
    if Ez_counter==0 or Ez_counter==number:
        flux_qe = flux_qe + total(Ez)*(dEz*ucev_to_j)*0.5 # [W/m2]
    else:
        flux_qe = flux_qe + total(Ez)*(dEz*ucev_to_j) # [W/m2]

    # integration for Ez, quantum tunneling current, Eq (3.8)
    if Ez_counter==0 or Ez_counter==number:
        Jqe = Jqe + total_current(Ez)*(dEz*ucev_to_j)*0.5 # [A/m2]
    else:
        Jqe = Jqe + total_current(Ez)*(dEz*ucev_to_j) # [A/m2]


    # Wmax = Ezmax
    # thermionic emission current, Eq (3.3)
    JTE = JL(Ezmax)-JR(Ezmax) # [A/m2]

    # thermionic flux, Eq (3.9)
    QTE = JTE*(Ezmax/sq) + 2*kb*((JL(Ezmax)*temperature_L-JR(Ezmax)*temperature_R)/sq) # [W/m2]

    # Ez update
    Ez = Ez + dEz

    # counter update
    Ez_counter = Ez_counter + 1

# output
#cond = flux_qe*Atip/delta_temperature # [W/K], [(W/m2)*m2/K]
#cond_therm = QTE*Atip/delta_temperature # [W/K], [(W/m2)*m2/K]

#print(str(flux)) # flux [W/m2]
print("Gap:{:.2f}".format(gapmax/ucnano) + "[nm]")
print("Qqe:{:.2f}".format(flux_qe*np.power(uccm,2.0)) + "[W/cm2]")
print("Qtherm:{:.4f}".format(QTE*np.power(uccm,2.0)) + "[W/cm2]")
print("Jqe:{:.2f}".format(Jqe*np.power(uccm,2.0)) + "[A/cm2]")
print("Jtherm:{:.2f}".format(JTE*np.power(uccm,2.0)) + "[A/cm2]")

# output
f4.write('nm V W/cm2 W/cm2 A/cm2 A/cm2')
f4.write('\n')
f4.write(str(gapmax/ucnano)) # [nm]
f4.write(str(' '))
f4.write(str(volt)) # [V]
f4.write(str(' '))
f4.write(str(flux_qe*np.power(uccm,2.0))) # [W/cm2], electron tunneling
f4.write(str(' '))
f4.write(str(QTE*np.power(uccm,2.0))) # [W/cm2], thermionic
f4.write(str(' '))
f4.write(str(Jqe*np.power(uccm,2.0))) # [A/cm2], electron tunneling
f4.write(str(' '))
f4.write(str(JTE*np.power(uccm,2.0))) # [A/cm2], thermionic
f4.write('\n')

# file close
f1.close
f2.close
f3.close
f4.close

# end
end()

# time display
elapsed_time = time.time()-start
print("elapsed_time:{:.2f}".format(elapsed_time) + "[sec]")
