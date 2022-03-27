# Electron tunneling by Devon's formula
# Gap: fixed value
# Coded by Takuro TOKUNAGA
# Last modified: January 27 2019
# Double Trapezoidal

# gold work function:
# P. A. Anderson, Phys. Rev. 115 553 (1959)
# gold fermi energy:
# N. W. Ashcroft and N. D. Mermin, Solid State Physics, 1976

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
kb = 1.38064852*np.power(10.,-23)            # Boltzmann constant
ch = 6.62607004*np.power(10.,-34)            # Planck constant
rh = ch/(2*np.pi)                            # reduced Planck constant
c = 299792458                                # light speed, [m]
eps_0 = 8.854187817*np.power(10.,-12)        # [F/m]
sq = 1.602176620898*np.power(10.,-19)        # elementary cahrge [C], small q
me = 9.10938356*np.power(10.,-31)            # electron mass, [kg]

# mathematical Parameters
criteria = 708 # criteria

# gap
number = 500 # [-], integral parameter, initial: 10000
nzmax = 1000 # [-], integral parameter, initial: 1000

small = 0.001*ucnano # zero [m]
gapmin = small # [m]
gap = gapmin   # [m]
gapmax = 0.401*ucnano-small   # [m], minimum limit of calculation
gapmaxmax = 3.0*ucnano       # [m], max of gapmax
dgap = (gapmax-gapmin)/number # [m]

## Parameters (start) ##
# work function
#phi_e = 2.10 # [eV], emitter work function
#phi_c = 1.80 # [eV], collector work function
wgold = 5.10 # [eV]
#wgold = 1.00 # [eV]
phi_e = wgold # [eV], emitter work function, gold
phi_c = wgold # [eV], collector work function, gold

volt = -0.6 # [V], load (vias) voltage
#volt = 0.0 # [V], load (vias) voltage

se = 1 # [eV/V], single electron
volt = se*volt # [eV]

# temperature
#temperature_L = 1575 # [K], emitter
#temperature_R = 1000 # [K], collector
#temperature_L = 305 # [K], emitter
#temperature_R = 300 # [K], collector
#temperature_L = 295 # [K], emitter
#temperature_R = 195 # [K], collector
temperature_L = 280 # [K], emitter
temperature_R = 120 # [K], collector

delta_temperature = temperature_L-temperature_R # [K]

# fermi energy
#Ef_L = 0 # [eV], emitter (Devon's Ph.D. thesis)
#Ef_R = volt # [eV], collector:
Ef_gold = 5.53 # [eV], relative to the emitter fermi energy (Devon's Ph.D. thesis, P.36)
#Ef_gold = 1.12*0.5 # [eV]
Ef_L = Ef_gold
Ef_R = Ef_gold

# area
#radius = 450*ucnano # [m], 170~250 [nm]
radius = 30*ucnano # [m], 170~250 [nm]
Atip = np.pi*np.power(radius,2.0) # [m2]

## Parameters (end) ##

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
    ibp = phi_e-(phi_e-phi_c-volt)*(arg_x/gapmax) # [eV], Devon's Ph.D. thesis

    # space charge
    scbp = 0 # in case of nanosacle, Devon's Ph.D. thesis

    # image charge barrier profile
    term1 = np.power(sq,2.0)/(16*np.pi*eps_0*gapmax) # [J]
    term1 = term1/ucev_to_j # [eV]
    term2 = -2*special.digamma(1) # [-]
    term3 = special.digamma(arg_x/gapmax) # [-]
    term4 = special.digamma(1-arg_x/gapmax) # [-]
    icbp = term1*(term2+term3+term4) # [eV]
    #icbp = 0 # to see the effect of image charge barrier profile

    total = ibp+scbp+icbp

    # consider fermi level
    total = total + Ef_gold

    return total # [eV]

# main start
begin()

# file open
f1 = open('results_d.txt', 'w')
f1.write('nm V W/cm2 W/cm2 A/cm2 A/cm2 nW/K nW/K W/m2K W/m2K A/m2 A/m2')
f1.write('\n')

# gapmax loop
while gapmax < gapmaxmax:
# initialization
    old = 0
    gap = gapmin # [m]
    dgap = (gapmax-gapmin)/number # [m]

    while gap < gapmax: # [m]
        wz = wz_total(gap) # [eV]

        # find Wmax(Ezmax)
        new = wz # [eV]
        dif = new-old # [eV], difference

        if dif>0:
            Wmax = new # [eV]

        old = new

        # gap update
        gap = gap + dgap # [m]

    # z integral calculation
    # Ez discretization (don't change the order)
    Ezmin = 0 # [eV], integral lower: change here, depends on condition
    Ez = Ezmin # [eV]
    Ezmax = Wmax #[eV]
    factor = 0.9995 # [-]
    dEz = factor*(Ezmax-Ezmin)/number # [eV]
    Ez_counter = 0 # [-]
    z_counter = 0 # [-]

    # z1 initialization
    z1 = gapmin # [m]
    z2 = gapmax # [m]

    # flux
    flux_qe = 0 # [W/m2], initialization
    Jqe = 0 # [A/m2], initialization

    # Ez integral calculation
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

        # define integrand
        def integrand_z(arg_z):
            integrand = wz_total(arg_z)-Ez # [eV]
            integrand = (wz_total(arg_z)-Ez)*ucev_to_j # [J]
            #print(str(Ez))
            integrand = np.sqrt(integrand) # [sqrt(J)]
            return integrand

        #print(str(z2))

        # transmission for each Ez
        term1 = -np.sqrt(8*me)/rh # [sqrt(kg)/(J*s)]

        # Integral by QUADPACK
        #term2 = quad(integrand_z, z1, z2)[0] # [sqrt(J)*m]

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

        #print(str(sz))

        term3 = term1*term2 # [-]
        #print(str(term3))
        trans = np.exp(term3) # transmission, [-]

        # output

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

            # interpretatio of term 2
            #term2 = -(arg_Ez-volt-Ef_L)*ucev_to_j/(kb*temperature_L) # [-] eV to J
            term2 = -(arg_Ez-Ef_L)*ucev_to_j/(kb*temperature_L) # [-] eV to J

            # overflow avoidance
            if term2 > criteria:
                integrand = 0
            else:
                integrand = term1*np.log(1+np.exp(term2)) # P.35, Devon's Ph.D. thesis

            return integrand

        # define Ez right integrand
        def NR(arg_Ez): # [eV] collector
            #term1 = me*kb*temperature_R/(2*np.power(np.pi,2.0)*np.power(rh,3.0))
            term1 = 4*np.pi*me*kb*temperature_R/np.power(ch,3.0)

            # interpretatio of term 2
            term2 = -(arg_Ez-Ef_R)*ucev_to_j/(kb*temperature_R) # [-] eV to J
            #term2 = -(arg_Ez-Ef_R-volt)*ucev_to_j/(kb*temperature_R) # [-] eV to J

            # overflow avoidance
            if term2 > criteria:
                integrand = 0
            else:
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

        # Electron & thermionic conductance
        cond_qe = flux_qe*Atip/delta_temperature # [W/K], [(W/m2)*m2/K]
        cond_therm = QTE*Atip/delta_temperature # [W/K], [(W/m2)*m2/K]

        # Ez update
        Ez = Ez + dEz

        # counter update
        Ez_counter = Ez_counter + 1

        # counter reset (transmission z1, z2 integral)
        z_counter = 0
        sz = 0
        dz = 0

    # output
    f1.write(str(gapmax/ucnano)) # [nm]
    f1.write(str(' '))
    f1.write(str(volt))          # [V]
    f1.write(str(' '))
    f1.write(str(Jqe*np.power(uccm,2.0))) # [A/cm2], electron tunneling
    f1.write(str(' '))
    f1.write(str(JTE*np.power(uccm,2.0))) # [A/cm2], thermionic
    f1.write(str(' '))
    f1.write(str(flux_qe*np.power(uccm,2.0))) # [W/cm2], electron tunneling
    f1.write(str(' '))
    f1.write(str(QTE*np.power(uccm,2.0)))     # [W/cm2], thermionic
    f1.write(str(' '))
    f1.write(str(cond_qe/ucnano))             # [nW/K], electron tunneling
    f1.write(str(' '))
    f1.write(str(cond_therm/ucnano))          # [nW/K], thermionic
    f1.write(str(' '))
    f1.write(str(cond_qe/Atip))               # [W/m2K], electron tunneling
    f1.write(str(' '))
    f1.write(str(cond_therm/Atip))            # [W/m2K], thermionic
    f1.write(str(' '))
    f1.write(str(Jqe)) # [A/m2], electron tunneling
    f1.write(str(' '))
    f1.write(str(JTE)) # [A/m2], thermionic
    f1.write('\n')

    #print(str(flux)) # flux [W/m2]
    print("Gap:{:.2f}".format(gapmax/ucnano) + "[nm]")
    print("Qqe:{:.2f}".format(flux_qe*np.power(uccm,2.0)) + "[W/cm2]")
    print("Qtherm:{:.4f}".format(QTE*np.power(uccm,2.0)) + "[W/cm2]")
    print("Jqe:{:.2f}".format(Jqe*np.power(uccm,2.0)) + "[A/cm2]")
    print("Jtherm:{:.2f}".format(JTE*np.power(uccm,2.0)) + "[A/cm2]")
    print("Gqe:{:.2f}".format(cond_qe/ucnano) + "[nW/K]")
    print("Gtherm:{:.2f}".format(cond_therm/ucnano) + "[nW/K]")
    print("\n")

    # counter reset

    # dgampax update
    #if gapmax<0.9*ucnano:
    if gapmax<2.0*ucnano:
        dgapmax = 0.1*ucnano
    #elif gapmax>=1.0*ucnano:
    elif gapmax>=2.0*ucnano:
        dgapmax = 1.0*ucnano

    gapmax = gapmax + dgapmax

# file close
f1.close

# end
end()

# time display
elapsed_time = time.time()-start
print("elapsed_time:{:.2f}".format(elapsed_time) + "[sec]")
