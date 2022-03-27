# test code
# Coded by Takuro TOKUNAGA
# Last modified: July 27 2019

import numpy as np
import cmath
import time
import sys
from scipy import special, optimize

start = time.time()
z1 = 3
z2 = 100
nzmax = 100
dz = (z2-z1)/nzmax
z_counter = 0
sz = z1

while z_counter <= nzmax:

    sz = sz + dz
    z_counter = z_counter + 1

print(str(sz))
print(str(z_counter))

# time display
elapsed_time = time.time()-start
print("elapsed_time:{:.2f}".format(elapsed_time) + "[sec]")
