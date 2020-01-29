#!/usr/bin/env python

from glob import glob
from shutil import copyfile


hd5 = glob("*.hdf5")


for f in hd5:

    par = f[9:-11]

    new = "pce_{}.hdf5".format(par)

    copyfile(f, new)

    print(new)
