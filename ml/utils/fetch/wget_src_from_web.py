#!/usr/bin/env python

import sys,os

util = 'wget '
src  = 'py_src'
code = 'fd1d_heat_explicit'
site = 'https://people.sc.fsu.edu/~jburkardt/' + str(src) + '/' + str(code) + '/'
#print site


fname = sys.argv[1]
#print fname

if (__name__ == '__main__'):

    with open(fname) as f:
        lines = f.readlines()
   
    for line in lines:
        cmd = str(util) + str(site) + str(line)
        #print cmd
        os.system(cmd)
