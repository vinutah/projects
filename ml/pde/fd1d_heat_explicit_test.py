#!/usr/bin/env python

import logging
logging.basicConfig(level=logging.DEBUG)

import argparse

weightsFile = ''

def fd1d_heat_explicit_test(path, mode, weightsFile):
  """fd1d_heat_explicit_test tests the fd1d_heat_explicit library."""

  from fd1d_heat_explicit_test01 import fd1d_heat_explicit_test01 
  #from fd1d_heat_explicit_test02 import fd1d_heat_explicit_test02

  print ''
  print 'FD1D_HEAT_EXPLICIT_TEST:'
  print '  Python version.'
  print '  Test the FD1D_HEAT_EXPLICIT library.'

  fd1d_heat_explicit_test01(path,mode, weightsFile)
  #fd1d_heat_explicit_test02 ( )

  print ''
  print 'FD1D_HEAT_EXPLICIT_TEST:'
  print '  Normal end of execution.'

  return

if ( __name__ == '__main__' ):
  parser = argparse.ArgumentParser( description = "this program is the wrapper to the tests")\

  parser.add_argument("-solve",   help='path to store outputs',required=True)
  parser.add_argument("-mode",   help='path to store outputs',required=True)
  parser.add_argument("-weights",   help='path to store outputs',required=False)

  args = parser.parse_args()

  from timestamp import timestamp
  timestamp ( )
  if args.solve:  
    path = args.solve
    mode = args.mode
    weightsFile = args.weights
    fd1d_heat_explicit_test(path,mode, weightsFile)
  timestamp ( )
