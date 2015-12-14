#!/usr/bin/env python

import logging
logging.basicConfig(level=logging.DEBUG)

import argparse

def fd1d_heat_explicit_test(path):
  """fd1d_heat_explicit_test tests the fd1d_heat_explicit library."""

  from fd1d_heat_explicit_test01 import fd1d_heat_explicit_test01 
  #from fd1d_heat_explicit_test02 import fd1d_heat_explicit_test02

  print ''
  print 'FD1D_HEAT_EXPLICIT_TEST:'
  print '  Python version.'
  print '  Test the FD1D_HEAT_EXPLICIT library.'

  fd1d_heat_explicit_test01(path)
  #fd1d_heat_explicit_test02 ( )

  print ''
  print 'FD1D_HEAT_EXPLICIT_TEST:'
  print '  Normal end of execution.'

  return

if ( __name__ == '__main__' ):
  parser = argparse.ArgumentParser( description = "this program is the wrapper to the tests")\

  parser.add_argument("-solve",   help='path to store outputs',required=True)

  args = parser.parse_args()

  from timestamp import timestamp
  timestamp ( )
  if args.solve:  
    path = args.solve
    fd1d_heat_explicit_test(path)
  timestamp ( )
