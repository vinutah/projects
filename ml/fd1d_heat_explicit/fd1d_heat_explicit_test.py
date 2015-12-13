#!/usr/bin/env python

def fd1d_heat_explicit_test ( ):
  """fd1d_heat_explicit_test tests the fd1d_heat_explicit library."""

  from fd1d_heat_explicit_test01 import fd1d_heat_explicit_test01 
  from fd1d_heat_explicit_test02 import fd1d_heat_explicit_test02

  print ''
  print 'FD1D_HEAT_EXPLICIT_TEST:'
  print '  Python version.'
  print '  Test the FD1D_HEAT_EXPLICIT library.'

  fd1d_heat_explicit_test01 ( )
  fd1d_heat_explicit_test02 ( )

  print ''
  print 'FD1D_HEAT_EXPLICIT_TEST:'
  print '  Normal end of execution.'

  return

if ( __name__ == '__main__' ):
  from timestamp import timestamp
  timestamp ( )
  fd1d_heat_explicit_test ( )
  timestamp ( )
