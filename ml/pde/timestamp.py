#!/usr/bin/env python

def timestamp ( ):
  import time

  t = time.time ( )
  print time.ctime ( t )

  return None

def timestamp_test ( ):
  print ''
  print 'TIMESTAMP_TEST:'
  print '  Python version:'
  print '  TIMESTAMP prints a timestamp of the current date and time.'
  print ''

  timestamp ( )

  print ''
  print 'TIMESTAMP_TEST:'
  print '  Normal end of execution.'

if ( __name__ == '__main__' ):
  timestamp_test ( )
