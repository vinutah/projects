#!/usr/bin/env python

def fd1d_heat_explicit_test01(path,mode,weightsFile):
  """fd1d_heat_explicit_test01 does a simple test problem"""

  from fd1d_heat_explicit import fd1d_heat_explicit
  from fd1d_heat_explicit_cfl import fd1d_heat_explicit_cfl
  from r8mat_write import r8mat_write
  from r8vec_write import r8vec_write
  from mpl_toolkits.mplot3d import Axes3D
  from matplotlib import cm
  from matplotlib.ticker import LinearLocator, FormatStrFormatter
  import matplotlib.pyplot as plt
  import numpy as np

  print ''
  print 'FD1D_HEAT_EXPLICIT_TEST01:'
  print '  Compute an approximate solution to the time-dependent'
  print '  one dimensional heat equation:'
  print ''
  print '    dH/dt - K * d2H/dx2 = f(x,t)'
  print ''
  print '  Run a simple test case.'

  
  """Heat coefficient"""
 
  k = k_test01 ( )

#
#  X_NUM is the number of equally spaced nodes to use between 0 and 1.
#
  x_num = 21
  x_min = 0.0
  x_max = 1.0
  dx = ( x_max - x_min ) / ( x_num - 1 )
  x = np.linspace ( x_min, x_max, x_num )

#
#  T_NUM is the number of equally spaced time points between 0 and 10.0.
#
  t_num = 201
  t_min = 0.0
  t_max = 80.0
  dt = ( t_max - t_min ) / ( t_num - 1 )
  t = np.linspace ( t_min, t_max, t_num )

#
#  Get the CFL coefficient.
#
  cfl = fd1d_heat_explicit_cfl ( k, t_num, t_min, t_max, x_num, x_min, x_max )

  print ''
  print '  Number of X nodes = %d' % ( x_num )
  print '  X interval is [%f,%f]' % ( x_min, x_max )
  print '  X spacing is %f' % ( dx )
  print '  Number of T values = %d' % ( t_num )
  print '  T interval is [%f,%f]' % ( t_min, t_max )
  print '  T spacing is %f' % ( dt )
  print '  Constant K = %g' % ( k )
  print '  CFL coefficient = %g' % ( cfl )
#
#  Running the code produces an array H of temperatures H(t,x),
#  and vectors x and t.
#
  hmat = np.zeros ( ( x_num, t_num ) )

  for j in range ( 0, t_num ):
    if ( j == 0 ):
      h = ic_test01 ( x_num, x, t[j] ,mode)
      h = bc_test01 ( x_num, x, t[j], h ,mode)
      
    else:
      h = fd1d_heat_explicit ( x_num, x, t[j-1], dt, cfl, rhs_test01, bc_test01, h, mode, weightsFile )
    
    for i in range ( 0, x_num ):
      hmat[i,j] = h[i]
#
#  Plot the data.
#
  tmat, xmat = np.meshgrid ( t, x )
  fig = plt.figure ( )
  ax = fig.add_subplot ( 111, projection = '3d' )
  ax = Axes3D ( fig )
  surf = ax.plot_surface ( xmat, tmat, hmat )
  plt.xlabel ( '<---X--->' )
  plt.ylabel ( '<---T--->' )
  plt.title ( 'U(X,T)' )
  save_at  = path.strip()
  print path, save_at
  filename = str(save_at) + 'plot_test_' + str(mode)  + '.png'
  print filename
  plt.savefig (filename)
  #plt.show ( )
#
#  Write the data to files.
#
  filename = str(save_at) + 'h_test01.txt'
  r8mat_write ( filename, x_num, t_num, hmat )
  #filename = str(save_at) + 't_test01.txt'
  #r8vec_write ( filename, t_num, t )
  #filename = str(save_at) + 'x_test01.txt'
  #r8vec_write ( filename, x_num, x )

  print ''
  print '  H(X,T) written to "h_test01.txt"'
  print '  T values written to "t_test01.txt"'
  print '  X values written to "x_test01.txt"'
#
#  Terminate.
#
  print ''
  print 'FD1D_HEAT_EXPLICIT_TEST01:'
  print '  Normal end of execution.'

  return

def bc_test01 ( x_num, x, t, h, mode ):
# bc_test01 evaluates the boundary conditions for problem 1.
#    Input, integer X_NUM, the number of nodes.
#    Input, real X(X_NUM,1), the node coordinates.
#    Input, real T, the current time.
#    Input, real H(X_NUM), the current heat values.
#    Output, real H(X_NUM), the current heat values, after boundary
#    conditions have been imposed.
#
  #for uniform
  if mode == 'original_uni_1':
    h[0]       = 90.0
    h[x_num-1] = 70.0

  if mode == 'original_uni_2':
    h[0]       = 50.0
    h[x_num-1] = 50.0
  
  if mode == 'original_uni_3':
    h[0]       = 25.0
    h[x_num-1] = 85.0
  
  if mode == 'original_uni_4':
    h[0]       = 75.0
    h[x_num-1] = 80.0

  #for tri
  if mode == 'original_tri_1':
    h[0]       = 0.0
    h[x_num-1] = 0.0

  if mode == 'original_tri_2':
    h[0]       = 5.0
    h[x_num-1] = 5.0
  
  if mode == 'original_tri_3':
    h[0]       = 20.0
    h[x_num-1] = 20.0
  
  if mode == 'original_tri_4':
    h[0]       = 10.0
    h[x_num-1] = 10.0

  #for pwl  
  if mode == 'original_pwl_1':
    h[0]       = 0.0
    h[x_num-1] = 50.0

  if mode == 'original_pwl_2':
    h[0]       = 00.0
    h[x_num-1] = 90.0
  
  if mode == 'original_pwl_3':
    h[0]       = 0.0
    h[x_num-1] = 60.0
  
  if mode == 'original_pwl_4':
    h[0]       = 50.0
    h[x_num-1] = 0.0

  return h

def ic_test01 ( x_num, x, t , mode):
# ic_test01 evaluates the initial condition for problem 1.
#    Input, integer X_NUM, the number of nodes.
#    Input, real X(X_NUM), the node coordinates.
#    Input, real T, the initial time.
#    Output, real H(X_NUM), the heat values at the initial time.
#
  import numpy as np

  h = np.zeros ( x_num )

  for i in range ( 0, x_num ):
    #for uniform
    if mode == 'original_uni_1':
      h[i] = 50.0
    if mode == 'original_uni_2':
      h[i] = 25.0
    if mode == 'original_uni_3':
      h[i] = 0.0
    if mode == 'original_uni_4':
      h[i] = 10.0
    
    if mode == 'original_tri_1':
      A = 50
      if (i< (float(x_num/2))):
        h[i] = float(2*A*i/x_num)
      else:
        h[i] = -1 * (float(2*A*i/x_num)) + 2*A
 
    if mode == 'original_tri_2':
      A = 20
      if (i< (float(x_num/2))):
        h[i] = float(2*A*i/x_num)
      else:
        h[i] = -1 * (float(2*A*i/x_num)) + 2*A
 
    if mode == 'original_tri_3':
      A = 50
      if (i< (float(x_num/2))):
        h[i] = float(2*A*i/x_num)
      else:
        h[i] = -1 * (float(2*A*i/x_num)) + 2*A
    
    if mode == 'original_tri_4':
      A = 60
      if (i< (float(x_num/2))):
        h[i] = float(2*A*i/x_num)
      else:
        h[i] = -1 * (float(2*A*i/x_num)) + 2*A
    
    if mode == 'original_pwl_1':
          
      if (i < (float(x_num)/2) ):
        h[i] = 0.0
      else:
        h[i] = 50.0
        
    if mode == 'original_pwl_2':
      if (i < (float(x_num*3/4)) ):
        h[i] = 0.0
      else:
        h[i] = 70.0
        
    if mode == 'original_pwl_3':
      if (i < (float(x_num*3/4)) ):
        h[i] = 20.0
      else:
        h[i] = 0.0
        
    if mode == 'original_pwl_4':
      if (i < (float(x_num*3/4)) ):
        h[i] = 50.0
      else:
        h[i] = 0.0
        

  return h

def k_test01 ( ):
    """ 
    k_test01 evaluates the conductivity for problem 1.
    Output, real K, the conducitivity.
    """
    k = 0.002
    return k

def rhs_test01 ( x_num, x, t ):

# RHS_TEST01 evaluates the right hand side for problem 1.
#    Input, integer X_NUM, the number of nodes.
#    Input, real X(X_NUM,1), the node coordinates.
#    Input, real T, the current time.
#    Output, real VALUE(X_NUM,1), the source term.
  import numpy as np

  value = np.zeros ( x_num )

  return value

if ( __name__ == '__main__' ):
  from timestamp import timestamp
  timestamp ( )
  fd1d_heat_explicit_test01 (path, mode, weightsFile)

  timestamp ( )

