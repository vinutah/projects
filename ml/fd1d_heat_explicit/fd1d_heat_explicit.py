#!/usr/bin/env python

def save_training_data(dep_var, feature_1, feature_2, feature_3):
    
    filename = "./data/training/train.svm"
    with open(filename,'a') as f:
        line  =         str(dep_var)
        line += ' 1:' + str(feature_1) 
        line += ' 2:' + str(feature_2) 
        line += ' 3:' + str(feature_3)
        line += '\n'
        f.write(line)


def fd1d_heat_explicit ( x_num, x, t, dt, cfl, rhs, bc, h ):
## FD1D_HEAT_EXPLICIT: Finite difference solution of 1D heat equation.
#  Discussion:
#
#    This program takes one time step to solve the 1D heat equation 
#    with an explicit method.
#
#    This program solves
#
#      dUdT - k * d2UdX2 = F(X,T)
#
#    over the interval [A,B] with boundary conditions
#
#      U(A,T) = UA(T),
#      U(B,T) = UB(T),
#
#    over the time interval [T0,T1] with initial conditions
#
#      U(X,T0) = U0(X)
#
#    The code uses the finite difference method to approximate the
#    second derivative in space, and an explicit forward Euler approximation
#    to the first derivative in time.
#
#    The finite difference form can be written as
#
#      U(X,T+dt) - U(X,T)                  ( U(X-dx,T) - 2 U(X,T) + U(X+dx,T) )
#      ------------------  = F(X,T) + k *  ------------------------------------
#               dt                                   dx * dx
#
#    or, assuming we have solved for all values of U at time T, we have
#
#      U(X,T+dt) = U(X,T) + cfl * ( U(X-dx,T) - 2 U(X,T) + U(X+dx,T) ) + dt * F(X,T) 
#
#    Here "cfl" is the Courant-Friedrichs-Loewy coefficient:
#
#      cfl = k * dt / dx / dx
#
#    In order for accurate results to be computed by this explicit method,
#    the CFL coefficient must be less than 0.5!
#
#    Input, integer X_NUM, the number of points to use in the spatial dimension.
#    Input, real X(X_NUM,1), the coordinates of the nodes.
#    Input, real T, the current time.
#    Input, real DT, the size of the time step.
#    Input, real CFL, the Courant-Friedrichs-Loewy coefficient,
#    computed by FD1D_HEAT_EXPLICIT_CFL.
#    Input, real H(X_NUM,1), the solution at the current time.
#    Input, @RHS, the function which evaluates the right hand side.
#    Input, @BC, the function which evaluates the boundary conditions.
#    Output, real H_NEW(X_NUM,1), the solution at time T+DT.
#
  import numpy as np

  h_new = np.zeros ( x_num )

  f = rhs ( x_num, x, t )

  for c in range ( 1, x_num - 1 ):
    l = c - 1
    r = c + 1


    h_new[c] = h[c] + cfl * ( h[l] - 2.0 * h[c] + h[r] ) + dt * f[c]
    save_training_data(h_new[c] , h[l] , h[c] , h[r] )

  h_new = bc ( x_num, x, t + dt, h_new )

  return h_new
