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


def readweights(weightsFile,w):
    with open(weightsFile,'r') as f:
        W = f.readlines()
        w.append(W[-1].strip())
        w.append(W[-2].strip())
        w.append(W[-3].strip())
    f.close()
    return w

def fd1d_heat_explicit ( x_num, x, t, dt, cfl, rhs, bc, h, mode , weightsFile):
  import numpy as np

  h_new = np.zeros ( x_num )

  f = rhs ( x_num, x, t )

  for c in range ( 1, x_num - 1 ):
    l = c - 1
    r = c + 1


    if mode == 'native': 
        h_new[c] = h[c] + cfl * ( h[l] - 2.0 * h[c] + h[r] ) + dt * f[c]
        save_training_data(h_new[c] , h[l] , h[c] , h[r] )

    if mode == 'ml_model':
        w = list()
        w = readweights(weightsFile,w)
        #print 'w[0]=%f' % ( float(str(w[0])) )
        #print 'w[1]=%f' % ( float(str(w[1])) )
        #print 'w[2]=%f' % ( float(str(w[2])) )
        #
        #print 'h[l]=%f' % ( float(str(h[l])) )
        #print 'h[c]=%f' % ( float(str(h[c])) )
        #print 'h[r]=%f' % ( float(str(h[r])) )

        w1 =  float(str(w[0]))
        w2 =  float(str(w[1]))
        w3 =  float(str(w[2]))
        
        f1 =  float(str(h[l]))
        f2 =  float(str(h[c]))
        f3 =  float(str(h[r]))

        h_new[c] = w1*f1 + w2*f2 + w3*f3
    
  h_new = bc ( x_num, x, t + dt, h_new )

  return h_new
