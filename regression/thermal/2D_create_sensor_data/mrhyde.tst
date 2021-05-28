#!/usr/bin/env python2.7
#-------------------------------------------------------------------------------

import sys, os
import subprocess as sp
import string
import shutil
import numpy as np
sys.path.append("../../scripts")
from mrhyde_test_support import *

# ==============================================================================
# Parsing input

# No reason to format the description as it will be reformatted by optparse.
desc = ''' gradient check for non-mortar thermal problem
       '''

its = mrhyde_test_support(desc)

print('Because of the diff test on the log file, this test needs ')
print('to run with "-v".  There is a buffering issue.')
print('Setting the verbosity to True.')
its.opts.verbose = True

#-------------------------------------------------------------------------------
# Problem Parameters

root = 'mrhyde'   # root filename for test
aeps = 5.0e-15     # absolute error tolerance
reps = 1.0e-12     # relative error tolerance
fdtol= 5.0e-10     # finite difference gradient tolerance

# These comments are for testing with the runtest.py utility.
#TESTING active
#TESTING -n 1
#TESTING -k thermal,sensors

# ==============================================================================
status = 0

# ------------------------------
if its.opts.preprocess:
  if its.opts.verbose != 'none': print('---> Preprocessing %s' % (root))
  status += its.call('echo "  No preprocessing, yet."')

status += its.call('mpiexec -n 1 ../../mrhyde >& mrhyde.log')
hostname = os.getenv('HOSTNAME') 
if hostname.find('weaver') != -1: 
  its.call('sed -i \'1,11d;\' mrhyde.log') 
  its.call('sed -i \'/weaver/d\' mrhyde.log')


err = 0.0

flog = np.genfromtxt('sensor.objval.out')
reflog = np.genfromtxt('sensor.objval.gold')
for j in range(flog.size):
  err += abs(flog[j]-reflog[j])

flog = np.genfromtxt('sensor.objgrad.out')
reflog = np.genfromtxt('sensor.objgrad.gold')
for j in range(flog.size):
  err += abs(flog[j]-reflog[j])

if err > aeps:
  status += 1


# ------------------------------
if its.opts.baseline and not status:
  if its.opts.verbose != 'none': print('---> Baseline %s' % (root))
  try :
    shutil.copy2('%s.ocs' %(root), 'ref/%s.ocs' %(root))
  except (IOError, os.error) as why:
    print(why)
    status += 1

  try :
    shutil.copy2('%s.rst' %(root), 'ref/%s.rst' %(root))
  except (IOError, os.error) as why:
    print(why)
    status += 1

  try :
    shutil.copy2('%s.adj.rst' %(root), 'ref/%s.adj.rst' %(root))
  except (IOError, os.error) as why:
    print(why)
    status += 1

# ------------------------------
if its.opts.graphics and not status:
  if its.opts.verbose != 'none': print('---> Graphics %s' % (root))
  status += its.call('echo "  No graphics, yet."')

# ------------------------------
if its.opts.clean and not status:
  if its.opts.verbose != 'none': print('---> Clean %s' % (root))
  os.chdir('obj-org')
  status += its.call('ichos_clean')
  status += its.call('rm -rf shot.*')
  os.chdir('..')
  status += its.call('ichos_clean')

# ==============================================================================
if status == 0: print('Success.')
else:           print('Failure.')
sys.exit(status)
