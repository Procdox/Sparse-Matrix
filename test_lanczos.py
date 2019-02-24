from numpy import linalg as la
import time

from matrix import Vector, Matrix, Sparse_Matrix
from utilities import sortValues, cleanDisplay

import sys

#target mtx file to profile with
if len(sys.argv) > 1:
  Profile_Target = sys.argv[1]
else:
  print('Please supply a target data file')
  sys.exit()
  


# Read in from MTX and translate to CSR
dense = Matrix()
dense.readMTX(Profile_Target, True, True)
dense.t().writeCSR('./output.csr')

#Read in from CSR
sparse = Sparse_Matrix('./output.csr')

lanczos_sparse = Vector(sparse.w)
lanczos_sparse[0] = 1.0

sparse_result = sparse.lanczos(lanczos_sparse, 10)

print('Lanczos Calculation')

for i in range(1,12):
  lanczos_sparse = Vector(sparse.w)
  lanczos_sparse[0] = 1.0
  time_start = time.time()
  sparse_result = sparse.lanczos(lanczos_sparse, i*5)
  lanczos_time = time.time() - time_start
  T = sparse_result[1].dense()
  time_start = time.time()
  v, w = la.eig(T.data)
  eig_time = time.time() - time_start
  ids = sortValues(v)
  print('M: ' + str(i*5))
  print('    Time: ' + str(lanczos_time))
  print('    max e-values:' + '   '.join(map(cleanDisplay, v[ids][-5:])))
  print('    min e-values:' + '   '.join(map(cleanDisplay, v[ids][:5])))

time_start = time.time()
v, w = la.eig(dense.data)
eig_time = time.time() - time_start
ids = sortValues(v)
print('Real')
print('    Time: ' + str(eig_time))
print('    max e-values:' + '   '.join(map(cleanDisplay, v[ids][-5:])))
print('    min e-values:' + '   '.join(map(cleanDisplay, v[ids][:5])))