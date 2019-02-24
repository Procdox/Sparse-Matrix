from numpy import linalg as la
import time
from decimal import Decimal

from matrix import Vector, Matrix, Sparse_Matrix


#target mtx file to profile with
if len(sys.argv > 1):
  Profile_Target = sys.argv[1]
else:
  print('Please supply a target data file')
  sys.exit()
  


# Read in from MTX and translate to CSR
dense = Matrix()
dense.readMTX(Profile_Target, True, True)
dense.t().writeCSR('../output.csr')

#Read in from CSR
sparse = Sparse_Matrix('../output.csr')


print('Vector Multiplication x 10')

test = Vector(dense.w)
test.data = [n-3 for n in range(0,dense.w)]

time_start = time.time()
for _ in range(0,10):
    sparse_result = sparse.leftMultiplyVector(test)
time_mid = time.time()
for _ in range(0,10):
    dense_result = dense.leftMultiplyVector(test)
time_end = time.time()

print("Sparse Time:", time_mid - time_start)
print("Dense Time:", time_end - time_mid)
print('\n')


print('Calculate Transform x10')

time_start = time.time()
for _ in range(0,10):
    sparse_result = sparse.t()
time_mid = time.time()
for _ in range(0,10):
    dense_result = dense.t()
time_end = time.time()

print("Sparse Time:", time_mid - time_start)
print("Dense Time:", time_end - time_mid)
print('\n')

print('Matrix Multiplication (may be slow)')

test_sparse = Sparse_Matrix('../output.csr')
test_dense = Matrix()
test_dense.readCSR('../output.csr')

time_start = time.time()
sparse_result = sparse.leftMultiplyMatrix(test_sparse)
time_mid = time.time()
dense_result = dense.leftMultiplyMatrix(test_dense, True)
time_end = time.time()

print("Sparse Time:", time_mid - time_start)
print("Dense Time:", time_end - time_mid)
print('\n')