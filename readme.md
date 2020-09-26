# Sparse Matrix

Sparse Matrix is an implementation of basic matrix operations with a dense and sparse representation.

The dense matrix allows for reading from mtx or csr files, with in-line parameters for specifying if the file is symmetrix, and pattern based.

The sparse representation allows for reading from csr files.

This project does NOT include mtx/csr files, but test file target paths can be specified through the first command line parameter.

# Operations Implementation
The project utilizes python lists only for storing data within either format. Dense algorithms operate exactly as one would naively expect. All sparse operations operate by iterating across rows nested in iterating across columns. For multiplication, the transform is calculated for the right matrix, then both the left and right are iterated across simultaneously.

# Usage
Note that, as this library is primarily aimed at profiling the difference in cost between matrix formats, and demonstrating the implementation of the Lanczos algorithm, it does not provide general operators intended for manipulation of matricies. I would recommend you stick with the popular Eigen library.

The main classes are Matrix and SparseMatrix of matrix.py. Matrix can be loaded from either MTX or CSR formats via the methods readMTX and readCSR respectively. SparseMatrix must be constructed with a CSR format file though.
See test_profile or test_lanczos for an example usage of the classes.

# Lanczos Implementation
The other major component of this project is the implementation of Lanczos. This algorithm can be performed with either matrix representation class. It of course is much faster with Sparse_Matrix, when appropriate. 

This algorithm is implemented for a matrix A, by multiplying with a unit vector b s.t. Ab = y we then subtract x from b s.t. (b - x) is orthogonal to all y's. We then normalize this vector to become our new b. This process is continued, and the series of b's and y's allow us to compute both a tridiagonal matrix T and a change of basis Q. Practically this is implemented by taking the dot product a of b and y, and setting b<sub>1</sub> to normalize(b<sub>0</sub> - a<sub>0</sub>y<sub>0</sub>) or further b<sub>n>1</sub> to normalize(b<sub>n-1</sub> - a<sub>n</sub>y<sub>n</sub> - b<sub>n-2</sub>a<sub>n-1</sub>).

The algorithm is quite successful at converging to a tridiagonal matrix that is similar to an input matrix. For 1138_bus, a matrix of size (1138,1138) found through online resources provided to the class, it manages to converge to the top 5 eigenvalues over 25 iterations, with accuraccy improving dramatically at till around 45 iterations. This particular implementation does not supply novel orthogonal vectors if the initial input vector becomes dependent (and therfor set to 0). This means that for particular test matricies, the default E0 input vector may not allow for many iterations. 

Specifically:
For Lanczos Calculation

|M| λ<sub>4</sub>|λ<sub>3</sub>|λ<sub>2</sub>|λ<sub>1</sub>|λ<sub>0</sub>|Time (s)|
|-|-|-|-|-|-|-|
|5|6.54E-2|-1.47E+0|1.58E+0|2.85E+0|3.87E+0|0.040|
|10|3.44E+0|3.98E+0|4.59E+0|4.97E+0|5.57E+0|0.060|
|15|3.44E+0|3.98E+0|4.59E+0|4.97E+0|5.57E+0|0.076|
|25| 4.71E+0|4.89E+0|5.16E+0|5.61E+0|5.81E+0|0.124|
|35|5.13E+0|5.35E+0|5.67E+0|5.81E+0|5.92E+0|0.168|
|45| 5.29E+0|5.41E+0|5.67E+0|5.81E+0|6.19E+0|0.220|
|real|5.40E+0|5.41E+0|5.67E+0|5.81E+0|6.19E+0|2.863|

M is the number of iterations.
Eigenvalues are listed as the largest 5 in increasing order.
Time is calculated as the sum of time spent calculating T and time spent calculating eigenvalues from T (or from A in the case of real).
