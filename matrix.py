###########################################################
# Andrew Blackledge 2018
# Sparse Matrix and Lanczos Algorithm Implementation
#
# Several methods have not yet been implemented
# In the future this will support complex matricies
###########################################################

import math
import time


#dense vector representation
class Vector:
    #default constructor, requires dimension
    def __init__(self, height): 
        self.h = height
        self.data = [0.0 for _ in range(0,height)]

    #access operators
    def __setitem__(self, key, item): 
        self.data[key] = item
    def __getitem__(self, key):
        return self.data[key]

    #copies elements of a list like object
    def copy(self, like):
        self.data = [x for x in like]

    #computes dot product with target
    def dot(self, target):
        if target.h != self.h:
            raise ArithmeticError('Size mismatch during vector.dot')
        
        total = 0
        for y in range(0,self.h):
            total += self.data[y] * target.data[y]
        return total

    #multiplies this vector by a scalar
    def scale(self, factor):
        for y in range(0,self.h):
            self.data[y] *= factor

    #returns the norm of the vector        
    def len(self):
        total = 0
        for y in range(0,self.h):
            total += self.data[y] * self.data[y]

        return math.sqrt(total)

    #sets the norm of the vector to 1, without changing the direction
    def normalize(self):
        self.scale(1.0/self.len())

#dense representation of a matrix
class Matrix:
    #default constructor, can be supplied with dimensions
    def __init__(self, width=0, height=0):
        self.w = width
        self.h = height
        self.data = [[0.0 for _ in range(0,self.w)] for _ in range(0,self.h)]

    #access operators
    def __setitem__(self, key, item):
        self.data[key] = item
    def __getitem__(self, key):
        return self.data[key]

    #read a csr file into this matrix, resizing if nessecary
    def readCSR(self, file_name):
        file = open(file_name, "r")
        #read and seperate
        info = file.readline().split(',')
        self.w = int(info[0])
        self.h = int(info[1])

        self.data = [[0.0 for _ in range(0,self.w)] for _ in range(0,self.h)]

        entries = file.readline().split(",")[:]
        counts = file.readline().split(",")[1:]
        offsets = file.readline().split(",")[:]

        #convert
        for x in range(0,len(entries)):
            entries[x] = float(entries[x])
        for x in range(0,len(counts)):
            counts[x] = int(counts[x])
        for x in range(0,len(offsets)):
            offsets[x] = int(offsets[x])

        #write entries to dense format
        p = 0
        line = 0
        for limit in counts:
            while p < limit:
                self.data[line][offsets[p]] = entries[p]
                p += 1
            line += 1

    #read an mtx file into this matrix, resizing if nessecary
    def readMTX(self, file_name, symmetric = False):
        file = open(file_name, 'r')

        line = ''
        #skip header
        while True:
            line = file.readline()
            if line[0]!='%':
                break

        info = line.split(' ')

        self.w = int(info[1])
        self.h = int(info[0])
        self.data = [[0.0 for _ in range(0,self.w)] for _ in range(0,self.h)]
        while True:
            line = file.readline()
            if line=='':
                break

            data = line.split(' ')
            self.data[int(data[1])-1][int(data[0])-1] = float(data[2])
            if symmetric:
                self.data[int(data[0])-1][int(data[1])-1] = float(data[2])
                
    #write this matrix to a csr file
    def writeCSR(self,file_name):
        entries = []
        counts = [0]
        offsets = []

        p = 0
        
        for y in range(0,self.h):
            for x in range(0,self.w):
                if(self.data[y][x] != 0):
                    entries.append(self.data[y][x])
                    offsets.append(x)
                    p += 1
            counts.append(p)

        file = open(file_name,'w')
        file.write(str(self.w) + "," + str(self.h) + '\n')
        file.write(','.join(str(e) for e in entries) + '\n')
        file.write(','.join(str(e) for e in counts) + '\n')
        file.write(','.join(str(e) for e in offsets))

    #write this matrix to an mtx file
    def writeMTX(self,file_name):
        raise NotImplementedError("writing to mtx files is not implemented")

    #calculate and return the transpose/conjugate
    def t(self):
        transpose = Matrix(self.h, self.w)
        for y in range(0,self.h):
            for x in range(0,self.w):
                transpose.data[x][y] = self.data[y][x]
        return transpose
    def leftMultiplyMatrix(self, target):
        if self.w != target.h:
            raise ArithmeticError('Size mismatch during matrix.leftMultiplyMatrix')

        result = Matrix(target.w, self.h)
        for y in range(0,self.h):
            if y%100 == 0:
                print(y)
            for x in range(0,target.w):
                row_sum = 0
                for t in range(0,self.w):
                    row_sum += self.data[y][t] * target.data[t][x]
                result.data[y][x] = row_sum

        return result
    def leftMultiplyVector(self, target):
        if self.w != target.h:
            raise ArithmeticError('Size mismatch during matrix.leftMultiplyVector')

        result = Vector(target.h)
        for y in range(0,self.h):
            row_sum = 0
            for x in range(0,self.w):
                row_sum += self.data[y][x] * target.data[x]
            result.data[y] = row_sum

        return result
    def isSymmetric(self):
        if(self.w != self.h):
            return False
        for y in range(1,self.h):
            for x in range(y+1,self.h):
                if self.data[y][x] != self.data[x][y]:
                    return False
        return True
    def scale(self, factor):
        for y in range(0,self.h):
            for x in range(0,self.w):
                self.data[y][x] /= factor
    def add(self, factor):
        for y in range(0,self.h):
            for x in range(0,self.w):
                self.data[y][x] += factor

    def lanczos(self, i, c):
        if self.w != i.h:
            raise ArithmeticError('Size mismatch during matrix.lanczos')
        if not self.isSymmetric():
            raise ValueError('Lanczos can only be calculated for hermeitan matrix')
        #v = [0 for _ in range(0,c+1)]
        b = [0 for _ in range(0, c)]
        a = [0 for _ in range(0, c)] 
        v = [0 for _ in range(0, c)]
        v[0] = i
        
        w = self.leftMultiplyVector(v[0])
        a[0] = w.dot(v[0])
        #print('wb:', w.data)

        for x in range(0,self.w):
            w[x] = w[x] - (a[0] * v[0][x])

        #print('wa:', w.data)
        for n in range(1,c):
            b[n] = w.len()
            
            #if(b[n]==0)
            #else
            w.scale(1.0/b[n])
            v[n] = w
            w = self.leftMultiplyVector(v[n])
            a[n] = w.dot(v[n])
            for x in range(0,self.w):
                w[x] = w[x] - (a[n] * v[n][x]) - (b[n] * v[n-1][x])
        
        #print('a:', a)
        #print('b:', b)

        #print('A:', self.data)
        V = Matrix(c, self.w)
        for y in range(0,c):
            for x in range(0,self.w):
                V[x][y] = v[y][x]
        #print('V:',V.data)
        T = Matrix(c,c)

        T[0][0] = a[0]
        for n in range(1,c):
            T[n][n] = a[n]
            T[n-1][n] = b[n]
            T[n][n-1] = b[n]
        
        #print('T:   ',T.data)
        
        
        #print('V*A: ',V.t().leftMultiplyMatrix(self).data)
        #print('V*AV:', V.t().leftMultiplyMatrix(self).leftMultiplyMatrix(V).data )
        
class Sparse_Matrix:
    #default constructor, supply with a filename to read from a csr
    def __init__(self, file_name=''):
        if file_name == '':
            return

        file = open(file_name, "r")

        info = file.readline().split(',')
        self.w = int(info[0])
        self.h = int(info[1])
        
        #read and seperate
        self.entries = file.readline().split(",")[:]
        self.counts = file.readline().split(",")[:]
        self.offsets = file.readline().split(",")[:]

        #convert
        for x in range(0,len(self.entries)):
            self.entries[x] = float(self.entries[x])
        for x in range(0,len(self.counts)):
            self.counts[x] = int(self.counts[x])
        for x in range(0,len(self.offsets)):
            self.offsets[x] = int(self.offsets[x])
        
    #write this matrix to a csr file
    def write_csr(self, file_name):
        raise NotImplementedError("writing to csr files is not implemented")

    def leftMultiplyVector(self, target):
        if self.w != target.h:
            raise ArithmeticError('Size mismatch during matrix.leftMultiplyVector')

        result = Vector(target.h)
        for y in range(0,self.h):
            start = self.counts[y]
            end = self.counts[y+1]

            row_sum = 0

            for x in range(start, end):
                row_sum += self.entries[x] * target.data[self.offsets[x]]

            result.data[y] = row_sum
        
        return result

    #calculate and return the transpose/conjugate
    def t(self):
        entry_bins = [[] for _ in range(0,self.w)]
        offset_bins = [[] for _ in range(0,self.w)]

        for y in range(0, self.h):
            start = self.counts[y]
            end = self.counts[y+1]

            for x in range(start, end):

                entry_bins[self.offsets[x]].append(self.entries[x])
                offset_bins[self.offsets[x]].append(y)

        result = Sparse_Matrix()
        result.entries = [0 for _ in range(0,len(self.entries))]
        result.offsets = [0 for _ in range(0,len(self.entries))]
        result.counts = [0 for _ in range(0,self.w + 1)]

        p = 0

        for x in range(0, self.w):
            result.counts[x] = p
            for r in range(0,len(entry_bins[x])):
                result.entries[p] = entry_bins[x][r]
                result.offsets[p] = offset_bins[x][r]
                p += 1

        result.counts[self.w] = p

        result.w = self.h
        result.h = self.w
        return result
    def leftMultiplyMatrix(self, target):
        result = Sparse_Matrix()
        result.entries = []
        result.offsets = []
        result.counts = []

        pair = target.t()
        p = 0
        for y in range(0, self.h):
            result.counts.append(p)
            start_self = self.counts[y]
            end_self = self.counts[y+1]

            for x in range(0, pair.h):
                start_pair = pair.counts[x]
                end_pair = pair.counts[x+1]

                mark_self = start_self
                mark_pair = start_pair

                total = 0

                while mark_self < end_self and mark_pair < end_pair:
                    offset_self = self.offsets[mark_self]
                    offset_pair = pair.offsets[mark_pair]
                    if offset_self == offset_pair:
                        total += self.entries[mark_self] * pair.entries[mark_pair]
                        mark_self += 1
                        mark_pair += 1
                    elif offset_self > offset_pair:
                        mark_pair += 1
                    else:
                        mark_self += 1
                
                if total != 0:
                    result.entries.append(total)
                    result.offsets.append(x)

        result.counts.append(p)
        return result
    def isSymmetric(self, transpose = None):
        if transpose == None:
            transpose = self.t()

        return transpose.entries == self.entries and transpose.counts == self.counts and transpose.offsets == self.offsets

    def lanczos(self, i, c):
        if self.w != i.h:
            raise ArithmeticError('Size mismatch during matrix.lanczos')
        if not self.isSymmetric():
            raise ValueError('Lanczos can only be calculated for symmetric/hermeitan matrix')
        #v = [0 for _ in range(0,c+1)]
        b = [0 for _ in range(0, c)]
        a = [0 for _ in range(0, c)] 
        v = [0 for _ in range(0, c)]
        v[0] = i
        
        w = self.leftMultiplyVector(v[0])
        a[0] = w.dot(v[0])
        #print('wb:', w.data)

        for x in range(0,self.w):
            w[x] = w[x] - (a[0] * v[0][x])

        #print('wa:', w.data)
        for n in range(1,c):
            b[n] = w.len()
            
            if b[n] == 0:
                raise ValueError('missing behavior!')

            w.scale(1.0/b[n])
            v[n] = w
            w = self.leftMultiplyVector(v[n])
            a[n] = w.dot(v[n])
            for x in range(0,self.w):
                w[x] = w[x] - (a[n] * v[n][x]) - (b[n] * v[n-1][x])
        
        #print('a:', a)
        #print('b:', b)

        #print('A:', self.data)
        V = Matrix(c, self.w)
        for y in range(0,c):
            for x in range(0,self.w):
                V[x][y] = v[y][x]
        #print('V:',V.data)

        T = Sparse_Matrix()
        T.counts = [0,2] + [2+n*3 for n in range(0,c-2)] + [c*3-2]
        T.entries = [0 for _ in range(0,c*3-2)]
        T.offsets = [0 for _ in range(0,c*3-2)]

        T.entries[0] = a[0]
        T.offsets[0] = 0
        for n in range(1,c):
            T.entries[n*3-2] = b[n]
            T.entries[n*3-1] = b[n]
            T.entries[n*3] = a[n]
            T.offsets[n*3-2] = n
            T.offsets[n*3-1] = n-1
            T.offsets[n*3] = n

        #VAV = V.t().leftMultiplyMatrix(self).leftMultiplyMatrix(V)
        return (V,T)




if __name__ == "__main__":
    dense = Matrix()
    dense.readMTX('datas/494_bus.mtx', True)
    dense.t().writeCSR('output.csr')

    sparse = Sparse_Matrix('output.csr')

    test = Vector(dense.w)
    test.data = [n-3 for n in range(0,dense.w)]

    time_start = time.time()
    sparse_result = sparse.leftMultiplyVector(test)
    time_mid = time.time()
    dense_result = dense.leftMultiplyVector(test)
    time_end = time.time()

    print('Vector Multiplication')
    print("Sparse Time:", time_mid - time_start)
    print("Dense Time:", time_end - time_mid)
    print('\n')

    time_start = time.time()
    sparse_result = sparse.t()
    time_mid = time.time()
    dense_result = dense.t()
    time_end = time.time()

    print('Calculate Transform')
    print("Sparse Time:", time_mid - time_start)
    print("Dense Time:", time_end - time_mid)
    print('\n')

    test_sparse = Sparse_Matrix('output.csr')
    test_dense = Matrix()
    test_dense.readCSR('output.csr')

    time_start = time.time()
    sparse_result = sparse.leftMultiplyMatrix(test_sparse)
    time_mid = time.time()
    #dense_result = dense.leftMultiplyMatrix(test_dense)
    time_end = time.time()

    print('Matrix Multiplication')
    print("Sparse Time:", time_mid - time_start)
    #print("Dense Time:", time_end - time_mid)
    print("Dense takes to long, don't bother profiling")
    print('\n')

    lanczos_sparse = Vector(sparse.w)
    lanczos_dense = Vector(sparse.w)
    lanczos_sparse[0] = 1.0
    lanczos_dense[0] = 1.0

    time_start = time.time()
    sparse_result = sparse.lanczos(lanczos_sparse, 10)
    time_mid = time.time()
    dense_result = dense.lanczos(lanczos_dense, 10)
    time_end = time.time()

    print('Lanczos Calculation')
    print("Sparse Time:", time_mid - time_start)
    print("Dense Time:", time_end - time_mid)
    print('\n')