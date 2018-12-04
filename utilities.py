from math import sqrt
from decimal import Decimal

def mag(n):
    return sqrt(n.real ** 2 + n.imag ** 2)

def sortValues(v):
    ids = [x for x in range(0,len(v))]
    def srt(a):
        return mag(v[a])

    ids.sort(key=srt)
    return ids

def cleanDisplay(value):
    #return "{:.2E}+{:.2E}j".format(Decimal(value.real),Decimal(value.imag))
    return "{:.2E}".format(Decimal(value.real))