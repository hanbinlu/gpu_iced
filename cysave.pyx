import cython
import numpy as np
cimport numpy as np
from libc.stdio cimport *

def cysave_file(row, col, data, char* outfile):
    
    cdef:
        int i, row_num = len(row)
        FILE* OUTFILE = fopen(outfile,'w')
        np.ndarray[int, ndim=1, mode='c'] crow = row
        np.ndarray[int, ndim=1, mode='c'] ccol = col
        np.ndarray[double, ndim=1, mode='c'] cdata = data    
        
    for i in range(row_num):
        fprintf(OUTFILE, "%d\t%d\t%f\n", crow[i], ccol[i], cdata[i])
    
    fclose(OUTFILE)
