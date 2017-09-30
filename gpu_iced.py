#%matplotlib notebook
from numba import cuda, jit, void, int32, int64, float32, float64, boolean
import accelerate.cuda.sparse as cusparse
import accelerate.cuda.blas as cublas
import scipy.sparse
import numpy as np
import pandas
import HiCPlotter_py3
import pyximport; pyximport.install(setup_args = {"include_dirs":np.get_include()})
import cysave


@cuda.jit('void(float32[:], float32[:])')
def zero_remove(d_in_array, d_out_array):
    '''
    Replacing the 0 value elements into ones
    '''
    i=cuda.grid(1)
    if i < d_in_array.shape[0]:
        if d_in_array[i] == 0:
            d_out_array[i] = 1
            
@cuda.jit('void(float32[:], float32[:], float32)')
def array_times_scalar(d_in_array, d_out_array, scalar):
    '''
    elementwise product of array times scalar
    '''
    i=cuda.grid(1)
    if i < d_in_array.shape[0] and  d_in_array[i] != 0:
        d_out_array[i] = d_in_array[i] * scalar

@cuda.jit('void(float32[:], float32[:],int32[:], int32[:])')
def matrix_norm(d_matrix_value, d_bias, d_col_index, d_row_index):
    '''
    Normalizing the element Xij in the matrix by ith and jth rows' bias production.
    '''
    i=cuda.grid(1)
    if i < d_matrix_value.shape[0]:
        d_matrix_value[i] /= d_bias[d_col_index[i]] * d_bias[d_row_index[i]]

@jit([float32[:](int32[:], int32[:], float32[:], boolean[:]), int32[:](int32[:], int32[:], int32[:],boolean[:])])
def filter_low(indptr, indice, data, filter_bool):
    '''
    filter lower coverage bins to avoid statistical instability when ice
    '''
    j = 0
    for i, col in enumerate(indice):
        if i >= indptr[j+1]:
            j += 1
        if filter_bool[col] or filter_bool[j]:
            data[i] = 0
    return data

    
# Define an object to manage the data and methods for ICE normalization    
class ice_norm(object):
    '''
    GPU implementation of ice normalization for HiC matrix.
    Initiate with a sparse matrix in triplet format. Since the matrix
    (raw/normalized) is in the memory, it enables convenient query of
    the matrix by plotting heatmap
    '''
    
    def __init__(self,triplet_table):
        '''
        Initiate an ice_norm instance by specifying the file name of 
        the raw matrix in triplet format. Currently only supports
        upper triangule of symmetric matrix
        '''
        self.input = triplet_table
        
    def make_matrix(self, reset=0, coerce=False, out_dtype='np.float32',scale = 1, filterlow_percent=0.02, include_zero=True):
        '''
        Read in triplet table and make sparse matrix. Reports properties
        of the matrix as reference. Matrix size and sparsity can be used
        to determine ice normalization parameters.
        '''
        # Read in table from disk
        if reset == 0:
            if coerce == False:
                # store the element value as float
                h_triplet = pandas.read_csv(self.input, header=None, sep="\t").as_matrix() * scale
            else:
                # store the element value as int even
                # it is float in the input file
                # for memory saving reason
                h_triplet = pandas.read_csv(self.input, header=None, sep="\t").as_matrix().astype(np.int32) * scale

            self.out_dtype = out_dtype
            self.ice_track = 0 # ice on iced matrix is allowed but with warning
            
        if reset == 0:
            #construct the csr_matrix from the triplet table in the GPU device
            self.d_matrix = cusparse.csr_matrix((h_triplet[:,2].astype(eval(out_dtype)),
                            (h_triplet[:,0].astype(np.int32), h_triplet[:,1].astype(np.int32))))
        # reset parameter is deprecated

        # filter low counts
        if filterlow_percent != 0:
            h_matrix =  self.d_matrix.copy_to_host()
            raw_row_sum = np.array(h_matrix.sum(axis=0)).flatten() + np.array(h_matrix.sum(axis=1)).flatten() \
                            - np.array(h_matrix.diagonal()).flatten()
            row_sum_sort = raw_row_sum.copy()
            row_sum_sort.sort()
            row_num = raw_row_sum.shape[0]
            #find out the cut through of low count
            if include_zero:
                cut_through = row_sum_sort[int(row_num * filterlow_percent)]
            else:
                zero_counts = sum(row_sum_sort == 0)
                cut_through = row_sum_sort[zero_counts + int((row_num - zero_counts) * filterlow_percent)]
            filter_bool = raw_row_sum < cut_through

            #filter
            self.d_matrix.data = cuda.to_device(filter_low(h_matrix.indptr, h_matrix.indices, h_matrix.data, filter_bool))

        # get properties of the matrix
        self.shape=self.d_matrix.shape[0]
        self.nnz=self.d_matrix.nnz * 2 - self.shape
        self.d_row_coo = cuda.device_array(self.d_matrix.indices.shape[0],dtype=np.int32)
        cusparse.Sparse().Xcsr2coo(self.d_matrix.indptr, self.nnz, 
                                       self.d_matrix.indptr.shape[0]-1, self.d_row_coo)
        self.matrix_descr =cusparse.Sparse().matdescr(matrixtype='S', fillmode='U')
        self.sparsity = self.nnz  / self.shape ** 2
        self.seq_depth = np.sum(self.d_matrix.data.copy_to_host())            
        print("Matrix properties:")
        print("size: {0} X {0}".format(self.shape))
        print("non-zero elements: {0}".format(self.nnz))
        print("sparsity: {0}".format(self.sparsity))
        print("sequence depth: {0}".format(self.seq_depth))

    def ice(self, epoches=1000, eps=1, read_depth=None, block_size=256, verbose=50):
        '''
        Perform ICE normalization
        '''
        if self.ice_track:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("Matrix had been ICE normalized.")
            print("Further ICE will perform on the existed normalized matrix.")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        
        if read_depth == None:
            read_depth=self.seq_depth
        self.ice_track += 1
        d_row_sum_operator = cuda.to_device(np.ones(shape=self.shape,dtype=np.float32))
        bias = np.ones(self.shape,dtype=np.float32)
        old_bias = None
        for i in range(epoches):
            # Calculate bias
            # row sums are calculated using csr_matrix multiply row sum operator [1,1,1....,1]
            # method provided by cusparse
            d_row_sum = cuda.device_array(self.shape,dtype=np.float32) 
            d_updated_bias = cuda.to_device(np.zeros(self.shape,dtype=np.float32)) # allocate memories
            cusparse.Sparse().csrmv("N", self.shape, self.shape, self.nnz, 1, self.matrix_descr, 
                                self.d_matrix.data, self.d_matrix.indptr, self.d_matrix.indices,
                                d_row_sum_operator, 0, d_row_sum)
            # Setup cuda thread structure for bias correction
            blck = (block_size,1)
            grid = (self.shape//block_size+1, 1)
            # Correct the d_row_sum into bias
            # remove 0 to avoid numerical unstability
            zero_remove[grid,blck](d_row_sum,d_updated_bias)
            d_updated_bias_nnz = d_updated_bias.shape[0]-cublas.Blas().asum(d_updated_bias)
            d_mean = cublas.Blas().asum(d_row_sum)/d_updated_bias_nnz
            # scaling the raw row sum into 
            array_times_scalar[grid,blck](d_row_sum,d_updated_bias,1/d_mean)
            bias *= d_updated_bias.copy_to_host()
            # Matrix correction
            mgrid = (self.nnz//block_size+1, 1)
            matrix_norm[mgrid,blck](self.d_matrix.data, d_updated_bias, self.d_matrix.indices, self.d_row_coo)
            # scale the normalized matrix
            scale_factor = read_depth / cublas.Blas().asum(self.d_matrix.data)
            bias *= np.sqrt(cublas.Blas().asum(self.d_matrix.data) / read_depth) 
            array_times_scalar[mgrid,blck](self.d_matrix.data,self.d_matrix.data,scale_factor)
            # eps checkpoint
            if old_bias is not  None:
                current_eps = np.sum(np.abs(old_bias-bias))
            if verbose and i != 0:    
                if i % verbose == 0 or i == epoches-1:
                    print("@ "+str(i+1)+ " round iteration, current eps: "+str(current_eps))
            if i != 0:
                if current_eps < eps:
                    print("Convergent at "+str(i+1)+" round iteration")
                    break
            old_bias = bias.copy()
            
    def plotHeatmap_on_the_fly(self,**kwargs):
        '''
        By incorporating the versatile HiC-Plotter package, Heatmap plottings are enabled
        on the fly. With proper setting of jupyter, heapmap can be plotted inline.
        Plotting parameters are the same as HiCPlotter, use the full name of the parameter.
        Currently multi-heatmap are not supported.
        '''
        h_matrix = self.d_matrix.copy_to_host()
        kwargs['files'] = h_matrix
        HiCPlotter_py3.HiCplotter(**kwargs)
                
    def write_matrix(self, file_name):
        '''
        Write out the matrix in triplet table format
        '''
        h_matrix = self.d_matrix.copy_to_host()
        h_matrix = h_matrix.tocoo()
        cysave.cysave_file(h_matrix.row, h_matrix.col, h_matrix.data.astype('float'), bytes(file_name.encode('utf-8')))
                       
if __name__ == "__main__":
    test_matrix = ice_norm("rep1_1000000.matrix")
    test_matrix.make_matrix()
    test_matrix.ice()
