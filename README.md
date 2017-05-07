# gpu_iced
As the sequence cost going down quickly, the sequence depth of HiC experiment increase from 10 million reads in 2009 to billions of reads.
The higher sequence depth can enable higher resolution view of genome 3D organization. In order to get correct interpretation of HiC
periment, the bias of raw matrix need to correct. One widely used correction method, iterative correction (ICE), correct the bias 
according to equal visibility. The elementwise correction of matrix using by ICE is very computationally intensive. This program aims at
speeding up ICE using GPU. Thanks to the huge performance increase by GPU, the correction process can be fully interactive. Moreover，
by incorporating the HiCplotter function, the matrix (raw, intermedia-corrected, finalized) are visualized instantly and interactively
to query the correction process, find the plotting parameters, and get different locus 3D architecture in jupyter matplotlib inline mode.

# Sample analysis


# Dependency
It depends on Numbapro to use the CUDA libraries. Numbapro are now in anaconda. If you successful make the Numbapro runned, you are pretty 
much have everything seted up for the program. Of course, the prerequisition is a GPU. Possibly your gaming GPU can work decently.
