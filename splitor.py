# Author: Bing Shi
# Function: split data to subsets

# import os                      
import sys
import numpy as np

def main(fn):
    """  """
    keyp = np.loadtxt(fn+'.kps.1.txt', dtype=np.float32)    
    clusters = np.loadtxt(fn+'_clusters.txt', dtype=np.int16)
    if len(clusters) != keyp.shape[0] : print('data error!'); return

    num = max(clusters)+1 
    for i in range(num):
        subset = keyp[clusters==i]
        ind = np.argwhere(clusters==i)
        print('subset:%d'%i)  
        subsetname1 = fn+'_%d_%d'%(i, subset.shape[0])+'.txt'        
        subsetname2 = fn+'_%d_ind'%i+'.txt'        
        np.savetxt(subsetname1, subset, fmt="%f", delimiter=' ', newline='\r\n')
        np.savetxt(subsetname2, ind, fmt="%d", delimiter=' ', newline='\r\n')
    print('OK!')
if __name__ == "__main__":
    main(sys.argv[1])     # first name of two files
