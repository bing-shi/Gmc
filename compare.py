# Author: Bing Shi
# Function: Result compare
                   
import sys
import numpy as np

def main(f1, f2):
    """  """
    f1  = np.loadtxt(f1+'.result.txt', dtype=np.int16)
    f2  = np.loadtxt(f2+'.result.txt', dtype=np.int16)
    if f1.shape[1] != f2.shape[1]: print('data error!'); return
    result = 0
    #max1 = np.maximum(f1)
    #max2 = np.maximum(f2)
    #max = max1
    #if max2 > max: max = max2
    #everyresult = np.zeros(max, np.int8)   
    for i in range(len(f1)):
        if f1[i] == f2[i]:
            result += 1
    print('result:%d %d'%(result, len(f1)))  
    ans = [result, len(f1)]
    np.savetxt("compare.txt", np.array(ans), fmt="%i", delimiter=' ', newline='\r\n')  
if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])     # main name of two image files
