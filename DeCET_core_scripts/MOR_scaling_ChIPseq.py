import numpy as np
import torch

def scale_function(t):
    """Median-of-ratios scaling for the ChIP-seq data tensor.

    For each assay the median-of-ratios method is used to scale
    the data across patients and conditions. The subtensor for
    each assay is then scaled to sum to one. This function 
    assume the tensor indices are ordered as condition, patient,
    assay, location.
    """
    dim = t.size()
    d = [0.0 for i in range(0,dim[3])]
    s = np.ones((dim[0],dim[1],dim[2]))
    # Median of ratios scaling.
    for k in range(0,dim[2]):
        d = [torch.prod(t[:,:,k,l])**(1.0/(dim[0]*dim[1])) for l in range(dim[3])]
        assay_weight = 0.0
        for i in range(0,dim[0]):
            for j in range(0,dim[1]):
                scale_list = []
                for l in range(0,dim[3]):
                    if(d[l] != 0.0):
                        scale_list.append(t[i,j,k,l]/d[l])
                s[i,j,k] = np.median(scale_list)
                print("MOR scale {},{},{}: {}".format(i,j,k,s[i,j,k]))
                ###
                t[i,j,k,:] /= s[i,j,k]
                ###
                # Get the total weight of each subtracted ChIP-seq data set
                # after MOR scaling and add this to the total assay tensor weight.
                assay_weight += sum(t[i,j,k,:])
        # Scale each assay subtensor to have total weight 1.
        print("assay total {}: {}".format(k, assay_weight))
        t[:,:,k,:] /= assay_weight
    return t
