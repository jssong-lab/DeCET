import numpy as np
import torch

def scale_function(t):
    """Median-of-ratios scaling for the ChIP-seq data tensor.

    For each assay the median-of-ratios method is used to scale
    the data across patients (the first index). The subtensor for
    each assay is then scaled to sum to one. This function 
    assumes the tensor indices are ordered as patient,
    assay, location.
    """
    dim = t.size()
    d = [0.0 for i in range(0,dim[2])]
    s = np.ones((dim[0],dim[1]))
    # Median of ratios scaling.
    for k in range(0,dim[1]):
        d = [torch.prod(t[:,k,l])**(1.0/dim[0]) for l in range(dim[2])]
        assay_weight = 0.0
        for i in range(0,dim[0]):
            scale_list = []
            for l in range(0,dim[2]):
                if(d[l] != 0.0):
                    scale_list.append(t[i,k,l]/d[l])
            s[i,k] = np.median(scale_list)
            print("MOR scale {},{}: {}".format(i,k,s[i,k]))
            ###
            t[i,k,:] /= s[i,k]
            ###
            # Get the total weight of each subtracted ChIP-seq data set
            # after MOR scaling and add this to the total assay tensor weight.
            assay_weight += sum(t[i,k,:])
        # Scale each assay subtensor to have total weight 1.
        print("assay total {}: {}".format(k, assay_weight))
        t[:,k,:] /= assay_weight
    return t
