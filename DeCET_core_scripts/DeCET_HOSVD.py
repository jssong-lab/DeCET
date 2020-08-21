import sys
import numpy as np
import time
import tensorly
import torch
import argparse
import importlib

def l1_scaling(X):
    """l1 normalize along the last index.
    
    Fix all but the last index and scale the resulting
    vector to unit l1 norm. Can be used to scale sequencing
    data to the same read depth.
    """
    Xn = torch.norm(X, p=1, dim=-1, keepdim=True)
    return X.div(Xn)

def full_tensor_product(X,U):
    """Compute the n-mode product with the list of matrices.

    This function takes a tensor X and a list of matrices U
    for each vector space and computes the corresponding 
    n-mode product of the tensor with each of the matrices
    X x_1 U^(1) x_2 U^(2)...x_p U^(p)
    Be careful about using the correct transposition for the
    context.
    """
    p = len(U)
    s = tensorly.tenalg.mode_dot(X,U[0],0)
    for i in range(1,p):
        s = tensorly.tenalg.mode_dot(s,U[i],i)
    return s

def HOSVD(X):
    """Compute the higher order singular value decomposition of X.

    Given a tensor X, obtain the HOSVD of the tensor
    and return the orthogonal matrix U for each subspace.
    Note that this function utilizes rank restrictions to
    reduce the SVD computation when applicable.  
    """
    U = []
    for i in range(0,len(dim)):
        X_n = tensorly.base.unfold(X,i)
        l = X_n.size()
        print(i)
        print( time.asctime( time.localtime(time.time()) ))
        sys.stdout.flush()
        if(l[0] <= l[1]):
            T = torch.mm(X_n,torch.t(X_n))
            E = torch.symeig(T,eigenvectors=True)
            eig_sort = torch.sort(E[0],0, descending=True)
            U.append(E[1][:,eig_sort[1]])
        else:
            T = torch.mm(torch.t(X_n),X_n)
            E = torch.symeig(T,eigenvectors=True)
            eig_sort = torch.sort(E[0],0, descending=True)
            V = torch.mm(X_n,E[1][:,eig_sort[1]])
            # Remove eigenvectors with zero eigenvalue
            # to prevent division by zero.
            nzi = torch.nonzero(eig_sort[0])
            U.append(V[:,nzi[:,0]]/torch.sqrt(eig_sort[0][nzi[:,0]]))
    S = full_tensor_product(X,[torch.t(U[i]) for i in range(0,len(U))])
    return U, S

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--data_tensor", type=str, required=True,
                        help=("Input data tensor. This is a PyTorch tensor storing     \n"
                              "the ChIP-seq data set. The only assumption made about   \n"
                              "the tensor is that the last index represents the genomic\n"
                              "location. This is used when projecting the data onto the\n"
                              "factor matrix for this space. Required"))
    parser.add_argument("--loc_cut", type=int, default=None,
                        help=("The number of location vectors to keep after decomposing.  \n"
                              "Setting this value to be less than the product of the      \n"
                              "dimensions of all but the last vector space will result    \n"
                              "in a truncated approximation to the tensor. This can       \n"
                              "save time and space when writing the output if only the    \n"
                              "first few location vectors are of interest. Default is     \n"
                              "to use the full rank given by the product of the dimensions\n"
                              "of the other vector spaces."))
    parser.add_argument("--scale_function", nargs='?', const="l1_scaling", default=None,
                        help=("Specify a method for scaling the input data tensor. \n"
                              "If this argument is given with no value, the column \n"
                              "vectors obtained from fixing all but the last index \n"
                              "will be scaled to unit l1 norm. Otherwise the name  \n"
                              "of a module can be given. The module must specify a \n"
                              "function named scale_function that takes an input   \n"
                              "tensor and returns a scaled version of this tensor. \n"
                              "Default is to not scale the input tensor in any way"))
    parser.add_argument("--cpu", const=True, default=False, action='store_const',
                        help=("Run the HOSVD on CPU. Default is to run on GPU."))
    parser.add_argument("-o", "--output", type=str, required=True,
                        help=("Prefix to use for writing the output files. Required."))

    args = parser.parse_args()

    output_name = args.output
    print("output_name: {}".format(output_name))
    print("input tensor: {}".format(args.data_tensor))
    print( time.asctime( time.localtime(time.time()) ))
    sys.stdout.flush()

    t = torch.load(args.data_tensor)
    dim = t.size()
    print( "loaded input tensor of shape {} at time {}".format(dim, time.asctime( time.localtime(time.time()) )))
    sys.stdout.flush()

    loc_cut = args.loc_cut
    if(loc_cut==None):
        loc_cut=np.prod(dim[:-1])
    print("Number of location vectors to keep: ", loc_cut)

    if(args.scale_function=="l1_scaling"):
        print("Using l1")
        t = l1_scaling(t)
    elif(args.scale_function!=None):
        print("Using {} for scaling".format(args.scale_function))
        scale_module = importlib.import_module(args.scale_function)
        scale_function = scale_module.scale_function
        t = scale_function(t)

    tensorly.set_backend('pytorch')
    print( 'moving to device' )
    print( time.asctime( time.localtime(time.time()) ))
    sys.stdout.flush()

    if(args.cpu):
        print("Using CPU")
        device = torch.device('cpu')
    else:
        print("Using GPU")
        device = torch.device('cuda')
    t = t.to(device)
    print(time.asctime( time.localtime(time.time()) ))
    sys.stdout.flush()

    print('Starting HOSVD')
    print( time.asctime( time.localtime(time.time()) ))
    sys.stdout.flush()

    U, S = HOSVD(t) 

    print('Finished HOSVD')   
    print( time.asctime( time.localtime(time.time()) ))
    sys.stdout.flush()

    if(loc_cut > U[-1].size()[1]):
        # Adjust the number of location vectors
        # in the event of some right singular 
        # vectors having zero eigenvalue. 
        print("Some location vectors had zero eigenvalue")
        loc_cut == U[-1].size()[1]

    ##########################
    # Check how well the HOSVD decomposition reproduces the 
    # input data tensor.

    x_h = full_tensor_product(S, U)
    print(torch.dist(t,x_h,p=2)/torch.norm(t,p=2),'fraction')
    print(torch.norm(t,p=2),'data norm')
    sys.stdout.flush()
    del x_h
    ##########################

    print( 'getting projections' )
    sys.stdout.flush()

    # Calculate the projections onto the decomposition
    # for the last index
    B = tensorly.tenalg.mode_dot(S,U[0],0)
    for i in range(1, len(dim)-1):
        B = tensorly.tenalg.mode_dot(B,U[i],i)

    print( 'finished projections' )
    sys.stdout.flush()

    # Write the projections onto the factor matrix for
    # the last index. Save as both a PyTorch tensor
    # and a text file.
    torch.save(B[Ellipsis,:loc_cut].to('cpu'), output_name + '_projections.pt')

    output_project = open(output_name + '_projections.txt','w+') 
    for index in np.ndindex(dim[:-1]):
        for i in range(len(dim)-2):
            output_project.write("{},".format(index[i]))
        output_project.write("{} : ".format(index[-1]))
        for k in range(loc_cut-1):
            output_project.write("{},".format(B[index][k]))
        output_project.write("{}\n".format(B[index][loc_cut-1]))
    output_project.close()

    # Write all the factor matrices and the core tensor
    torch.save(S[Ellipsis,:loc_cut].to('cpu'), output_name + '_core_tensor.pt')
    for i in range(len(dim)-1):
        torch.save(U[i].to('cpu'), output_name + '_factor_matrix_' + str(i) + '.pt')
    torch.save(U[-1][:,:loc_cut].to('cpu'), output_name + '_factor_matrix_' + str(len(dim)-1) + '.pt')

    print( time.asctime( time.localtime(time.time()) ))
