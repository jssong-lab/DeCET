import sys
import numpy as np
import time
import torch
import math
import argparse
from sklearn.linear_model import LinearRegression
import pickle

def select_genome(genome):
    """Specify the genome chromosomes and lengths to be used."""
    if(genome == 'hg19'):
        # hg19 genome with mitochondrial DNA not included
        chr_length = {'chrY': 59373566, 'chrX': 155270560, 'chr13': 115169878, 'chr12': 133851895,
                      'chr11': 135006516, 'chr10': 135534747, 'chr17': 81195210, 'chr16': 90354753,
                      'chr15': 102531392, 'chr14': 107349540, 'chr19': 59128983, 'chr18': 78077248,
                      'chr22': 51304566, 'chr20': 63025520, 'chr21': 48129895,
                      'chr7': 159138663, 'chr6': 171115067, 'chr5': 180915260, 'chr4': 191154276,
                      'chr3': 198022430, 'chr2': 243199373, 'chr1': 249250621, 'chr9': 141213431,
                      'chr8': 146364022}
        chr_list = {'chr1':0, 'chr2':1, 'chr3':2, 'chr4':3, 'chr5':4,
                    'chr6':5, 'chr7':6, 'chr8':7, 'chr9':8, 'chr10':9,
                    'chr11':10, 'chr12':11, 'chr13':12, 'chr14':13, 'chr15':14,
                    'chr16':15, 'chr17':16, 'chr18':17, 'chr19':18, 'chr20':19,
                    'chr21':20, 'chr22':21, 'chrX':22, 'chrY':23}  
    elif(genome == 'hg19X'):
        # This is just the hg19 genome without chrM and chrY
        chr_length = {'chrX': 155270560, 'chr13': 115169878, 'chr12': 133851895,
                      'chr11': 135006516, 'chr10': 135534747, 'chr17': 81195210, 'chr16': 90354753,
                      'chr15': 102531392, 'chr14': 107349540, 'chr19': 59128983, 'chr18': 78077248,
                      'chr22': 51304566, 'chr20': 63025520, 'chr21': 48129895,
                      'chr7': 159138663, 'chr6': 171115067, 'chr5': 180915260, 'chr4': 191154276,
                      'chr3': 198022430, 'chr2': 243199373, 'chr1': 249250621, 'chr9': 141213431,
                      'chr8': 146364022}
        chr_list = {'chr1':0, 'chr2':1, 'chr3':2, 'chr4':3, 'chr5':4,
                    'chr6':5, 'chr7':6, 'chr8':7, 'chr9':8, 'chr10':9,
                    'chr11':10, 'chr12':11, 'chr13':12, 'chr14':13, 'chr15':14,
                    'chr16':15, 'chr17':16, 'chr18':17, 'chr19':18, 'chr20':19,
                    'chr21':20, 'chr22':21, 'chrX':22}
    elif(genome == 'hg19S'):
        # The somatic hg19 genome
        chr_length = {'chr13': 115169878, 'chr12': 133851895,
                      'chr11': 135006516, 'chr10': 135534747, 'chr17': 81195210, 'chr16': 90354753,
                      'chr15': 102531392, 'chr14': 107349540, 'chr19': 59128983, 'chr18': 78077248,
                      'chr22': 51304566, 'chr20': 63025520, 'chr21': 48129895,
                      'chr7': 159138663, 'chr6': 171115067, 'chr5': 180915260, 'chr4': 191154276,
                      'chr3': 198022430, 'chr2': 243199373, 'chr1': 249250621, 'chr9': 141213431,
                      'chr8': 146364022}
        chr_list = {'chr1':0, 'chr2':1, 'chr3':2, 'chr4':3, 'chr5':4,
                    'chr6':5, 'chr7':6, 'chr8':7, 'chr9':8, 'chr10':9,
                    'chr11':10, 'chr12':11, 'chr13':12, 'chr14':13, 'chr15':14,
                    'chr16':15, 'chr17':16, 'chr18':17, 'chr19':18, 'chr20':19,
                    'chr21':20, 'chr22':21}
    elif(genome == 'hg38'):
        # hg38 genome
        chr_length = {'chr1': 248956422, 'chr2': 242193529, 'chr3': 198295559, 'chr4': 190214555, 
                      'chr5': 181538259, 'chr6': 170805979, 'chr7': 159345973, 'chr8': 145138636,
                      'chr9': 138394717, 'chr10': 133797422, 'chr11': 135086622, 'chr12': 133275309,
                      'chr13': 114364328, 'chr14': 107043718, 'chr15': 101991189, 'chr16': 90338345,
                      'chr17': 83257441, 'chr18': 80373285, 'chr19': 58617616, 'chr20': 64444167,
                      'chr21': 46709983, 'chr22': 50818468, 'chrX': 156040895, 'chrY': 57227415}
        chr_list = {'chr1':0, 'chr2':1, 'chr3':2, 'chr4':3, 'chr5':4,
                    'chr6':5, 'chr7':6, 'chr8':7, 'chr9':8, 'chr10':9,
                    'chr11':10, 'chr12':11, 'chr13':12, 'chr14':13, 'chr15':14,
                    'chr16':15, 'chr17':16, 'chr18':17, 'chr19':18, 'chr20':19,
                    'chr21':20, 'chr22':21, 'chrX':22, 'chrY':23}  
    elif(genome == 'mm10'):
        # mm10 genome
        chr_length = {'chr1':195471971, 'chr2': 182113224, 'chr3': 160039680, 'chr4': 156508116,
                      'chr5': 151834684, 'chr6': 149736546, 'chr7': 145441459, 'chr8': 129401213,
                      'chr9': 124595110, 'chr10': 130694993, 'chr11': 122082543, 'chr12': 120129022,
                      'chr13': 120421639, 'chr14': 124902244, 'chr15': 104043685, 'chr16': 98207768,
                      'chr17': 94987271, 'chr18': 90702639, 'chr19': 61431566, 'chrX': 171031299,
                      'chrY': 91744698}
        chr_list = {'chr1':0, 'chr2':1, 'chr3':2, 'chr4':3, 'chr5':4,
                    'chr6':5, 'chr7':6, 'chr8':7, 'chr9':8, 'chr10':9,
                    'chr11':10, 'chr12':11, 'chr13':12, 'chr14':13, 'chr15':14,
                    'chr16':15, 'chr17':16, 'chr18':17, 'chr19':18, 'chrX':19, 'chrY':20}
    elif(genome == 'custom'):
        # Read in a custom genome file
        genome_file = open('custom_genome.txt','r')   
        chr_length = {}
        chr_list = {}
        chr_index = 0
        for line in genome_file:
            l = line.strip().split()
            chr_length[l[0]] = int(l[1])
            chr_list[l[0]] = chr_index
            chr_index += 1
        genome_file.close()

    return chr_length, chr_list

def bin_bed_data(bed_file):
    """Obtain a genome wide histogram of reads from a bed file

    The global bin_size variable specifies the size of the bins.
    Each read is shifted 100bp in the 3 prime direction of the 
    strand to which the read aligned.
    """
    print("Binning file: ", bed_file) 
    sys.stdout.flush()

    # If no control is present return 0 vector
    if(bed_file == '.'):
        print("Warning: No control file specified for one of the treatments")
        sys.stdout.flush()
        tot_len = 0
        for c in chr_list:
            tot_len += math.ceil(chr_length[c]/bin_size) + 1
        return np.array([0.0 for i in range(tot_len)])

    pt_hist = [[0.0 for i in range(math.ceil(chr_length[chro]/bin_size) + 1)] for chro in chr_list]
    bf = open(bed_file,'r')
    for line in bf:
        l = line.strip().split('\t')
        if(l[0] not in chr_list):
            continue
        if(l[5] == '+'):
            shift = 100
            pt_hist[chr_list[l[0]]][math.floor((int(l[1]) + shift)/bin_size)] += 1
        elif(l[5] == '-'):
            shift = int(l[2]) - int(l[1]) - 100 
            # Avoid shifting a read beyond the beginning of the chromosome
            if(int(l[1]) + shift < 0):
                pt_hist[chr_list[l[0]]][0] += 1
                continue
            pt_hist[chr_list[l[0]]][math.floor((int(l[1]) + shift)/bin_size)] += 1
        else:
            print >> sys.stderr, "Unrecognized strand"
            sys.exit(0)
    bf.close()

    tot_len = 0
    for c in chr_list:
        tot_len += math.ceil(chr_length[c]/bin_size) + 1
    pt_hist_flat = np.zeros(tot_len, np.double)
    counter = 0
    for c in chr_list:
        for l in range(len(pt_hist[chr_list[c]])):
            pt_hist_flat[counter] = pt_hist[chr_list[c]][l]
            counter += 1
    del pt_hist
    return pt_hist_flat

def residual(x, y, slope, intercept):
    """Calculate the residual sum of squares for linear regression"""
    residual = 0.0
    L = len(x)
    for l in range(L):
        residual += (y[l] - slope*x[l] - intercept)**2
    residual /= L - 1
    return residual

def compute_scale(TR_vector, CT_vector):
    """Scale control data to signal data.

    Implementaion of the Signal Extraction Scaling method to scale
    the control data to the corresponding signal data set. 
    """

    # For empty control (zero reads) return 0 as the scale factor.
    if(sum(CT_vector) == 0.0):
        print("Warning: Empty control file, setting scale factor to zero")
        sys.stdout.flush()
        return 0.0

    X = list(TR_vector)
    X.sort()

    L = len(X)
    Ydata = []
    Xdata = []
    R = list(range(20,99))
    R.extend([99, 99.2 , 99.4, 99.6, 99.8, 100])

    for k in R:
        cut = min(int(L*k*.01), L-1)

        upper1 = X[cut]
        S1  = 0
        S2 = 0
        for j in range(len(TR_vector)):
            if(TR_vector[j] <= upper1):
                S1 += TR_vector[j]
                S2 += CT_vector[j]

        Ydata.append(S1*1.0/S2) # IP
        Xdata.append(k) # Input

    xdata = Xdata[20:61]
    ydata = Ydata[20:61]
    
    clf = LinearRegression(fit_intercept=True,
                           normalize=False).fit(np.array(xdata).reshape(-1, 1), np.array(ydata).reshape(-1, 1))
    intercept = clf.intercept_
    slope = clf.coef_
    sigma = np.sqrt(residual(xdata, ydata, slope[0][0], intercept[0]))
    #print(slope, sigma, clf.score(np.array(xdata).reshape(-1, 1),np.array(ydata).reshape(-1, 1)))
    
    cut = 0
    scale = 0
    for k in range(0, len(R)):
        predicted = slope* R[k] + intercept
        #print(R[k], predicted)

    for k in range(51, len(R)):
        predicted = slope* R[k] + intercept
        if abs(predicted - Ydata[k]) > 2*sigma:
            cut = R[k]
            scale = Ydata[k]
            break
    if cut == 0:
        cut = R[-1]
        scale = Ydata[-1]
    print("scale within: ", scale)
    sys.stdout.flush()

    return scale

def fill_tensor(vector_list, index_list, shape):
    """Arrange the vector samples into a tensor."""
    filled_t = torch.zeros(shape, dtype=torch.double)
    for i, index in enumerate(index_list):
        filled_t[tuple(index)] = torch.from_numpy(vector_list[i])
    return filled_t

def fill_control_tensor(uc_vector_list, index_list, shape, index_dic):
    """Arrange the control vectors into a tensor."""
    filled_t = torch.zeros(shape, dtype=torch.double)
    for i, index in enumerate(index_list):
        filled_t[tuple(index)] = torch.from_numpy(uc_vector_list[index_dic[tuple(index)]])
    return filled_t

def get_unique_control(control_list, index_list):
    """Get the unique control files."""
    unique_control_list = []
    control_dic = {}
    for i, cfile in enumerate(control_list):
        if(cfile not in unique_control_list):
            unique_control_list.append(cfile)
    unique_control_dic = {}
    for i, ucfile in enumerate(unique_control_list):
        unique_control_dic[ucfile] = i
    for i, cfile in enumerate(control_list):
        control_dic[tuple(index_list[i])] = unique_control_dic[cfile]
    return unique_control_list, control_dic

def dim_from_index(index_list, genome_length):
    """Infer the tensor shape from the specified indices."""
    L = len(index_list[0])
    N = len(index_list)
    dim_tmp = [0 for l in range(L+1)]
    for l in range(L):
        dim_tmp[l] = max([index_list[i][l] for i in range(N)]) + 1
    dim_tmp[-1] = genome_length
    return tuple(dim_tmp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--file_list", type=str,
                        help=("A text file specifying the ChIP-seq signal and control files   \n"
                              "containing aligned reads in bed format and the corresponding   \n"
                              "indices used to fill the data tensor. The input bed files do   \n"
                              "not need to be sorted. This should be a white space delimited  \n"
                              "file with the first entry specifying the ChIP-seq signal file, \n"
                              "the second entry the ChIP-seq control file, and then the white \n"
                              "space separated index of the subtensor that will store the     \n"
                              "binned signal for this sample. The second entry can be replaced\n"
                              "with a . if no control file is available. In this case the     \n"
                              "ChIP-seq signal reads will be binned and the result returned   \n"
                              "with no subtraction of matched control. Required"))
    parser.add_argument("-o", "--output", type=str,
                   help=("Name of the output file. Required."))
    parser.add_argument("-g", "--genome", type=str, choices=["hg19","hg19X","hg19S","hg38","mm10","custom"], default="hg19",
                        help=("Specify a genome to use. This specifies the chromosomes to be  \n"
                              "used and the length of these chromosomes. Reads mapping to     \n"
                              "unspecified chromosomes will be removed. Options are hg19,     \n"
                              "hg19X, hg19S, hg38, mm10, or custom. If custom is specified,   \n"
                              "the working directory will be searched for a file named        \n"
                              "custom_genome.txt that specifies the chromosomes and           \n"
                              "corresponding lengths of a genome. Each line should specify a  \n"
                              "chromosome name that will appear in the corresponding bed files\n"
                              "and the length of this chromosome separated by a white space,  \n"
                              "e.g. chr1	249250621"))
    parser.add_argument("-b", "--bin_size", type=int, default=2000,
                        help=("Size of the bin to use for binning up the genome. Default 2000"))
    parser.add_argument("-p", "--processors", type=int, default=1,
                        help=("Number of processors to use. Parallelization allows multiple   \n"
                              "bed files to be read and binned simultaneously. The scaling    \n"
                              "factors are also calculated in parallel. Default 1"))

    args = parser.parse_args()

    treatment_list = args.file_list
    chr_length, chr_list = select_genome(args.genome)
    bin_size = args.bin_size
    p = args.processors
    output_name = args.output

    print("File specifying a list of treatment files: ", treatment_list)
    print("Genome to ues: {}".format(args.genome))
    print("Bin size: {}".format(bin_size))
    print("Output: {}".format(bin_size))
    print("N processors: {}".format(p))

    # Multiprocessing is only imported if more than one
    # processors are available.
    if(p > 1):
        import multiprocessing

    sys.stdout.flush()

    # Get the length of the binned genome.
    genome_length = 0
    for c in chr_list:
        genome_length += math.ceil(chr_length[c]/bin_size) + 1

    print( time.asctime( time.localtime(time.time()) ))
    sys.stdout.flush()
    
    # Read in the file list storing the input bed files and the 
    # corresponding indices used to structure the data into a 
    # tensor  
    treatment_file_list = []
    control_file_list = []
    treatment_index_list = []
    fl = open(treatment_list,'r')
    for line in fl:
        l=line.strip().split()
        index = [int(l[i]) for i in range(2,len(l))]
        print(index)
        sys.stdout.flush()
        print(l[0])
        treatment_file_list.append(l[0])
        control_file_list.append(l[1])
        treatment_index_list.append(index)
    fl.close()

    dim = dim_from_index(treatment_index_list, genome_length)
    print("Tensor shape: ", dim)
    sys.stdout.flush()

    if(p > 1):
        # Set up the pool for multiprocessing
        pool = multiprocessing.Pool(processes=p)
        treatment_vector_list = pool.map(bin_bed_data, treatment_file_list)
        pool.close()
        pool.join()
    else:
        treatment_vector_list = [bin_bed_data(f) for f in treatment_file_list]

    print("Filling treatment tensor")
    treatment_tensor = fill_tensor(treatment_vector_list, treatment_index_list, dim)
    del(treatment_vector_list)

    # Get a list of unique control files to avoid binning these
    # multiple times.
    unique_control_list, control_dic = get_unique_control(control_file_list, treatment_index_list)

    if(p > 1):
        pool = multiprocessing.Pool(processes=p)
        unique_control_vector_list = pool.map(bin_bed_data, unique_control_list)
        pool.close()
        pool.join()
    else:
        unique_control_vector_list = [bin_bed_data(f) for f in unique_control_list]

    # Fill the control tensor.
    control_tensor = fill_control_tensor(unique_control_vector_list, treatment_index_list, dim, control_dic)
    del(unique_control_vector_list)

    print(sys.getsizeof(treatment_tensor), sys.getsizeof(control_tensor))
    sys.stdout.flush()

    def subtractControl(index):
        """A mutliprocessing compatible function to compute the scaling factors"""
        print("Starting scale: ", index)
        sys.stdout.flush()
        global treatment_tensor
        global control_tensor
        return np.array(compute_scale(treatment_tensor[tuple(index)].tolist(), control_tensor[tuple(index)].tolist()))

    # Compute the scaling factors for each treatment file and
    # format these into a tensor. 
    if(p > 1): 
        pool = multiprocessing.Pool(processes=p)
        scale_factors = pool.map(subtractControl, treatment_index_list)
        pool.close()
        pool.join()
    else:
        scale_factors = [subtractControl(ind) for ind in treatment_index_list]

    scale_tensor = fill_tensor(scale_factors, treatment_index_list, dim[:-1])

    # Save the scale factors
    torch.save(scale_tensor, output_name + '_scale_tensor' + '.pt')

    # Save the number of reads in each control file
    torch.save(torch.sum(control_tensor, dim=-1), output_name + '_net_control_tensor' + '.pt')

    # Subtract the scaled control from the treatment tensor
    for index in treatment_index_list:
        print("Scale: ", scale_tensor[tuple(index)], tuple(index))
        print("Subtracting: ", index)
        sys.stdout.flush()
        treatment_tensor[tuple(index)] = treatment_tensor[tuple(index)] - scale_tensor[tuple(index)]*control_tensor[tuple(index)]

    # Set resulting negative entries to zero
    treatment_tensor = torch.clamp(treatment_tensor, min=0.0)

    # Save the treatment_tensor
    print("saving tensor")
    sys.stdout.flush()
    torch.save(treatment_tensor, output_name + '.pt')

    # Save a list specifying the genomic coordinates of the
    # bin corresponding to the genomic location index used 
    # in the tensor.
    genomic_bin = [[] for l in range(0,genome_length)]
    counter = 0
    for chrom in chr_list:
        for l in range(0, int(np.ceil(chr_length[chrom]/bin_size))+1):
            genomic_bin[counter] = [chrom, l*bin_size, (l+1)*bin_size]
            counter += 1
    f = open(output_name + '_index_to_genomic.pkl', 'wb+')
    pickle.dump(genomic_bin, f)
    f.close()

