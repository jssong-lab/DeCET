import numpy as np
import math

##### Auxiliary Functions #####
# Some useful functions for working with genomic coordinate
# files (bed files)

def getChrList():
    """ An index dictionary for the hg19 genome."""
    chr_list = {'chr1':0, 'chr2':1, 'chr3':2, 'chr4':3, 'chr5':4,
                'chr6':5, 'chr7':6, 'chr8':7, 'chr9':8, 'chr10':9,
                'chr11':10, 'chr12':11, 'chr13':12, 'chr14':13, 'chr15':14,
                'chr16':15, 'chr17':16, 'chr18':17, 'chr19':18, 'chr20':19,
                'chr21':20, 'chr22':21, 'chrX':22, 'chrY':23}    
    return chr_list

def getChrIndex(chrom):
    """Return the index for the chromosome."""
    chr_list = getChrList()
    index = chr_list[chrom]
    return index

def isInt(var):
    try:
        int(var)
        return True
    except ValueError:
        return False
    
def isFloat(var):
    try:
        int(var)
        return True
    except ValueError:
        return False

##### Reading Functions #####

def readBed(bed_file):
    """Read a bed file into a python list.
    
    The list will consist of seperate lists
    for each chromosome.
    Note the returned lists for each chromosome
    are not sorted in any particular order.
    """
    bf = open(bed_file,'r')
    chr_list = getChrList()
    bed_array = [[] for l in chr_list]
    for line in bf:
        if(line[0] == '#'):
            continue
        l = line.strip().split('\t')
        if(l[0] not in chr_list):
            continue
        bed_array[chr_list[l[0]]].append([])
        for i in range(0,len(l)):
            if(isInt(l[i])):
                bed_array[chr_list[l[0]]][-1].append(int(l[i]))
            elif(isFloat(l[i])):
                bed_array[chr_list[l[0]]][-1].append(float(l[i]))
            else:
                bed_array[chr_list[l[0]]][-1].append(l[i])
    bf.close()
    return bed_array

def readBedList(bed_file):
    """Reads the bed file into a python list without separating by chromosome."""
    bed_array = readBed(bed_file)
    chr_list = getChrList()
    L = 0
    for l in range(len(chr_list)):
        L += len(bed_array[l])
    bed_list = [[] for l in range(L)]
    list_ind = 0
    for l in range(len(chr_list)):
        for i in range(len(bed_array[l])):
            bed_list[list_ind] = bed_array[l][i]
            list_ind += 1
    return bed_list

##### Sorting Functions #####

def sortBedFirstCoord(bed_array):
    """Sort a bed array by first coordinate along each chromosome."""
    chr_list = getChrList()
    sort_array = [[[] for i in range(len(bed_array[c]))] for c in range(len(chr_list))]
    for c in range(len(chr_list)):
        c_bed_L = len(bed_array[c])
        sort_ind = np.argsort([bed_array[c][i][1] for i in range(0,c_bed_L)])
        for i in range(c_bed_L):
            sort_array[c][i] = bed_array[c][sort_ind[i]]
    return sort_array

##### Overlapping Functions #####
# The overlap functions here use the half-open
# zero indexed convention.

def doesOverlap(s_reg, region_list):
    """Test if a region overlaps any region in a list."""
    chr_list = getChrList()
    reg_chr_i = chr_list[s_reg[0]]
    for j in range(len(region_list[reg_chr_i])):
        if(int(s_reg[2]) <= int(region_list[reg_chr_i][j][1])):
            continue
        if(int(s_reg[1]) >= int(region_list[reg_chr_i][j][2])):
            continue
        return True
    return False

def intersectBedFile(bed1_file, bed2_file):
    bed1 = sortBedFirstCoord(readBed(bed1_file))
    bed2 = sortBedFirstCoord(readBed(bed2_file))
    chr_list = getChrList()
    output = [[] for c in range(len(chr_list))]
    for c in range(len(chr_list)):
        for i in range(len(bed1[c])):
            for j in range(len(bed2[c])):
                if(bed1[c][i][1] >= bed2[c][j][2]):
                    continue
                if(bed1[c][i][2] <= bed2[c][j][1]):
                    break
                output[c].append([bed1[c][i][0],max(bed1[c][i][1],bed2[c][j][1]),min(bed1[c][i][2],bed2[c][j][2])])
    return output

def intersectBed(bed1_array, bed2_array):
    """Obtain the intersection of two bed arrays."""
    bed1 = sortBedFirstCoord(bed1_array)
    bed2 = sortBedFirstCoord(bed2_array)
    chr_list = getChrList()
    output = [[] for c in range(len(chr_list))]
    for c in range(len(chr_list)):
        for i in range(len(bed1[c])):
            for j in range(len(bed2[c])):
                if(bed1[c][i][1] >= bed2[c][j][2]):
                    continue
                if(bed1[c][i][2] <= bed2[c][j][1]):
                    break
                output[c].append([bed1[c][i][0],max(bed1[c][i][1],bed2[c][j][1]),min(bed1[c][i][2],bed2[c][j][2])])
    return output

def unionOfRegions(bed, ext=0):
    """Get the union of genomic regions.
    
    The ext parameter can be used to merge regions
    within the provided extension distance.
    """
    chr_list = getChrList()
    sort_bed = sortBedFirstCoord(bed)
    merged = [[] for c in range(len(chr_list))]
    for c in range(len(chr_list)):
        if(len(sort_bed[c]) == 0):
            continue
        merged[c] = [[sort_bed[c][0][0],sort_bed[c][0][1],sort_bed[c][0][2]]]
        for i in range(len(sort_bed[c])):
            if(sort_bed[c][i][1] <= merged[c][-1][2] + ext):
                merged[c][-1][2] = max(merged[c][-1][2], sort_bed[c][i][2])
            else:
                merged[c].append([sort_bed[c][i][0],sort_bed[c][i][1],sort_bed[c][i][2]])
    return merged

def writeBed(bed, output_name):
    """Write a bed array to an output file."""
    output = open(output_name,'w+')
    chr_list = getChrList()
    for c in range(len(chr_list)):
        for i in range(len(bed[c])):
            for l in range(len(bed[c][i])-1):
                output.write("{}\t".format(bed[c][i][l]))
            output.write("{}\n".format(bed[c][i][len(bed[c][i])-1]))
    output.close()
    return
