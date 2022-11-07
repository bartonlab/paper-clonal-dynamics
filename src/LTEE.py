# LIBRARIES

import numpy as np
import pandas as pd
import math
from scipy import stats
from scipy import interpolate
import sys

DATA_DIRECTORY = './data/LTEE-metagenomic-master/data_files/'

all_lines = ['m5','p2','p4','p1','m6','p5','m1','m2','m3','m4','p3','p6']
complete_nonmutator_lines = ['m5','m6','p1','p2','p4','p5']

starting_idx = 0
base_table = {'A':'T','T':'A','G':'C','C':'G'}
codon_table = { 'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A', 'CGT': 'R', 'CGC': 'R', 'CGA':'R',
'CGG':'R', 'AGA':'R', 'AGG':'R', 'AAT':'N', 'AAC':'N', 'GAT':'D', 'GAC':'D', 'TGT':'C', 'TGC':'D', 'CAA':'Q', 'CAG':'Q', 'GAA':'E', 'GAG':'E', 'GGT':'G', 'GGC':'G', 'GGA':'G', 'GGG':'G', 'CAT':'H', 'CAC':'H', 'ATT':'I', 'ATC':'I', 'ATA':'I', 'TTA':'L', 'TTG':'L', 'CTT':'L', 'CTC':'L', 'CTA':'L', 'CTG':'L', 'AAA':'K', 'AAG':'K', 'ATG':'M', 'TTT':'F', 'TTC':'F', 'CCT':'P', 'CCC':'P', 'CCA':'P', 'CCG':'P', 'TCT':'S', 'TCC':'S', 'TCA':'S', 'TCG':'S', 'AGT':'S', 'AGC':'S', 'ACT':'T', 'ACC':'T', 'ACA':'T', 'ACG':'T', 'TGG':'W', 'TAT':'Y', 'TAC':'Y', 'GTT':'V', 'GTC':'V', 'GTA':'V', 'GTG':'V', 'TAA':'!', 'TGA':'!', 'TAG':'!' }

# calculate number of synonymous opportunities for each codon
codon_synonymous_opportunity_table = {}
for codon in codon_table.keys():
    codon_synonymous_opportunity_table[codon] = {}
    for i in range(0,3):
        codon_synonymous_opportunity_table[codon][i] = -1 # since G->G is by definition synonymous, but we don't want to count it
        codon_list = list(codon)
        for base in ['A','C','T','G']:
            codon_list[i]=base
            new_codon = "".join(codon_list)
            if codon_table[codon]==codon_table[new_codon]:
                # synonymous!
                codon_synonymous_opportunity_table[codon][i]+=1

bases = set(['A','C','T','G'])
substitutions = []
for b1 in bases:
    for b2 in bases:
        if b2==b1:
            continue

        substitutions.append( '%s->%s' % (b1,b2) )

codon_synonymous_substitution_table = {}
codon_nonsynonymous_substitution_table = {}
for codon in codon_table.keys():
    codon_synonymous_substitution_table[codon] = [[],[],[]]
    codon_nonsynonymous_substitution_table[codon] = [[],[],[]]

    for i in range(0,3):
        reference_base = codon[i]

        codon_list = list(codon)
        for derived_base in ['A','C','T','G']:
            if derived_base==reference_base:
                continue
            substitution = '%s->%s' % (reference_base, derived_base)
            codon_list[i]=derived_base
            new_codon = "".join(codon_list)
            if codon_table[codon]==codon_table[new_codon]:
                # synonymous!
                codon_synonymous_substitution_table[codon][i].append(substitution)
            else:
                codon_nonsynonymous_substitution_table[codon][i].append(substitution)

clade_hmm_states = {'A':0,'E':1,'FB':2,'FM':3, 'Fm':4,'PB':5,'PM':6,'Pm':7,'PB*':8}
well_mixed_hmm_states = {'A':0,'E':1,'F':2,'P':3}

UNBORN = clade_hmm_states['A']
EXTINCT= clade_hmm_states['E']
ANCESTRAL_FIXED = clade_hmm_states['FB']
MINOR_FIXED=clade_hmm_states['Fm']
MAJOR_FIXED=clade_hmm_states['FM']
ANCESTRAL_POLYMORPHIC=clade_hmm_states['PB']
MINOR_POLYMORPHIC=clade_hmm_states['Pm']
MAJOR_POLYMORPHIC=clade_hmm_states['PM']

clade_extinct_states = set([clade_hmm_states['A'],clade_hmm_states['E']])
clade_fixed_states = set([clade_hmm_states['FB'], clade_hmm_states['FM'], clade_hmm_states['Fm'], clade_hmm_states['PB*']])

clade_majority_states = set([clade_hmm_states['FB'], clade_hmm_states['FM'], clade_hmm_states['PB'],clade_hmm_states['PM'],clade_hmm_states['PB*']])

clade_polymorphic_states = set([clade_hmm_states['PB'], clade_hmm_states['PM'], clade_hmm_states['Pm']])

well_mixed_extinct_states = set([well_mixed_hmm_states['A'], well_mixed_hmm_states['E']])
well_mixed_polymorphic_states = set([well_mixed_hmm_states['P']])
well_mixed_fixed_states = set([well_mixed_hmm_states['F']])

FIXED = well_mixed_hmm_states['F']
POLYMORPHIC = well_mixed_hmm_states['P']


def calculate_W(Ne0,Nr0,Nef,Nrf,cycles=1):
    return np.log(Nef*(100.0**(0))/Ne0)/np.log(Nrf*(100.0**(0))/Nr0)


def calculate_X(Ne0,Nr0,Nef,Nrf,cycles=1):
    return np.log( (Nef/Nrf) / (Ne0/Nr0) ) / (6.64*cycles)


def parse_ancestor_fitnesses(filename="./LTEE-metagenomic-master/additional_data/Concatenated.LTEE.data.all.csv"):
    line_data = {line: [] for line in all_lines}
    file = open(filename,"r")
    file.readline() # skip headers
    for line in file:
        #print line
        items = line.split(",")
        t = int(items[0])

        subitems = items[3].split()
        idx=int(subitems[2])
        if subitems[1]=='-':
            line_idx='m%d' % idx
        else:
            line_idx='p%d' % idx

        if items[5] == '' or items[8]=='':
            continue


        R0c = float(items[5])
        W0c = float(items[6])
        D0 = float(items[7])
        Rfc = float(items[8])
        Wfc = float(items[9])
        Df = float(items[10])
        theirW = float(items[11])

        R0 = 1.0*R0c*D0
        W0 = 1.0*W0c*D0
        Rf = 1.0*Rfc*Df
        Wf = 1.0*Wfc*Df

        if items[1][0]=='A':
            # evolved clone is red pop
            Ne0 = R0
            Nef = Rf
            Nr0 = W0
            Nrf = Wf
        else:
            Ne0 = W0
            Nef = Wf
            Nr0 = R0
            Nrf = Rf

        W = calculate_W(Ne0,Nr0,Nef,Nrf)

        if math.fabs(W-theirW) > 1e-02:
            # should never happen!
            print("Fitnesses don't match up!")

        line_data[line_idx].append((t,Ne0,Nr0,Nef,Nrf))

    file.close()

    # process data

    new_line_data = {line: [{},{}] for line in all_lines}
    for line in all_lines:
        for t,Ne0,Nr0,Nef,Nrf in line_data[line]:
            W = calculate_W(Ne0,Nr0,Nef,Nrf)
            X = calculate_X(Ne0,Nr0,Nef,Nrf)
            if not t in new_line_data[line][0]:
                new_line_data[line][0][t] = []
                new_line_data[line][1][t] = []
            new_line_data[line][0][t].append(X)
            new_line_data[line][1][t].append(W)

    trajectories = {line: None for line in all_lines}
    for line in all_lines:
        ts = list(new_line_data[line][0].keys())
        sorted(ts)
        xs = np.zeros_like(ts)*1.0
        error_xs = np.zeros_like(ts)*1.0
        ws = np.zeros_like(ts)*1.0
        for i in range(0,len(ts)):
            xs[i] = np.array(new_line_data[line][0][ts[i]]).mean()
            error_xs[i] = new_line_data[line][0][ts[i]][0] - xs[i]
            ws[i] = np.array(new_line_data[line][1][ts[i]]).mean()
        trajectories[line] = [ts,xs,ws,error_xs]

    return trajectories, line_data


def parse_annotated_timecourse(population, only_passed=True, min_coverage=5, data_directory=None):

    mutations = []

    if data_directory is None:
        timecourse_filename = DATA_DIRECTORY + ("%s_annotated_timecourse.txt" % population)
    else:
        timecourse_filename = data_directory + ("%s_annotated_timecourse.txt" % population)
    file = open(timecourse_filename, "r")

    header_line = file.readline()
    items = header_line.strip().split(",")

    times = []
    for i in range(13,len(items),2):
        times.append(int(items[i].split(":")[1]))
    times = np.array(times)

    # depth line
    depth_line = file.readline()
    items = depth_line.strip().split(",")
    avg_depths = []
    for i in range(13,len(items),2):
        avg_depths.append(float(items[i+1]))
    avg_depths = np.array(avg_depths)

    population_avg_depth_times = times[times<1000000]
    population_avg_depths = avg_depths[times<1000000]
    clone_avg_depth_times = times[times>1000000]-1000000
    clone_avg_depths = avg_depths[times>1000000]

    for line in file:
        items = line.strip().split(",")
        location = int(items[0])
        gene_name = items[1].strip()
        allele = items[2].strip()
        var_type = items[3].strip()
        test_statistic = float(items[4])
        pvalue = float(items[5])
        cutoff_idx = int(items[6])
        depth_fold_change = float(items[7])
        depth_change_pvalue = float(items[8])

        duplication_idx = int(items[9])
        fold_increase = float(items[10])
        duplication_pvalue = float(items[11])

        passed_str = items[12]
        if passed_str.strip()=='PASS':
            passed = True
        else:
            passed = False

        alts = []
        depths = []

        for i in range(13,len(items),2):
            alts.append(int(float(items[i])))
            depths.append(int(float(items[i+1])))

        alts = np.array(alts)
        depths = np.array(depths)

        # zero out timepoints with individual coverage lower than some threshold
        alts *= (depths>=min_coverage)*(avg_depths>=min_coverage)
        depths *= (depths>=min_coverage)*(avg_depths>=min_coverage)

        pop_times = times[(times<1000000)]
        pop_alts = alts[(times<1000000)]
        pop_depths = depths[(times<1000000)]

        clone_times = times[(times>1000000)]-1000000
        clone_alts = alts[(times>1000000)]
        clone_depths = depths[(times>1000000)]

        if passed or (not only_passed):
            mutations.append((location, gene_name, allele, var_type, test_statistic, pvalue, cutoff_idx, depth_fold_change, depth_change_pvalue, pop_times, pop_alts, pop_depths, clone_times, clone_alts, clone_depths))

    file.close()

    # sort by position
    keys = [mutation[0] for mutation in mutations]
    keys, mutations = (list(t) for t in zip(*sorted(zip(keys, mutations))))
    return mutations, (population_avg_depth_times, population_avg_depths, clone_avg_depth_times, clone_avg_depths)


def parse_haplotype_timecourse(population, data_directory = None):

    if data_directory is None:
        haplotype_filename = DATA_DIRECTORY + ('%s_haplotype_timecourse.txt' % population)
    else:
        haplotype_filename = data_directory + ('%s_haplotype_timecourse.txt' % population)
    file = open(haplotype_filename,"r")

    times = np.array([float(item) for item in file.readline().split(",")])
    fmajors = np.array([float(item) for item in file.readline().split(",")])
    fminors = np.array([float(item) for item in file.readline().split(",")])
    file.readline()
    file.readline()
    file.readline()
    file.readline()
    file.readline()
    file.readline()
    haplotypes = []
    for line in file:
        Ls = np.array([float(item) for item in line.split(",")])
        haplotypes.append(Ls)
    file.close()
    return times, fmajors, fminors, haplotypes


def mask_timepoints(times, alts, depths, var_type, cutoff_idx, depth_fold_change, depth_change_pvalue, min_depth=5):

    # first make a copy of alts and depths
    # so that we can modify in place without worrying
    masked_alts = np.copy(alts)
    masked_depths = np.copy(depths)

    # zero out timepoints that don't pass depth threshold
    masked_alts[masked_depths<min_depth] = 0
    masked_depths[masked_depths<min_depth] = 0

    #masked_alts -= masked_alts*(masked_depths < min_depth)
    #masked_depths -= masked_depths*(masked_depths < min_depth)

    # did we infer that a deletion happened?
    #if (var_type=='sv' and depth_fold_change < -2 and depth_change_pvalue < 1e-04) or (var_type!='sv' and depth_fold_change < -1 and depth_change_pvalue < 1e-03):
    if depth_change_pvalue < 1e-02:
       # deletion nearby, trim timecourse
       masked_alts[cutoff_idx:] = 0
       masked_depths[cutoff_idx:] = 0

    good_idxs = np.nonzero(masked_depths>0.5)[0]

    return good_idxs, masked_alts, masked_depths


def estimate_frequencies(alts,depths):
    """
    Naive frequency estimator, # alts / # depths (0 if no depth)
    """
    return alts * 1.0 / (depths + (depths == 0))


def create_interpolation_function(times, freqs, tmax=100000, kind='linear'):
    # can create it for anything!

    padded_times = np.zeros(len(times)+1)
    padded_freqs = np.zeros(len(times)+1)
    padded_times[0:len(times)] = times
    padded_freqs[0:len(times)] = freqs
    padded_times[-1] = tmax
    padded_freqs[-1] = freqs[-1]

    #xis = np.log((alts+1e-01)/(depths-alts+1e-01))
    #interpolating_function = interp1d(times, xis, kind='linear',bounds_error=False)
    #interpolated_xis = interpolating_function(theory_times)
    #interpolated_freqs = 1.0/(1+np.exp(-interpolated_xis))
    #freqs = 1.0/(1+np.exp(-xis))

    interpolating_function = interpolate.interp1d(padded_times, padded_freqs, kind=kind,bounds_error=True)

    return interpolating_function


def parse_gene_list(reference_sequence=None, filename="LTEE-metagenomic-master/additional_data/REL606.6.gbk"):

    features = set(['CDS','gene','tRNA','rRNA','repeat_region'])

    if reference_sequence==None:
        reference_sequence=parse_reference_genome()

    observed_gene_names = set()

    gene_names = []
    start_positions = []
    end_positions = []
    promoter_start_positions = []
    promoter_end_positions = []
    gene_sequences = []
    strands = []

    file = open(filename,"r")
    line = file.readline()
    while line!="":

        items = line.split()
        feature = items[0]

        gene_name = ""
        feature_location = ""

        if feature=='CDS':
            feature_location = items[1]

            line = file.readline().strip()

            gene_name=""
            locus_name=""
            is_pseudo=False

            while line.split()[0] not in features:

                if line.startswith('/gene'):
                    gene_name = line.split('=')[1].strip()[1:-1]
                if line.startswith('/locus_tag'):
                    locus_name = line.split('=')[1].strip()[1:-1]
                if line.startswith('/pseudo'):
                    is_pseudo=True

                line = file.readline().strip()

            if gene_name=="":
                gene_name = locus_name

            if is_pseudo:
                gene_name = ""

            # done here

        elif feature=='tRNA' or feature=='rRNA':

            feature_location = items[1]
            #print feature_location

            while not line.strip().startswith('/gene'):
                line = file.readline().strip()
            gene_name = line.split('=')[1].strip()[1:-1]
            gene_name = '%s:%s' % (feature, gene_name)


        else:
            # nothing to see here
            line = file.readline().strip()

        # If the element has a feature location string and a name
        # it should either be a gene, tRNA, or rRNA, so let's get details
        if feature_location!="" and gene_name!="":

            location_str = feature_location.lstrip("complement(").lstrip("join(").rstrip(")")
            location_strs = location_str.split(",")

            for location_str in location_strs:

                locations = [int(subitem) for subitem in location_str.split("..")]

                gene_start = locations[0]
                gene_end = locations[1]

                if feature=="CDS":
                    gene_sequence = reference_sequence[gene_start-1:gene_end]
                else:
                    gene_sequence = ""

                strand = 'forward'
                promoter_start = gene_start - 100 # by arbitrary definition, we treat the 100bp upstream as promoters
                promoter_end = gene_start - 1


                if gene_sequence!="" and (not len(gene_sequence)%3==0):
                    pass
                    print(gene_name, gene_start, "Not a multiple of 3")

                if feature_location.startswith('complement'):
                    strand='reverse'
                    promoter_start = gene_end+1
                    promoter_end = gene_end+100

                    if gene_sequence=="":
                        promoter_end = promoter_start


                # record information

                # first make sure gene name is unique
                i = 1
                old_gene_name = gene_name
                while gene_name in observed_gene_names:
                    i+=1
                    gene_name = "%s_%d" % (old_gene_name,i)

                start_positions.append(gene_start)
                end_positions.append(gene_end)
                promoter_start_positions.append(promoter_start)
                promoter_end_positions.append(promoter_end)
                gene_names.append(gene_name)
                gene_sequences.append(gene_sequence)
                strands.append(strand)
                observed_gene_names.add(gene_name)

    file.close()

    # sort genes based on start position

    gene_names, start_positions, end_positions, promoter_start_positions, promoter_end_positions, gene_sequences, strands = (list(x) for x in zip(*sorted(zip(gene_names, start_positions, end_positions, promoter_start_positions, promoter_end_positions, gene_sequences, strands), key=lambda pair: pair[1])))

    return gene_names, np.array(start_positions), np.array(end_positions), np.array(promoter_start_positions), np.array(promoter_end_positions), gene_sequences, strands


def parse_repeat_list(filename="LTEE-metagenomic-master/additional_data/REL606.6.gbk"):

    repeat_names = []
    start_positions = []
    end_positions = []
    complements = []

    file = open(filename,"r")
    line = file.readline()
    while line!="":
        items = line.split()
        feature = items[0]

        if feature == 'repeat_region':
            feature_location = items[1]

            # Get name of mobile element
            repeat_name = 'unknown'

            line = file.readline()
            while line.strip()[0]=='/':
                if line.strip().startswith('/mobile_element'):
                    repeat_name = line.split('=')[1].strip()[1:-1]
                line = file.readline()

            # Finished at next non '/' entry, presumably next feature

            if feature_location.startswith('complement'):
                complement = True
            else:
                complement = False

            location_str = feature_location.lstrip("complement(").lstrip("join(").rstrip(")")
            location_strs = location_str.split(",")
            for location_str in location_strs:

                locations = [int(subitem) for subitem in location_str.split("..")]
                start_positions.append(locations[0])
                end_positions.append(locations[1])
                repeat_names.append(repeat_name)
                complements.append(complement)

        else:

            line = file.readline()
    file.close()

    return repeat_names, np.array(start_positions), np.array(end_positions), complements


def parse_reference_genome(filename="LTEE-metagenomic-master/additional_data/REL606.6.gbk"):
    reference_sequences = []

    # GBK file
    if filename[-3:] == 'gbk':
        file = open(filename,"r")
        origin_reached = False
        for line in file:
            if line.startswith("ORIGIN"):
                origin_reached=True
            if origin_reached:
                items = line.split()
                if items[0].isdigit():
                    reference_sequences.extend(items[1:])
        file.close()

    # FASTA file
    else:
        file = open(filename,"r")
        file.readline() # header
        for line in file:
            reference_sequences.append(line.strip())
        file.close()

    reference_sequence = "".join(reference_sequences).upper()
    return reference_sequence


def parse_mask_list(filename="LTEE-metagenomic-master/additional_data/REL606.L20.G15.P0.M35.mask.gd"):

    # Add masks calculated in Tenaillon et al (Nature, 2016)
    # Downloaded from barricklab/LTEE-Ecoli/reference/ github repository

    mask_start_positions = []
    mask_end_positions = []

    file = open(filename,"r")
    file.readline() # header
    for line in file:
        items = line.split()
        start = int(items[4])
        length = int(items[5])
        mask_start_positions.append(start)
        mask_end_positions.append(start+length-1)

    # Add masking of prophage elements (Methods section of Tenaillon et al (Nature, 2016))
    mask_start_positions.append(880528)
    mask_end_positions.append(904682)

    return np.array(mask_start_positions), np.array(mask_end_positions)


def create_gene_size_map(effective_gene_lengths=None):

    if effective_gene_lengths==None:
        reference_sequence = parse_reference_genome()
        gene_data = parse_gene_list()
        repeat_data = parse_repeat_list()
        mask_data = parse_mask_list()
        gene_names, start_positions, end_positions, promoter_start_positions, promoter_end_positions, gene_sequences, strands = gene_data

        position_gene_map, effective_gene_lengths, substitution_specific_synonymous_fraction = create_annotation_map(gene_data, repeat_data, mask_data)


    excluded_genes=set(['synonymous','nonsynonymous','noncoding','masked'])

    gene_size_map = {}
    for gene_name in effective_gene_lengths.keys():

        #if gene_name.startswith('tRNA'):
        #    print gene_name

        if gene_name in excluded_genes:
            continue

        gene_size_map[gene_name] = effective_gene_lengths[gene_name]

    return gene_size_map


def get_pretty_name(line):
    if line[0]=='m':
        return 'Ara-%s' % line[1]
    else:
        return 'Ara+%s' % line[1]


def parse_well_mixed_state_timecourse(population):

    haplotype_filename = DATA_DIRECTORY+('%s_well_mixed_state_timecourse.txt' % population)

    file = open(haplotype_filename,"r")

    times = np.array([float(item) for item in file.readline().split(",")])
    num_unborn = np.array([float(item) for item in file.readline().split(",")])
    num_extinct = np.array([float(item) for item in file.readline().split(",")])
    num_fixed = np.array([float(item) for item in file.readline().split(",")])
    num_polymorphic = np.array([float(item) for item in file.readline().split(",")])

    states = []
    for line in file:
        Ls = np.array([float(item) for item in line.split(",")])
        states.append(Ls)
    file.close()
    return times, states


def calculate_appearance_time(ts, fs, state_Ls, Ls):
    # Three choices: either from freqs, simple HMM, or clade HMM
    #return calculate_appearance_time_from_freqs(ts,fs)
    #return calculate_appearance_time_from_hmm(ts,fs,state_Ls)
    return calculate_appearance_fixation_time_from_clade_hmm(ts,fs,Ls)[0]


def calculate_appearance_fixation_time_from_clade_hmm(times,fs,Ls):

    extinct_idxs = np.array([l in clade_extinct_states for l in Ls])
    polymorphic_idxs = np.array([l in clade_polymorphic_states for l in Ls])
    non_polymorphic_idxs = np.logical_not(polymorphic_idxs)
    num_polymorphic_idxs = (polymorphic_idxs).sum()

    # Calculate time tstar at which f(t) is largest
    if Ls[-1] in clade_fixed_states:

        # If fixed, this is final timepoint
        tstar = times[-1]

    else:
        # Otherwise, pick point where f(t) is largest
        # (restrict to polymorphic timepoints so that we
        #  don't focus on an error fluctuation)
        # print(polymorphic_idxs.astype(int).dtype, len(polymorphic_idxs))
        # print(type(polymorphic_idxs))
        # print(polymorphic_idxs.astype(int))
        # print([fs[i] for i in range(len(polymorphic_idxs)) if polymorphic_idxs[i] == True])
        # print(fs[polymorphic_idxs.astype(int)])
        # print(fs[polymorphic_idxs].argmax())
        # print(len(times[polymorphic_idxs]))
        test = np.array([fs[i] for i in range(len(polymorphic_idxs)) if polymorphic_idxs[i] == True])
        # print(test.argmax())
        tstar = (times[polymorphic_idxs])[test.argmax()]


    appearance_time = times[np.nonzero((times<=tstar)*extinct_idxs)[0][-1]]+250

    later_non_polymorphic_idxs = np.nonzero((times>=appearance_time)*non_polymorphic_idxs)[0]


    if len(later_non_polymorphic_idxs) == 0:
        # Stays polymorphic until the end
        fixation_time = 1000000 # never fixes
        transit_time = times[-1]+250 - appearance_time
    else:
        fixation_time = times[later_non_polymorphic_idxs[0]]-250
        transit_time = fixation_time - appearance_time

    return appearance_time, fixation_time, transit_time


def create_annotation_map(gene_data=None, repeat_data=None, mask_data=None):

    if gene_data==None:
        gene_data = parse_gene_list()
        repeat_data = parse_repeat_list()
        mask_data = parse_mask_list()


    gene_names, gene_start_positions, gene_end_positions, promoter_start_positions, promoter_end_positions, gene_sequences, strands = gene_data
    repeat_names, repeat_start_positions, repeat_end_positions, repeat_complements = repeat_data
    mask_start_positions, mask_end_positions = mask_data

    position_gene_map = {}
    gene_position_map = {}

    num_masked_sites = 0

    # first mark things that are repeats
    # this takes precedence over all other annotations
    for start,end in zip(repeat_start_positions,repeat_end_positions):
        for position in range(start,end+1):
            if position not in position_gene_map:
                position_gene_map[position]='repeat'
                num_masked_sites+=1

    # then mark masked things
    for start,end in zip(mask_start_positions, mask_end_positions):
        for position in range(start,end+1):
            if position not in position_gene_map:
                position_gene_map[position]='repeat'
                num_masked_sites+=1


    # then greedily annotate genes at remaining sites
    for gene_name,start,end in zip(gene_names,gene_start_positions,gene_end_positions):
        for position in range(start,end+1):
            if position not in position_gene_map:
                position_gene_map[position] = gene_name
                if gene_name not in gene_position_map:
                    gene_position_map[gene_name]=[]
                gene_position_map[gene_name].append(position)

    # remove 'partial' genes that have < 10bp unmasked sites
    for gene_name in list(sorted(gene_position_map.keys())):
        if len(gene_position_map[gene_name]) < 10:
            for position in gene_position_map[gene_name]:
                position_gene_map[position] = 'repeat'
            del gene_position_map[gene_name]

    # count up number of synonymous opportunities
    effective_gene_synonymous_sites = {}
    effective_gene_nonsynonymous_sites = {}

    substitution_specific_synonymous_sites = {substitution: 0 for substitution in substitutions}
    substitution_specific_nonsynonymous_sites = {substitution: 0 for substitution in substitutions}

    for gene_name,start,end,gene_sequence,strand in zip(gene_names, gene_start_positions, gene_end_positions, gene_sequences, strands):

        if gene_name not in gene_position_map:
            continue

        if strand=='forward':
            oriented_gene_sequence = gene_sequence
        else:
            oriented_gene_sequence = calculate_reverse_complement_sequence(gene_sequence)

        for position in gene_position_map[gene_name]:

            if gene_name not in effective_gene_synonymous_sites:
                effective_gene_synonymous_sites[gene_name]=0
                effective_gene_nonsynonymous_sites[gene_name]=0

            if gene_name.startswith('tRNA') or gene_name.startswith('rRNA'):
                pass

            else:

                # calculate position in gene
                if strand=='forward':
                    position_in_gene = position-start
                else:
                    position_in_gene = end-position

                # calculate codon start
                codon_start = int(position_in_gene/3)*3
                codon = gene_sequence[codon_start:codon_start+3]
                position_in_codon = position_in_gene%3

                #print gene_name, start, end, position, codon,position_in_codon

                effective_gene_synonymous_sites[gene_name] += codon_synonymous_opportunity_table[codon][position_in_codon]/3.0
                effective_gene_nonsynonymous_sites[gene_name] += 1-codon_synonymous_opportunity_table[codon][position_in_codon]/3.0

                for substitution in codon_synonymous_substitution_table[codon][position_in_codon]:
                    substitution_specific_synonymous_sites[substitution] += 1

                for substitution in codon_nonsynonymous_substitution_table[codon][position_in_codon]:
                    substitution_specific_nonsynonymous_sites[substitution] += 1

    substitution_specific_synonymous_fraction = {substitution: substitution_specific_synonymous_sites[substitution]*1.0/(substitution_specific_synonymous_sites[substitution]+substitution_specific_nonsynonymous_sites[substitution]) for substitution in substitution_specific_synonymous_sites.keys()}

    # then annotate promoter regions at remaining sites
    for gene_name,start,end in zip(gene_names,promoter_start_positions,promoter_end_positions):
        for position in range(start,end+1):
            if position not in position_gene_map:
                # position hasn't been annotated yet

                if gene_name not in gene_position_map:
                    # the gene itself has not been annotated
                    # so don't annotate the promoter
                    pass
                else:
                    position_gene_map[position] = gene_name
                    gene_position_map[gene_name].append(position)

    # calculate effective gene lengths
    effective_gene_lengths = {gene_name: len(gene_position_map[gene_name])-effective_gene_synonymous_sites[gene_name] for gene_name in gene_position_map.keys()}
    effective_gene_lengths['synonymous'] = sum([effective_gene_synonymous_sites[gene_name] for gene_name in gene_position_map.keys()])
    effective_gene_lengths['nonsynonymous'] = sum([effective_gene_nonsynonymous_sites[gene_name] for gene_name in gene_position_map.keys()])
    effective_gene_lengths['masked'] = num_masked_sites
    effective_gene_lengths['noncoding'] = calculate_genome_length()-effective_gene_lengths['synonymous']-effective_gene_lengths['nonsynonymous']-effective_gene_lengths['masked']

    return position_gene_map, effective_gene_lengths, substitution_specific_synonymous_fraction


def calculate_reverse_complement_sequence(dna_sequence):
    return "".join(base_table[base] for base in dna_sequence[::-1])


def calculate_genome_length(reference_sequence=None):
    if reference_sequence==None:
        reference_sequence=parse_reference_genome()
    return len(reference_sequence)


def parse_reference_genome(filename="./LTEE-metagenomic-master/additional_data/REL606.6.gbk"):
    reference_sequences = []

    # GBK file
    if filename[-3:] == 'gbk':
        file = open(filename,"r")
        origin_reached = False
        for line in file:
            if line.startswith("ORIGIN"):
                origin_reached=True
            if origin_reached:
                items = line.split()
                if items[0].isdigit():
                    reference_sequences.extend(items[1:])
        file.close()

    # FASTA file
    else:
        file = open(filename,"r")
        file.readline() # header
        for line in file:
            reference_sequences.append(line.strip())
        file.close()

    reference_sequence = "".join(reference_sequences).upper()
    return reference_sequence


def parse_gene_list(reference_sequence=None, filename="./LTEE-metagenomic-master/additional_data/REL606.6.gbk"):

    features = set(['CDS','gene','tRNA','rRNA','repeat_region'])

    if reference_sequence==None:
        reference_sequence=parse_reference_genome()

    observed_gene_names = set()

    gene_names = []
    start_positions = []
    end_positions = []
    promoter_start_positions = []
    promoter_end_positions = []
    gene_sequences = []
    strands = []

    file = open(filename,"r")
    line = file.readline()
    while line!="":

        items = line.split()
        feature = items[0]

        gene_name = ""
        feature_location = ""

        if feature=='CDS':
            feature_location = items[1]

            line = file.readline().strip()

            gene_name=""
            locus_name=""
            is_pseudo=False

            while line.split()[0] not in features:

                if line.startswith('/gene'):
                    gene_name = line.split('=')[1].strip()[1:-1]
                if line.startswith('/locus_tag'):
                    locus_name = line.split('=')[1].strip()[1:-1]
                if line.startswith('/pseudo'):
                    is_pseudo=True

                line = file.readline().strip()

            if gene_name=="":
                gene_name = locus_name

            if is_pseudo:
                gene_name = ""

            # done here

        elif feature=='tRNA' or feature=='rRNA':

            feature_location = items[1]
            #print feature_location

            while not line.strip().startswith('/gene'):
                line = file.readline().strip()
            gene_name = line.split('=')[1].strip()[1:-1]
            gene_name = '%s:%s' % (feature, gene_name)


        else:
            # nothing to see here
            line = file.readline().strip()

        # If the element has a feature location string and a name
        # it should either be a gene, tRNA, or rRNA, so let's get details
        if feature_location!="" and gene_name!="":

            location_str = feature_location.lstrip("complement(").lstrip("join(").rstrip(")")
            location_strs = location_str.split(",")

            for location_str in location_strs:

                locations = [int(subitem) for subitem in location_str.split("..")]

                gene_start = locations[0]
                gene_end = locations[1]

                if feature=="CDS":
                    gene_sequence = reference_sequence[gene_start-1:gene_end]
                else:
                    gene_sequence = ""

                strand = 'forward'
                promoter_start = gene_start - 100 # by arbitrary definition, we treat the 100bp upstream as promoters
                promoter_end = gene_start - 1


                if gene_sequence!="" and (not len(gene_sequence)%3==0):
                    pass
                    print(gene_name, gene_start, "Not a multiple of 3")

                if feature_location.startswith('complement'):
                    strand='reverse'
                    promoter_start = gene_end+1
                    promoter_end = gene_end+100

                    if gene_sequence=="":
                        promoter_end = promoter_start


                # record information

                # first make sure gene name is unique
                i = 1
                old_gene_name = gene_name
                while gene_name in observed_gene_names:
                    i+=1
                    gene_name = "%s_%d" % (old_gene_name,i)

                start_positions.append(gene_start)
                end_positions.append(gene_end)
                promoter_start_positions.append(promoter_start)
                promoter_end_positions.append(promoter_end)
                gene_names.append(gene_name)
                gene_sequences.append(gene_sequence)
                strands.append(strand)
                observed_gene_names.add(gene_name)

    file.close()

    # sort genes based on start position

    gene_names, start_positions, end_positions, promoter_start_positions, promoter_end_positions, gene_sequences, strands = (list(x) for x in zip(*sorted(zip(gene_names, start_positions, end_positions, promoter_start_positions, promoter_end_positions, gene_sequences, strands), key=lambda pair: pair[1])))

    return gene_names, np.array(start_positions), np.array(end_positions), np.array(promoter_start_positions), np.array(promoter_end_positions), gene_sequences, strands


def calculate_parallelism_statistics(convergence_matrix, allowed_populations=complete_nonmutator_lines,Lmin=0):

    allowed_populations = set(allowed_populations)

    gene_statistics = {}

    # Now calculate gene counts
    Ltot = 0
    Ngenes = 0
    ntot = 0
    for gene_name in sorted(convergence_matrix.keys()):

        times = []
        selections_muts = []

        L = max([convergence_matrix[gene_name]['length'],Lmin])
        n = 0

        num_pops = 0

        for population in allowed_populations:

            new_muts = len(convergence_matrix[gene_name]['mutations'][population])

            if new_muts > 0.5:

                num_pops += 1

                n += new_muts

                for t,l,lclade,f,s in convergence_matrix[gene_name]['mutations'][population]:
                    times.append(t)
                    selections_muts.append(s)


        Ltot += L
        ntot += n

        gene_statistics[gene_name] = {}
        gene_statistics[gene_name]['length'] = L
        gene_statistics[gene_name]['observed'] = n
        gene_statistics[gene_name]['selections'] = selections_muts
        gene_statistics[gene_name]['mean_selection'] = np.mean(selections_muts)

        if len(times) > 0:
            gene_statistics[gene_name]['median_time'] = np.median(times)
        else:
            gene_statistics[gene_name]['median_time'] = 0

        gene_statistics[gene_name]['nonzero_populations'] = num_pops

    Lavg = Ltot*1.0/len(convergence_matrix.keys())

    for gene_name in gene_statistics.keys():

        gene_statistics[gene_name]['expected'] = ntot*gene_statistics[gene_name]['length']/Ltot

        gene_statistics[gene_name]['multiplicity'] = gene_statistics[gene_name]['observed']*1.0/gene_statistics[gene_name]['length']*Lavg

        gene_statistics[gene_name]['g'] = np.log(gene_statistics[gene_name]['observed']*1.0/gene_statistics[gene_name]['expected']+(gene_statistics[gene_name]['observed']==0))

    return gene_statistics
