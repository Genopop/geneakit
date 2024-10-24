import pandas as pd
import cgeneo
import time

# Compute

def phi(gen, **kwargs):
    pro = kwargs.get('pro', None)
    if pro is None:
        pro = cgeneo.get_proband_ids(gen)
    verbose = kwargs.get('verbose', False)
    compute = kwargs.get('compute', True)
    if not compute:
        required_memory = cgeneo.get_required_memory_for_kinships(gen, pro)
        required_memory = round(required_memory, 2)
        print(f'You will require at least {required_memory} GB of RAM.')
        return
    if verbose:
        begin = time.time()
    cmatrix = cgeneo.compute_kinships(gen, pro, verbose)
    kinship_matrix = pd.DataFrame(
        cmatrix, index=pro, columns=pro, copy=False
    )
    if verbose:
        end = time.time()
        elapsed_time = round(end - begin, 2)
        print(f'Elapsed time: {elapsed_time} seconds')
    return kinship_matrix

def phiMean(kinship_matrix):
    mean = 0
    for i in range(kinship_matrix.shape[0]):
        for j in range(i):
            mean += kinship_matrix.iloc[i, j]
    mean /= (kinship_matrix.shape[0] * (kinship_matrix.shape[0] - 1) / 2)
    return mean

def phiOver(phiMatrix, threshold):
    pairs = []
    for i in range(phiMatrix.shape[0]):
        for j in range(i):
            if phiMatrix.iloc[i, j] > threshold:
                pairs.append([i, j, phiMatrix.index[i], phiMatrix.columns[j],
                              phiMatrix.iloc[i, j]])
    return pd.DataFrame(pairs, columns=['line', 'column', 'pro1',
                                        'pro2', 'kinship'])

def f(gen, **kwargs):
    pro = kwargs.get('pro', None)
    if pro is None:
        pro = cgeneo.get_proband_ids(gen)
    cmatrix = cgeneo.compute_inbreedings(gen, pro)
    inbreeding_matrix = pd.DataFrame(
        cmatrix, index=pro, copy=False
    )
    return inbreeding_matrix

def gc(pedigree, **kwargs):
    pro = kwargs.get('pro', None)
    ancestors = kwargs.get('ancestors', None)
    if pro is None:
        pro = cgeneo.get_proband_ids(pedigree)
    if ancestors is None:
        ancestors = cgeneo.get_founder_ids(pedigree)
    cmatrix = cgeneo.compute_genetic_contributions(pedigree, pro, ancestors)
    kinship_matrix = pd.DataFrame(
        cmatrix, index=pro, columns=ancestors, copy=False
    )
    return kinship_matrix