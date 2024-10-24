import cgeneo

# Extract a subpedigree from a given pedigree

def branching(gen, **kwargs):
    pro = kwargs.get('pro', None)
    ancestors = kwargs.get('ancestors', None)
    if pro is None:
        pro = cgeneo.get_proband_ids(gen)
    if ancestors is None:
        ancestors = cgeneo.get_founder_ids(gen)
    extracted_pedigree = cgeneo.extract_pedigree(gen, pro, ancestors)
    return extracted_pedigree

def lineages(gen, **kwargs):
    pro = kwargs.get('pro', None)
    if pro is None:
        pro = cgeneo.get_proband_ids(gen)
    maternal = kwargs.get('maternal', True)
    extracted_pedigree = cgeneo.extract_lineages(gen, pro, maternal)
    return extracted_pedigree