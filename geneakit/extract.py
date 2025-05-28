import cgeneakit

def branching(gen, **kwargs):
    """Extract a subpedigree containing specified individuals and ancestors
    
    Creates a focused genealogy containing only the specified probands
    and their ancestral paths to the designated founders.

    Args:
        gen (cgeneakit.Pedigree): Source genealogy object
        pro (list, optional): Target proband IDs to include. 
            Defaults to all probands in the genealogy.
        ancestors (list, optional): Ancestor IDs to retain.
            Defaults to all founders in the original pedigree.
            
    Returns:
        cgeneakit.Pedigree: New pedigree object containing only:
            - Specified probands
            - Specified ancestors
            - All connecting individuals
            
    Examples:
        >>> import geneakit as gen
        >>> from geneakit import genea140
        >>> ped = gen.genealogy(genea140)
        >>> sub_ped = gen.branching(ped, pro=[409033, 408728])
        >>> print(len(sub_ped))
        1543
        
    Notes:
        - Maintains original sex/relationship data
        - If no probands specified, includes all childless individuals
        - Empty ancestor list will result in minimal pedigree
        - Runtime scales with ancestor/proband count
    """
    pro = kwargs.get('pro', None)
    ancestors = kwargs.get('ancestors', None)
    if pro is None:
        pro = cgeneakit.get_proband_ids(gen)
    if ancestors is None:
        ancestors = cgeneakit.get_founder_ids(gen)
    extracted_pedigree = cgeneakit.extract_pedigree(gen, pro, ancestors)
    return extracted_pedigree

def lineages(gen, **kwargs):
    """Extract maternal or paternal lineages from pedigree
    
    Creates a lineage pedigree containing only the specified probands'
    direct maternal or paternal ancestry.

    Args:
        gen (cgeneakit.Pedigree): Source genealogy object
        pro (list, optional): Target proband IDs to trace.
            Defaults to all probands in the genealogy.
        maternal (bool): Trace maternal lineage if True (default),
                         paternal lineage if False.
            
    Returns:
        cgeneakit.Pedigree: Lineage pedigree containing:
            - Specified probands
            - Direct maternal/paternal ancestors
            - No collateral relatives
            
    Examples:
        >>> import geneakit as gen
        >>> from geneakit import genea140
        >>> ped = gen.genealogy(genea140)
        >>> mat_lineage = gen.lineages(ped, maternal=True)
        >>> pat_lineage = gen.lineages(ped, pro=[717634], maternal=False)
        
    Notes:
        - Only follows one parental line (mothers for maternal,
          fathers for paternal)
        - Useful for uniparental inheritance analysis
        - Probands without specified parent lineage are excluded
    """
    pro = kwargs.get('pro', None)
    if pro is None:
        pro = cgeneakit.get_proband_ids(gen)
    maternal = kwargs.get('maternal', True)
    extracted_pedigree = cgeneakit.extract_lineages(gen, pro, maternal)
    return extracted_pedigree