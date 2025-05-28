import pandas as pd
import cgeneakit

def genout(gen, **kwargs):
    """Convert genealogy object to structured DataFrame
    
    Exports pedigree data in a tabular format suitable for analysis
    or file export. Maintains original genealogy structure unless
    sorted.

    Args:
        gen (cgeneakit.Pedigree): Initialized genealogy object
        sorted (bool, optional): Sort individuals numerically by ID.
            Defaults to False (maintain original load order).
            
    Returns:
        pd.DataFrame: Pedigree table with columns:
            - ind: Individual ID (integer)
            - father: Paternal ID (0 = unknown)
            - mother: Maternal ID (0 = unknown)
            - sex: Biological sex (1=male, 2=female, 0=unknown)
            
    Examples:
        >>> import geneakit as gen
        >>> from geneakit import geneaJi
        >>> ped = gen.genealogy(geneaJi)
        >>> df_raw = gen.genout(ped)
        >>> print(df_raw.head(3))
           ind  father  mother  sex
        0   17       0       0    1
        1   19       0       0    1
        2   25       0       0    1
        
        >>> df_sorted = gen.genout(ped, sorted=True)
        >>> print(df_sorted.head(3))
           ind  father  mother  sex
        0    1       4       3    1
        1    2       4       3    1
        2    3       7       5    2

    Notes:
        - Maintains original data types from genealogy object
        - Sorting preserves family relationships (no chronological sorting)
        - Reset index after sorting for clean row numbering
        
    See Also: 
        gen.genealogy: For creating pedigree objects from raw data
    """
    sorted = kwargs.get('sorted', False)
    output = cgeneakit.output_pedigree(gen)
    dataframe = pd.DataFrame(
        output, columns=['ind', 'father', 'mother', 'sex'], copy=False)
    if sorted:
        dataframe = dataframe.sort_values(by='ind').reset_index(drop=True)
    return dataframe