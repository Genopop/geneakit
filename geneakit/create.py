import os
import pandas as pd
import cgeneakit

def genealogy(input, **kwargs):
    """Create a Genealogy object from pedigree data

    Constructs a genealogical structure from either a pandas DataFrame
    or a pedigree text file. Handles parent-child relationships and
    foundational genealogy metrics.

    Args:
        input (pd.DataFrame | str): Input data source, either:
            - DataFrame with columns: ['ind', 'father', 'mother', 'sex']
            - Path to text file with same columns (tab or space separated)
        sorted (bool, optional): Sort individuals chronologically with 
            parents before children. Defaults to False.
            
    Returns:
        cgeneakit.Pedigree: Compiled genealogy object with familial links
        and metadata
        
    Raises:
        FileNotFoundError: If provided file path doesn't exist
        ValueError: If input DataFrame has incorrect columns
        
    Examples:
        >>> import geneakit as gen
        >>> import pandas as pd
        
        ### From DataFrame
        >>> df = pd.DataFrame({
        ...     'ind': [1,2,3],
        ...     'father': [0,0,1],
        ...     'mother': [0,0,2],
        ...     'sex': [1,2,1]
        ... })
        >>> ped = gen.genealogy(df)
        
        ### From file
        >>> ped = gen.genealogy("family_data.csv", sorted=True)

    Notes:
        - Required column order: ID, Father ID, Mother ID, Sex
        - Unknown parents should be marked with 0
        - Sex encoding: 1/M = Male, 2/F = Female, 0/U = Unknown
        - File format should be UTF-8 encoded with header row
    """
    sorted = kwargs.get('sorted', False)
    if type(input) == pd.DataFrame:
        dataframe = input
        ids = dataframe.iloc[:, 0].values
        father_ids = dataframe.iloc[:, 1].values
        mother_ids = dataframe.iloc[:, 2].values
        sexes = dataframe.iloc[:, 3].values
        sexes = [1 if sex == 'M' else 2 if sex == 'F' else 0 if sex == 'U'
                 else sex for sex in sexes]
        pedigree = cgeneakit.load_pedigree_from_vectors(
            ids, father_ids, mother_ids, sexes, sorted)
        return pedigree
    elif type(input) == str:
        file_path = input
        if not os.path.exists(input):
            raise FileNotFoundError('File not found: ' + file_path)
        pedigree = cgeneakit.load_pedigree_from_file(file_path, sorted)
        return pedigree   