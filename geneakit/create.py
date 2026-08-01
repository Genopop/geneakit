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
        sorted (bool, optional): Declare that the input is already in
            topological order, every parent listed before their children.
            The sorting pass is then skipped and the given order is kept.
            Defaults to False, which sorts the pedigree by depth-first
            search. This flag asserts a property of the input rather than
            asking for one: passing True for a pedigree that is not already
            sorted raises IndexError.

    Returns:
        cgeneakit.Pedigree: Compiled genealogy object with familial links
            and metadata

    Raises:
        FileNotFoundError: If provided file path doesn't exist
        ValueError: If input DataFrame has incorrect columns
        IndexError: If sorted=True but a child is listed before one of its
            parents

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
        >>> ped = gen.genealogy("family_data.csv")

        ### From a file already ordered with parents before children
        >>> ped = gen.genealogy("sorted_data.csv", sorted=True)

    Notes:
        - Required column order: ID, Father ID, Mother ID, Sex
        - The `sorted` flag of gen.genout() is unrelated: it asks for the
          output to be sorted by ID, whereas this one declares that the
          input is already in topological order
        - Unknown parents should be marked with 0
        - Sex encoding: 1/M = Male, 2/F = Female, 0/U = Unknown
        - File format should be UTF-8 encoded with header row
    """
    sorted = kwargs.get('sorted', False)
    if isinstance(input, pd.DataFrame):
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
    elif isinstance(input, str):
        file_path = input
        if not os.path.exists(input):
            raise FileNotFoundError('File not found: ' + file_path)
        pedigree = cgeneakit.load_pedigree_from_file(file_path, sorted)
        return pedigree
