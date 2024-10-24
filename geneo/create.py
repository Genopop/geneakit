import os
import pandas as pd
import cgeneo

# Create a pedigree from a dataframe or a file

def genealogy(input, **kwargs):
    sorted = kwargs.get('sorted', False)
    if type(input) == pd.DataFrame:
        dataframe = input
        ids = dataframe.iloc[:, 0].values
        father_ids = dataframe.iloc[:, 1].values
        mother_ids = dataframe.iloc[:, 2].values
        sexes = dataframe.iloc[:, 3].values
        pedigree = cgeneo.load_pedigree_from_vectors(
            ids, father_ids, mother_ids, sexes, sorted)
        return pedigree
    elif type(input) == str:
        file_path = input
        if not os.path.exists(input):
            raise FileNotFoundError('File not found: ' + file_path)
        pedigree = cgeneo.load_pedigree_from_file(file_path, sorted)
        return pedigree   