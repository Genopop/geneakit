import pandas as pd
import cgeneo

# Output a pedigree to a dataframe 

def genout(gen, **kwargs):
    sorted = kwargs.get('sorted', False)
    output = cgeneo.output_pedigree(gen)
    dataframe = pd.DataFrame(
        output, columns=['ind', 'father', 'mother', 'sex'], copy=False
    )
    if sorted:
        dataframe = dataframe.sort_values(by='ind').reset_index(drop=True)
    return dataframe