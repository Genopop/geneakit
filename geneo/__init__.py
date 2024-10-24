import os
path = os.path.dirname(os.path.realpath(__file__)) + "/datasets/"
genea140 = path + "genea140.csv"
geneaJi = path + "geneaJi.csv"
pop140 = path + "pop140.csv"

from .create import *
from .output import *
from .identify import *
from .extract import *
from .describe import *
from .compute import *