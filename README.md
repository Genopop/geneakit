# GENEO Toolkit

A set of functions for pedigree analysis, designed for use with data from the [GÉNÉO portal](https://portailgeneo.ca). Based on the functionality of [GENLIB](https://cran.r-project.org/web/packages/GENLIB/index.html): see the article by Gauvin et al. (2015) \<[doi:10.1186/s12859-015-0581-5](https://doi.org/10.1186/s12859-015-0581-5)>.

## Documentation

The [GENLIB reference manual](https://cran.r-project.org/web/packages/GENLIB/GENLIB.pdf) and this README file are sufficient to learn how to use the GENEO toolkit.

## Aims

* Easily port R code using GENLIB into Python code using the GENEO toolkit;
* Integrate with Python libraries such as Pandas and NumPy;
* Provide speed and convenience;
* Present a modular structure for further development.

## Functions

* Create a pedigree structure from a file or `DataFrame`;
* Output a pedigree as a `DataFrame`;
* Identify individuals in a pedigree, such as probands and founders;
* Extract a subpedigree from a pedigree;
* Describe a pedigree, such as the number of individuals and its completeness;
* Compute information about a pedigree, such as the pairwise kinship coefficients of probands and the genetic contributions of ancestors;
* (Eventually) Simulate information about pedigrees and individuals.

## Installation

* Clone this repository, `cd` into it, then run `pip install .`. Alternatively, without cloning, run `pip install https://github.com/GPhMorin/geneo/archive/main.zip`. Both options install two packages, `geneo` and `cgeneo` (used by the former internally), and their dependencies. You will need a compiler that supports C++17.

* If OpenMP is found during installation, the `geneo.phi()` function will run in parallel, making it the fastest implementation of kinship computation that we know of. If you use macOS, you may need to follow [these instructions](https://www.scivision.dev/cmake-openmp/) to enable OpenMP.

* On Windows 11, the toolkit was tested using Microsoft Visual C++ 2022.

## Data

* If the pedigree is loaded from a file, the file *must* start with an irrelevant line (such as `ind father mother sex`) and the following lines must contain, as digits, each individual's ID, their father's ID (`0` if unknown), their mother's ID (`0` if unknown), and their sex (`0` if unknown, `1` if male, `2` if female), in that order. Each information must be separated by anything but digits (tabs, spaces, commas, etc.), with one line per individual.

* Three datasets come from the GENLIB source code: `geneo.geneaJi`, `geneo.genea140`  and `geneo.pop140`. They are part of the project for testing and practice. More information on these datasets is available in the [GENLIB reference manual](https://cran.r-project.org/web/packages/GENLIB/GENLIB.pdf).

## Comparison with GENLIB

The function calls are almost verbatim copies of GENLIB's. For instance:

```r
# With GENLIB
library(GENLIB)
data(genea140)
ped <- gen.genealogy(genea140)
pro <- gen.pro(ped)
phi <- gen.phi(ped, pro=pro)
mean <- gen.phiMean(phi)
mrca <- gen.findMRCA(ped, c(802424, 868572))
dist <- gen.find.Min.Distance.MRCA(mrca)
out <- gen.genout(ped, sorted=TRUE)
```


```python
# With the GENEO toolkit
import geneo as gen
genea140 = gen.genea140
ped = gen.genealogy(genea140)
pro = gen.pro(ped)
phi = gen.phi(ped, pro=pro)
mean = gen.phiMean(phi)
mrca = gen.findMRCA(ped, [802424, 868572])
dist = gen.find_Min_Distance_MRCA(mrca)
out = gen.genout(ped, sorted=True)
```

## GENLIB Functions Not Included

| Function                    | Description                                                                    |
| --------------------------- | ------------------------------------------------------------------------------ |
| `gen.graph`                 | Pedigree graphical tool                                                        |
|                             |                                                                                |
| `gen.simuHaplo`             | Gene dropping simulations - haplotypes                                         |
| `gen.simuHaplo_convert`     | Convert proband simulation results into sequence data given founder haplotypes |
| `gen.simuHaplo_IBD_compare` | Compare proband haplotypes for IBD sharing                                     |
| `gen.simuHaplo_traceback`   | Trace inheritance path for results from gene dropping simulation               |
| `gen.simuProb`              | Gene dropping simulations - Probabilities                                      |
| `gen.simuSample`            | Gene dropping simulations - Sample                                             |
| `gen.simuSampleFreq`        | Gene dropping simulations - Frequencies                                        |
| `gen.simuSet`               | Gene dropping simulations with specified transmission probabilities            |
|                             |                                                                                |
| `gen.fCI`                   | Average inbreeding coefficient confidence interval                             |
| `gen.phiCI`                 | Average kinship confidence interval                                            |
| `gen.completenessVar`       | Variance of completeness index                                                 |
| `gen.implexVar`             | Variance of genealogical implex                                                |
| `gen.meangendepthVar`       | Variance of genealogical depth                                                 |
