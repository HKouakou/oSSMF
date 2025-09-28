## Overview

This repository contains the code associated with the paper **"Online simplex-structured matrix factorization"**.

## Contents

### Data

- **Basis vectors:**

   - `basisVectors.mat`: Basis vectors used in the paper.

- **Coefficient matrices:**

  - `coeff_3.csv`: A coefficient matrix, each row was sampled from a Dirichlet distribution with parameters 1,1,1;  maximum purity level = 0.7.
  - `coeff_7.csv`: A coefficient matrix, each row was sampled from a Dirichlet distribution with parameters 1,1,1,1,1,1,1;  maximum purity level = 0.7.

### CodeOssmf
   - Contains the code for the OSSMF algorithm, using SISAL as a baseline.



## Usage

To run the code, execute `Test.m`. This script illustrates the use of oSSMF using SISAL as a baseline, with an example that uses one of the coefficient matrices listed above. You can adjust the signal-to-noise ratio as well as other parameters, as explained in `Test.m` (see comments for details).

## Citation

If you use this code, please cite the paper as follows:

```bibtex
@article{kouakou2025online,
  title={Online simplex-structured matrix factorization},
  author={Kouakou, Hugues and de Morais Goulart, Jos{\'e} Henrique and Vitale, Raffaele and Oberlin, Thomas and Rousseau, David and Ruckebusch, Cyril and Dobigeon, Nicolas},
  journal={IEEE Signal Processing Letters},
  year={2025},
  publisher={IEEE}
}
