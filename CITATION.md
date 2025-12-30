# Citing ts2net

## Python ts2net

If you use this Python implementation in your research, please cite:

```bibtex
@software{ts2net_python,
  title = {ts2net: Time Series to Networks in Python},
  author = {Jones, K.},
  year = {2025},
  url = {https://github.com/yourusername/ts2net},
  note = {Python implementation with univariate and multivariate analysis}
}
```

## Original R ts2net Package

The **multivariate functionality** in this Python package is directly inspired by and implements the API from Leonardo N. Ferreira's original R package. Please also cite:

```bibtex
@article{ferreira2024,
  author = {Ferreira, Leonardo N.},
  title = {From time series to networks in R with the ts2net package},
  journal = {Applied Network Science},
  year = {2024},
  volume = {9},
  number = {1},
  pages = {32},
  doi = {10.1007/s41109-024-00642-2},
  url = {https://doi.org/10.1007/s41109-024-00642-2}
}
```

**R Package:**
- GitHub: https://github.com/lnferreira/ts2net
- CRAN: https://cran.r-project.org/package=ts2net

## Acknowledgments

### Multivariate Analysis (Multiple Time Series → Graph)
The distance functions (`tsdist_*`) and network construction methods (`net_knn`, `net_enn`, `net_weighted`) implement the API designed by Leonardo N. Ferreira in the R ts2net package. 


This Python implementation:

- Matches the R API for distance functions and network builders
- Adds Numba acceleration for performance (10-200x speedups)
- Extends with parallel processing using joblib
- Provides HPC batch processing support

### Univariate Analysis (Single Time Series → Graph)
The univariate methods (HVG, NVG, Recurrence Networks, Transition Networks) are based on established algorithms from the complex networks literature:

- **Visibility Graphs**: Lacasa et al. (2008)
- **Recurrence Networks**: Marwan et al. (2009)
- **Transition Networks**: Zhang & Small (2006)
- **False Nearest Neighbors**: Kennel et al. (1992)

## Key References

### Visibility Graphs
```bibtex
@article{lacasa2008,
  title={From time series to complex networks: The visibility graph},
  author={Lacasa, Lucas and Luque, Bartolo and Ballesteros, Fernando and Luque, Jordi and Nuno, Juan Carlos},
  journal={Proceedings of the National Academy of Sciences},
  volume={105},
  number={13},
  pages={4972--4975},
  year={2008}
}
```

### Recurrence Networks
```bibtex
@article{marwan2009,
  title={Complex network approach for recurrence analysis of time series},
  author={Marwan, Norbert and Donges, Jonathan F and Zou, Yong and Donner, Reik V and Kurths, J{\"u}rgen},
  journal={Physics Letters A},
  volume={373},
  number={46},
  pages={4246--4254},
  year={2009}
}
```

### Ordinal Patterns (Transition Networks)
```bibtex
@article{bandt2002,
  title={Permutation entropy: a natural complexity measure for time series},
  author={Bandt, Christoph and Pompe, Bernd},
  journal={Physical Review Letters},
  volume={88},
  number={17},
  pages={174102},
  year={2002}
}
```

### False Nearest Neighbors
```bibtex
@article{kennel1992,
  title={Determining embedding dimension for phase-space reconstruction using a geometrical construction},
  author={Kennel, Matthew B and Brown, Reggie and Abarbanel, Henry DI},
  journal={Physical Review A},
  volume={45},
  number={6},
  pages={3403},
  year={1992}
}
```

## License

This Python package is released under the MIT License, maintaining compatibility with the original R package (also MIT License).

## Contact

**Python Implementation:**
- GitHub Issues: [Report bugs or request features]

**Original R Package:**
- Author: Leonardo N. Ferreira
- Email: ferreira@leonardonascimento.com
- Website: https://leonardoferreira.com
- GitHub: https://github.com/lnferreira/ts2net

