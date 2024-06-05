# pyDowker: Dowker complexes with Python

Under development!

## Installation

Clone the repository and install via:

``` 
git clone https://github.com/nihell/pyDowker.git
cd pyDowker
pip install -e . 
```

## Usage
The main feature that sets this implementation apart from the ones that were previously available is the ability to handle arbitrary, user-specified, relations.
Encoding them as numpy-arrays, this package allows to compute persistence via integration with gudhi and bipersistence via rivet.

```
import gudhi as gd
from pyDowker.DowkerComplex import DowkerComplex
dow = DowkerComplex(rel)  #assuming rel is the numpy array storing the function values which induces a filtration of relations via sublevels
st = dow.create_simplex_tree(max_dimension = 4, filtration = "Sublevel") #constructs the 4-skeleton filtered by sublevels of function values.
st.compute_persistence() # from here on, the usual gudhi commands apply
dgm = gd.persistence_intervals_in_dimension(3)
gd.plot_persistence_diagram(dgm) #plot the persistence diagram
```

## References

Niklas Hellmer and Jan Spaliński. "Density Sensitive Bifiltered Dowker Complexes via Total Weight". arXiv, 24 May 2024. http://arxiv.org/abs/2405.15592.
