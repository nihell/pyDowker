import numpy as np
from pyDowker import DowkerComplex

def test_total_weight_filtration() -> None:
    relation = np.array([[1,1,1,0],
                         [1,0,0,1],
                         [0,0,1,1],
                         [0,1,1,0],
                         [1,1,0,0]],dtype=bool)
    complex = DowkerComplex.DowkerComplex(relation).create_simplex_tree(filtration='TotalWeight', max_dimension=3)
    assert complex.filtration([0]) == -3
    assert complex.filtration([0,4]) == -2
    assert complex.filtration([0,1,4]) == -1