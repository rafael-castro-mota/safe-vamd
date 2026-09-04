import numpy as np
from numpy.typing import ArrayLike
from typing import Sequence

from safe_vamd.mathematical_functions.mathematical_func import MathFunc
from safe_vamd.mathematical_functions.polynomial_func import PolyFunc


class QuadBasisFunc(MathFunc):
    """
    Function for the quadratic basis function for a selected node in a finite-element,
    implemented as a subclass of MathFunc.

    Attributes
    ----------
    bfid : int
        Selector for the basis functions (0 for the basis function associated with the first node,
        2 for the one associated with the second node, etc.)
    node_coords : Sequence[float]
        List of z-coordinates (height, in meters) for the element's nodes.

    Notes
    ----------
    The piece-wise quadratic basis functions have value of 1 on the associated node and 0 on the other two.
    """

    def __init__(self, bfid: int, node_coords: Sequence[float]) -> None:
        """
        initializes an instance of QuadBasisFunc.
        """

        self.bfid = bfid
        self.node_coords = node_coords

    def value(self, z: ArrayLike) -> np.ndarray:
        """
        Evaluates the basis function at z.
        Parameters
        ----------
        z:float
            Height(s) (in meters).
        Returns
        -------
        float
            Value(s) of the basis function at z(s).
        """

        z = np.array(z)

        # Converting global coordinates to local ones (standard element spans from -1 to 1)
        z1 = self.node_coords[0]
        zc = self.node_coords[1]
        z3 = self.node_coords[2]
        length = abs(z3 - z1)
        z = 2*(z - zc)/length

        val = 0
        if self.bfid == 0:
            poly1 = PolyFunc(np.array([0, -0.5, 0.5]))
            val = poly1.value(z)
        elif self.bfid == 1:
            poly2 = PolyFunc(np.array([1, 0, -1]))
            val = poly2.value(z)
        elif self.bfid == 2:
            poly3 = PolyFunc(np.array([0, 0.5, 0.5]))
            val = poly3.value(z)
        return val

    def dvalue(self, z: ArrayLike) -> np.ndarray:
        """
        Evaluates the derivative of the basis function at z.
        Parameters
        ----------
        z:float
          Height(s) (in meters).
        Returns
        -------
        float
            Value(s) of the derivative of the basis function at z(s).
        """

        # Converting global coordinates to local ones
        z1 = self.node_coords[0]
        zc = self.node_coords[1]
        z3 = self.node_coords[2]
        lenght = abs(z3 - z1)
        z = 2 * (z - zc) / lenght

        val = 0
        if self.bfid == 0:
            dpoly1 = PolyFunc((2/lenght)*np.array([-0.5, 1]))
            val = dpoly1.value(z)
        elif self.bfid == 1:
            dpoly2 = PolyFunc((2/lenght)*np.array([0, -2]))
            val = dpoly2.value(z)
        elif self.bfid == 2:
            dpoly3 = PolyFunc((2/lenght)*np.array([0.5, 1]))
            val = dpoly3.value(z)
        return val

