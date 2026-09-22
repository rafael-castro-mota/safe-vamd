import numpy as np
from typing import Sequence

from safe_vamd.mathematical_functions.mathematical_func import MathFunc
from safe_vamd.mathematical_functions.polynomial_func import PolyFunc
from safe_vamd.mathematical_functions.derivative_func import DerivFunc

from safe_vamd.mesh.node import Node
from safe_vamd.mesh.quadratic_basis_function import QuadBasisFunc


class Element1DArbitraryOrderLagrange:
    """
    A single one-dimensional element (part of a finite-element mesh)
    with piece-wise Lagrange polynomials of arbitrary order.

    **Class attributes:**

    Attributes
    ----------
    eid : int
        The Element Identification Number.
    nodes : list[Node]
        A list of *Node* objects associated with the element.
    degree : int
        The degree of the piece-wise Lagrange polynomials.
    bf : Sequence[MathFunc]
        List of the Lagrange polynomials (each implemented as a *MathFunc*).
    dbf  : Sequence[MathFunc]
        List of derivatives for the Lagrange polynomials (each implemented as a *MathFunc*).
    length : float
        The length of the element (in :math:`m`).
    """
    def __init__(self, eid: int, nodes: Sequence[Node], degree: int) -> None:
        """

        Parameters
        ----------
        eid : int
            The Element Identification Number.
        nodes : Sequence[Node]
            A list of *Node* objects associated with the element.
        degree : int
            The degree of the piece-wise Lagrange polynomials.
        bf : Sequence[MathFunc]
            A list of the Lagrange polynomials (each implemented as a *MathFunc*).
        dbf  : Sequence[MathFunc]
            A list of derivatives for the Lagrange polynomials (each implemented as a *MathFunc*).
        length : float
            The length of the element (in :math:`m`).

        """
        self.eid = eid
        self.nodes = nodes
        self.degree = degree
        self.bf = self.create_bf()
        self.dbf = self.create_dbf()
        self.length = self.calculate_length()

    def create_bf(self) -> Sequence[MathFunc]:
        """
        Creates a list with the element's piece-wise Lagrange polynomials (each implemented as a *MathFunc*).

        Returns
        -------
        bf : Sequence[MathFunc]
            A list of the element's basis functions (each implemented as a *MathFunc*).
        """

        if self.degree == 2:  # If the polynomials are quadratic,
            # an appropriate pre-existing class type is used (QuadBasisFunc)

            znodes = [node.z for node in self.nodes]
            bf = [QuadBasisFunc(i, znodes) for i in range(0, 3)]

        else: # If the polynomials are not quadratic, the coefficients
            # for the polynomials are computed through a system of equations

            # Assembling the system of equations
            system = np.linalg.inv(
                np.array([[node.z ** h for h in range(self.degree + 1)] for node in self.nodes]))
            ind = np.identity(self.degree + 1)

            # Computing the coefficients
            bf = [PolyFunc(np.matmul(system, ind[i, :])) for i in range(self.degree + 1)]
        return bf

    def create_dbf(self) -> Sequence[MathFunc]:
        """
        Creates a list with the derivatives for the finite-element's piece-wise Lagrange polynomials
        (each implemented as a MathFunc).

        Returns
        -------
        dbf : Sequence[MathFunc]
            A list of the derivatives for the Lagrange polynomials (each implemented as a *MathFunc*).
        """

        # Computing the coefficients and passing them to the computation domain element
        dbf = [DerivFunc(self.bf[i]) for i in range(self.degree + 1)]
        return dbf

    def calculate_length(self) -> float:
        """
        Computes the length of the element (in :math:`m`).

        Returns
        -------
        float
            The length of the element (in :math:`m`).
        """

        znodes = [node.z for node in self.nodes]
        return np.abs(np.max(znodes)-np.min(znodes))

