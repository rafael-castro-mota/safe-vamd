from typing import Sequence

from safe_vamd.mesh.node import Node
from safe_vamd.mesh.kirby_abs import KirbyABS
from safe_vamd.mesh.element_1d_arbitrary_order_lagrange import Element1DArbitraryOrderLagrange


class Element1DArbitraryOrderLagrangePML(Element1DArbitraryOrderLagrange):
    """
    One-dimensional arbitrary-order Lagrange finite element in a Perfectly Matched Layer (PML).

    This class extends ``Element1DArbitraryOrderLagrange`` by incorporating a stretching function "pml"
    (Perfectly Matched Layer).


    Attributes
    ----------

    z_ref : float
        z-coordinate (height, in meters) of the physical domain/PML interface.
    pml : KirbyABS
        Ray Kirby's stretching function for the Perfectly Matched Layer.

    Notes
    -----
    The attributes ``eid``, ``nodes``, and ``degree`` are inherited from
    ``Element1DArbitraryOrderLagrange``.
    """

    def __init__(self, eid: int, nodes: Sequence[Node], degree: int, z_ref: float,
                 pml_thickness: float, tau1: float = 4, tau2: float = 4) -> None:
        """
        Initializes an instance of ``Element1DArbitraryOrderLagrangePML``.

        Parameters
        ----------
        pml_thickness: float
            The thickness (in meters) of the Perfectly Matched Layer.
        tau1: float
            A factor for the stretching function.
        tau2: float
            A factor for the stretching function.

         Notes
        -----
        The factors tau1 and tau2 are given the value of 4 in https://doi.org/10.1121/10.0002912.
        """

        super().__init__(eid, nodes, degree)
        self.z_ref = z_ref
        self.pml = KirbyABS(tau1, tau2, z_ref, pml_thickness)
