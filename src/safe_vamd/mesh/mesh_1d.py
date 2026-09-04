import numpy as np
from typing import Sequence

from safe_vamd.mesh.node import Node

from safe_vamd.mesh.element_1d_arbitrary_order_lagrange import Element1DArbitraryOrderLagrange
from safe_vamd.mesh.element_1d_arbitrary_order_lagrange_pml import Element1DArbitraryOrderLagrangePML


class Mesh1D:
    """
    A one-dimensional computational mesh, comprised of finite-elements and nodes.

    Attributes
    ----------

    nodes : Sequence[Node]
        A list of Node objects associated with the mesh.
    elements : Sequence[Element1DArbitraryOrderLagrange]
        A list of finite-element objects associated with the mesh.

    """
    def __init__(self, elements: Element1DArbitraryOrderLagrange = None, nodes: Sequence[Node] = None) -> None:
        """
        Initializes an instance of class Mesh1D.
        """

        self.nodes = nodes
        self.elements = elements

    def load_from_element_bounds(self, element_bounds: Sequence[float], degree: int) -> None:
        """
        Takes a list of interface heights for the desired finite-elements,
        places the appropriate interior nodes (instances of Node), creates the mesh nodes (instance of Node) and
        finite-elements (instances of Element1DArbitraryOrderLagrange) and associates them with the mesh object.

        Parameters
        ----------
        element_bounds: Sequence[float]
            An ordered (ascending) list of floats representing the bounds of the elements.
        degree: int
            The degree of the piece-wise Lagrange polynomials in the finite-elements.

        Returns
        -------
        None
        """

        # Number of desired finite-elements
        n_elements = int(len(element_bounds)-1)

        # Inserting the interior nodes (degree+1 total of nodes for each element)
        coords = []
        for i in range(1, n_elements+1):
            v1 = np.linspace(element_bounds[i - 1], element_bounds[i], degree + 1, endpoint=True)
            coords.extend(v1[0:-1])
        coords.append(element_bounds[- 1])

        # Creating the nodes (Node object)
        nids = range(0, len(coords))    # Node Identification Numbers
        nodes = [Node(nids[i], 0, 0, coords[i]) for i in range(0, len(coords))]

        # Creating the elements (Element1DArbitraryOrderLagrange)
        eids = range(0, n_elements)     # Element Identification Numbers
        elements = [Element1DArbitraryOrderLagrange(eids[i],
                    nodes[int(degree * i):int(degree * i + degree)+1], degree) for i in range(0, n_elements)]

        self.elements = elements
        self.nodes = nodes
        return None

    def add_top_pml(self, thick: float, n_elements: int, degree: int, tau1: float = 4, tau2: float = 4) -> None:
        """
        Adds a Perfectly Matched Layer (PML) of a certain thickness and number of arbitrary-order Lagrange piece-wise
        finite-elements to the top of the computational domain.

        Parameters
        ----------
        thick:float
            The thickness (in meters) of the PML.
        n_elements:int
            The number of elements in the PML.
        degree:int
            The degree of the piece-wise Lagrange polynomials in the PML's finite-elements.
        tau1: float
            A factor for the stretching function in the PML.
        tau2: float
            A factor for the stretching function in the PML.

        Returns
        -------
        None
        """

        n_elements = int(n_elements)
        element_bonds = self.nodes[-1].z + np.linspace(0, thick, n_elements + 1, endpoint=True)

        # Number of Elements
        n_elements = int(len(element_bonds) - 1)

        # Inserting extra nodes for the "degree+1"th order Lagrange Elements
        coords = []
        for i in range(1, n_elements + 1):
            v1 = np.linspace(element_bonds[i - 1], element_bonds[i], degree + 1, endpoint=True)
            coords.extend(v1[0:-1])
        coords.append(element_bonds[- 1])

        # Creating the PML nodes
        nids = range(self.nodes[-1].nid, self.nodes[-1].nid + len(coords))     # Node Ident. Number
        nodes = [Node(nids[i], 0, 0, coords[i]) for i in range(1, len(coords))]

        nodes.insert(0, self.nodes[-1])     # Adding the top node of the existing mesh

        # Creating the PML elements (instances of Element1DArbitraryOrderLagrangePML)
        eids = range(self.elements[-1].eid + 1, self.elements[-1].eid + n_elements + 1)     # Element Ident. Number
        z_ref = self.nodes[-1].z
        pml_elements = [Element1DArbitraryOrderLagrangePML(eids[i], nodes[int(degree * i):int(degree * i + degree) + 1],
                        degree, z_ref, thick, tau1, tau2) for i in range(0, n_elements)]

        # Adding the PML elements and nodes to the mesh
        self.elements = np.concatenate((self.elements, pml_elements))
        self.nodes = np.concatenate((self.nodes, nodes[1:]))

        return None
