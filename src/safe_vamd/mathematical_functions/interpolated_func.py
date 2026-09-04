
import numpy as np

from safe_vamd.mathematical_functions.mathematical_func import MathFunc
from safe_vamd.mesh.mesh_1d import Mesh1D


class InterpolatedFunc(MathFunc):
    """
    An approximation of mathematical function with the finite elements in a mesh, implemented as a subclass of MathFunc.

     Attributes
    ----------
    f1 : MathFunc
        Function to be approximated.
    mesh : Mesh1D
        Mesh with the elements and respective basis functions defining the approximation.
    """

    def __init__(self, mesh: Mesh1D, f1: MathFunc) -> None:
        """
        Initializes an instance of InterpolatedFunc
        """

        self.mesh = mesh
        self.f1 = f1

    def value(self, z: float, eid: str | int = 'find') -> float | complex:
        """
        Finds (if needed) the element in which the desired height is located and approximates the mathematical function
        locally with the element's basis functions.

        Parameters
        ----------
        z: float
            Height (in meters) at which the approximation is evaluated.
        eid: str or int
            Element index at which the desired height is located. If eid is set to 'find', the element is looked for.
        Returns
        -------
        float or complex
            Value of the approximation at the desired height.
        """

        if eid == 'find':
            element_found = self.find_element(z)
            element = self.mesh.elements[element_found]
        else:
            element = self.mesh.elements[eid]

        node_vals = np.array([self.f1.value(node.z) for node in element.nodes])
        bf_vals = np.array([b.value(z) for b in element.bf])
        val = np.sum(node_vals * bf_vals)
        return val

    def dvalue(self, z: float, eid: str | int = 'find') -> float | complex:
        """
        Finds (if needed) the element in which the desired height is located and approximates the derivative with respect to height of the
        mathematical function locally with the element's basis functions.

        Parameters
        ----------
        z: float
            Height (in meters) at which the approximation is evaluated.
        eid: str or int
            Element index at which the desired height is located. If eid is set to 'find', the element is looked for.
        Returns
        -------
        float or complex
            Value of the approximation for the derivative of the mathematical function at the desired height.
        """

        # Element
        if eid == 'find':
            element_found = self.find_element(z)
            element = self.mesh.elements[element_found]
        else:
            element = self.mesh.elements[eid]

        node_vals = np.array([self.f1.value(node.z) for node in element.nodes])
        bf_vals = np.array([d.value(z) for d in element.dbf])
        val = np.sum(node_vals * bf_vals)
        return val

    def find_element(self, z: float) -> int:
        """
        Finds the element at which the desired height is located.

        Parameters
        ----------
        z: float
            Height (in meters)
        Returns
        -------
        int
            The index of the element at which the desired height is located.
        """

        if z < self.mesh.elements[0].nodes[-1].z:
            return self.mesh.elements[0].eid

        for element in self.mesh.elements:
            if element.nodes[0].z < z <= element.nodes[-1].z:
                return element.eid

        return element

