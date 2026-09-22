

class Node:
    """
    A node in a finite-element mesh.

    **Class attributes:**

    Attributes
    ----------
    nid : int
        Node Identification Number.
    x : float
        X-coordinate (in :math:`m`) for the node.
    y : float
        Y-coordinate (in :math:`m`) for the node.
    z : float
        Z-coordinate (in :math:`m`) for the node.

    """
    def __init__(self, nid: int = None, x: float = None, y: float = None, z: float = None) -> None:
        """

        Parameters
        ----------
        nid : int
            Node Identification Number.
        x : float
            X-coordinate (in :math:`m`) for the node.
        y : float
            Y-coordinate (in :math:`m`) for the node.
        z : float
            Z-coordinate (in :math:`m`) for the node.
        """

        self.nid = nid
        self.x = x
        self.y = y
        self.z = z
