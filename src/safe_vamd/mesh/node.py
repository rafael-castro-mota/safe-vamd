

class Node:
    """
    A node in a computational mesh.

    Attributes
    ----------

    nid : int
        Node Identification Number.
    x : float
        X-coordinate (in meters) for the node.
    y : float
        Y-coordinate (in meters) for the node.
    z : float
        Z-coordinate (in meters) for the node.

    """
    def __init__(self, nid: int = None, x: float = None, y: float = None, z: float = None) -> None:
        """
        Initializes an instance of class Node.
        """

        self.nid = nid
        self.x = x
        self.y = y
        self.z = z
