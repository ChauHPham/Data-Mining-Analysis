class Node(object):
    """
    A class representing a node in a decision tree.

    Attributes:
        is_leaf (bool): Indicates if the node is a leaf node.
        name (Optional[str]): The feature name for decision nodes.
        children (Dict[str, 'Node']): The children of the decision node.
        is_numerical (Optional[bool]): Indicates if the feature is numerical.
        threshold (Optional[float]): The threshold value for numerical decision nodes.
        node_class (str): The class of the node, set as the mode of the classes of data in this node.
        size (int): The number of data samples in this node.
        depth (int): The depth of the node in the tree.
        single_class (bool): Indicates if all data in this node belongs to a single class.
    """

    def __init__(self, node_size: int, node_class: str, depth: int, single_class:bool = False):
        """
        Initializes a Node object.

        :param node_size: Number of data samples in this node
        :param node_class: The class of the node
        :param depth: Depth of the node in the tree
        :param single_class: Indicates if all data in this node belongs to a single class
        """

        # Every node is a leaf unless you set its 'children'
        self.is_leaf = True
        # Each 'decision node' has a name. It should be the feature name
        self.name = None
        # All children of a 'decision node'. Note that only decision nodes have children
        self.children = {}
        # Whether corresponding feature of this node is numerical or not. Only for decision nodes.
        self.is_numerical = None
        # Threshold value for numerical decision nodes. If the value of a specific data is greater than this threshold,
        # it falls under the 'ge' child. Other than that it goes under 'l'. Please check the implementation of
        # get_child_node for a better understanding.
        self.threshold = None
        # The class of a node. It determines the class of the data in this node. In this assignment it should be set as
        # the mode of the classes of data in this node.
        self.node_class = node_class
        # Number of data samples in this node
        self.size = node_size
        # Depth of a node
        self.depth = depth
        # Boolean variable indicating if all the data of this node belongs to only one class. This is condition that you
        # want to be aware of so you stop expanding the tree.
        self.single_class = single_class

    def set_children(self, children):
        """
        Sets the children of the node and marks it as a non-leaf node.

        :param children: Dictionary of child nodes
        """
        self.is_leaf = False
        self.children = children

    def get_child_node(self, feature_value) -> 'Node':
        """
        Returns the appropriate child node based on the feature value.

        :param feature_value: The value of the feature to determine the child node
        :return: The child node corresponding to the feature value
        """
        # For numerical features, compare with the threshold
        if self.is_numerical:
            if feature_value >= self.threshold:
                return self.children['ge']
            else:
                return self.children['l']
        # For categorical features, return the corresponding child node
        else:
            return self.children.get(feature_value, None)

