from utils.LeafNode import LeafNode


class InternalNode:
    """
    internal node for Hoeffding Tree classifier

    Attributes
    ----------
    class_distribution : dictionary of of tuples (class value, count)
        to store class distribution for a leaf node
    children : dictionary of of tuples (attribute value, node)
        store children nodes
    split_attribute : String
        the attribute that the node split on
    parent_node : InternalNode
        the parent node for this leaf node
    branch_name : String
        The branch leading from the parent to the node.
    """

    def __init__(self, class_distribution={}, children={}, split_attribute=None, parent_node=None, branch_name=None):
        self.class_distribution = class_distribution
        self.children = children
        self.split_attribute = split_attribute
        self.parent_node = parent_node
        self.branch_name = branch_name


    def sort_instance(self, x):
        """
        sort instance to appropriate leaf node

        Parameters
        ----------
        x : array
            instance x

        Returns
        -------
        LeafNode
            leaf node of instance x
        """
        if isinstance(self, LeafNode):
            return self
        else:
            x_attr = x[self.split_attribute]
            x_chil = self.children[x_attr]
            return x_chil.sort_instance(x)
