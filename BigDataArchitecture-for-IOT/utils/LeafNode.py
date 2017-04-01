from utils.Attribute import Attribute
from operator import attrgetter, itemgetter


class LeafNode:
    """
    Leaf node for Hoeffding Tree classifier

    Attributes
    ----------
    class_distribution : dictionary of of tuples (class value, count)
        to store class distribution for a leaf node
    node_statistics : nested dictionary : (attribute name , ( class value , (attribute value, count )))
        to store sufficient statistics of attributes for a leaf node
    list_split_attributes : Array
        list of attributes that can a node split on
    majority_class : string
        The predicted class value for this leaf node
    nb_of_seen_instances : int
        The number of instances a leaf observe
    parent_node : InternalNode
        the parent node for this leaf node
    branch_name : String
        The branch leading from the parent to the node.

    """

    def __init__(self, class_distribution={}, node_statistics={}, list_split_attributes=[], majority_class=None,
                 nb_of_seen_instances=0, parent_node= None, branch_name=None):
        self.class_distribution = class_distribution
        self.node_statistics = node_statistics
        self.list_split_attributes = list_split_attributes
        self.majority_class = majority_class
        self.nb_of_seen_instances = nb_of_seen_instances
        self.parent_node = parent_node
        self.branch_name = branch_name

    def update(self, x, y):
        """
        update class distribution, node statistics and number of seen instances after observing x

         Parameters
        ----------
        x : array
            instance x
        """

        self.class_distribution[y] += 1
        self.nb_of_seen_instances += 1
        if self.node_statistics == {}:
            for i in range(len(x)):
                self.node_statistics[i] = {}
        for i in range(len(x)):
            if y not in self.node_statistics[i].keys():
                self.node_statistics[i][y] = {}
            if x[i] not in self.node_statistics[i][y].keys():
                self.node_statistics[i][y][x[i]] = 0;

            self.node_statistics[i][y][x[i]] += 1

        # majority class in the node
        self.majority_class = max(self.class_distribution.iteritems(), key=itemgetter(1))[0]

    def get_sorted_attributes(self):
        """
        sort all attributes after calculating evaluation metric (Gini)

        Returns
        -------
        array
            array containing sorted attributes
        """
        list_attributes =[]
        for i in self.list_split_attributes:
            attr = Attribute(name=self.list_split_attributes[i])
            attr.compute_post_distribution(self.class_distribution, self.node_statistics)
            attr.compute_metric()
            list_attributes.append(attr)

        return self.sort(list_attributes)

    def sort(self, list_attributes):
        """
        sort in descending order the list of attributes according to Gini value

        Parameters
        ----------
        list_attributes: Array
            Array of attribute that a node can split on

        Returns
        -------
        sorted_att_list : att
            list in descending order
        """

        return list_attributes.sort(key=attrgetter('metric_value'), reverse=True)
