class Attribute:
    """
    class attribute to handle information and statistics of attributes

    Attributes
    ----------
    name : str
        The name of the attribute
    att_values : Array
        the values of an attribute
    post_class_dist :  Array
        the class distribution for each value of an attribute
    num_splits : int
        number of possible splits
    metric_value : float
        the value of gini/entroy for this attribute
    majority_class : str
        the value of the majority class
    """

    def __init__(self, name=None, num_splits=None, metric_value=None,
                 post_class_dist=[], att_values=[], majority_class=None):
        self.name = name
        self.num_splits = num_splits
        self.metric_value = metric_value
        self.post_class_dist = post_class_dist
        self.att_values = att_values
        self.majority_class= majority_class

    def compute_post_distribution(self, class_distribution, node_statistics):
        """
        calculate post distribution of classes value after splitting on this attribute

        Parameters
        ----------
        class_distribution: dict
            class distribution before split

        node_statistics: nested dict
            sufficient statistics of a node

        """
        class_att_value_dic = node_statistics[self.name]
        split_dists = {}
        for class_val, att_dist in class_att_value_dic.items():
            for att_val, att_count in att_dist.items():
                cls_dist = split_dists.get(att_val, None)
                if cls_dist is None:
                    cls_dist = {}
                    split_dists[att_val] = cls_dist

                cls_count = cls_dist.get(class_val, None)
                if cls_count is None:
                    cls_dist[class_val] = cls_count
                cls_count += att_count
        for att_val, dist in split_dists.items():
            self.att_values.append(att_val)
            self.post_class_dist.append(dist)
        self.num_splits = len(self.att_values)

    def compute_metric(self):
        """
        compute the Gini value

        """
        total_count = 0.0
        dist_count = []
        for i in range(len(self.post_class_dist)):
            dist_count.append(self.sum(self.post_class_dist[i]))
            total_count += dist_count[i]
        gini_metric = 0
        for i in range(len(self.post_class_dist)):
            gini_metric += (dist_count[i] / total_count) * self.gini(self.post_class_dist[i], dist_count[i])
        self.metric_value = 1.0 - gini_metric

    def gini(self, val_class_dist, dist_count=None):
        if dist_count is None:
            dist_count = self.sum(val_class_dist)
        gini_metric = 1.0
        for class_value, count in val_class_dist.items():
            frac = count / dist_count
            gini_metric -= frac * frac
        return gini_metric
