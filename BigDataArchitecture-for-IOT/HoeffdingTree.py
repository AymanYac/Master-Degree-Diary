from sklearn.base import BaseEstimator
from utils.LeafNode import LeafNode
from utils.InternalNode import InternalNode
import math


class HoeffdingTreeClassifier(BaseEstimator):
    """
    A Hoeffding Tree Classifier

    A Hoeffding tree is an incremental, anytime decision tree induction algorithm
    that is capable of learning from massive data streams, assuming that the
    distribution generating examples does not change over time. Hoeffding trees
    exploit the fact that a small sample can often be enough to choose an optimal
    splitting attribute. This idea is supported mathematically by the Hoeffding
    bound, which quantiﬁes the number of observations (in our case, examples)
    needed to estimate some statistics within a prescribed precision (in our
    case, the goodness of an attribute).</p> <p>A theoretically appealing feature
    of Hoeffding Trees not shared by other incremental decision tree learners is
    that it has sound guarantees of performance. Using the Hoeffding bound one
    can show that its output is asymptotically nearly identical to that of a
    non-incremental learner using inﬁnitely many examples.

    Parameters
    ----------
    split_confidence : double
        The allowable error in split decision, values closer to 0 will take longer to decide.

    Attributes
    ----------
    root_ : node
        The root node of heoffding tree

    """

    def __init__(self, split_confidence=1.0):
        self.split_confidence = split_confidence

    def partial_fit(self, x, y):
        """
        Update the Hoeffding Tree classifier with the given instance.


        This method is expected to be called several times consecutively
        on different instances

        Parameters
        ----------
        X : array, shape = [n_features]
            The training input sample.
        y : array, shape = 1
            The target values :class labels.

        Returns
        -------
        self : object
            Returns self (current tree).

        """
        if self.root_ is None:
            list_of_attributes = []
            for i in range(len(x)):
                list_of_attributes.append(i)
            self.root_ = LeafNode(list_split_attributes=list_of_attributes)

        if isinstance(self.root_, InternalNode):
            actual_node = self.root_.sort_instance(x)
        else:
            actual_node = self.root_

        actual_node.update(x,y)
        if len(actual_node.list_split_attributes) != 0:
            attributes_stat = actual_node.get_sorted_attributes()
            best_attribute =  attributes_stat[0];
            second_best_attribute = attributes_stat[1];
            hoeffding_bound = self.calculate_hoeff_bound(actual_node.class_distribution, actual_node.nb_of_seen_instances)

            if best_attribute.metric_value - second_best_attribute.metric_value > hoeffding_bound:
                new_internal_node = InternalNode(class_distribution=actual_node.class_distribution,
                                                 split_attribute=best_attribute.name, parent_node=actual_node.parent_node,
                                                 branch_name=actual_node.branch_name)
                for i in range(best_attribute.num_splits):
                    branch_name = best_attribute.att_values[i]
                    new_list_of_split_attributes = actual_node.list_split_attributes.remove(best_attribute.name)
                    new_child = LeafNode(class_distribution=best_attribute.post_class_dist[i],
                                         list_split_attributes=new_list_of_split_attributes,
                                         parent_node=new_internal_node,
                                         branch_name=branch_name, majority_class=best_attribute.majority_class)
                    new_internal_node.children[branch_name] = new_child

                if actual_node.parent_node is None:
                    self.root_ = new_internal_node
                else:
                    actual_node.parent_node.children[actual_node.branch_name]=new_internal_node


        # Return the classifier
        return self

    def calculate_hoeff_bound(self, class_distribution, nb_instances):
        """
        calculate hoeffding bound

        Parameters
        ----------
        class_distribution : dict
            class value distribution of the node
        nb_instances : int
            number of observed instances in the node
        Returns
        -------
        float :
            hoeffding bound value
        """

        num_classes = len(class_distribution)
        if num_classes < 2:
            num_classes = 2

        max_value = math.log2(num_classes)

        return math.sqrt(((max_value * max_value) * math.log(1.0 / self.split_confidence)) / (2.0 * nb_instances))

    def predict(self, x):
        """
        Predict class value for X.

        For a classification model, the predicted class for each sample in X is
        returned.

        Parameters
        ----------
        X : array , shape = [n_features]
            The input instance that we want to predict its class value

        Returns
        -------
        y : array, shape = 1
            The predicted classe
        """
        instance_leaf_node = self.root_.sort_instance(x)

        return instance_leaf_node.majority_class









