from __future__ import division
__author__ = 'Ayman YACHAOUI'
import pandas as pd
import numpy as np
from math import log
import sys
import getopt
import re
import random
def entropy(data_classes, base=2):

    if not isinstance(data_classes, pd.core.series.Series):
        raise AttributeError('input array should be a pandas series')
    classes = data_classes.unique()
    N = len(data_classes)
    ent = 0  

    
    for c in classes:
        partition = data_classes[data_classes == c]  
        proportion = len(partition) / N
        
        ent -= proportion * log(proportion, base)

    return ent

def cut_point_information_gain(dataset, cut_point, feature_label, class_label):
    '''
    Return de information gain obtained by splitting a numeric attribute in two according to cut_point
    :param dataset: pandas dataframe with a column for attribute values and a column for class
    :param cut_point: threshold at which to partition the numeric attribute
    :param feature_label: column label of the numeric attribute values in data
    :param class_label: column label of the array of instance classes
    :return: information gain of partition obtained by threshold cut_point
    '''
    if not isinstance(dataset, pd.core.frame.DataFrame):
        raise AttributeError('input dataset should be a pandas data frame')

    entropy_full = entropy(dataset[class_label])  

    
    data_left = dataset[dataset[feature_label] <= cut_point]
    data_right = dataset[dataset[feature_label] > cut_point]
    (N, N_left, N_right) = (len(dataset), len(data_left), len(data_right))

    gain = entropy_full - (N_left / N) * entropy(data_left[class_label]) - \
        (N_right / N) * entropy(data_right[class_label])

    return gain






class MDLP_Discretizer(object):
    def __init__(self, dataset, class_label, out_path_data, out_path_bins, features=None):


        if not isinstance(dataset, pd.core.frame.DataFrame):  
            raise AttributeError('input dataset should be a pandas data frame')

        self._data_raw = dataset 

        self._class_name = class_label

        self._classes = self._data_raw[self._class_name].unique()


        
        if features:
            self._features = [f for f in features if f in self._data_raw.columns]  
            missing = set(features) - set(self._features)  
            if missing:
                print 'WARNING: Les features %s introuvables dans la source' % str(missing)
        else:  
            numeric_cols = self._data_raw._data.get_numeric_data().items
            self._features = [f for f in numeric_cols if f != class_label]
        
        self._ignored_features = set(self._data_raw.columns) - set(self._features)

        
        self._data = self._data_raw.loc[:, self._features + [class_label]]
        
        self._boundaries = self.compute_boundary_points_all_features()
        
        self._cuts = {f: [] for f in self._features}
        
        self.all_features_accepted_cutpoints()
        
        self.apply_cutpoints(out_data_path=out_path_data, out_bins_path=out_path_bins)

    def MDLPC_criterion(self, data, feature, cut_point):
        '''
        Determines whether a partition is accepted according to the MDLPC criterion
        :param feature: feature of interest
        :param cut_point: proposed cut_point
        :param partition_index: index of the sample (dataframe partition) in the interval of interest
        :return: True/False, whether to accept the partition
        '''
        
        data_partition = data.copy(deep=True)
        data_left = data_partition[data_partition[feature] <= cut_point]
        data_right = data_partition[data_partition[feature] > cut_point]

        
        cut_point_gain = cut_point_information_gain(dataset=data_partition, cut_point=cut_point,
                                                    feature_label=feature, class_label=self._class_name)
        
        N = len(data_partition) 
        partition_entropy = entropy(data_partition[self._class_name])
        k = len(data_partition[self._class_name].unique())
        k_left = len(data_left[self._class_name].unique())
        k_right = len(data_right[self._class_name].unique())
        entropy_left = entropy(data_left[self._class_name])  
        entropy_right = entropy(data_right[self._class_name])
        delta = log(3 ** k, 2) - (k * partition_entropy) + (k_left * entropy_left) + (k_right * entropy_right)

        
        gain_threshold = (log(N - 1, 2) + delta) / N

        if cut_point_gain > gain_threshold:
            return True
        else:
            return False

    def feature_boundary_points(self, data, feature):
        '''
        Given an attribute, find all potential cut_points (boundary points)
        :param feature: feature of interest
        :param partition_index: indices of rows for which feature value falls whithin interval of interest
        :return: array with potential cut_points
        '''
        
        data_partition = data.copy(deep=True)
        data_partition.sort(feature, ascending=True, inplace=True)

        boundary_points = []

        
        data_partition['class_offset'] = data_partition[self._class_name].shift(1)  
        data_partition['feature_offset'] = data_partition[feature].shift(1)  
        data_partition['feature_change'] = (data_partition[feature] != data_partition['feature_offset'])
        data_partition['mid_points'] = data_partition.loc[:, [feature, 'feature_offset']].mean(axis=1)

        potential_cuts = data_partition[data_partition['feature_change'] == True].index[1:]
        sorted_index = data_partition.index.tolist()

        for row in potential_cuts:
            old_value = data_partition.loc[sorted_index[sorted_index.index(row) - 1]][feature]
            new_value = data_partition.loc[row][feature]
            old_classes = data_partition[data_partition[feature] == old_value][self._class_name].unique()
            new_classes = data_partition[data_partition[feature] == new_value][self._class_name].unique()
            if len(set.union(set(old_classes), set(new_classes))) > 1:
                boundary_points += [data_partition.loc[row]['mid_points']]

        return set(boundary_points)

    def compute_boundary_points_all_features(self):
        '''
        Computes all possible boundary points for each attribute in self._features (features to discretize)
        :return:
        '''
        boundaries = {}
        for attr in self._features:
            data_partition = self._data.loc[:, [attr, self._class_name]]
            boundaries[attr] = self.feature_boundary_points(data=data_partition, feature=attr)
        return boundaries

    def boundaries_in_partition(self, data, feature):
        '''
        From the collection of all cut points for all features, find cut points that fall within a feature-partition's
        attribute-values' range
        :param data: data partition (pandas dataframe)
        :param feature: attribute of interest
        :return: points within feature's range
        '''
        range_min, range_max = (data[feature].min(), data[feature].max())
        return set([x for x in self._boundaries[feature] if (x > range_min) and (x < range_max)])

    def best_cut_point(self, data, feature):
        '''
        Selects the best cut point for a feature in a data partition based on information gain
        :param data: data partition (pandas dataframe)
        :param feature: target attribute
        :return: value of cut point with highest information gain (if many, picks first). None if no candidates
        '''
        candidates = self.boundaries_in_partition(data=data, feature=feature)
        
        if not candidates:
            return None
        gains = [(cut, cut_point_information_gain(dataset=data, cut_point=cut, feature_label=feature,
                                                  class_label=self._class_name)) for cut in candidates]
        gains = sorted(gains, key=lambda x: x[1], reverse=True)

        return gains[0][0] 

    def single_feature_accepted_cutpoints(self, feature, partition_index=pd.DataFrame().index):
        '''
        Computes the cuts for binning a feature according to the MDLP criterion
        :param feature: attribute of interest
        :param partition_index: index of examples in data partition for which cuts are required
        :return: list of cuts for binning feature in partition covered by partition_index
        '''
        if partition_index.size == 0:
            partition_index = self._data.index  

        data_partition = self._data.loc[partition_index, [feature, self._class_name]]

        
        if data_partition[feature].isnull().values.any:
            data_partition = data_partition[~data_partition[feature].isnull()]

        
        if len(data_partition[feature].unique()) < 2:
            return
        
        cut_candidate = self.best_cut_point(data=data_partition, feature=feature)
        if cut_candidate == None:
            return
        decision = self.MDLPC_criterion(data=data_partition, feature=feature, cut_point=cut_candidate)

        
        if not decision:
            return  
        if decision:
            
            
            left_partition = data_partition[data_partition[feature] <= cut_candidate]
            right_partition = data_partition[data_partition[feature] > cut_candidate]
            if left_partition.empty or right_partition.empty:
                return 
            self._cuts[feature] += [cut_candidate]  
            self.single_feature_accepted_cutpoints(feature=feature, partition_index=left_partition.index)
            self.single_feature_accepted_cutpoints(feature=feature, partition_index=right_partition.index)
            
            self._cuts[feature] = sorted(self._cuts[feature])
            return

    def all_features_accepted_cutpoints(self):
        '''
        Computes cut points for all numeric features (the ones in self._features)
        :return:
        '''
        for attr in self._features:
            self.single_feature_accepted_cutpoints(feature=attr)
        return

    def apply_cutpoints(self, out_data_path=None, out_bins_path=None):
        '''
        Discretizes data by applying bins according to self._cuts. Saves a new, discretized file, and a description of
        the bins
        :param out_data_path: path to save discretized data
        :param out_bins_path: path to save bins description
        :return:
        '''
        bin_label_collection = {}
        for attr in self._features:
            if len(self._cuts[attr]) == 0:
                self._data[attr] = 'All'
                bin_label_collection[attr] = ['All']
            else:
                cuts = [-np.inf] + self._cuts[attr] + [np.inf]
                start_bin_indices = range(0, len(cuts) - 1)
                bin_labels = ['%s_to_%s' % (str(cuts[i]), str(cuts[i+1])) for i in start_bin_indices]
                bin_label_collection[attr] = bin_labels
                self._data[attr] = pd.cut(x=self._data[attr].values, bins=cuts, right=False, labels=bin_labels,
                                          precision=6, include_lowest=True)

        
        if self._ignored_features:
            to_return = pd.concat([self._data, self._data_raw[list(self._ignored_features)]], axis=1)
            to_return = to_return[self._data_raw.columns] 
        else:
            to_return = self._data

        
        if out_data_path:
            to_return.to_csv(out_data_path)
        
        if out_bins_path:
            with open(out_bins_path, 'w') as bins_file:
                print>>bins_file, 'Description of bins in file: %s' % out_data_path
                for attr in self._features:
                    print>>bins_file, 'attr: %s\n\t%s' % (attr, ', '.join([bin_label for bin_label in bin_label_collection[attr]]))



def main(argv):
    out_path_data, out_path_bins, return_bins, class_label, features = None, None, False, None, None

    
    try:
        parameters, _ = getopt.getopt(argv, shortopts='', longopts=['source=', 'destination=', 'features=', 'labels_classes=', 'return_bins'])
    except:
        print 'Parametres: ./MDLP --source=path --destination=path --features=f1,f2,f3... ' \
              '--labels_classes=temperature'
        sys.exit(2)

    for opt, value in parameters:
        if opt == '--source':
            data_path = value
            if not data_path.endswith('csv') or data_path.endswith('CSV'):
                print 'Input data doit etre un csv valide'
                sys.exit(2)
            print 'Input : %s' % data_path
        elif opt == '--destination':
            out_path_data = value
            if not data_path.endswith('csv') or data_path.endswith('CSV'):
                out_path_data = '%s.csv' % out_path_data
            print 'Output sera enregistre dans: %s' % out_path_data
        elif opt == '--features':
            features = re.split(r',', value)
            features = [f for f in features if f]
        elif opt == '--return_bins':
            return_bins = True
        elif opt == '--labels_classes':
            class_label = value

    if return_bins:
        bins_name = ''.join(re.split(r'\.', out_path_data)[:-1])
        out_path_bins = '%s_bins.txt' % bins_name
        print 'Bins : %s' % out_path_bins

    if not class_label:
        print 'Un label de classe doit etre specifie --labels_classes= labelx'
        sys.exit(2)

    
    data = pd.read_csv(data_path)
    discretizer = MDLP_Discretizer(dataset=data, class_label=class_label, features=features, out_path_data=out_path_data, out_path_bins=out_path_bins)

if __name__ == '__main__':
    main(sys.argv[1:])