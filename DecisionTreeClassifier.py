import numpy as np
import pandas as pd
import random


class DecisionTreeClassifier():
     
    def __init__(self, tree_type="classic", splitter="best", min_samples=2, max_depth=5, criterion="gini"):
        
        if not isinstance(tree_type, str):
            raise ValueError("tree_type must be str format")
        if not isinstance(splitter, str):
            raise ValueError("splitter must be str formt")
        if splitter not in {'best', 'fast'}:
            raise ValueError("splitter {} is not specified".format(splitter))
        if tree_type not in {'classic', 'oblivious'}:
            raise ValueError("tree_type {} is not specified".format(tree_type))
        if not isinstance(min_samples, int):
            raise ValueError("min_samples must be int format")
        if not isinstance(max_depth, int):
            raise ValueError("max_depth must be int format")
        if not isinstance(criterion, str):
            raise ValueError("criterion must be str format")
            if criterion not in {'gini', 'entropy'}:
                raise ValueError("criterion {} is not specified".format(criterion))
                
        self.tree_type = tree_type
        self.splitter = splitter
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.criterion = criterion
        
    # Helper methods
        
    def _is_pure(self, labels):
        """ Check: do the labels in subsample have the same values?
        Params
        _________________________
        : param labels: list, List containing labels
        _________________________
        : return true/false"""
    
        if len(np.unique(labels)) == 1:
            return True
        else:
            return False
    def _majority_voting_rule(self, labels):
        """ Majority classification in node
        Params
        ____________________________
        : param labels: list, List containing labels
        ____________________________
        : return classification: str, Str containing label"""
    

        unique_classes = list(set(labels))
        if self._is_pure(labels):
            return str(unique_classes[0])
        else:
            _, classes_count = np.unique(labels, return_counts = True)
            classification = unique_classes[np.argmax(classes_count)]
            return classification
    
    
    # Information criterion and IGain
    
    def _get_gini(self, labels):
        _, label_count = np.unique(labels, return_counts=True)
        gini = 1 - sum([np.power(count/len(labels), 2) for count in label_count])
        return gini
    
    def _get_entropy(self,labels):
        # count of objects for each label value
        _, label_count = np.unique(labels, return_counts=True)
        entropy = sum([- count/len(labels) * np.log2(count/len(labels)) if count else 0 for count in label_count])
        return entropy
    
    def _get_information_gain(self, data_below, data_above, labels, criterion):
        p_data_below = len(data_below)/len(labels)
        p_data_above = len(data_above)/len(labels)
        #print(data_below.index.tolist())
        if criterion == "entropy":
            total_entropy = self._get_entropy(labels)
            info_gain = total_entropy - (p_data_below * self._get_entropy(labels[data_below.index.tolist()]) + 
                                           p_data_above * self._get_entropy(labels[data_above.index.tolist()]))
        elif criterion == "gini":
            total_gini = self._get_gini(labels)
            info_gain = total_gini - (p_data_below * self._get_gini(labels[data_below.index.tolist()]) + 
                                           p_data_above * self._get_gini(labels[data_above.index.tolist()]))
        
        return info_gain
    
    
    def _get_weighted_sum_of_errors(self, data_below, data_above, labels):
        classification = pd.Series(data=np.ones((labels.shape[0],)), index=labels.index.tolist())
        vector_of_errs = pd.Series(data=np.ones((labels.shape[0],)), index=labels.index.tolist())
        
        info_about_split = {}
        classification.at[data_below.index] = 0
        vector_of_errs.at[classification == labels] = 0
        N = self.sample_weights * vector_of_errs
        N = N.fillna(0).sum()
        info_about_split['bincount'] = np.bincount(classification)
        info_about_split['classification'] = classification[:10]
        info_about_split['labels'] = labels[:10]
        info_about_split['errors'] = vector_of_errs[:10]
        info_about_split['accuracy'] = 1 - sum(vector_of_errs)/len(vector_of_errs)
        info_about_split['N'] = N
        return N, info_about_split
        
    
    # Split methods
    
    def _best_split(self, data, labels, potential_splits, criterion):
        max_gain = 0
        min_error = np.inf
        for column_index in potential_splits:
            for value in potential_splits[column_index]:
                data_below, data_above = self._split_data(data, column_index, value)
                
                # Если веса объектов не заданы, то максимизируем Gain
                #if self.sample_weights.any() != True:
                if type(self.sample_weights) != pd.core.series.Series:
                    info_gain = self._get_information_gain(data_below, data_above, labels, criterion)
                    if info_gain > max_gain:
                        max_gain = info_gain
                        best_split_column = column_index
                        best_split_value = value
                        #best_info_gain = info_gain\
                        
                # Если веса объектов заданы, то минимизируем взвешенную сумму ошибок
                else:
                  
                    N, info_about_split = self._get_weighted_sum_of_errors(data_below, data_above, labels)
                    if N < min_error:
                        min_error = N
                        best_split_column = column_index
                        best_split_value = value
                        best_info = info_about_split
                        #best_inequal = inequal
        self.N = min_error
        return best_split_column, best_split_value
    
    def _potential_split(self, data):
        """
        Params
        _______________
        : param data: Subsample
        : param labels: labels in given subsample
        _______________
        :return potential_splits: Dictionary containing feature number as key and list of potential split points as value """
        potential_splits = {}
        n = data.shape[1]
        max_gain = 0
        
        # Если был выбран способ разбиения "best", то перебираем все значения признака
        if self.splitter == 'best':
            for feature in range(n):
                unique_values = np.unique(data.iloc[:, feature])
                feature_type = self.feature_types[feature]
        
                if feature_type == 0:
                    potential_splits[feature] = []
                    for value in range(1, len(unique_values)):
            
                        right_num  = unique_values[value]
                        left_num = unique_values[value-1]
                        split_point = (left_num + right_num)/2
                        potential_splits[feature].append(split_point)
                elif len(unique_values) > 1 :
                    potential_splits[feature] = unique_values
                    
        # Если способ разделения "fast", то перебираем 10 значений в промежутке от минимального до максимального для признака
        else:
            
            for feature in range(n):
                unique_values = np.unique(data.iloc[:, feature])
                feature_type = self.feature_types[feature]
                if feature_type == 0:
                    potential_splits[feature] = []
                    max_value = np.max(data.iloc[:, feature])
                    min_value = np.min(data.iloc[:, feature])
                    n_steps = 10
                    step_size = (max_value - min_value)/n_steps
                    
                    for j in range(0, n_steps):
                        split_point = min_value + step_size*j
                        potential_splits[feature].append(split_point)
                
                elif len(unique_values) > 1:
                    potential_splits[feature] = unique_values
                
            
            
        return potential_splits
    
    def _split_data(self, data, split_column, split_value):
    
        column_values = np.array(data.iloc[:, split_column])
        feature_type = self.feature_types[split_column]
    
        if feature_type == 0:
            data_below = data.loc[data.iloc[:, split_column] <= split_value]
            data_above = data.loc[data.iloc[:, split_column] > split_value]
    
        # feature is categorical
        else:
            data_below = data.loc[data.iloc[:, split_column] == split_value]
            data_above = data.loc[data.iloc[:, split_column] != split_value]
    
        return data_below, data_above
    
    
    # Features type checking method(numeric, categorical)
    
    def _get_list_of_feature_types(self, feature_counts, cat_features):
        """ Returns binary array containing the information about column's type
        Params
        ____________________________
        :param feature_counts: int, count of features
        :param cat_features: list, List containing indexes of categorial features
        ____________________________
        :return feature_types: ndarray, Binary array. 0 - if feature is continuous, 1 - if feature is categorical
    
        """
        feature_types = np.zeros(feature_counts)
        for elem in range(len(cat_features)):
            feature_types[cat_features[elem]] = 1
    
        return feature_types
    
    def fit(self, df, labels, cat_features=[], sample_weights = None):
        self.sample_weights = sample_weights
        
        if self.tree_type == "classic":
            self.tree = self._DecisionTreeClassifierFit(df, labels, self.min_samples, self.max_depth,
                               cat_features, self.criterion, counter = 0)
         
        elif self.tree_type == "oblivious":
            self.tree = self._ObliviousTreeClassifierFit(df, labels, self.min_samples, self.max_depth,
                                                         cat_features, self.criterion, counter = 0)
        
        
    def _DecisionTreeClassifierFit(self, df, labels, min_samples, max_depth,
                               cat_features, criterion, counter):
        
        if counter == 0:
            
            #global COLUMN_HEADERS, FEATURE_TYPES
            self.column_headers = df.columns
            
            self.feature_types = self._get_list_of_feature_types(len(self.column_headers), cat_features)
            
        # Stop criterion
        if self._is_pure(labels) or (len(df) < min_samples) or (counter == max_depth):
            classification = self._majority_voting_rule(labels)
            return classification
        
        else:
            
            counter += 1
    
            # Looking for the best split rule
            potential_splits = self._potential_split(df)
            split_feature, split_value = self._best_split(df, labels, potential_splits, criterion)
            
            # Splitting
            data_below, data_above = self._split_data(df, split_feature, split_value)
            # Empty splitting checking
            if len(data_below) == 0 or len(data_above) == 0:
                classification = self._majority_voting_rule(labels)
                return classification
            
            # Creating a subtree
            column_name = self.column_headers[split_feature]
            feature_type = self.feature_types[split_feature]
            if feature_type == 0:
                rule = "{} <= {}".format(column_name, np.round(split_value, 2))
                sub_tree = {rule: []}
            # feature is categorical
            else:
                rule = "{} = {}".format(column_name, np.round(split_value, 2))
                sub_tree = {rule: []}
            # Recursive part
            left_branch = self._DecisionTreeClassifierFit(data_below, labels[data_below.index.tolist()], min_samples, max_depth,
                                                          cat_features, criterion, counter)
            right_branch = self._DecisionTreeClassifierFit(data_above, labels[data_above.index.tolist()], min_samples, max_depth,
                                                          cat_features, criterion, counter)
            sub_tree[rule].append(left_branch)
            sub_tree[rule].append(right_branch)
            return sub_tree
        
        
        
        
    def _ObliviousTreeClassifierFit(self, df, labels, min_samples, max_depth,
                               cat_features, criterion, counter):
        
        if counter == 0:
            
            self.oblivious_splits = {a: None for a in range(1,max_depth+1)}
            self.column_headers = df.columns
            
            self.feature_types = self._get_list_of_feature_types(len(self.column_headers), cat_features)
            
        # Stop criterion
        if self._is_pure(labels) or (counter == max_depth):
            classification = self._majority_voting_rule(labels)
            return classification
        
        else:
            
            counter += 1
            
            # if we are at the level that has a splitting rule already, then apply the same rule
            if self.oblivious_splits[counter]:
                split_feature, split_value = self.oblivious_splits[counter]
                data_below, data_above = self._split_data(df,split_feature , split_value)
            # else we are gonna find the best splitting rule and add it to the dictionary
            else:
                potential_splits = self._potential_split(df)
                split_feature, split_value = self._best_split(df, labels, potential_splits, criterion)
                self.oblivious_splits[counter]  = [split_feature, split_value]
                
            
            # splitting
            data_below, data_above = self._split_data(df, split_feature, split_value)
            
            # Empty splitting checking
            if len(data_below) == 0 or len(data_above) == 0:
                classification = self._majority_voting_rule(labels)
                return classification
            
             # Creating a subtree
            column_name = self.column_headers[split_feature]
            feature_type = self.feature_types[split_feature]
            if feature_type == 0:
                rule = "{} <= {}".format(column_name, np.round(split_value, 2))
                sub_tree = {rule: []}
            # feature is categorical
            else:
                rule = "{} = {}".format(column_name, np.round(split_value, 2))
                sub_tree = {rule: []}
            
            # Recursive part
            left_branch = self._ObliviousTreeClassifierFit(data_below, labels[data_below.index.tolist()], min_samples, max_depth,
                                                          cat_features, criterion, counter)
            right_branch = self._ObliviousTreeClassifierFit(data_above, labels[data_above.index.tolist()], min_samples, max_depth,
                                                          cat_features, criterion, counter)
            
            sub_tree[rule].append(left_branch)
            sub_tree[rule].append(right_branch)
        
            return sub_tree
        
       
    
    # Predict methods 
    
    def _predict_one_example(self, example, tree):
    
        question = list(tree.keys())[0]
        feature_name, comparison, value = question.split()
     
        
        # Применяем пороговое правило
        if comparison == "<=":
            if(example[feature_name] <= float(value)):
                answer = tree[question][0]
            else:
                answer = tree[question][1]
        elif comparison == "=":
            if str(example[feature_name]) == value:
                answer = tree[question][0]
            else:
                answer = tree[question][1]
            
        # Если попали в лист, то возвращаем результат
        if not isinstance(answer, dict):
            return answer
        # Иначе рекурсивный проход по поддеревьям
        else:
            residual_tree = answer
            return self._predict_one_example(example, residual_tree)
        
      
    def predict(self, data):
        pred = data.apply(self._predict_one_example, axis=1, args = (self.tree,))
        return pred
    
    
   
    # Quality metrics
    
    def _get_bool_answers(self, y_pred, y_true):
        """ 
        Params
        ___________
        :param: y_pred: series, Series containing prediction results
        
        """
        bool_answers = y_pred == y_true
        true_object_idx = []
        for i in bool_answers:
            if i == True:
                true_object_idx.append(i)
        return bool_answers, true_object_idx
    
    
    def accuracy(self, y_pred, y_true):
        bool_answers, true_object_idx = self._get_bool_answers(y_pred, y_true)
        return len(true_object_idx)/len(bool_answers)