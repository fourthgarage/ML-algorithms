import numpy as np
import pandas as pd
from tree import DecisionTreeClassifier as MyTreeRealization
from sklearn.decomposition import PCA



class RotationForestClassifier():
    def __init__(self,
                 n_estimators = 10,
                 base_estimator = MyTreeRealization(splitter='fast', max_depth = 2),
                 random_state = None):
        self.n_estimators = n_estimators
        self.base_estimator = base_estimator
        if not random_state:
            self.random_state = np.random.randint(0,1000)
        else:
            self.random_state = random_state
        self.models, self.rotation_matrices, self.feature_subspaces = None, None, None
        
        
    # Helper function. Receives dataset, returns the dataset with renamed columns
    def _rename_columns_as_in_original_dataset(self,X):
        for column_ind in range(X.shape[1]):
            X = X.rename(columns = {column_ind:self.column_names[column_ind]})
        return X
    
    
    
    def _get_random_subset(self, features, k):
        '''
        Params
        :param features: list, List containing the column indexes
        :param k: int, dimension of subspace
        ___________________________________________________________
        :return subsets: list, List containing lists as elements(subsets of features)
        '''
        subsets = []
        iteration = 0
        np.random.seed(self.random_state)
        np.random.shuffle(features)
        subset = 0
        limit = len(features)/k
        
        while iteration < limit:
            if k <= len(features):
                subset = k
            else:
                subset = len(features)
            
            subsets.append(features[-subset:])
            del features[-subset:]
            iteration+=1
        return subsets
        
    
        
    def _build_rotation_tree_model(self, X, y, k):
        '''
        Params
        :params X and y: pd.DataFrame and pd.Series, train dataset and train labels
        :param k: int, dimension of subspace
        ___________________________________________________________________________
        :return models: list, List of base models
        :return rotation_matrices: list, List of rotation matrices
        :return feature_subspaces: list, List of feature subspaces
        '''
        models = []
        rotation_matrices = []
        feature_subspaces = []
        
        for algo in range(self.n_estimators):
            x,_,_,_ = train_test_split(X, y, test_size = 0.3, random_state = self.random_state)
            
            
            
            # Features idx
            feature_idx = list(range(x.shape[1]))
            
            # Get random subspace
            random_k_subset = self._get_random_subset(feature_idx, k)
            feature_subspaces.append(random_k_subset)
            
            # Rotation matrix
            R_matrix = np.zeros((x.shape[1], x.shape[1]), dtype = float)
            # fit PCA for each subspace and fill the R_matrix with the appropriate values of components
            for each_subset in range(len(random_k_subset)):
                pca = PCA()
                x_subset = x.iloc[:, random_k_subset[each_subset]]
                pca.fit(x_subset)
                for ii in range(len(pca.components_)):
                    for jj in range(len(pca.components_)):
                        R_matrix[random_k_subset[each_subset][ii], random_k_subset[each_subset][jj]] = pca.components_[ii,jj]
                        
            # Multiply the original matrix by the rotation matrix       
            x_transformed = X@R_matrix
            
            # Rename columns of x_transformed as in original dataframe
            x_transformed = self._rename_columns_as_in_original_dataset(x_transformed)
            
            # Fit the base_estimator on transformered dataset
            model = self.base_estimator
            model.fit(x_transformed, y)
            
            # Append model to the models list
            models.append(model)
            # Append rotation matrix to the list
            rotation_matrices.append(R_matrix)
        return models, rotation_matrices, feature_subspaces
         
    
            
    def fit(self, X, y, k):
        self.column_names = X.columns
        self.models, self.rotation_matrices, self.feature_subspaces = self._build_rotation_tree_model(X, y, k)
        
    def predict(self, X):
        '''
        Params
        :param X: pd.Series, test labels
        ________________________________
        :return final_pred: pd.Series, predictions
        '''
        predict_vectors = []
        classification_list = []
        
        for i,model in enumerate(self.models):
            x_mod = X@self.rotation_matrices[i]
            
            # Rename columns of x_transformed as in original dataframe 
            x_mod = self._rename_columns_as_in_original_dataset(x_mod)
            
            y_pred = model.predict(x_mod)
            predict_vectors.append(y_pred)
            
        
        predicted_matrix = np.asmatrix(predict_vectors)
        final_pred = []
        
        for i in range(len(X)):
            pred_vector_for_one_sample = np.ravel(predicted_matrix[:, i])
            non_zero_pred = np.nonzero(pred_vector_for_one_sample)[0]
            
            is_one = len(non_zero_pred) > len(self.models)/2
            final_pred.append(int(is_one))
        
        # converting from list to ps.Series format 
        final_pred = pd.Series(data=np.array(final_pred), index=X.index.tolist())
        
        
        return final_pred
