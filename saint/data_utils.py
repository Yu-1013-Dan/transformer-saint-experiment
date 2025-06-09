import openml
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from torch.utils.data import Dataset
from saint.models.layers import compute_bins
import warnings

# Suppress pandas warnings for cleaner output
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

def simple_lapsed_time(text, lapsed):
    hours, rem = divmod(lapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    print(text+": {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))


def task_dset_ids(task):
    dataset_ids = {
        'binary': [1487,44,1590,42178,1111,31,42733,1494,1017,4134],
        'multiclass': [188, 1596, 4541, 40664, 40685, 40687, 40975, 41166, 41169, 42734],
        'regression':[541, 42726, 42727, 422, 42571, 42705, 42728, 42563, 42724, 42729]
    }
    return dataset_ids[task]

def concat_data(X,y):
    return pd.concat([pd.DataFrame(X['data']), pd.DataFrame(y['data'][:,0].tolist(),columns=['target'])], axis=1)


def data_split(X,y,nan_mask,indices):
    x_d = {
        'data': X.values[indices],
        'mask': nan_mask.values[indices]
    }
    
    if x_d['data'].shape != x_d['mask'].shape:
        raise ValueError('Shape of data not same as that of nan mask!')
        
    y_d = {
        'data': y[indices].reshape(-1, 1)
    } 
    return x_d, y_d


def data_prep_custom(X, y, cat_cols, con_cols, task, datasplit=[.65, .15, .2]):
    """
    Improved data preparation function with better pandas handling
    """
    # Create a deep copy to avoid any warnings
    X = X.copy()
    
    cat_idxs = [X.columns.get_loc(col) for col in cat_cols]
    con_idxs = [X.columns.get_loc(col) for col in con_cols]

    # Convert categorical columns to object type safely
    X_new = X.copy()
    for col in cat_cols:
        X_new[col] = X_new[col].astype("object")
    X = X_new

    # Add split column
    X["Set"] = np.random.choice(["train", "valid", "test"], p = datasplit, size=(X.shape[0],))

    train_indices = X[X.Set=="train"].index
    valid_indices = X[X.Set=="valid"].index
    test_indices = X[X.Set=="test"].index

    X = X.drop(columns=['Set'])
    
    # Create missing value mask before any modifications
    nan_mask = X.notna().astype(int)
    
    cat_dims = []
    # Process categorical columns
    for col in cat_cols:
        X_col = X[col].fillna("MissingValue")
        l_enc = LabelEncoder() 
        X[col] = l_enc.fit_transform(X_col.values)
        cat_dims.append(len(l_enc.classes_))
    
    # Process continuous columns
    for col in con_cols:
        mean_val = X.loc[train_indices, col].mean()
        X[col] = X[col].fillna(mean_val)

    # Handle target variable
    if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
        y = y.values
    if task != 'regression':
        l_enc = LabelEncoder() 
        y = l_enc.fit_transform(y)

    X_train, y_train = data_split(X, y, nan_mask, train_indices)
    X_valid, y_valid = data_split(X, y, nan_mask, valid_indices)
    X_test, y_test = data_split(X, y, nan_mask, test_indices)

    train_mean = np.array(X_train['data'][:,con_idxs], dtype=np.float32).mean(0)
    train_std = np.array(X_train['data'][:,con_idxs], dtype=np.float32).std(0)
    train_std = np.where(train_std < 1e-6, 1e-6, train_std)
    
    # Compute bins for PLE
    X_train_tensor = torch.from_numpy(X_train['data'][:, con_idxs].astype(np.float32))
    y_train_tensor = torch.from_numpy(y_train['data'].flatten())
    bins = compute_bins(X_train_tensor, n_bins=64, y=y_train_tensor, regression=(task=='regression'))
    
    # Prepend [CLS] token dimension
    cat_dims = [1] + cat_dims
    
    return cat_dims, cat_idxs, con_idxs, X_train, y_train, X_valid, y_valid, X_test, y_test, train_mean, train_std, bins


def data_prep_openml(ds_id, seed, task, datasplit=[.65, .15, .2]):
    np.random.seed(seed) 
    dataset = openml.datasets.get_dataset(ds_id)
    
    X, y, categorical_indicator, attribute_names = dataset.get_data(dataset_format="dataframe", target=dataset.default_target_attribute)
    if ds_id == 42178:
        categorical_indicator = [True, False, True,True,False,True,True,True,True,True,True,True,True,True,True,True,True,False, False]
        tmp = [x if (x != ' ') else '0' for x in X['TotalCharges'].tolist()]
        X['TotalCharges'] = [float(i) for i in tmp ]
        y = y[X.TotalCharges != 0]
        X = X[X.TotalCharges != 0]
        X.reset_index(drop=True, inplace=True)
        print(y.shape, X.shape)
    if ds_id in [42728,42705,42729,42571]:
        X, y = X[:50000], y[:50000]
        X.reset_index(drop=True, inplace=True)
    categorical_columns = X.columns[list(np.where(np.array(categorical_indicator)==True)[0])].tolist()
    cont_columns = list(set(X.columns.tolist()) - set(categorical_columns))
    
    return data_prep_custom(X, y, categorical_columns, cont_columns, task, datasplit)


class DataSetCatCon(Dataset):
    def __init__(self, X, Y, cat_cols,task='clf',continuous_mean_std=None):
        
        cat_cols = list(cat_cols)
        X_mask =  X['mask'].copy()
        X = X['data'].copy()
        con_cols = list(set(np.arange(X.shape[1])) - set(cat_cols))
        self.X1 = X[:,cat_cols].copy().astype(np.int64) #categorical columns
        self.X2 = X[:,con_cols].copy().astype(np.float32) #numerical columns
        self.X1_mask = X_mask[:,cat_cols].copy().astype(np.int64) #categorical columns
        self.X2_mask = X_mask[:,con_cols].copy().astype(np.int64) #numerical columns
        if task == 'clf':
            self.y = Y['data']#.astype(np.float32)
        else:
            self.y = Y['data'].astype(np.float32)
        self.cls = np.zeros((len(self.y),1),dtype=int)
        self.cls_mask = np.ones((len(self.y),1),dtype=int)
        if continuous_mean_std is not None:
            mean, std = continuous_mean_std
            self.X2 = (self.X2 - mean) / std

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        # X1 has categorical data, X2 has continuous
        return np.concatenate((self.cls[idx], self.X1[idx])), self.X2[idx],self.y[idx], np.concatenate((self.cls_mask[idx], self.X1_mask[idx])), self.X2_mask[idx] 