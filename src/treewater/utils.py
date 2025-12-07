import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Tuple, List, Optional, Union
from dataclasses import dataclass, field
from functools import lru_cache
from sklearn.preprocessing import StandardScaler


@dataclass
class FeatureConfig:
    """Configuration for feature columns"""
    time_varying: List[str] = field(
        default_factory=lambda: ['twd', 'pr', 'at', 'ws', 'dp', 'sr', 'lr']
    )
    time_varying_target: List[str] = field(
        default_factory=lambda: ['twd']
    )
    time_varying_no_target: List[str] = field(
        default_factory=lambda: ['pr', 'at', 'ws', 'dp', 'sr', 'lr']
    )
    static: List[str] = field(
        default_factory=lambda: [
            'mch_elevation', 'mch_easting', 'mch_northing',
            'Carpinus betulus', 'Corylus avellana', 'Fagus sylvatica',
            'Picea abies', 'Pinus sylvestris', 'Pseudotsuga menziesii'
        ]
    )
    cols_to_normalize: List[str] = field(
        default_factory=lambda: ['pr', 'at', 'ws', 'dp', 'sr', 'lr', 
            'mch_elevation', 'mch_easting', 'mch_northing']
    )

def create_training_test_set_optimized(
    df: pd.DataFrame,
    train_size: float = 0.7,
    val_size: float = 0.15,
    feature_window_size: Optional[int] = None,
    autoregressive: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Optimized train/val/test split handling groups efficiently
    """
    # Pre-sort data once
    df = df.sort_values(['site_name', 'species', 'ts'])
    
    # Group data efficiently
    grouped = df.groupby(['site_name', 'species'])
    splits = {'train': [], 'val': [], 'test': []}
    
    for _, group_df in grouped:
        group_df = group_df.reset_index(drop = True)
        n = len(group_df)
        train_end = int(n * train_size)
        val_end = int(n * (train_size + val_size))
        
        # Handle autoregressive window overlap
        val_start = train_end - (feature_window_size if autoregressive else 0)
        
        splits['train'].append(group_df.iloc[:train_end])
        splits['val'].append(group_df.iloc[val_start:val_end])
        splits['test'].append(group_df.iloc[val_end:])


    return tuple(pd.concat(splits[key]) for key in ['train', 'val', 'test'])

# @lru_cache(maxsize=32)
def get_feature_windows(
    df: pd.DataFrame,
    feature_window_size: int,
    label_window_size: int = 1,
    shift: int = 1,
    autoregressive: bool = False,
    config: Optional[FeatureConfig] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates sliding windows efficiently using vectorized operations
    """
    config = config or FeatureConfig()
    features_ts_cols = config.time_varying
    features_ts_cols_wo_twd = config.time_varying_no_target
    features_static_cols = config.static
    label_input = df['twd']
    if autoregressive:
        feature_input = df
        n_feature = (feature_window_size +1) * len(features_ts_cols) + len(features_static_cols) -1  # +1 because we would like to keep the current selection, -1 to exclude twd that we would like to predict

    else:
        feature_input = df.drop(columns=['twd'])
        features_ts_cols = features_ts_cols_wo_twd
        n_feature = (feature_window_size +1) * len(features_ts_cols) + len(features_static_cols)

    n_sample = feature_input.shape[0]
    
    # Create feature windows using stride tricks
    n_windows = n_sample - 2*feature_window_size - label_window_size - shift + 1

    # we also need to make sure location and other static features are only included for once...

    # Initialize arrays
     
    features = np.zeros((n_windows, n_feature))
    labels = np.zeros((n_windows, label_window_size))
    
    for i in range(n_windows):
        # Feature window
        start_idx = i
        end_idx = i + feature_window_size
        features[i] = np.concatenate([feature_input[features_ts_cols].iloc[start_idx:end_idx, ].to_numpy().reshape(-1), 
                                         feature_input[features_ts_cols_wo_twd+features_static_cols].iloc[end_idx, ].to_numpy().reshape(-1)])
        
        # Label window
        label_start = i + feature_window_size + shift - 1
        label_end = label_start + label_window_size
        labels[i] = label_input[label_start:label_end]

    
            
    return features, labels


def get_feature_windows_LSTM(
    df: pd.DataFrame,
    feature_window_size: int,
    label_window_size: int = 1,
    shift: int = 1,
    autoregressive: bool = False,
    config: Optional[FeatureConfig] = None
):  
    

    n_tvt = len(config.time_varying)
    n_other = len(config.time_varying_no_target)
    n_static = len(config.static)


    n_sample = len(df)
    window_len = feature_window_size + 1 
    n_windows = n_sample - feature_window_size - label_window_size - shift + 1
    per_row_cols = config.time_varying + config.time_varying_no_target + config.static
    

    # index of twd inside the time_varying block
    idx_twd_in_tvt = config.time_varying.index("twd")

    if n_windows <= 0:
        # return empty arrays with sensible shapes
        tv_cols = n_tvt * feature_window_size
        return (np.empty((0, tv_cols)), np.empty((0, n_other)), np.empty((0, n_static)), np.empty((0,)))

    # convert to numpy once
    arr = df[per_row_cols].to_numpy(dtype=float)  # shape (n_sample, cols)

    # build sliding windows: shape (n_windows, window_len, cols)
    windows = np.stack([arr[i : i + window_len] for i in range(n_windows)], axis=0)

    # column indices inside per-row block
    idx_tvd = list(range(0, n_tvt))

    if not autoregressive:
        idx_tvd = [i for i in idx_tvd if i != idx_twd_in_tvt ]
    idx_other = list(range(n_tvt, n_tvt + n_other))
    idx_static = list(range(n_tvt + n_other, n_tvt + n_other + n_static))

    start = 0
    end = start + feature_window_size  # exclusive end for slicing past lags; current index = end

    # flattened lagged time-varying features for all windows at this step
    tv_block = windows[:, start:end, :][:, :, idx_tvd]

    # current-day non-target and static features (at index 'end')
    pred_day_other_feats = windows[:, end, :][:, idx_other] if n_other > 0 else np.empty((n_windows, 0))
    static_feats = windows[:, end, :][:, idx_static] if n_static > 0 else np.empty((n_windows, 0))

    # label index inside window to read true or overwrite with prediction
    label_start = start + feature_window_size + shift - 1
    label_end = label_start + label_window_size

    labels = windows[:, label_start, idx_twd_in_tvt]

    return tv_block, pred_day_other_feats, static_feats, labels 


def get_dataset_LSTM(
    df: pd.DataFrame,
    feature_window_size: int,
    label_window_size: int = 1,
    shift: int = 1,
    autoregressive: bool = False,
    # batch_size: int = 64
    config: Optional[FeatureConfig] = None
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:

    config = config or FeatureConfig()
    tv_block_list = []
    pred_day_other_feats_list = []
    static_feats_list = []
    labels_list = []
     
    sites = df['site_name'].unique()
    for site in sites:
        df_site = df[df['site_name'] == site]
        species = df_site['species'].unique()
        
        for sp in species:
            df_sp = df_site[df_site['species'] == sp]\
                .sort_values(by='ts', ascending = True)\
                .drop(["species", "site_name", "ts"], axis = 1).reset_index(drop = True)


            tv_block, pred_day_other_feats, static_feats, labels =  get_feature_windows_LSTM(df_sp, feature_window_size, label_window_size, shift, autoregressive, config)

            tv_block_list.append(tv_block)
            pred_day_other_feats_list.append(pred_day_other_feats)
            static_feats_list.append(static_feats)
            labels_list.append(labels)


            # concatenate or produce empty arrays with correct shapes
    if tv_block_list:
        tv_block_arr = np.concatenate(tv_block_list, axis=0)
        pred_day_other_feats_arr = np.concatenate(pred_day_other_feats_list, axis=0)
        static_feats_arr = np.concatenate(static_feats_list, axis=0)
        labels_arr = np.concatenate(labels_list, axis=0)
    else:
        n_tvt = len(config.time_varying) if autoregressive else len(config.time_varying_no_target)
        n_other = len(config.time_varying_no_target)
        n_static = len(config.static)
        tv_block_arr = np.empty((0, feature_window_size,  n_tvt), dtype=float)
        pred_day_other_feats_arr = np.empty((0, n_other), dtype=float)
        static_feats_arr = np.empty((0, n_static), dtype=float)
        labels_arr = np.empty((0,), dtype=float)

    tv_block_tf = tf.convert_to_tensor(tv_block_arr, dtype=tf.float32)
    pred_day_other_feats_tf = tf.convert_to_tensor(pred_day_other_feats_arr, dtype=tf.float32)
    static_feats_tf = tf.convert_to_tensor(static_feats_arr, dtype=tf.float32)
    labels_tf = tf.convert_to_tensor(labels_arr, dtype=tf.float32)
    
    return tv_block_tf, pred_day_other_feats_tf, static_feats_tf, labels_tf

def spliting_windows_df(df: pd.DataFrame, 
                                        feature_window_size: int ,
                                        label_window_size: int,
                                        shift = 1,
                                        autoregressive: bool =False,
                                        feature_index = None,
                                        config: Optional[FeatureConfig] = None):
    
    
    config = config or FeatureConfig()
    features_list = []
    labels_list = []
     
    sites = df['site_name'].unique()
    for site in sites:
        df_site = df[df['site_name'] == site]
        species = df_site['species'].unique()
        
        for sp in species:
            df_sp = df_site[df_site['species'] == sp]\
                .sort_values(by='ts', ascending = True)\
                .drop(["species", "site_name", "ts"], axis = 1).reset_index(drop = True)
            features, labels =  get_feature_windows(df_sp,
                                                    feature_window_size,
                                                    label_window_size,
                                                    shift,
                                                    autoregressive,
                                                    config)
             

            features_list.append(features)
            labels_list.append(labels)

    # Concatenate all features and labels
    all_features = np.concatenate(features_list, axis=0)
    all_labels = np.concatenate(labels_list, axis=0)

        
    if feature_index is not None:
        all_features = all_features[:, feature_index]                                             
        
    return all_features, all_labels

def create_dataset(
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: Optional[int] = None,
    as_tensor: bool = False
) -> Union[Tuple[np.ndarray, np.ndarray], tf.data.Dataset]:
    """
    Creates final dataset format efficiently
    """
    if as_tensor:
        features = tf.convert_to_tensor(features, dtype=tf.float32)
        labels = tf.convert_to_tensor(labels, dtype=tf.float32)
        
        if batch_size:
            return tf.data.Dataset.from_tensor_slices((features, labels))\
                     .batch(batch_size)\
                     .prefetch(tf.data.AUTOTUNE)
    
    return features, labels

def compute_recursive_predictions(
    model,
    df: pd.DataFrame,
    feature_window_size: int,
    label_window_size: int = 1,
    shift: int = 1,
    config: Optional[FeatureConfig] = None
) -> Tuple[List[float], List[float]]:
    """
    Compute recursive predictions on validation/test set by using
    predicted values as inputs for subsequent predictions
    
    Args:
        model: Trained model with predict method
        df: DataFrame with features and target
        feature_window_size: Size of sliding window for features
        label_window_size: Size of prediction window
        shift: Number of steps to shift between windows
        config: Feature configuration
        
    Returns:
        Tuple of predictions and true values
    """
    config = config or FeatureConfig()
    predictions = []
    true_values = []
    
    # Group by site and species
    for site in df.site_name.unique():
        df_site = df.loc[df['site_name'] == site,]
        species = df_site['species'].unique()   
        for sp in species:
            df_group = df_site[df_site['species'] == sp].sort_values(by='ts', ascending = True).reset_index(drop = True)
            
            # Get features and labels
            feature_input = df_group
            label_input = df_group['twd']
            
            n_sample = len(df_group)
            n_windows = n_sample - 2*feature_window_size - label_window_size - shift + 1
            
            for i in range(n_windows):
                # Get window of data
                window_df = feature_input.iloc[i:i+2*feature_window_size+1, ].reset_index(drop=True).copy()
                window_labels = label_input[i:i+2*feature_window_size+1, ].reset_index(drop=True)
                
                # Recursive prediction for each step in window
                for step in range(0, feature_window_size + 1):
                    # Prepare features
                    start_idx = step
                    end_idx = step + feature_window_size
                    
                    time_varying = window_df[config.time_varying].iloc[start_idx:end_idx].values.reshape(-1)
                    static = window_df[config.static].iloc[end_idx].values
                    other_features = window_df[config.time_varying_no_target].iloc[end_idx].values
                    

                    # features= np.concatenate([window_df[config.time_varying].iloc[start_idx:end_idx, ].to_numpy().reshape(-1), 
                    #                         window_df[config.time_varying_no_target +config.static].iloc[end_idx, ].to_numpy().reshape(-1)])
                    features = np.concatenate([time_varying, other_features, static])
                    
                    label_start = step + feature_window_size + shift - 1
                    label_end = label_start + label_window_size
                    label= window_labels[label_start:label_end]

                    # Make prediction
                    pred = model.predict(features.reshape(1, -1))[0]
                    
                    # Store final prediction or update window
                    if step == feature_window_size:
                        predictions.append(pred)
                        true_values.append(label)
                    else:
                        window_df.at[label_start, 'twd'] = pred
    
    return np.array(predictions), np.array(true_values)



def compute_recursive_predictions_fast(
    model,
    df: pd.DataFrame,
    feature_window_size: int,
    label_window_size: int = 1,
    shift: int = 1,
    config: Optional[FeatureConfig] = None,
    batch_size: int = 64,
    tensor = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized autoregressive recursive predictions.
    - Builds sliding windows per (site, species) once.
    - Predicts in batches: one model.predict call per recursion step across all windows.
    - Updates the target inside the windows with predicted values so subsequent steps use preds.
    Returns (preds, trues) as 1D numpy arrays (matching original function semantics).
    """
    config = config or FeatureConfig()
    preds_all = []
    trues_all = []


    per_row_cols = config.time_varying + config.time_varying_no_target + config.static
    n_tvt = len(config.time_varying)
    n_other = len(config.time_varying_no_target)
    n_static = len(config.static)

    # index of twd inside the time_varying block
    idx_twd_in_tvt = config.time_varying.index("twd")

    for site in df.site_name.unique():
        df_site = df.loc[df['site_name'] == site, :]
        for sp in df_site['species'].unique():
            df_group = (
                df_site[df_site['species'] == sp]
                .sort_values('ts', ascending=True)
                .reset_index(drop=True)
            )
            n_sample = len(df_group)
            window_len = 2 * feature_window_size + 1 
            n_windows = n_sample - 2*feature_window_size - label_window_size - shift + 1
            if n_windows <= 0:
                continue
            # convert to numpy once
            arr = df_group[per_row_cols].to_numpy(dtype=float)  # shape (n_sample, cols)

            # build sliding windows: shape (n_windows, window_len, cols)
            windows = np.stack([arr[i : i + window_len] for i in range(n_windows)], axis=0)

            # column indices inside per-row block
            idx_tvd = list(range(0, n_tvt))
            idx_other = list(range(n_tvt, n_tvt + n_other))
            idx_static = list(range(n_tvt + n_other, n_tvt + n_other + n_static))

            # recursive steps: predict step-by-step, updating windows with preds
            for step in range(0, feature_window_size + 1):
                start = step
                end = start + feature_window_size  # exclusive end for slicing past lags; current index = end

                # flattened lagged time-varying features for all windows at this step
                tv_block = windows[:, start:end, :][:, :, idx_tvd].reshape(n_windows, -1)

                # current-day non-target and static features (at index 'end')
                other_feats = windows[:, end, :][:, idx_other] if n_other > 0 else np.empty((n_windows, 0))
                static_feats = windows[:, end, :][:, idx_static] if n_static > 0 else np.empty((n_windows, 0))

                X_batch = np.concatenate([tv_block, other_feats, static_feats], axis=1)

                # one predict call for all windows at this step
                if tensor:
                    y_batch = model.predict(X_batch, batch_size=batch_size,  verbose=0).reshape(-1)
                else: 
                    y_batch = model.predict(X_batch).reshape(-1)

                # label index inside window to read true or overwrite with prediction
                label_start = step + feature_window_size + shift - 1
                label_end = label_start + label_window_size

                if step == feature_window_size:
                    # final step: collect predictions and true labels
                    true_labels = windows[:, label_start:label_end, idx_twd_in_tvt].reshape(-1)
                    preds_all.append(y_batch)
                    trues_all.append(true_labels)
                else:
                    # update the 'twd' position for all windows with predicted values (autoregressive feed)
                    windows[:, label_start, idx_twd_in_tvt] = y_batch

    if len(preds_all) == 0:
        return np.array([]), np.array([])

    preds = np.concatenate(preds_all, axis=0)
    trues = np.concatenate(trues_all, axis=0)
    return preds, trues


def compute_recursive_predictions_fast_LSTM(
    model,
    df: pd.DataFrame,
    feature_window_size: int,
    label_window_size: int = 1,
    shift: int = 1,
    config: Optional[FeatureConfig] = None,
    batch_size: int = 64
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized autoregressive recursive predictions for an LSTM model.
    The model is expected to accept three inputs: 
      [past_dynamic (N, timesteps, n_tvt), current_day_exog (N, n_other), static (N, n_static)]
    Returns (preds, trues) as 1D numpy arrays.
    """
    config = config or FeatureConfig()
    preds_all = []
    trues_all = []

    per_row_cols = config.time_varying + config.time_varying_no_target + config.static
    n_tvt = len(config.time_varying)
    n_other = len(config.time_varying_no_target)
    n_static = len(config.static)

    # index of twd inside the time_varying block
    idx_twd_in_tvt = config.time_varying.index("twd")

    for site in df.site_name.unique():
        df_site = df.loc[df['site_name'] == site, :]
        for sp in df_site['species'].unique():
            df_group = (
                df_site[df_site['species'] == sp]
                .sort_values('ts', ascending=True)
                .reset_index(drop=True)
            )
            n_sample = len(df_group)
            window_len = 2 * feature_window_size + 1
            n_windows = n_sample - 2*feature_window_size - label_window_size - shift + 1
            if n_windows <= 0:
                continue

            # numpy array (n_sample, cols)
            arr = df_group[per_row_cols].to_numpy(dtype=float)

            # sliding windows: (n_windows, window_len, cols)
            windows = np.stack([arr[i : i + window_len] for i in range(n_windows)], axis=0)

            # column indices inside per-row block
            idx_tvd = list(range(0, n_tvt))
            idx_other = list(range(n_tvt, n_tvt + n_other))
            idx_static = list(range(n_tvt + n_other, n_tvt + n_other + n_static))

            # recursive steps
            for step in range(0, feature_window_size + 1):
                start = step
                end = start + feature_window_size  # exclusive end for slicing past lags; current index = end

                # keep 3D dynamic block: (n_windows, feature_window_size, n_tvt)
                tv_block = windows[:, start:end, :][:, :, idx_tvd]

                # current-day non-target and static features (at index 'end')
                other_feats = windows[:, end, :][:, idx_other] if n_other > 0 else np.empty((n_windows, 0))
                static_feats = windows[:, end, :][:, idx_static] if n_static > 0 else np.empty((n_windows, 0))

                # predict in batch using LSTM model inputs
                y_batch = model.predict([tv_block, other_feats, static_feats], batch_size=batch_size,
                                         verbose=0).reshape(-1)

                # label indices
                label_start = step + feature_window_size + shift - 1
                label_end = label_start + label_window_size

                if step == feature_window_size:
                    true_labels = windows[:, label_start:label_end, idx_twd_in_tvt].reshape(-1)
                    preds_all.append(y_batch)
                    trues_all.append(true_labels)
                else:
                    # update autoregressive twd in all windows for next step
                    windows[:, label_start, idx_twd_in_tvt] = y_batch

    if len(preds_all) == 0:
        return np.array([]), np.array([])

    preds = np.concatenate(preds_all, axis=0)
    trues = np.concatenate(trues_all, axis=0)
    return preds, trues


def standardize_dataset(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: Optional[FeatureConfig] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Standardizes specified columns in train, val, and test DataFrames
    using StandardScaler from sklearn.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        cols_to_normalize: List of column names to standardize
        
    Returns:
        Tuple of standardized (train_df, val_df, test_df) and scaler parameters
    """
    # use default config
    config = FeatureConfig()
    cols_to_normalize = config.cols_to_normalize
    scaler = StandardScaler()
    
    
    # Fit and transform the training data
    train_df[cols_to_normalize] = scaler.fit_transform(train_df[cols_to_normalize])
    
    # Transform validation and test data using the same scaler
    val_df[cols_to_normalize] = scaler.transform(val_df[cols_to_normalize])
    test_df[cols_to_normalize] = scaler.transform(test_df[cols_to_normalize])
    
    # Store scaler parameters for later use if needed
    scaler_params = {
        'mean_': scaler.mean_,
        'scale_': scaler.scale_
    }
    
    return train_df, val_df, test_df


def build_autoregressive_training_data_fast_LSTM(
    model,
    df: pd.DataFrame,
    feature_window_size: int,
    label_window_size: int = 1,
    shift: int = 1,
    config: Optional[FeatureConfig] = None,
    batch_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build training data where the *history* has been partially replaced by the
    model's own predictions (offline autoregressive training).

    For each (site, species), we:
      - build sliding windows of length 2*feature_window_size + 1
      - run recursive prediction across step = 0..feature_window_size
      - at *each* step we extract a training sample:
          X_dyn:  (n_windows, feature_window_size, n_tvt)
          X_day:  (n_windows, n_other)
          X_stat: (n_windows, n_static)
          y_true: (n_windows,) true label returned
        where the dynamic history includes predictions from previous steps.
    """
    config = config or FeatureConfig()

    per_row_cols = config.time_varying + config.time_varying_no_target + config.static
    n_tvt = len(config.time_varying)
    n_other = len(config.time_varying_no_target)
    n_static = len(config.static)

    idx_twd_in_tvt = config.time_varying.index("twd")

    X_dyn_list = []
    X_day_list = []
    X_static_list = []
    y_list = []

    for site in df.site_name.unique():
        df_site = df.loc[df["site_name"] == site, :]
        for sp in df_site["species"].unique():
            df_group = (
                df_site[df_site["species"] == sp]
                .sort_values("ts", ascending=True)
                .reset_index(drop=True)
            )
            n_sample = len(df_group)
            window_len = 2 * feature_window_size + 1
            n_windows = n_sample - 2 * feature_window_size - label_window_size - shift + 1
            if n_windows <= 0:
                continue

            # (n_sample, cols)
            arr = df_group[per_row_cols].to_numpy(dtype=float)

            # (n_windows, window_len, cols)
            windows = np.stack(
                [arr[i : i + window_len] for i in range(n_windows)],
                axis=0,
            )

            # column indices inside per-row block
            idx_tvd = list(range(0, n_tvt))
            idx_other = list(range(n_tvt, n_tvt + n_other))
            idx_static = list(range(n_tvt + n_other, n_tvt + n_other + n_static))

            # recursive steps: update windows in-place with predictions
            for step in range(0, feature_window_size + 1):
                start = step
                end = start + feature_window_size  # history indices [start:end), current day = end

                # dynamic history (n_windows, feature_window_size, n_tvt)
                tv_block = windows[:, start:end, :][:, :, idx_tvd]

                # current-day exog + static at index 'end'
                other_feats = (
                    windows[:, end, :][:, idx_other]
                    if n_other > 0
                    else np.empty((n_windows, 0))
                )
                static_feats = (
                    windows[:, end, :][:, idx_static]
                    if n_static > 0
                    else np.empty((n_windows, 0))
                )

                # label indices inside window
                label_start = step + feature_window_size + shift - 1
                label_end = label_start + label_window_size
                if label_start < 0 or label_end > window_len:
                    # safety guard for weird shift/label_window combos
                    continue

                # true labels for this step (read BEFORE overwriting)
                true_labels = windows[:, label_start:label_end, idx_twd_in_tvt].reshape(-1)



                if step == feature_window_size:
                    X_dyn_list.append(tv_block)
                    X_day_list.append(other_feats)
                    X_static_list.append(static_feats)
                    y_list.append(true_labels)

                # if this isn't the last step, do AR update for the *next* step
                if step < feature_window_size:
                    y_batch = model.predict(
                        [tv_block, other_feats, static_feats],
                        batch_size=batch_size,
                        verbose=0,
                    ).reshape(-1)

                    # overwrite twd with predictions
                    windows[:, label_start, idx_twd_in_tvt] = y_batch

                


    if not X_dyn_list:
        return (
            np.empty((0, feature_window_size, n_tvt)),
            np.empty((0, n_other)),
            np.empty((0, n_static)),
            np.empty((0,)),
        )

    X_dyn_ar = np.concatenate(X_dyn_list, axis=0)
    X_day_ar = np.concatenate(X_day_list, axis=0)
    X_static_ar = np.concatenate(X_static_list, axis=0)
    y_ar = np.concatenate(y_list, axis=0)
    

    return X_dyn_ar, X_day_ar, X_static_ar, y_ar
