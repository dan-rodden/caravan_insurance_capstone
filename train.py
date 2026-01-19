#!/usr/bin/env python
# coding: utf-8
"""
Training script for Caravan Insurance Customer Prediction Model

This script trains an XGBoost classifier to predict potential caravan insurance customers.
The model and DictVectorizer are saved as a pickle file for deployment.

Usage:
    python train.py

Output:
    xgb_model_eta=0.05_depth=3_min-child=6_v{version}.bin
"""

import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
import xgboost as xgb

# Configuration
DATA_PATH = "data/caravan-insurance-challenge.csv"  # Update path as needed
MODEL_VERSION = "0.0"
RANDOM_STATE = 1

# XGBoost parameters (tuned values from notebook)
XGB_PARAMS = {
    'eta': 0.05,
    'max_depth': 3,
    'min_child_weight': 6,
    'objective': 'binary:logistic',
    'eval_metric': ['auc', 'aucpr'],
    'seed': RANDOM_STATE
}
NUM_BOOST_ROUNDS = 200


def load_data(filepath: str) -> pd.DataFrame:
    """Load and prepare the dataset."""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Remove ORIGIN column if present (used only for train/test split in original dataset)
    if 'ORIGIN' in df.columns:
        df = df[df['ORIGIN'] == 'train'].copy()
        del df['ORIGIN']
    
    # Convert MOSTYPE and MOSHOOFD to categorical
    df['MOSTYPE'] = df['MOSTYPE'].astype('category')
    df['MOSHOOFD'] = df['MOSHOOFD'].astype('category')
    
    print(f"Loaded {len(df)} records with {len(df.columns)} features")
    return df


def prepare_data(df: pd.DataFrame):
    """Split data and prepare features."""
    print("Preparing train/validation/test splits...")
    
    # Stratified split to maintain class balance
    df_full_train, df_test = train_test_split(
        df, test_size=0.2, random_state=RANDOM_STATE, stratify=df['CARAVAN']
    )
    df_train, df_val = train_test_split(
        df_full_train, test_size=0.25, random_state=RANDOM_STATE, stratify=df_full_train['CARAVAN']
    )
    
    # Reset indices
    df_full_train = df_full_train.reset_index(drop=True)
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    
    # Extract target variable
    y_full_train = df_full_train['CARAVAN'].values
    y_train = df_train['CARAVAN'].values
    y_val = df_val['CARAVAN'].values
    y_test = df_test['CARAVAN'].values
    
    # Remove target from features
    del df_full_train['CARAVAN']
    del df_train['CARAVAN']
    del df_val['CARAVAN']
    del df_test['CARAVAN']
    
    print(f"Training set: {len(df_train)} samples")
    print(f"Validation set: {len(df_val)} samples")
    print(f"Test set: {len(df_test)} samples")
    print(f"Full training set: {len(df_full_train)} samples")
    
    return (df_full_train, df_train, df_val, df_test, 
            y_full_train, y_train, y_val, y_test)


def train_model(df_train, df_val, y_train, y_val, params, num_rounds):
    """Train XGBoost model with validation monitoring."""
    print("\nTraining XGBoost model...")
    
    # Convert to dictionaries for DictVectorizer
    dv = DictVectorizer(sparse=False)
    
    train_dict = df_train.to_dict(orient='records')
    val_dict = df_val.to_dict(orient='records')
    
    X_train = dv.fit_transform(train_dict)
    X_val = dv.transform(val_dict)
    
    # Create DMatrix objects
    dtrain = xgb.DMatrix(X_train, label=y_train, 
                         feature_names=list(dv.get_feature_names_out()))
    dval = xgb.DMatrix(X_val, label=y_val, 
                       feature_names=list(dv.get_feature_names_out()))
    
    # Training with validation monitoring
    watchlist = [(dtrain, 'train'), (dval, 'validation')]
    
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_rounds,
        verbose_eval=50,
        evals=watchlist
    )
    
    return model, dv


def train_final_model(df_full_train, y_full_train, dv, params, num_rounds):
    """Train final model on full training data."""
    print("\nTraining final model on full training data...")
    
    # Transform using existing DictVectorizer
    full_train_dict = df_full_train.to_dict(orient='records')
    X_full_train = dv.transform(full_train_dict)
    
    # Create DMatrix
    dfull_train = xgb.DMatrix(X_full_train, label=y_full_train,
                              feature_names=list(dv.get_feature_names_out()))
    
    # Train final model
    model = xgb.train(
        params=params,
        dtrain=dfull_train,
        num_boost_round=num_rounds,
        verbose_eval=50
    )
    
    return model


def evaluate_model(model, dv, df_test, y_test):
    """Evaluate model on test set."""
    from sklearn.metrics import roc_auc_score, average_precision_score
    
    print("\nEvaluating model on test set...")
    
    test_dict = df_test.to_dict(orient='records')
    X_test = dv.transform(test_dict)
    
    dtest = xgb.DMatrix(X_test, feature_names=list(dv.get_feature_names_out()))
    y_pred = model.predict(dtest)
    
    roc_auc = roc_auc_score(y_test, y_pred)
    avg_precision = average_precision_score(y_test, y_pred)
    
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print(f"Average Precision Score: {avg_precision:.4f}")
    
    return roc_auc, avg_precision


def save_model(model, dv, params, version):
    """Save model and DictVectorizer to pickle file."""
    eta = params['eta']
    depth = params['max_depth']
    min_child = params['min_child_weight']
    
    output_file = f"xgb_model_eta={eta}_depth={depth}_min-child={min_child}_v{version}.bin"
    
    print(f"\nSaving model to {output_file}...")
    
    with open(output_file, 'wb') as f_out:
        pickle.dump((dv, model), f_out)
    
    print(f"Model saved successfully!")
    return output_file


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("Caravan Insurance Customer Prediction - Model Training")
    print("=" * 60)
    
    # Load data
    df = load_data(DATA_PATH)
    
    # Prepare data splits
    (df_full_train, df_train, df_val, df_test,
     y_full_train, y_train, y_val, y_test) = prepare_data(df)
    
    # Train model with validation
    model, dv = train_model(df_train, df_val, y_train, y_val, 
                            XGB_PARAMS, NUM_BOOST_ROUNDS)
    
    # Evaluate on test set
    evaluate_model(model, dv, df_test, y_test)
    
    # Train final model on full training data
    final_model = train_final_model(df_full_train, y_full_train, dv,
                                    XGB_PARAMS, NUM_BOOST_ROUNDS)
    
    # Save model
    output_file = save_model(final_model, dv, XGB_PARAMS, MODEL_VERSION)
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Model saved to: {output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()