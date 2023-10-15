import optuna
from optuna.samplers import TPESampler

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn.pipeline import Pipeline

# Models
from xgboost import XGBClassifier
from sklearn.multioutput import MultiOutputClassifier
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold, RepeatedMultilabelStratifiedKFold
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, GradientBoostingClassifier

import core
import warnings
from pathlib import Path
from core.pre_process.data_processor import DataProcessor

warnings.filterwarnings("ignore")
import os
import pandas as pd
import numpy as np

file_names = []
directory = Path(core.DATA_PATH)

for dirpath, dirnames, filenames in os.walk(directory):
    for filename in filenames:
        file_path = os.path.join(dirpath, filename)
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        globals()[file_name] = pd.read_csv(file_path)
        # print(file_name)

train = pd.read_csv(Path(core.DATA_PATH)/"Enzyme_Substrate/train.csv")
test = pd.read_csv(Path(core.DATA_PATH)/"Enzyme_Substrate/test.csv")
mixed_desc = pd.read_csv(Path(core.DATA_PATH)/"Enzyme_Substrate/mixed_desc.csv")
sample_submission = pd.read_csv(Path(core.DATA_PATH)/"Enzyme_Substrate/sample_submission.csv")



train.drop(columns=["id"], inplace=True)
test.drop(columns=["id"], inplace=True)
mixed_desc.drop(columns=["CIDs"], inplace=True)
col = "EC1_EC2_EC3_EC4_EC5_EC6"

mixed_desc[col.split("_")] = mixed_desc[col].str.split('_', expand=True).astype(int)
mixed_desc.drop(col, axis=1, inplace=True)

original = mixed_desc[train.columns]

train = pd.concat([train, original]).reset_index(drop=True)
train.drop(columns=col.split("_")[2:], inplace=True)


print(train.head())

from sklearn.mixture import GaussianMixture


def get_gmm_class_feature(feat, n):
    gmm = GaussianMixture(n_components=n, random_state=42)
    gmm.fit(train[feat].values.reshape(-1, 1))
    train[f'{feat}_class'] = gmm.predict(train[feat].values.reshape(-1, 1))
    test[f'{feat}_class'] = gmm.predict(test[feat].values.reshape(-1, 1))


get_gmm_class_feature("BertzCT", 4)
get_gmm_class_feature("Chi1", 4)
get_gmm_class_feature("Chi1n", 3)
get_gmm_class_feature("Chi1v", 3)
get_gmm_class_feature("Chi2v", 4)
get_gmm_class_feature("Chi3v", 3)
get_gmm_class_feature("Chi4n", 3)
get_gmm_class_feature("EState_VSA1", 2)
get_gmm_class_feature("EState_VSA2", 4)
get_gmm_class_feature("ExactMolWt", 3)
get_gmm_class_feature("FpDensityMorgan1", 3)
get_gmm_class_feature("FpDensityMorgan2", 3)
get_gmm_class_feature("FpDensityMorgan3", 3)
get_gmm_class_feature("HallKierAlpha", 4)
get_gmm_class_feature("HeavyAtomMolWt", 3)
get_gmm_class_feature("Kappa3", 1)
get_gmm_class_feature("MaxAbsEStateIndex", 3)
get_gmm_class_feature("MinEStateIndex", 2)
get_gmm_class_feature("NumHeteroatoms", 3)
get_gmm_class_feature("PEOE_VSA10", 3)
get_gmm_class_feature("PEOE_VSA14", 4)
get_gmm_class_feature("PEOE_VSA6", 4)
get_gmm_class_feature("PEOE_VSA7", 4)
get_gmm_class_feature("PEOE_VSA8", 6)
get_gmm_class_feature("SMR_VSA10", 2)
get_gmm_class_feature("SMR_VSA5", 3)
get_gmm_class_feature("SlogP_VSA3", 3)
get_gmm_class_feature("VSA_EState9", 3)




num=['BertzCT', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2n', 'Chi2v', 'Chi3v', 'Chi4n',
       'EState_VSA1', 'EState_VSA2', 'ExactMolWt', 'FpDensityMorgan1',
       'FpDensityMorgan2', 'FpDensityMorgan3', 'HallKierAlpha',
       'HeavyAtomMolWt', 'Kappa3', 'MaxAbsEStateIndex', 'MinEStateIndex',
        'PEOE_VSA10', 'PEOE_VSA14', 'PEOE_VSA6', 'PEOE_VSA7',
       'PEOE_VSA8', 'SMR_VSA10', 'SMR_VSA5', 'SlogP_VSA3', 'VSA_EState9']

train['sum']=train[num].sum(axis=1)
train['mean']=train[num].mean(axis=1)
train['min']=train[num].min(axis=1)
train['max']=train[num].max(axis=1)
train['std']=train[num].std(axis=1)
train['var']=train[num].var(axis=1)

test['sum']=test[num].sum(axis=1)
test['mean']=test[num].mean(axis=1)
test['min']=test[num].min(axis=1)
test['max']=test[num].max(axis=1)
test['std']=test[num].std(axis=1)
test['var']=test[num].var(axis=1)


def divide_with_check(a,b):
    result = np.where(b != 0, np.divide(a, b), 0)
    return result

def fe(df):
    df['BertzCT_MaxAbsEStateIndex_Ratio']= df['BertzCT'] / (df['MaxAbsEStateIndex'] + 1e-12)
    df['BertzCT_ExactMolWt_Product']= df['BertzCT'] * df['ExactMolWt']
    df['NumHeteroatoms_FpDensityMorgan1_Ratio']= df['NumHeteroatoms'] / (df['FpDensityMorgan1'] + 1e-12)
    df['VSA_EState9_EState_VSA1_Ratio']= df['VSA_EState9'] / (df['EState_VSA1'] + 1e-12)
    df['PEOE_VSA10_SMR_VSA5_Ratio']= df['PEOE_VSA10'] / (df['SMR_VSA5'] + 1e-12)
    df['Chi1v_ExactMolWt_Product']= df['Chi1v'] * df['ExactMolWt']
    df['Chi2v_ExactMolWt_Product']= df['Chi2v'] * df['ExactMolWt']
    df['Chi3v_ExactMolWt_Product']= df['Chi3v'] * df['ExactMolWt']
    df['EState_VSA1_NumHeteroatoms_Product']= df['EState_VSA1'] * df['NumHeteroatoms']
    df['PEOE_VSA10_Chi1_Ratio']= df['PEOE_VSA10'] / (df['Chi1'] + 1e-12)
    df['MaxAbsEStateIndex_NumHeteroatoms_Ratio']= df['MaxAbsEStateIndex'] / (df['NumHeteroatoms'] + 1e-12)
    df['BertzCT_Chi1_Ratio']= df['BertzCT'] / (df['Chi1'] + 1e-12)



fe(train)
fe(test)


def generate_features(train, test, cat_cols, num_cols):
    df = pd.concat([train, test], axis=0, copy=False)
    for c in cat_cols + num_cols:
        df[f'count_{c}'] = df.groupby(c)[c].transform('count')
    for c in cat_cols:
        for n in num_cols:
            df[f'mean_{n}_per_{c}'] = df.groupby(c)[n].transform('median')

    return df.iloc[:len(train), :], df.iloc[len(train):, :]


target_cols = ['EC1', 'EC2']
cols_to_drop = ['id']

features = [c for c in train.columns if c not in target_cols + cols_to_drop]

cat_cols = ['EState_VSA2','HallKierAlpha','NumHeteroatoms','PEOE_VSA10','PEOE_VSA14','PEOE_VSA6',
            'PEOE_VSA7','PEOE_VSA8', 'SMR_VSA10','SMR_VSA5','SlogP_VSA3','fr_COO','fr_COO2']

num_cols = [c for c in features if c not in cat_cols]


X_train = train[features]
Y_train = train[target_cols]
X_test = test[features]

X_train, X_test = generate_features(X_train, X_test, cat_cols, num_cols)

y  = Y_train
X  = X_train

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
# from sklearn.model_selection import RepeatedMultilabelStratifiedKFold
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

# XGBoost classifier parameters
xgb_params = {'n_estimators': 100,
              'tree_method': 'hist',
              'max_depth': 4,
              'reg_alpha': 0.06790740746476749,
              'reg_lambda': 0.03393770327994609,
              'min_child_weight': 1,
              'gamma': 2.5705812096617772e-05,
              'learning_rate': 0.07132617944894756,
              'colsample_bytree': 0.11664298814833247,
              'colsample_bynode': 0.9912092923877247,
              'colsample_bylevel': 0.29178614622079735,
              'subsample': 0.7395301853144935,
              'random_state': 42
              }

# LightGBM classifier parameters
lgbm_params = {'n_estimators': 200,
               'boosting_type': 'gbdt',
               'max_depth': 10,
               'reg_alpha': 6.720380454685094,
               'reg_lambda': 7.074828689930955e-05,
               'min_child_samples': 15,
               'subsample': 0.5182995486972547,
               'learning_rate': 0.027352422199502537,
               'colsample_bytree': 0.2257179878033366,
               'colsample_bynode': 0.7098194984886731,
               'random_state': 84315}

# Define the classifiers
xgb_classifier = MultiOutputClassifier(XGBClassifier(**xgb_params))
lgbm_classifier = MultiOutputClassifier(LGBMClassifier(**lgbm_params))
# GBC_classifier = MultiOutputClassifier(GradientBoostingClassifier(n_estimators=100))

# Create the pipelines
xgb_clf = Pipeline([('classifier', xgb_classifier)])
lgbm_clf = Pipeline([('classifier', lgbm_classifier)])
# GBC_clf = Pipeline([('classifier', GBC_classifier)])

# Initialize variables
oof_preds_xgb = np.zeros(y.shape)
oof_preds_lgbm = np.zeros(y.shape)
# oof_preds_GBC = np.zeros(y.shape)

test_preds_xgb = np.zeros((test.shape[0], y.shape[1]))
test_preds_lgbm = np.zeros((test.shape[0], y.shape[1]))
# test_preds_GBC = np.zeros((test.shape[0], y.shape[1]))

oof_losses_xgb = []
oof_losses_lgbm = []
# oof_losses_GBC = []

n_splits = 10
kf = RepeatedMultilabelStratifiedKFold(n_splits=n_splits, n_repeats=1, random_state=42)
train_losses_xgb = []
train_losses_lgbm = []
train_losses_GBC = []

over_train = []
over_valid = []
# Loop over folds
for fn, (trn_idx, val_idx) in enumerate(kf.split(X, y)):
    print('Starting fold:', fn)
    X_train, X_val = X.iloc[trn_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[trn_idx], y.iloc[val_idx]

    # Train and predict with XGBoost classifier
    xgb_clf.fit(X_train, y_train)
    train_preds_xgb = xgb_clf.predict_proba(X_train)
    train_preds_xgb = np.array(train_preds_xgb)[:, :, 1].T
    # train_loss_xgb = roc_auc_score(np.ravel(y_train), np.ravel(train_preds_xgb))
    # train_losses_xgb.append(train_loss_xgb)

    val_preds_xgb = xgb_clf.predict_proba(X_val)
    val_preds_xgb = np.array(val_preds_xgb)[:, :, 1].T
    oof_preds_xgb[val_idx] = val_preds_xgb
    loss_xgb = roc_auc_score(np.ravel(y_val), np.ravel(val_preds_xgb))
    oof_losses_xgb.append(loss_xgb)
    preds_xgb = xgb_clf.predict_proba(X_test)
    preds_xgb = np.array(preds_xgb)[:, :, 1].T
    test_preds_xgb += preds_xgb / n_splits

    # Train and predict with LightGBM classifier
    lgbm_clf.fit(X_train, y_train)
    train_preds_lgbm = lgbm_clf.predict_proba(X_train)
    train_preds_lgbm = np.array(train_preds_lgbm)[:, :, 1].T
    # train_loss_lgbm = roc_auc_score(np.ravel(y_train), np.ravel(train_preds_lgbm))
    # train_losses_lgbm.append(train_loss_lgbm)

    val_preds_lgbm = lgbm_clf.predict_proba(X_val)
    val_preds_lgbm = np.array(val_preds_lgbm)[:, :, 1].T
    oof_preds_lgbm[val_idx] = val_preds_lgbm

    loss_lgbm = roc_auc_score(np.ravel(y_val), np.ravel(val_preds_lgbm))
    oof_losses_lgbm.append(loss_lgbm)
    preds_lgbm = lgbm_clf.predict_proba(X_test)
    preds_lgbm = np.array(preds_lgbm)[:, :, 1].T
    test_preds_lgbm += preds_lgbm / n_splits
    """
    # Train and predict with GBC classifier
    GBC_clf.fit(X_train, y_train)
    train_preds_GBC = GBC_clf.predict_proba(X_train)
    train_preds_GBC = np.array(train_preds_GBC)[:, :, 1].T
    #train_loss_lgbm = roc_auc_score(np.ravel(y_train), np.ravel(train_preds_lgbm))
    #train_losses_lgbm.append(train_loss_lgbm)

    val_preds_GBC = GBC_clf.predict_proba(X_val)
    val_preds_GBC = np.array(val_preds_GBC)[:, :, 1].T
    oof_preds_GBC[val_idx] = val_preds_GBC

    loss_GBC = roc_auc_score(np.ravel(y_val), np.ravel(val_preds_lgbm))
    oof_losses_GBC.append(loss_GBC)
    preds_GBC = GBC_clf.predict_proba(X_test)
    preds_GBC = np.array(preds_GBC)[:, :, 1].T
    test_preds_GBC += preds_GBC / n_splits
    """

    overall_train_preds = (train_preds_xgb + train_preds_lgbm) / 2
    overall_train_loss = roc_auc_score(np.ravel(y_train), np.ravel(overall_train_preds))
    overall_valid_preds = (val_preds_xgb + val_preds_lgbm) / 2
    overall_valid_loss = roc_auc_score(np.ravel(y_val), np.ravel(overall_valid_preds))
    over_train.append(overall_train_loss)
    over_valid.append(overall_valid_loss)
    print("overall_train", overall_train_loss)
    print("overall_valid", overall_valid_loss)

print("over_train", np.mean(over_train))
print("over_valid", np.mean(over_valid))

sample_submission.iloc[:,1:] = (test_preds_xgb+test_preds_lgbm)/2


sample_submission.to_csv("submission.csv",index=False)
