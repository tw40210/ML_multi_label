import numpy as np
import pandas as pd

import sys
import datetime
import time
import math

import lightgbm as lgb
import optuna.integration.lightgbm as lgbo

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, PowerTransformer, QuantileTransformer
le = preprocessing.LabelEncoder()
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error # 平均絶対誤差
from sklearn.metrics import mean_squared_error # 平均二乗誤差
from sklearn.metrics import mean_squared_log_error # 対数平均二乗誤差
from sklearn.metrics import r2_score # 決定係数
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from core.models.NN_model import NNModel
sns.set()

import missingno as msno
import plotly.express as px

import warnings
warnings.filterwarnings("ignore")


# Pandas setting to display more dataset rows and columns
pd.set_option('display.max_rows', 150)
pd.set_option('display.max_columns', 600)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

import core

sample_submission = pd.read_csv(core.REPO_PATH + '/data/Software_Defects/sample_submission.csv')
train = pd.read_csv(core.REPO_PATH + '/data/Software_Defects/train.csv')
test = pd.read_csv(core.REPO_PATH + '/data/Software_Defects/test.csv')
origin = pd.read_csv(core.REPO_PATH + '/data/Software_Defects/jm1.csv')


# train に含まれる特徴量を確認
# Check features included in train

feat_train = train.columns.drop('id').tolist()
feat_test = test.columns.drop('id').tolist()

origin['uniq_Op'][origin['uniq_Op'] == '?'] = np.nan
origin['uniq_Opnd'][origin['uniq_Opnd'] == '?'] = np.nan
origin['total_Op'][origin['total_Op'] == '?'] = np.nan
origin['total_Opnd'][origin['total_Opnd'] == '?'] = np.nan
origin['branchCount'][origin['branchCount'] == '?'] = np.nan

# 型を変換
# Change the type

origin['uniq_Op'] = origin['uniq_Op'].astype(float)
origin['uniq_Opnd'] = origin['uniq_Opnd'].astype(float)
origin['total_Op'] = origin['total_Op'].astype(float)
origin['total_Opnd'] = origin['total_Opnd'].astype(float)
origin['branchCount'] = origin['branchCount'].astype(float)

# Fill nan with mean
origin['uniq_Op'][origin['uniq_Op'].isna()] = origin['uniq_Op'].mean()
origin['uniq_Opnd'][origin['uniq_Opnd'].isna()] = origin['uniq_Opnd'].mean()
origin['total_Op'][origin['total_Op'].isna()] = origin['total_Op'].mean()
origin['total_Opnd'][origin['total_Opnd'].isna()] = origin['total_Opnd'].mean()
origin['branchCount'][origin['branchCount'].isna()] = origin['branchCount'].mean()

# trainとorigin、testを結合
# Concat train, origin and test

data_t_o = pd.concat([train, origin], ignore_index=True)
data_t_o = data_t_o.drop_duplicates() # 重複データを削除

data_all = pd.concat([data_t_o, test], ignore_index=True)

# データをトレーニング用と予測用に分けます
# Split train and test

train = data_all.loc[data_t_o.index[0]:data_t_o.index[-1]-1973]
test = data_all.loc[data_t_o.index[-1]+1-1973:]

# 型を変換
train['defects'] = train['defects'].astype(int)

# 外れ値のデータを削除
# Remove outliers

train = train.drop(train[train['loc'] > 3000].index)
train = train.drop(train[train['l'] > 1.2].index)
train = train.drop(train[train['e'] > 30000000].index)
train = train.drop(train[train['b'] > 25].index)
train = train.drop(train[train['t'] > 1500000].index)
train = train.drop(train[train['lOCode'] > 2500].index)
train = train.drop(train[train['locCodeAndComment'] > 100].index)
train = train.drop(train[train['total_Op'] > 5000].index)
train = train.drop(train[train['total_Opnd'] > 3000].index)

#----------------------------------------------------
# 基本設定（Base setting）
#----------------------------------------------------
test_size = 0.25
random_state = 0
objective = 'binary'
metric = 'binary_logloss' # binary_logloss, binary_error, auc

#----------------------------------------------------
# for optuna
#----------------------------------------------------
optuna_switch = 'off'
opt_count = 1
num_choose = 1

if opt_count < num_choose:
    num_choose = opt_count

#----------------------------------------------------
# for lightGBM
#----------------------------------------------------
learning_rate = 0.005 # 0.0001
num_iterations = 300000 # 100
max_depth = -1


# 学習用データと検証用データを作成する関数

def make_lgb_data(test_size, random_state, metric, X, value):
    X_train, X_test, t_train, t_test = train_test_split(
        X,
        value,
        test_size=test_size,
        random_state=random_state
    )

    lgb_train = lgb.Dataset(
        X_train,
        t_train
    )

    lgb_eval = lgb.Dataset(
        X_test,
        t_test,
        reference=lgb_train
    )

    dic_return = {
        'X_train': X_train,
        'X_test': X_test,
        't_train': t_train,
        't_test': t_test,
        'lgb_train': lgb_train,
        'lgb_eval': lgb_eval
    }

    return dic_return

def make_nn_data(test_size, random_state, metric, X, value):
    X_train, X_test, t_train, t_test = train_test_split(
        X,
        value,
        test_size=test_size,
        random_state=random_state
    )

    X_train = torch.tensor(X_train.to_numpy())
    X_test = torch.tensor(X_test.to_numpy())
    t_train = torch.tensor(t_train.to_numpy())
    t_test = torch.tensor(t_test.to_numpy())

    nn_train = {
        "data": X_train,
        "labels": t_train
    }
    nn_test = {
        "data": X_test,
        "labels": t_test
    }


    dic_return = {
        'X_train': X_train,
        'X_test': X_test,
        't_train': t_train,
        't_test': t_test,
        'nn_train': nn_train,
        'nn_eval': nn_test
    }

    return dic_return

# optuna

def tuneParams(test_size, random_state, objective, metric, X, value):
    opt_params = {
        "objective": objective,
        "metric": metric
    }

    dic = make_lgb_data(test_size, random_state, metric, X, value)
    lgb_train = dic['lgb_train']
    lgb_eval = dic['lgb_eval']

    opt = lgbo.train(
        opt_params,
        lgb_train,
        valid_sets=lgb_eval,
        verbose_eval=False,
        num_boost_round=10,
        early_stopping_rounds=10
    )

    return opt


# 学習（モデル作成）関数

def make_lgb_model(X, value, test_size, random_state, objective, metric, learning_rate, num_iterations, max_depth,
                   paramObj):
    dic = make_lgb_data(test_size, random_state, metric, X, value)
    lgb_train = dic['lgb_train']
    lgb_eval = dic['lgb_eval']
    X_test = dic['X_test']  # 検証用
    t_test = dic['t_test']  # 〃

    params = {
        'task': 'train',
        'objective': objective,
        'metric': metric,
        'boosting_type': 'gbdt',
        'learning_rate': learning_rate,
        'num_iterations': num_iterations,
        'max_depth': max_depth,
        'feature_pre_filter': paramObj['feature_pre_filter'],
        'lambda_l1': paramObj['lambda_l1'],
        'lambda_l2': paramObj['lambda_l2'],
        'num_leaves': paramObj['num_leaves'],
        'feature_fraction': paramObj['feature_fraction'],
        'bagging_fraction': paramObj['bagging_fraction'],
        'bagging_freq': paramObj['bagging_freq'],
        'min_child_samples': paramObj['min_child_samples'],
        'verbosity': -1
    }

    evaluation_results = {}  # 学習の経過を保存する
    model = lgb.train(
        params,
        valid_names=['train', 'valid'],  # 学習経過で表示する名称
        valid_sets=[lgb_train, lgb_eval],  # モデル検証のデータセット
        train_set=lgb_train,
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, ),
            lgb.record_evaluation(evaluation_results),
            lgb.log_evaluation(100)
        ]
    )

    resultObj = {'paramObj': paramObj,
                 'evaluation_results': evaluation_results,
                 'model': model,
                 'X_test': X_test,  # 検証用
                 't_test': t_test}  # 〃
    return resultObj

def make_nn_model(X, value, test_size, random_state, objective, metric, learning_rate, num_iterations, max_depth,
                   paramObj):
    dic = make_nn_data(test_size, random_state, metric, X, value)
    nn_train = dic['nn_train']
    nn_eval = dic['nn_eval']
    X_test = dic['X_test']  # 検証用
    t_test = dic['t_test']  # 〃

    params = {
        'task': 'train',
        'objective': objective,
        'metric': metric,
        'learning_rate': learning_rate,
        'num_iterations': num_iterations,
        'max_depth': max_depth,
        'lambda_l1': paramObj['lambda_l1'],
        'lambda_l2': paramObj['lambda_l2'],
    }

    evaluation_results = {}  # 学習の経過を保存する
    model = NNModel()
    model = model.train(
        params,
        names={"x":'data', "y":'labels'},  # 学習経過で表示する名称
        valid_sets=[nn_train, nn_eval],  # モデル検証のデータセット
        train_set=nn_train,
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, ),
            lgb.record_evaluation(evaluation_results),
            lgb.log_evaluation(100)
        ]
    )

    resultObj = {'paramObj': paramObj,
                 'evaluation_results': evaluation_results,
                 'model': model,
                 'X_test': X_test,  # 検証用
                 't_test': t_test}  # 〃
    return resultObj

# trainを学習用データセットと検証用データセットに分割

X = train[feat_test]
value = train['defects']

# optuna

if optuna_switch == 'on':
    param_ary = []
    for i in range(opt_count):
        print('=' * 80)
        print(f'【Round : {i + 1}】')
        print('=' * 80)
        opt = tuneParams(test_size, random_state, objective, metric, X, value)
        score = opt.best_score['valid_0'][metric]
        dic = {'score': score, 'params': opt.params}
        param_ary.append(dic)

    if metric == 'auc':
        # スコアの高い順にソート
        param_ary = sorted(param_ary, key=lambda x: x['score'], reverse=True)
    elif metric == 'binary_logloss':
        # スコアの低い順にソート
        param_ary = sorted(param_ary, key=lambda x: x['score'], reverse=False)

if optuna_switch == 'on':
    count = 0
    for dic in param_ary:
        score = dic['score']
        params = dic['params']
        count += 1
        print('')
        print('=' * 100)
        print(f'【Place : {count}】')
        print(f'opt_score : {score}')
        print(f'params : {params}')
        print('=' * 100)
        print('')

if optuna_switch == 'on':
    param_ary = param_ary[0: num_choose]
else:
    # optunaを使わないときはここにパラメーターをセット
    # In case of unusing optuna, set params here
    param_ary = [
        {'objective': 'binary', 'metric': 'binary_logloss', 'feature_pre_filter': False,
         'lambda_l1': 1.6314697848556082e-05, 'lambda_l2': 9.531962710059137e-05, 'num_leaves': 85,
         'feature_fraction': 1.0, 'bagging_fraction': 0.9991169270666491, 'bagging_freq': 6, 'min_child_samples': 50,
         'num_iterations': 10, 'early_stopping_round': None}
    ]

result_ary = []
for i in range(len(param_ary)):
    print()
    print('=' * 80)
    print(f'【Round : {i + 1}】')

    paramObj = param_ary[i]

    print(f'params : {paramObj}')
    print('-' * 80)
    resultObj = make_nn_model(X, value, test_size, random_state, objective, metric, learning_rate, num_iterations,
                              max_depth, paramObj)
    result_ary.append(resultObj)

# for i in range(len(param_ary)):
#     print()
#     print('=' * 80)
#     print(f'【Round : {i + 1}】')
#
#     if optuna_switch == 'on':
#         score = param_ary[i]['score']
#         print(f'opt score : {score}')
#         paramObj = param_ary[i]['params']
#     else:
#         paramObj = param_ary[i]
#
#     print(f'params : {paramObj}')
#     print('-' * 80)
#     resultObj = make_lgb_model(X, value, test_size, random_state, objective, metric, learning_rate, num_iterations,
#                                max_depth, paramObj)
#     result_ary.append(resultObj)




def show_roc_curve(X, model):
    for_verifi = model.predict(X)
    true = train['defects']
    fpr, tpr, thresholds = roc_curve(true, for_verifi)
    # plt.plot(fpr, tpr, marker='.')
    # plt.xlabel('FPR: False positive rate')
    # plt.ylabel('TPR: True positive rate')
    # plt.title('ROC Curve')
    # plt.show()

    return {'for_verifi': for_verifi,
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds}

# 混同行列
# Confusion matrix

def show_cm(for_verifi):
    cm = confusion_matrix(train['defects'], np.round(for_verifi))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt=',')
    plt.xlabel('Prediction')
    plt.ylabel('Result')
    plt.show()

for i in range(len(result_ary)):
    print('-'*80)
    print(f'Round {i+1} / {len(result_ary)}')
    model = result_ary[i]['model']
    val_score = model.best_score['valid'][metric]
    print(f'Valid Score : {val_score}')
    print('-'*80)

    # Rock curve
    obj = show_roc_curve(X, model)

    # Auc
    fpr = obj['fpr']
    tpr = obj['tpr']
    acc = auc(fpr, tpr)
    result_ary[i]['auc'] = acc # result_aryにaucを保存
    print(f'auc : {acc}')
    print()

    # Confusion Matrix
    print('Confusion Matrix')
    for_verify = obj['for_verifi']
    # show_cm(for_verify)

# 学習したモデルで予測
# Predict with learned model

def get_pred(test, feat_test):
    fold = []
    print()
    for i in range(len(result_ary)):
        paramObj = result_ary[i]['paramObj']
        model = result_ary[i]['model']
        auc = result_ary[i]['auc']
        if auc > 0.5:
            print(f'params : {paramObj}')
            print(f'auc : {auc}')
            print()
            result = model.predict(test[feat_test])
            fold.append(result)
    print('-'*80)
    print(f'Result : {fold}')
    print('-'*80)
    return fold

fold = get_pred(test, feat_test)
# 複数のモデルで予測した結果をアンサンブル
# Ensemble predicted results

df_result = pd.DataFrame(fold).transpose().mean(axis=1)

sample_submission['defects'] = df_result
sample_submission.to_csv("submission.csv",index=False)
