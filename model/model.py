"""
Machine learning example for landuse change modeling in Inner Mongolia

Steps:
1.) Read the data using pandas
2.) change the imbalance data to a balanced data by RandomOverSampler
3.) Use a logistic regression as benchmark
4.) Use Xgboost 
5.) Make a model evaluation
6.) Open the blackbox of the XGBoost using the SHAP library
步骤:
1.) 使用pandas库读取数据
2.) 使用RandomOverSampler方法将不平衡数据转换为平衡数据
3.) 使用逻辑回归模型作为性能基准
4.) 使用XGBoost模型进行训练
5.) 对模型进行性能评估
6.) 使用SHAP库来解释XGBoost这个“黑箱模型”
Batunacun 24.06.2020 GPLv3
"""

import pandas as pd
import numpy as np
import pylab as plt
from xgboost import plot_importance
from xgboost import plot_tree
from xgboost import XGBClassifier
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, precision_score
from sklearn.model_selection import train_test_split 
from random import sample
from collections import Counter
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from imblearn.over_sampling import RandomOverSampler

#=============CHANGE LD data===================
# adapt this path according your file structure
# 定义数据文件所在的相对路径
path='../data/'
# 使用pandas读取CSV文件，sep=';'表示文件中的数据是用分号分隔的
df=pd.read_csv(path+'data.csv', sep=';')

# original order, do not move
# 定义一个列表，包含所有用作模型输入的特征（驱动因素）的列名
inputs=['water','desertPolicy','gdp','road','slope','dem','urban','aspect',
        'sheep','mine','disforest','temp','rural','discrop','pop','prec',
        'disunused','dismode','disdens','dissparse']
         
# prepare data from sample data for machine learning
# --- 定义函数，用于从原始数据中提取输入特征(X) ---
# 定义一个名为gen_X的函数，参数inp是包含特征名称的列表
def gen_X(inp):
    """ 
    uses inp to select the data from df
    这个函数的作用是根据inp列表中的列名，从全局变量df中选择数据
    """
    global df  # df sample data # 声明函数将使用全局变量df（即上面读取的数据表）
    # 创建一个全为0的numpy数组X，行数与df相同，列数等于特征数量
    X=np.zeros((df.shape[0],len(inp))) # store the training data from sample 
    k=0 # count the names # 初始化一个计数器k，用作X数组的列索引
    for name in inp:  # 遍历inp列表中的每一个特征名称
        # 从df中提取名为'name'的列的所有数据，并存入X数组的第k列
        X[:,k]=df[name].values #sampling dataset
        k+=1 # 计数器加1，为下一列做准备
    # 返回整理好的、只包含输入特征的numpy数组X
    return X # return the sample point drivers

#generate Y data
# --- 定义函数，用于从原始数据中提取目标变量(Y) ---
# 定义一个名为gen_Y的函数，参数sel是目标变量的列名
def gen_Y(sel):
    global df # df, sample data # 声明函数将使用全局变量df
    Y=np.zeros((df.shape[0])) # 创建一个一维的全0数组Y，长度与df的行数相同
    # 从df中提取名为'sel'的列的所有数据，并存入Y数组
    Y[:]=df[sel].values # df include all sample data
    return Y # 返回只包含目标变量的numpy数组Y

# evaluate the trained models
# --- 定义函数，用于评估训练好的模型性能 ---
 # 定义score函数，参数为模型、训练集和测试集
def score(model,Xtrain,ytrain,Xtest,ytest):
    """ uses the model to calculate a set of scores """
    """ 这个函数使用训练好的模型来计算一系列性能分数 """
    # 打印模型在测试集上的默认得分（通常是准确率）
    print('Testing score     :',model.score(Xtest,Ytest))

    # 打印模型在训练集上的默认得分
    print('Training score    :',model.score(Xtrain,Ytrain))
    # 使用模型对测试集Xtest进行预测，得到预测结果Ym
    Ym=model.predict(Xtest)
    # accuracy_score
    # 计算并打印预测结果的准确率
    print('Testing score1    :',accuracy_score(Ytest,Ym))
    # 计算并打印预测结果的精确率
    print('Testing precision :', precision_score(Ytest,Ym))
    # 计算并打印预测结果的召回率
    print('Testing recall    :', recall_score(Ytest,Ym))
    # 计算并打印Kappa系数
    print('kappa             :',cohen_kappa_score(Ytest,Ym))
    # 计算并打印F1分数，%f用于格式化浮点数输出
    print('F1 score          : %f' % f1_score(Ytest,Ym))
    # ROC AUC
    # --- ROC AUC 计算 ---
    # 预测每个测试样本属于各个类别的概率
    Yp=model.predict_proba(Xtest)
    # 只保留属于正类（类别为1）的概率
    probs = Yp[:, 1]  # keep probabilities for the positive outcome only
    # 计算并打印ROC曲线下面积（AUC）
    print('ROC AUC           : %f' % roc_auc_score(Ytest, probs))
    #print auc of precision recall curve
    # --- PR曲线下面积 计算 ---
    # 获取计算PR曲线所需的精确率、召回率和阈值
    precision, recall, thresholds = precision_recall_curve(Ytest, probs)
    # calculate precision-recall AUC
    # 计算并打印PR曲线下面积
    print('area under PR     : %f' % auc(recall, precision))
# --- 执行数据提取 ---
#==========CHANGE different period=======
name='gdnew'  # 设置目标变量的列名为'gdnew'
#name='ldlrChange' # alternattive training fit
x=gen_X(inputs)  # 调用gen_X函数，准备输入特征数据x
y=gen_Y(name)# 'ldlr7500' is one filed in df, # 调用gen_Y函数，准备目标变量数据y

print('========================= Oversampling ============================')
# ======================= 3. 处理数据不平衡问题 =======================
# over sample quickly, increase value 1
# 快速进行过采样，增加类别为1的样本数量
# 打印过采样之前，目标变量y中各类别的样本数量
print('items before over sampling:',sorted(Counter(y).items()))
# 创建一个随机过采样器的实例，random_state=0保证每次结果可复现
ros = RandomOverSampler(random_state=0)
# 对特征x和标签y执行过采样操作，返回新的平衡后的数据X和Y
X, Y = ros.fit_resample(x, y)
# 打印过采样之后，新目标变量Y中各类别的样本数量
print('items after over sampling:',sorted(Counter(Y).items()))

# ======================= 4. 划分数据集并训练模型 =======================
# split train and test data # --- 划分训练集和测试集 ---
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.33)
# train the logistic regression
# --- 训练逻辑回归模型 (作为基准) ---
# 创建一个逻辑回归模型实例，并设置参数
lg = LogisticRegression(C=0.1,penalty='l2',max_iter=5000,
                           solver="lbfgs",multi_class='multinomial', 
                           random_state=0,)
# 使用训练集数据(Xtrain, Ytrain)来训练逻辑回归模型
lg.fit(Xtrain,Ytrain)

# score the model
# --- 评估逻辑回归模型 ---
print('====================== Logistic Regression ========================')
score(lg,Xtrain,Ytrain,Xtest,Ytest) # 调用score函数，评估逻辑回归模型的性能

# --- 训练XGBoost模型 ---
# 定义一个字典，包含了经过调优的XGBoost模型参数
# the parameter after tunning 
basicparameter={'base_score':0.5,
                'booster':'gbtree',
                'objective':'binary:logistic',
                'scale_pos_weight':1,
                'max_delta_step':5,
                'n_jobs':1,
                'random_state':0, 
                'max_depth':5,
                'min_child_weight':3,
                'n_estimators':300,                
                'subsample':1.0,#0.9,
                'colsample_bytree':0.5,
                'reg_lambda':10,
                'reg_alpha':0.1,
                'learning_rate':0.01,
                'gamma':0.1}                

# 创建一个XGBoost分类器实例，**basicparameter将字典中的参数解包传入
xgb=XGBClassifier(**basicparameter)
# xgb=XGBClassifier() # apply the default parameters 如果使用它，则会采用XGBoost的默认参数
xgb.fit(Xtrain,Ytrain)  # 使用训练集数据(Xtrain, Ytrain)来训练XGBoost模型

# --- 评估XGBoost模型 ---
#  score the model
print('============================= XGBoost =============================')
score(xgb,Xtrain,Ytrain,Xtest,Ytest) # 调用score函数，评估XGBoost模型的性能

# ======================= 5. 模型解释 (SHAP) =======================
print('============================== SHAP ===============================')

# 针对树模型(如XGBoost)创建一个SHAP解释器
explainer = shap.TreeExplainer(xgb)   # define the explainer
# 使用全部数据X来计算每个特征对每个样本预测结果的SHAP值
shap_values = explainer.shap_values(X)  # use all data for analysis

# --- 定义一个辅助函数，将numpy数组转为带列名的DataFrame，方便SHAP绘图 ---
def gen_data(inputs,X): # 定义gen_data函数
    """ creates a data Frame with inputs and X for statistics with shap """
    """ 这个函数的作用是为特征数据X创建一个带有列名(inputs)的DataFrame """
    df1=pd.DataFrame()  # 创建一个空的pandas DataFrame
    for i,name in enumerate(inputs):  # 遍历特征名列表inputs，同时获得索引i和名称name
        df1[name]=X[:,i]  # 将numpy数组X的第i列数据，添加到DataFrame中，并以name为列名
    return df1  # 返回创建好的DataFrame
df1=gen_data(inputs,X)  # 调用该函数，将特征数据X转换成带列名的DataFrame df1

# --- 绘制SHAP摘要图 ---
# 绘制SHAP摘要图，直观展示各特征的重要性及对预测结果的影响方向
shap.summary_plot(shap_values, df1)
#画出条形图形式的SHAP特征重要性排序
shap.summary_plot(shap_values, df1, plot_type="bar")
