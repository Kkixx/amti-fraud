#导库
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn import preprocessing, neighbors

#导入数据
train=pd.read_csv('train.csv')
#print(train)
test=pd.read_csv("test.csv")
#print(test)
#数据合并，进而数据预处理
data=pd.concat([train,test],axis=0)
#print(data)
#查看数据集摘要信息，由结果可知无缺失值
print(data.info())
#对数据唯一值个数进行处理
for col in data.columns:
    print(col, data[col].nunique())
#对不同类型的特征变量进行不同的处理
cat_columns=data.select_dtypes(include='object').columns   #类型为object的列include写字段类型
print(cat_columns)
#新建两个空列表，用于存放object类型的列以及相对的的unique值
oj_columns=[]
unique_valueo=[]
for col in cat_columns:
    oj_columns.append(col)
    unique_valueo.append(data[col].nunique())
#对含有？的字段进行处理
print(data['property_damage'].value_counts())
data['property_damage'] = data['property_damage'].map({'NO': 0, 'YES': 1, '?': 2})
print(data['property_damage'].value_counts())

data['police_report_available'].value_counts()
data['police_report_available'] = data['police_report_available'].map({'NO': 0, 'YES': 1, '?': 2})
data['police_report_available'].value_counts()

#对日期特征进行处理
# policy_bind_date, incident_date
data['policy_bind_date'] = pd.to_datetime(data['policy_bind_date'])   #将字符串转换为时间序列数据
data['incident_date'] = pd.to_datetime(data['incident_date'])
#查询日期中的最大最小值，用于进行编码
data['policy_bind_date'].min() # 1990-01-08
data['policy_bind_date'].max() # 2015-02-22
data['incident_date'].min() # 2015-01-01
data['incident_date'].max() # 2015-03-01
base_date = data['policy_bind_date'].min()#选取最小的日期
# 转换为date_diff,以最小的日期为基准
data['policy_bind_date_diff'] = (data['policy_bind_date'] - base_date).dt.days#计算相差天数
data['incident_date_diff'] = (data['incident_date'] - base_date).dt.days  #相差天数
data['incident_date_policy_bind_date_diff'] = data['incident_date_diff'] - data['policy_bind_date_diff']

# 去掉原始日期字段 policy_bind_date	incident_date
data.drop(['policy_bind_date', 'incident_date'], axis=1, inplace=True)
#print(data)
#去掉id，太有特征性
data.drop(['policy_id'],axis=1,inplace=True)
#特征编码
cat_columns=data.select_dtypes(include='object').columns
for col in cat_columns:
    enc=preprocessing.LabelEncoder()   #获取一个LabelEncoder
    data[col]=enc.fit_transform(data[col])
print(data.columns)
# 数据集切分
trainl = data[data['fraud'].notnull()]
#将训练集划分为训练集训练集和验证集,造成概率降低，欠拟合
#X_train,X_test,y_train,Y_test=train_test_split(trainl.iloc[:,:-1],trainl.iloc[:,-1],test_size=0.3,random_state=0)
testl = data[data['fraud'].isnull()]

#检查目标变量的分布，类别分类不平衡
print(trainl['fraud'].value_counts())

model_lgb = lgb.LGBMClassifier(
            num_leaves=2**5-1, reg_alpha=0.25, reg_lambda=0.25, objective='binary',
            max_depth=-1, learning_rate=0.005, min_child_samples=3, random_state=2022,
            n_estimators=2000, subsample=1, colsample_bytree=1,
        )
# 模型训练
model_lgb.fit(trainl.drop(['fraud'], axis=1), trainl['fraud'])
# AUC评测： 以proba进行提交，结果会更好
y_pred = model_lgb.predict_proba(testl.drop(['fraud'], axis=1))
print(y_pred)

result = pd.read_csv('submission.csv')
result['fraud'] = y_pred[:, 1]
result.to_csv('baseline.csv', index=False)
'''
# 过采样  概率降低0.0009
X=trainl.iloc[:,:-1]
y=trainl.iloc[:,-1]
def up_sample_data(df, percent=0.4):
    
    percent:少数类别样本数量的重采样的比例，可控制，一般不超过0.5，以免过拟合
    
    data1 = df[df['fraud'] == 0]  # 将多数类别的样本放在data1
    data0 = df[df['fraud'] == 1]  # 将少数类别的样本放在data0
    index = np.random.randint(
        len(data0), size=int(percent * (len(df) - len(data0))))  # 随机给定上采样取出样本的序号
    up_data0 = data0.iloc[list(index)]  # 上采样
    return (pd.concat([up_data0, data1]))


np.random.seed(28)
up_train_com=up_sample_data(trainl, percent=0.4)#上采样训练集
print(up_train_com['fraud'].value_counts())#变成5：2 减少样本不平衡带来的影响


model_lgb.fit(up_train_com.drop(['fraud'], axis=1), up_train_com['fraud'])
# AUC评测： 以proba进行提交，结果会更好
y_pred = model_lgb.predict_proba(testl.drop(['fraud'], axis=1))
print(y_pred)

result = pd.read_csv('submission.csv')
result['fraud'] = y_pred[:, 1]
result.to_csv('baseline.csv', index=False)
'''



'''
# 过采样 ADASYN  报错
X=trainl.iloc[:,:-1]
y=trainl.iloc[:,-1]

adasyn = ADASYN(random_state=10)

X_train_res, y_train_res = adasyn.fit_resample(X, y)
'''