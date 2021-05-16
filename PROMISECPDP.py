import os

import numpy as np
from sklearn.preprocessing import StandardScaler
from stacking_classifier import *
from sklearn.metrics import f1_score, roc_auc_score
import pandas as pd

target = [  'camel-1.4.csv'
          ]
source = [
'ant-1.6.csv'
          ]

classifier = StackingClassifier(
	base_classifiers=[
		LightGBMClassifier(),
		CatBoostClassifier(),
		AdaBoostClassifier(),
		StackingClassifier(
			base_classifiers=[
				SimpleMLPClassifer(train_params={'input_num': 20, 'class_num': 2}),  # 比如放这儿
				RandomForestClassifier(),
			],
			base_k_fold=10,  # 基分类器分拆份数,force_cv=True时生效，
			meta_k_fold=10,  # 元分类器分拆份数,force_cv=True时生效，
			meta_classifier=GradientBoostingClassifier(),
		)
	],
	meta_classifier=LogisticRegression(),
	base_k_fold=10,  # 基分类器分拆份数,force_cv=True时生效，
	meta_k_fold=10,  # 元分类器分拆份数,force_cv=True时生效，
)


global_F1_list = []


all_data1 = []
all_data2 = []

for i in range(len(source)):
	path = '..\\PROMISE datasets\\' + source[i]
	all_data1.append(pd.read_csv(path).iloc[:, 3:24])
train = pd.concat(all_data1)
x_train = train.drop('bug', axis=1).values
x_train = StandardScaler().fit_transform(x_train)
train[train['bug'] != 0] = 1
y_train = train['bug']
	
for i in range(len(target)):
	path = '..\\PROMISE datasets\\' + target[i]
	all_data2.append(pd.read_csv(path).iloc[:, 3:24])
test= pd.concat(all_data2)
x_test = test.drop('bug', axis=1).values
x_x_test = StandardScaler().fit_transform(x_test)
test[test['bug'] != 0] = 1
y_test = test['bug']



# 4. 模型训练
classifier.build_model()
classifier.fit(train_x=x_train, train_y=y_train)

predict_y = classifier.predict(x_test)
print(f1_score(y_test, predict_y, average='weighted'))

