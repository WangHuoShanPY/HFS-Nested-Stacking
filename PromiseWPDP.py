import os

import random
import shutil
import cv2 as cv
import numpy as np
import pandas as pd
import featuretools as ft
from PIL.Image import Image
from sklearn.metrics import roc_auc_score, f1_score

from stacking_classifier import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# 第一步:加载文件，获取文件路径


root_path = "..\\PROMISE datasets\\"
allpath = []

classifier = StackingClassifier(
    base_classifiers=[
        LightGBMClassifier(),
        CatBoostClassifier(),
        AdaBoostClassifier(),
        # StackingClassifier(
        #     base_classifiers=[
        #         SimpleMLPClassifer(train_params={'input_num':20,'class_num':2}),#比如放这儿
        #         RandomForestClassifier(),
        #     ],
        #     meta_classifier=GradientBoostingClassifier(),
        #     base_k_fold=10,#基分类器分拆份数,force_cv=True时生效，
        #     meta_k_fold=10,#元分类器分拆份数,force_cv=True时生效
        # )
    ],
    meta_classifier=LogisticRegression(),
    base_k_fold=10,#基分类器分拆份数,force_cv=True时生效，
    meta_k_fold=10,#元分类器分拆份数,force_cv=True时生效，

)

def get_lableandwav(path):
	dirs = os.listdir(path)
	for a in dirs:
		if os.path.isfile(path + "/" + a):
			allpath.append(dirs)
		else:
			get_lableandwav(str(path) + "/" + str(a))
	##循环遍历这个文件夹
	return allpath

allpath = get_lableandwav(root_path)
allpath = allpath[1]
print(allpath)

for i in range(len(allpath)):
	counter = 0
	allpath1 = str(allpath[i])[:-4]
	#print(allpath1)
	df = pd.read_csv(root_path+ allpath[i]).iloc[:, 3:24]
	
	x_data = df.drop('bug', axis=1).values
	x_data = StandardScaler().fit_transform(x_data)
	df[df['bug'] != 0] = 1
	y_data = df['bug']
	
	
	# 3. 数据划分
	X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, shuffle=True, random_state=42)
		
	# 4. 模型训练
	classifier.build_model()
	classifier.fit(train_x=X_train, train_y=y_train)
	predict_y = classifier.predict(X_test)
	print( f1_score(y_test, predict_y, average='weighted'))
