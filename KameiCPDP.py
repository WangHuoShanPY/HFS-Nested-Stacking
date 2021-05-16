from sklearn.preprocessing import StandardScaler
from stacking_classifier import *
from sklearn.metrics import f1_score, roc_auc_score
import pandas as pd

classifier = StackingClassifier(
	base_classifiers=[
		LightGBMClassifier(),
		CatBoostClassifier(),
		RandomForestClassifier(),
		AdaBoostClassifier(),
		StackingClassifier(
			base_classifiers=[
				SimpleMLPClassifer(train_params={'input_num': 18, 'class_num': 2}),  # 比如放这儿
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

# 四个评估指标
global_auc_list = []
global_F1_list = []

all_data = []
system_names = ['bugzilla', 'columba', 'jdt', 'mozilla', 'platform', 'postgres']

for i in range(len(system_names)):
	print(system_names[i])
	path = '..\\data\\' + system_names[i] + '.csv'
	# 1. 读取数据
	df = pd.read_csv(path).iloc[:, 2:17]
	
	# 2. 数据预处理
	func1 = lambda x, y: x / y
	df['ALA'] = list(map(func1, df['la'], df['npt']))
	df['ALD'] = list(map(func1, df['ld'], df['npt']))
	func2 = lambda x, y: abs(x - y)
	df['GEXP1'] = list(map(func2, df['rexp'], df['exp']))
	df['GEXP2'] = list(map(func2, df['exp'], df['sexp']))
	
	x_data = df.drop('bug', axis=1).values
	x_data = StandardScaler().fit_transform(x_data)
	y_data = df['bug']
	all_data.append([x_data, y_data])

xgbc = Classifier()

# ----------------关键代码-----------------------------------------
for i in range(len(system_names)):
	for j in range(len(system_names)):
		if i != j:
			# 构建训练集和测试集
			x_train, x_test = all_data[i][0], all_data[j][0]
			y_train, y_test = all_data[i][1], all_data[j][1]
			# 模型训练
			classifier.build_model()
			classifier.fit(train_x=x_train, train_y=y_train)
			# 模型预测
			predict_prob_y = classifier.predict_proba(x_test)[:, 1]
			predict_y = classifier.predict(x_test)
			# 模型评估
			auc = roc_auc_score(y_test, predict_prob_y)
			f1 = f1_score(y_test, predict_y, average='weighted')
			global_auc_list.append(auc)
			global_F1_list.append(f1)
	print(np.mean(global_auc_list))
	print(np.mean(global_F1_list))
