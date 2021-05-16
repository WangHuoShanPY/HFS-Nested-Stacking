from sklearn.preprocessing import StandardScaler
from stacking_classifier import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
import pandas as pd

system_names = ['bugzilla', 'columba', 'jdt', 'mozilla', 'platform', 'postgres']
classifier = StackingClassifier(
    base_classifiers=[
        LightGBMClassifier(),
        CatBoostClassifier(),
        AdaBoostClassifier(),
        # StackingClassifier(
        #     base_classifiers=[
        #         SimpleMLPClassifer(train_params={'input_num':18,'class_num':2}),#比如放这儿
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



for i in range(0, 6):
    # 读取配置文件
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

    # 3. 数据划分
    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, shuffle=True,random_state=42)

    # 4. 模型训练
    classifier.build_model()
    classifier.fit(train_x=X_train, train_y=y_train)

    predict_prob_y = classifier.predict_proba(X_test)[:, 1]
    predict_y = classifier.predict(X_test)
    print('AUC', roc_auc_score(y_test, predict_prob_y))
    print('F1-SCORE:', f1_score(y_test, predict_y, average='weighted'))
    