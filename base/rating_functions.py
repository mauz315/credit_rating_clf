# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 13:08:04 2019

@author: Mauricio Meza
"""
# Nuevo código RF classifier para ratings crediticios
# Réplica delas funciones de rating_prediction_functions

def is_string(s):
    try:
        float(s)
        return False
    except ValueError:
        return True


def feat_elim(data,feat_key, to_del):
    dummy = tuple(to_del)
    for ratio in dummy:
        to_del.append("Ratio" + str(int(ratio[-1:])+13))
    
    for column in to_del:
        del data[column]
            
    rem_feat = []
    for ratio in list(feat_key.Key):
        rem_feat.append(ratio in to_del)
    
    rem_feat = feat_key.index[rem_feat]        
    feat_key.drop(rem_feat, inplace=True)
    return(data, feat_key)

def model_training(data_em, feat_key, le, remove_nan, output_file,
                   model_file, train_set, sov_encoder_file, n_estimators,
                   min_samples_leaf, permut=True, shuffle_sample=False, conf_matrix=True):

    import numpy as np
    import pandas as pd
    import joblib
#    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.preprocessing import Imputer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import RobustScaler
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import QuantileTransformer
    from sklearn.preprocessing import PowerTransformer
    from sklearn.utils import check_random_state
    from sklearn.ensemble import RandomForestClassifier
#    from sklearn.feature_selection import SelectFromModel
    from sklearn import tree
    from sklearn import metrics    
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report
    import matplotlib.pyplot as plt
    from pprint import pprint
#    from sklearn.metrics import roc_curve, auc
    
    data_index = data_em.index # Se crea la variable data_index para publicar el output.
    y_ = np.array(data_em.pop('IssuerRating'))
    X_ = np.array(data_em[feat_key["Key"]])
    
    # Remove observations with no output
    ind_valid_out = [is_string(yi) for yi in y_]
    X = X_[ind_valid_out]
    y = y_[ind_valid_out]
    data_index = data_index[ind_valid_out]
    
    a = []
    for yi in y_:
        if is_string(yi):
            a.append(le.loc[yi])
        else:
            float('NaN')
    
    y = np.array(a)
    
    # Encode Sovereign Rating
    sr = feat_key[feat_key["Key"] == 'SovereignRating']
    if len(sr)>0:
        pos_sr = feat_key.index.get_loc(sr.index[0])# Position sovereign rating
        pos_str = [is_string(x) for x in X[:,pos_sr]]
        labels = np.unique(X[pos_str,pos_sr])
        le_X = LabelEncoder()
        le_X.fit(labels)
        X[pos_str,pos_sr] = le_X.transform(X[pos_str,pos_sr])
        joblib.dump(le_X, sov_encoder_file)# Save sovereign label encoder
    
    # Remove NaN
    if remove_nan:
        ind_not_na = [not np.isnan(np.sum(x)) for x in X]
        X = X[ind_not_na]
        y = y[ind_not_na]
        data_index = data_index[ind_not_na]
    # else:
    #    imp = Imputer(missing_values='NaN', strategy = 'mean', axis = 0)
    #    imp.fit(X = X_train)
    #     X = imp.transform(X)
    
    #Visualización de clases luego de retirar NaNs    
#    r = list()
#    for i in y:
#        r.append(le.index[le.iloc[i[0]][0]])
#    r = pd.DataFrame(r, columns=['Rating'])
#    r["Rating"].value_counts().plot(kind='bar')
#    plt.show()
    
    # Data Permutation: Para si es necesario volver 
    X_o = X
    y_o = y
    if permut:
        random_state = check_random_state(0)
        permutation = random_state.permutation(X.shape[0])
        
        X = X[permutation]
        y = y[permutation]
        
        data_index = data_index[permutation]
    
    # Train and test samples:
    perc_train_size = 0.8
    train_size = int(X.shape[0] * perc_train_size)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = train_size, shuffle = False)
    
    print('Muestra de entrenamiento: %d' % X_train.shape[0])
    print('Muestra de testing: %d' % X_test.shape[0])
    print('')
    
    scale = True
    if scale:
        sov_train = X_train[:,-1] 
        sov_test = X_test[:,-1]
        scaler = RobustScaler()
        scaler.fit(X_train[:,:-1])
        X_train = np.column_stack((scaler.transform(X_train[:,:-1]), sov_train))
        X_test = np.column_stack((scaler.transform(X_test[:,:-1]), sov_test))
        joblib.dump(scaler, 'model/my_scaler.sav')
    y = np.array(y)
    
    # Initializing model:
    clf = RandomForestClassifier(n_estimators = n_estimators, max_features = "auto", min_samples_leaf = min_samples_leaf)
    
#    Randomized Search CV Hyperparameter Optimization
    print('Parameters currently in use:\n')
    pprint(clf.get_params())
    
    #  Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(50, 300, num = 25)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 5, 10]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    pprint(random_grid)
    rf_random = RandomizedSearchCV(estimator = clf, param_distributions = random_grid,
                                   n_iter = 100, cv = 3, verbose=2, random_state=42, 
                                   n_jobs = -1)
#        
#    # Model fitting:
    rf_random.fit(X_train, np.ravel(y_train))
#    pprint(rf_random.best_params_)
#    clf.fit(X_train, np.ravel(y_train))
    # Print importances
#    importances = list(clf.feature_importances_)
#    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(list(feat_key.index), importances)]
#    
#    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
#    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]
    
    # Save model
    joblib.dump(rf_random, model_file)

    # Save training set for LIME explainer
    joblib.dump(X_train, train_set)
    
    # Prediction files por training and test sets
    pred_train = rf_random.predict(X_train)
    pred_test = rf_random.predict(X_test)
    pred_o = rf_random.predict(X_o)
    
    print('Train Accuracy:', metrics.accuracy_score(y_train, pred_train))
    print('Test Accuracy:', metrics.accuracy_score(y_test, pred_test))
    print('Original Set Accuracy:', metrics.accuracy_score(y_o, pred_o))
    print(classification_report(y_test, pred_test))
    
    mse_train = metrics.mean_squared_error(y_train, pred_train)
    mse_test = metrics.mean_squared_error(y_test, pred_test)
    print("Train MSE: {}".format(mse_train))
    print("Test MSE: {}".format(mse_test))
    
    #Confusion matrix of test data 
    if conf_matrix:
        conf_mat = confusion_matrix(y_test, pred_test)
        print(conf_mat)
    
    # output file:

    pred_calif = np.array([le.iloc[x == list(le.iloc[:,0]),0].index[0] for x in rf_random.predict(X_test)])
    y_test_calif = np.array([le.iloc[x == list(le.iloc[:,0]),0].index[0] for x in y_test])
    
    if len(sr)>0:
        X_test[:, pos_sr] = le_X.inverse_transform(X_test[:, pos_sr].astype('int')) # Inverse transform of sov. ratingsS

    data_test = pd.DataFrame(np.column_stack((np.column_stack((X_test, y_test_calif)), pred_calif)), columns = list(feat_key.index)+['Rating Test', 'Rating Predicc'], index = data_index[np.arange(train_size, data_index.shape[0])])

    # Output file:
    data_test.to_csv(output_file)
    
    
    # Compute ROC curve and ROC area for each class
#    fpr = dict()
#    tpr = dict()
#    roc_auc = dict()
#    for i in range(n_classes):
#        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
#        roc_auc[i] = auc(fpr[i], tpr[i])
#
#    # Compute micro-average ROC curve and ROC area
#    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
#    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Función de predicción
def rating_prediction(data, rf, rf_pure, feat_key, le, sov_lab_encoder, scaler, output_file):
    # rf: modelo base con riesgo soberano
    # rf_pure: modelo sin riesgo sobreano para DM, POR ELIMINAR
    
    import numpy as np
    import pandas as pd
    
    to_del = ["Ratio7", "Ratio13"]
    for ratio in to_del:
        rat1 = feat_key[feat_key.Key == ratio].index
        rat2 = feat_key[feat_key.Key == "Ratio" + str(int(ratio[-1:])+13)].index
        data = data.drop(rat1)
        data = data.drop(rat2)
            
    # Importando nueva data:
    X_new = np.array(data.loc[feat_key.index].T)
    X_new_pure = np.array(data.loc[feat_key.index[(feat_key != 'SovereignRating')['Key']]].T)
    
    sov_new = X_new[:,-1]
    X_new = np.column_stack((scaler.transform(X_new[:,:-1]), sov_new))

    # Transformando info soberanos a escala
    if sov_lab_encoder != None:
        pos_sr = feat_key.index.get_loc(feat_key[feat_key["Key"] == 'SovereignRating'].index[0])# Position sovereign rating
        sob_rating = X_new[:,pos_sr].copy()
        X_new[:,pos_sr] = sov_lab_encoder.transform(X_new[:,pos_sr])

    # Predicción primer modelo
    pred_calif = np.array([le.iloc[x == list(le.iloc[:,0]),0].index[0] for x in rf.predict(X_new)])
    X_new[:, pos_sr] = sov_lab_encoder.inverse_transform(X_new[:, pos_sr].astype('int')) # Inverse transform of sov. ratings
    
    # Predicción modelo pure / POR ELIMINAR
    pred_calif_pure = np.array([le.iloc[x == list(le.iloc[:,0]),0].index[0] for x in rf_pure.predict(X_new_pure)])

    rat_dist = np.array([int(le.loc[x]) for x in sob_rating]) - np.array([int(le.loc[x]) for x in pred_calif_pure])
    rat_trans = int(le.loc['AAA'])  - np.array([np.max([int(i),0]) for i in rat_dist])
    pred_calif_translate = [le[le['Value']==x].index[0] for x in rat_trans]

   # Tabla para print de resultados   
   # data_pred = pd.DataFrame(np.column_stack((np.column_stack((X_new, data.columns)), np.column_stack((pred_calif,np.column_stack((pred_calif_pure, pred_calif_translate)))))), columns = list(data.loc[feat_key.index].index)+['Periodo', 'Rating Predicc', 'Rating Pure', 'Rating Local Trad'])
    data_pred = pd.DataFrame(np.column_stack((np.column_stack((X_new, data.columns)),
                                              np.column_stack((pred_calif, pred_calif_translate)))),
                                            columns = list(data.loc[feat_key.index].index)+['Periodo',
                                                          'Rating Predicc', 'Rating Local Trad'])    
    
    print('Predicción Rating:')
    print('')
    print(data_pred[['Periodo', 'Rating Predicc', 'Rating Local Trad']])

    # Output file:
    data_pred.to_csv(output_file, index = False)
    return(None)