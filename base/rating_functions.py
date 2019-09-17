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
    
def model_training(data_em, feat_key, le, remove_nan, perc_train_size,
                   output_file, model_file, train_set, sov_encoder_file, n_estimators,
                   min_samples_leaf, permut=True, shuffle_sample=False):

    import numpy as np
    import pandas as pd
    import joblib
#    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import Imputer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.preprocessing import LabelEncoder
    from sklearn.utils import check_random_state
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import SelectFromModel
    from sklearn import tree
    from sklearn import metrics    

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
    
    # Data Permutation: PREGUNTAR QUE ES !!!!!
    if permut:
        random_state = check_random_state(0)
        permutation = random_state.permutation(X.shape[0])
        
        X = X[permutation]
        y = y[permutation]
        
        data_index = data_index[permutation]
    
    # Train and test samples:
    train_size = int(X.shape[0] * perc_train_size)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = train_size, shuffle = shuffle_sample)
    
    print('Muestra de entrenamiento: %d' % X_train.shape[0])
    print('Muestra de testing: %d' % X_test.shape[0])
    print('')
    y = np.array(y)
    
    # Model fitting:
    clf = RandomForestClassifier(n_estimators = n_estimators, max_features = "auto", min_samples_leaf = min_samples_leaf)
    clf.fit(X_train, np.ravel(y_train))
    
    # Save model
    joblib.dump(clf, model_file)

    # Save training set for LIME explainer
    joblib.dump(X_train, train_set)
    
    print('Train Accuracy:', metrics.accuracy_score(y_train, clf.predict(X_train)))
    print('Test Accuracy:', metrics.accuracy_score(y_test, clf.predict(X_test)))
    
    res = clf.predict(X_train)
    
    mse_train = metrics.mean_squared_error(y_train, res)
    mse_test = metrics.mean_squared_error(y_test, clf.predict(X_test))
    print("Train MSE: {}".format(mse_train))
    print("Test MSE: {}".format(mse_test))


# Función de predicción
def rating_prediction(data, rf, rf_pure, feat_key, le, sov_lab_encoder, output_file):
    # rf: modelo base con riesgo soberano
    # rf_pure: modelo sin riesgo sobreano para DM, POR ELIMINAR
    
    import numpy as np
    import pandas as pd

    # Importando nueva data:
    X_new = np.array(data.loc[feat_key.index].T)
    X_new_pure = np.array(data.loc[feat_key.index[(feat_key != 'SovereignRating')['Key']]].T)

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