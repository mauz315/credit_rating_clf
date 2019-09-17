# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 15:43:18 2019

@author: P900017
"""

# LIME explainer

def explain_tree(data, period, ratings, model, train_set, sov_lab_encoder, le, feat_key):
    
    import numpy as np
    from lime import lime_tabular
    import webbrowser
    
    X_new = np.array(data.loc[feat_key.index].T)
    if sov_lab_encoder is not None:
        pos_sr = feat_key.index.get_loc(feat_key[feat_key["Key"] == 'SovereignRating'].index[0])
        sob_rating = X_new[:, pos_sr].copy()
        X_new[:, pos_sr] = sov_lab_encoder.transform(X_new[:, pos_sr])
    
    # Predicting to check actual prediction
#    pred_calif = np.array([le.iloc[x == list(le.iloc[:,0]),0].index[0] for x in model.predict(X_new)])
    
    X_new = X_new.astype('float')
    
    # features_names = sum([feature_names_key], [])
    # print(features_names)
    class_names = list(le.index)[0:-1]
    class_names.reverse()
    feature_names = list(feat_key.index) # Usar .index (nombres muy largos) o usar .Key (Ratio y #)
    # Create the the Lime explainer and the lambda function
    explainer = lime_tabular.LimeTabularExplainer(train_set, mode='classification',
                                                  feature_names=feature_names,
                                                  class_names=class_names,
                                                  discretize_continuous=True)
    
    predict_fn_rf = lambda x: model.predict_proba(x).astype(float)
    
    # explainer = lime.lime_tabular.LimeTabularExplainer(X_train, feature_names=feature_names,
    #                                                   class_names=class_names, categorical_features=columns,
    #                                                   categorical_names=feature_names_cat, kernel_width=3)
    # Explaining prediction with Lime
    exp = explainer.explain_instance(X_new[period], model.predict_proba, num_features=5, top_labels=ratings)
    # exp.show_in_notebook(show_table=True, show_all=False)
    #print(exp.available_labels())
    exp.save_to_file('explainer/lime_output.html')
    
    av_lab = exp.available_labels()
    for lab in av_lab:
        print ('Explanation for class %s' % class_names[lab])
        print ('\n'.join(map(str, exp.as_list(label=lab))))
        print ()
#        
#    exp_html = 'explainer/lime_output.html'
#    webbrowser.open(exp_html,new=2)
