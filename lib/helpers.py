from collections import defaultdict
import gzip
import re
import pandas as pd
import numpy as np
from numpy import loadtxt
import xgboost
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn import metrics

def readGz(f):
    for l in gzip.open(f):
        yield eval(l)

def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)

def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')

def getTrainDF(removeBad=True, removeZero=False ):
    df = getDF('train.json.gz')
    df = get_full_df(df)
    
    return df

def getTestDF():
    test_df = getDF('../test_Helpful.json.gz')
    test_df = get_full_df(test_df, test_df=True)
    return test_df

        
def get_full_df(train_df, removeZero=False,test_df=False):
    train_df['review_length'] = train_df['reviewText'].apply(lambda x: len(str(x))) # get length of reviewText
    train_df['summary_length'] = train_df['summary'].apply(lambda x: len(str(x))) # get length of reviewText
    train_df['price'] = train_df['price'].fillna(train_df['price'].mean()) # fill all nas with zeros
    
    train_df['outOf'] = train_df['helpful'].apply(lambda x: x['outOf'])

    if not test_df:
        train_df['num_helpful'] = train_df['helpful'].apply(lambda x: x['nHelpful'])
        train_df['perc_helpful'] = train_df['num_helpful'] * 1.0 / train_df['outOf']
        avg_per_outOf = train_df.groupby('outOf').mean().perc_helpful
        outlier_num_helpfuls = avg_per_outOf[avg_per_outOf < 0.5].index.values
        print(outlier_num_helpfuls)
        train_df = train_df[~train_df.outOf.isin(outlier_num_helpfuls)]
        del train_df['perc_helpful']
        
    train_df['month'] = train_df['reviewTime'].apply(lambda x: int(x[:2]))
    train_df['day'] = train_df['reviewTime'].apply(lambda x: int(x[3:6].replace(',', '').replace(' ', '')))
    train_df['year'] = train_df['reviewTime'].apply(lambda x: int(x[-4:]))
    del train_df['reviewTime']
    
    train_df['review+summary'] = train_df['reviewText'] + ' ' + train_df['summary']
    train_df = train_df.dropna()
    
    
    if removeZero:
        train_df = train_df[train_df['outOf'] != 0]
    
    train_df.columns = ['ri_' + x for x in train_df.columns]
    
    train_df = train_df.reindex(sorted(train_df.columns), axis=1)
    
    return train_df

def train_model_df(train_df, removeZero=True):
        
    if removeZero:
        train_df = train_df[train_df['ri_outOf'] != 0]
        
    y = train_df.ri_num_helpful
    x_df = train_df
    del x_df['ri_num_helpful']
    print(x_df.shape)
    
    del train_df['ri_categories']
    del train_df['ri_reviewText']
    del train_df['ri_summary']
    del train_df['ri_review+summary']
    #del train_df['ri_reviewerID']
    #del train_df['ri_itemID']
    del train_df['ri_reviewHash']
    del train_df['ri_helpful']
    
    return x_df, y
        
# want to get mae score on val set
def train_mae(train_df, model_list, round=True, percent_helpful=False):
    train_df = train_df.copy()
    del train_df['ri_categories']
    del train_df['ri_reviewText']
    del train_df['ri_summary']
    del train_df['ri_review+summary']
    del train_df['ri_reviewHash']
    del train_df['ri_helpful']
    
    y = train_df.ri_num_helpful
    
    if percent_helpful:
        y = y / train_df['ri_outOf']
    
    x_df = train_df
    del x_df['ri_num_helpful']
    print(x_df.shape)
    
    for i in range(len(model_list) - 1):
        val = i+1
        model = model_list[i]
        predict_str = 'predict' + str(val)
        train_df[predict_str] = model.predict(x_df)
        
    
    final_model = model_list[-1]
    
    train_df['pred_train'] = final_model.predict(x_df)
    
    if percent_helpful:
        train_df['pred_train'] = train_df['pred_train'] * train_df['ri_outOf']
    
    
    train_df.loc[(train_df['ri_outOf'] == 0), 'pred_train'] = 0.0
    
    if(round):
        train_df['predicted'] = train_df['pred_train'].round()
        
    pred_train = train_df['predicted'].values
    print('train MAE: ' + str(metrics.mean_absolute_error(y, pred_train)))
    print('cross val MAE: ' + str(cross_val_score(final_model,x_df, y, scoring="neg_mean_absolute_error")))

    
def fullTestWorkflow(df, filename, model_list, percent=True):
    test_df = df.copy()
    del test_df['ri_categories']
    del test_df['ri_reviewText']
    del test_df['ri_summary']
    del test_df['ri_review+summary']
    del test_df['ri_reviewHash']
    del test_df['ri_helpful']
    print(test_df.columns)
    predict_test_df = test_df.copy()
    del predict_test_df['ri_reviewerID']
    del predict_test_df['ri_itemID']
    
    for i in range(len(model_list) - 1):
        val = i+1
        model = model_list[i]
        predict_str = 'predict' + str(val)
        print(predict_str)
        predict_test_df[predict_str] = model.predict(predict_test_df)
        
    
    final_model = model_list[-1]
    test_predict = final_model.predict(predict_test_df)
    
    test_df['predicted'] = test_predict
    test_df.loc[(test_df['ri_outOf'] == 0), 'predicted'] = 0.0
    
    if percent:
        test_df['predicted'] = test_df['predicted'] * test_df['ri_outOf']
    
    test_df['predicted'] = test_df.predicted.round()
    
    final_test_df = test_df[['ri_reviewerID', 'ri_itemID', 'ri_outOf', 'predicted']]
    final_test_df.columns =  ['userID', 'itemID', 'outOf', 'prediction']
    
    final_test_df.to_csv(filename, index=False)
    
    newf=""
    with open(filename,'r+') as f:
        for line in f:
            newline=line.replace(',','-',2) 
            newf+=newline
            #print(newline)
        f.close()

    with open(filename,'w') as f:
        f.write(newf)
        f.close()
        
        
    