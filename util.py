import pandas as pd
import numpy as np

#處理第一行不是 'open','high','low','close'
def transform2df(origin):
    l =  origin.columns.values
    ll = [float(a) for a in l]
    a = np.asarray(ll).reshape(1,-1)
    b = origin.loc[:,:].values
    c = np.concatenate((a,b))
    df = pd.DataFrame(c,columns=['open','high','low','close'])
    return df

#diff of previous day
def create_regression_dataset(df,lookahead_days=1):
    raw_X= []
    features=['open','high','low','close']
    raw_X = df.loc[:,features].values
    #raw_X = scaler.fit_transform(raw_X)
    X,y = [],[]


    n = len(df)
    
    for i in range(0,n-lookahead_days):
#
#        #if i == 0 :
#        #    avg10 = today_price
#        #elif i < 10:
#        #    avg10 = np.mean(raw_X[0:i,0])
#        #else:
#        #    avg10 = np.mean(raw_X[i-10:i,0])
#
        
        #mean of open in next k days
        _y = np.mean(raw_X[i+1:i+1+lookahead_days,0])

       
        y.append(_y)
    
    X = preparing_features(raw_X,0,n-lookahead_days)
    y = np.array(y)
    assert len(X) == len(y)

    return X,y


def preparing_features(raw_X,start,end):

    X = []
    for i in range(start,end):
        today_price = raw_X[i][0]
        avg5 = 0
        max5 = 0
        min5 = 0


        if i == 0 :
            avg5 = today_price
            max5 = today_price
            min5 = today_price

          
        elif i < 5:
            avg5 = np.mean(raw_X[0:i,0])
            max5 = np.max(raw_X[0:i,0])
            min5 = np.min(raw_X[0:i,0])

            
        else:
            avg5 = np.mean(raw_X[i-5:i,0]) 
            max5 = np.max(raw_X[i-5:i,0])
            min5 = np.min(raw_X[i-5:i,0])

           

        _X = np.append(raw_X[i],[avg5,max5,min5])
        X.append(_X)

    return  np.array(X)

def create_evaluation_data(path):
    df = transform2df(pd.read_csv(path))
    features=['open','high','low','close']
    rawX = df.loc[:,features].values
    evaX = preparing_features(rawX,0,len(rawX))
    return  evaX

def calculate_profit(testing_data,transformed_data,trader,output_filename = 'output.csv'):
    from evaluator import Evaluator

    print(testing_data.shape)
    eva = Evaluator(testing_data)

    with open(output_filename, 'w') as output_file:
        for i,row in enumerate(testing_data[0:-1]):
            action = trader.predict_action(transformed_data[i])
            output_file.write(action+"\n")
       

    eva.caculate_profit(output_filename)

    print('goal is %.3f'%(testing_data[-1][3]-testing_data[1][0]))
    print('Our profit is %.3f'%(eva.profit))
