import util

class Trader():
    def __init__(self,reg_tom,reg_trend):
        self.reg_tom = reg_tom
        self.reg_trend = reg_trend
        self.day_cnt = 0    
        self.current_slot = 0

    def train(self,training_data):
        pass

    def predict_action(self,datum):

        current_price = datum[0]
        next_price,mean3 = self.reg_tom.predict(datum.reshape(1,-1)), self.reg_trend.predict(datum.reshape(1,-1))


        self.day_cnt = self.day_cnt +1

        act = self.policy(current_price,next_price,mean3)
        return act
    #using hold price ? or ?
    def policy(self,current_price,next_price,trend):

        #當預測價格會上漲時(未來平均高於明天的價格) 買入
        #當預測價格會下跌時(未來平均低於明天的價格) 賣掉
        

        if self.current_slot == 1: #當預測價格會下跌時(未來平均低於明天的價格) 賣掉
            if next_price >  trend : 
                self.current_slot =0
                return '-1'

        elif self.current_slot == 0 :
            if next_price >  trend : 
                self.current_slot = -1
                return '-1'
            if  next_price < trend  : 
                self.current_slot = 1
                return '1'

        elif self.current_slot == -1 : # 今天跌 明天漲 -->賣空買回
            if next_price <  trend : 
                self.current_slot =0
                return '1'

        return '0'


if __name__ == '__main__':
    # You should not modify this part.
    import argparse
    from sklearn import linear_model
    from sklearn.preprocessing import PolynomialFeatures
    import pandas as pd

    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
                       default='training_data.csv',
                       help='input training data file name')
    parser.add_argument('--testing',
                        default='testing_data.csv',
                        help='input testing data file name')
    parser.add_argument('--output',
                        default='output.csv',
                        help='output file name')
    args = parser.parse_args()
    
    # The following part is an example.
    # You can modify it at will.

    lookahead_days = 3
    pdegree = 2
    poly = PolynomialFeatures(pdegree)
    train_set = util.transform2df(pd.read_csv(args.training))
    train_X1,train_y1 = util.create_regression_dataset(train_set,1)
    train_X2,train_y2 = util.create_regression_dataset(train_set,lookahead_days)
    transformed_trainX1 = poly.fit_transform(train_X1)
    transformed_trainX2 = poly.fit_transform(train_X2)

    predict_tom = linear_model.Ridge (alpha = 1) #predict tommorow price
    predict_trend = linear_model.Ridge (alpha = 1)

    predict_tom .fit(transformed_trainX1 ,train_y1)
    predict_trend.fit(transformed_trainX2 ,train_y2)


    testing_data = util.create_evaluation_data(args.testing)

   
    trader = Trader(predict_tom, predict_trend)
    #trader.train(training_data)
    



    with open(args.output, 'w') as output_file:
        for row in testing_data[0:-1]:
            # We will perform your action as the open price in the next day.
            
            datam = poly.fit_transform(row.reshape(1,-1))
            action = trader.predict_action(datam.ravel())

            output_file.write(action+"\n")

            # this is your option, you can leave it empty.
            #trader.re_training(i)
