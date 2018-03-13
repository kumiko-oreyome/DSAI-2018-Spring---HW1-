# DSAI HW1-AutoTrading

## Stragery
在這個作業我會使用到兩個regression model預測
- 明日open price  --> model1
- 未來三天的open price的平均(當作一個價格趨勢的指標) --> model2

假設今天有一筆資料(open,low.high,close)= X
會先將X做feature transformation成 f(X) = Z
然後將Z丟到 
model1(Z)-->得到 p1,代表預測的明天的開盤價
model2(Z)-->得到 p2,代表預測的未來三天的open price的平均(當作一個價格趨勢的指標)。

if slot == 1 代表持股,
 if p1 > p2 : 代表未來三天內股票很可能會跌,而且明天的股票價格會比趨勢還要高,則賣出。
 
if slot == 0 
 if p1 > p2 : 代表未來三天內股票很可能會跌,而且明天的價格會比趨勢還要高,則賣空(將來有機會可以買回來賺差價)。
 if p1 < p2 : 代表未來三天內股票很可能會漲,而且明天的價格會比趨勢還要低,則買入。
 
if slot == -1
 if p1 < p2 : 代表未來三天內股票很可能會上漲,而且明天的股票價格會比趨勢還要低,則買入賣空的股票。

在這個策略裡面並未考慮到當前持有股票的價錢,那是因為根據自己測試的結果,不考慮比較簡單且賺得比較多。


## Data preprocessing & feature engineering

input的每一筆資料會有每天open,high,low,close 的價格,
除了那4個feature外還會加入每一天過去五天的(如果是第一到五天則只看前0到四天,因為前五天要在第六天才有)
- average of open price
- max of open price
- min of open price
總共是7個feature

會加入這些feature是因為考量到feature數量可能不足以及有在validation set上面跑過,加入這些feature會對performance提升,
但由於時間不足,還未嘗試到更多可能更好的feature。

然後再對這些feature做二次的多項式的 的feature trasformation,因為只有用linear  的發現無法很好的fitting訓練資料(但是加入二次的多項式的transformation可能會導致overfitting)。

至於label就是明日open price和未來三天的open price的平均。
因為有兩個regression model所以有兩組training的dataset。(X相同y不同)

## Model
訓練兩個Ridge regression,regularization weight為1.0。

## 額外說明
有嘗試過使用
- Stack LSTM ,sequence size 從五到八,feature為open,low,high,close,但效果沒有特別好,
推測可能是因為資料不夠多所以嘗試使用簡單一點的regression。

想嘗試使用RL或者增加更多前幾天的feature,像是把前幾天的開盤價全部都當作feature。