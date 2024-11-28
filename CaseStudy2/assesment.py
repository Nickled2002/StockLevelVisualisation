import pandas as pd
from polygon import RESTClient
from polygon.rest.models import (Agg,)
import datetime
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import mplcursors
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.python.keras import models
from tensorflow.python.keras import layers
from sklearn.metrics import root_mean_squared_error, r2_score ,mean_absolute_error
from tensorflow.python.keras.engine import data_adapter
from sklearn.linear_model import SGDRegressor 


class stock:#stock object 
    def __init__(self, name,df,df2,df3):
        # stores the name of the stock and its contents so it doesnt have to be read multiple times
        self.name = name
        self.stockdf = df
        self.incomedf = df2
        self.stockdf100k = df3

class incomestatements:#income statements dataframe object 
    def __init__(self,df):
        # stores the name of the stock and its contents so it doesnt have to be read multiple times
        self.incomestatements = df
        
#TensorFlow Compatibility Fix
def _is_distributed_dataset(ds):
    return isinstance(ds, data_adapter.input_lib.DistributedDatasetSpec)

def retrievestockdata(name,time,date): #Retrieve stock data
    entries = []
    while not entries:
        for data in client.list_aggs(
            name,
            1,
            time,
            date,
            datetime.datetime.today().strftime('%Y-%m-%d'),
            limit=50000,
        ):
            entries.append(data)
        if not entries:
            print('Stock', name, 'does not exist try again')
            break
    return entries

def generalvisualisation(data,name):
    #=============================close price-time===============================================
    # Plot the adjusted close price
    plottable = pd.DataFrame()
    plottable['value'] = data['close']
    timestamp=pd.to_datetime(data['timestamp'])
    plottable = plottable.set_index(timestamp)
    plt.plot(plottable,label = name)
    plt.plot(figsize = (15,7))
    plt.gcf().autofmt_xdate()
    mplcursors.cursor(hover=True)
    # Define the label for the title of the figure
    plt.title("Close Price", fontsize=16)
    # Define the labels for x-axis and y-axis
    plt.ylabel('Price', fontsize=14)
    plt.xlabel('Year', fontsize=14)
    # Plot the grid lines
    plt.grid(which="major", color='k', linestyle='-.', linewidth=0.5)
    # Show the plot
    plt.legend()
    plt.show()
    #=============================volume of stock===============================================
    plottable = pd.DataFrame()

    plottable['value'] = data['volume']
    timestamp=pd.to_datetime(data['timestamp'])
    plottable = plottable.set_index(timestamp)
    plt.plot(plottable,label = name)
    plt.plot(figsize = (15,7))
    plt.gcf().autofmt_xdate()
    mplcursors.cursor(hover=True)
    plt.title("Volume of Stock traded", fontsize=16)
    plt.ylabel('Price', fontsize=14)
    plt.xlabel('Year', fontsize=14)
    plt.grid(which="major", color='k', linestyle='-.', linewidth=0.5)
    plt.legend()
    plt.show()
    #=============================Market Cap stock===============================================
    plottable = pd.DataFrame()
    plottable['data'] = data['open']*data['volume']
    timestamp=pd.to_datetime(data['timestamp'])
    plottable = plottable.set_index(timestamp)
    plt.plot(plottable,label= name)
    plt.plot(figsize = (15,7))
    plt.gcf().autofmt_xdate()
    mplcursors.cursor(hover=True)
    plt.title("Market Cap", fontsize=16)
    plt.ylabel('Price', fontsize=14)
    plt.xlabel('Year', fontsize=14)
    plt.grid(which="major", color='k', linestyle='-.', linewidth=0.5)
    plt.legend()
    plt.show()
    #=============================Volatility===============================================
    plottable = pd.DataFrame()
    plottable['returns'] = (data['close']/data['close'].shift(1)) -1
    plottable['returns'].hist(bins = 100, label = name, alpha = 0.5, figsize = (15,7))
    plt.title("Volatility", fontsize=16)
    plt.grid(which="major", color='k', linestyle='-.', linewidth=0.5)
    mplcursors.cursor(hover=True)
    plt.legend()
    plt.show()

def convertstockdataframe(entries):
    timestamp=[]
    openval=[]
    high=[]
    low=[] 
    close=[]
    volume=[]
    vwap=[]
    transactions=[]
    year=[]
    for entry in entries:
            # verify this is an agg
            if isinstance(entry, Agg):
                # verify this is an int
                if isinstance(entry.timestamp, int):
                    timestamp.append(datetime.datetime.fromtimestamp(entry.timestamp / 1000).strftime('%Y-%m-%d'))
                    openval.append(entry.open)
                    high.append(entry.high)
                    low.append(entry.low)
                    close.append(entry.close)
                    volume.append(entry.volume)
                    vwap.append(entry.vwap)
                    transactions.append(entry.transactions)
                    year.append(datetime.datetime.fromtimestamp(entry.timestamp / 1000).year)
    dfdata = {"timestamp":timestamp,"open":openval,"high":high,"low":low,"close":close,"volume":volume,"vwap":vwap,"transactions":transactions,"year":year,}
    df = pd.DataFrame(data=dfdata)
    return df

    
def incomevisualisation(plotdata):
    for entry in plotdata.index:
        fig = go.Figure(data=[go.Sankey(
        valueformat = ".0f",
        valuesuffix = " $ ",
        node = dict(
        pad = 100,
        thickness = 2,
        line = dict(color = "black", width = 0.001),
        label = ["Revenue","Cost Of Revenue", "Gross Profit","Operating Expenses","Operating Income","Administrative Operating Expenses", "Research and Development","Other Operating Expenses","Total Other Income","Income Before Tax","Income Tax Expense","Net Income"],
        color = ["green","Red","green","Red","green","Red","Red","Red","blue","green","Red","green"],
        ),
        link = dict(
        source = [0, 0, 2, 2, 3, 3, 3, 4, 8, 9, 9], 
        target = [1, 2, 3, 4, 5, 6, 7, 9, 9, 11, 10],
        value =  [abs(plotdata['grossProfit'][entry]), abs(plotdata['costOfRevenue'][entry]), abs(plotdata['totalOperatingExpenses'][entry])-abs(plotdata['costOfRevenue'][entry]), abs(plotdata['operatingIncome'][entry]), abs(plotdata['sellingGeneralAdministrative'][entry]), abs(plotdata['researchDevelopment'][entry]), abs(plotdata['otherOperatingExpenses'][entry]), abs(plotdata['incomeBeforeTax'][entry]), abs(plotdata['totalOtherIncomeExpenseNet'][entry]), abs(plotdata['incomeTaxExpense'][entry]), abs(plotdata['netIncome'][entry])],
        ))])
        fig.update_layout(title_text=plotdata['stock'][entry]+"'s Income Statement - "+plotdata['endDate'][entry], font_size=10)
        fig.show()        

def stockprediction(data,name):
    timestamp=pd.to_datetime(data['timestamp'])
    data = data.set_index(timestamp)
    preddataclose=data.filter(['close']).values
    scale = MinMaxScaler(feature_range=(0,1))
    closeprice_scaled = scale.fit_transform(preddataclose)
    period = 1
    trainsize = int(len(closeprice_scaled)*0.65)
    traindata, testdata = closeprice_scaled[0:trainsize,:], closeprice_scaled[trainsize-period:,:]
    xtrain= []
    ytrain= []
    for i in range(period, len(traindata)):
        xtrain.append(traindata[i-period:i,0])
        ytrain.append(traindata[i,0])
    xtrain,ytrain = np.array(xtrain),np.array(ytrain)
    xtrain = np.reshape(xtrain, (xtrain.shape[0], xtrain.shape[1],1))
    model = models.Sequential([
        layers.LSTMV1(50,input_shape=(xtrain.shape[1],1), return_sequences=True),
        layers.LSTMV1(50, return_sequences=False),
        layers.Dense(25),
        layers.Dense(1)
        ])
    print(model.summary())
    model.compile(optimizer='adam',loss='mean_squared_error')
    model.fit(xtrain,ytrain,batch_size=1,epochs=40)
    xtest , ytest =[],preddataclose[trainsize:,:]
    for i in range (period,len(testdata)):
        xtest.append(testdata[i-period:i,0])
    xtest = np.array(xtest)
    xtest = np.reshape(xtest, (xtest.shape[0],xtest.shape[1],1))
    pred = model.predict(xtest)
    pred = scale.inverse_transform(pred)
    print("Mean Absolute Error: ", mean_absolute_error(ytest, pred))
    print("Coefficient of Determination: ", r2_score(ytest, pred))
    print("Mean Squared Error: ",root_mean_squared_error(ytest,pred))
    train = data[:trainsize]
    test = data[trainsize:].copy()
    test['pred'] = pred
    plt.plot(figsize = (20,10))
    plt.grid(which="major", color='k', linestyle='-.', linewidth=0.5)
    plt.plot(train['close'])
    plt.plot(test[['close','pred']])
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Price', fontsize=14)
    plt.title(name+"'s Close Stock Price Predictions ", fontsize=16)
    plt.legend(['Train','Test','Predictions'])
    plt.gcf().autofmt_xdate()
    mplcursors.cursor(hover=True)
    plt.show()
    
    
def stockdatacheck(newstock,time,date,dfname): 
    if getattr(newstock, dfname).empty:
        if newstock.name == '':
            entries = []
            while not entries:
                stockname=input('Which stock would you like to explore (Use capitals):')
                try:
                    stockname = str(stockname)
                except:
                    print('Please use the stock name abbreviation.')
                    continue
                if len(stockname) < 1 or len(stockname) > 5:
                    print('Please use the stock name abbreviation.')
                    continue
                entries=retrievestockdata(stockname,time,date)
            newstock.name = stockname 
        else:
            entries=retrievestockdata(newstock.name,time,date) 
            if not entries:
                newstock.name=''
                return False
        df = convertstockdataframe(entries)
        setattr(newstock, dfname, df)
    return True

def stockprediction100k(data, name):
    timestamp=pd.to_datetime(data['timestamp'])
    data = data.set_index(timestamp)
    preddataclose=data.filter(['close']).values
    scale = MinMaxScaler(feature_range=(0,1))
    closeprice_scaled = scale.fit_transform(preddataclose)
    period = 10
    trainsize = int(len(closeprice_scaled)*0.65)
    traindata, testdata = closeprice_scaled[0:trainsize,:], closeprice_scaled[trainsize-period:,:]
    xtrain= []
    ytrain=[]
    for i in range(period, len(traindata)):
        xtrain.append(traindata[i-period:i,0])
        ytrain.append(traindata[i,0])
    xtrain,ytrain = np.array(xtrain),np.array(ytrain)
    model = SGDRegressor(max_iter=1000, alpha=0.0001, learning_rate='invscaling', random_state=0, shuffle=False)
    model.fit(xtrain,ytrain)
    xtest , ytest =[],preddataclose[trainsize:,:]
    for i in range (period,len(testdata)):
        xtest.append(testdata[i-period:i,0])
    xtest = np.array(xtest)
    pred = model.predict(xtest)
    pred = scale.inverse_transform(pred.reshape(-1, 1))
    print("Mean Absolute Error: ", mean_absolute_error(ytest, pred))
    print("Coefficient of Determination: ", r2_score(ytest, pred))
    print("Mean Squared Error: ",root_mean_squared_error(ytest,pred))
    train = data[:trainsize]
    test = data[trainsize:].copy()
    test['pred'] = pred
    plt.plot(figsize = (20,10))
    plt.grid(which="major", color='k', linestyle='-.', linewidth=0.5)
    plt.plot(train['close'])
    plt.plot(test[['close','pred']])
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Price', fontsize=14)
    plt.title(name+"'s Close Stock Price Predictions ", fontsize=16)
    plt.legend(['Train','Test','Predictions'])
    plt.gcf().autofmt_xdate()
    mplcursors.cursor(hover=True)
    plt.show()



def decisions():
    newstock = stock('',pd.DataFrame(),pd.DataFrame(),pd.DataFrame())
    # user input for choice of first operation and error handling
    while True:
        decision=input('What operation do you want to perform: (H)our precise Stock Sample Predictions[Year To Date], General (S)tock Visualisation, (P)rediction of Stock Level, (I)ncome Statement Visualisation, (T)ry anther stock, (E)xit: ')
        match decision:
            case 's' | 'S':
                if stockdatacheck(newstock, "day","2022-04-05", "stockdf"):
                    generalvisualisation(newstock.stockdf,newstock.name)
            case 'p' | 'P':
                if stockdatacheck(newstock, "day","2022-04-05", "stockdf"):
                    stockprediction(newstock.stockdf,newstock.name)
            case 'h' | 'H':
                today = datetime.datetime.today()
                ytd=str(int(today.strftime("%Y"))-1)+"-"+today.strftime("%m")+"-"+today.strftime("%d")
                if stockdatacheck(newstock, "hour",ytd, "stockdf100k"):
                    stockprediction100k(newstock.stockdf100k,newstock.name)
            case 'i' | 'I':
                if newstock.incomedf.empty:
                    if newstock.name == '':
                        while True:
                            stockname=input('Which stock would you like to explore:')
                            try:
                                stockname = str(stockname)
                            except:
                                print('Please use the stock name abbreviation.')
                                continue
                            if len(stockname) < 1 or len(stockname) > 5:
                                print('Please use the stock name abbreviation.')
                                continue
                            inentries=allstatements.incomestatements.query("stock == '"+stockname+"'")
                            if inentries.empty:
                                print("Sorry "+stockname+"'s statement isn't in the dataset try again")
                                continue
                            else:
                                newstock.incomedf = inentries
                                newstock.name = stockname
                                break
                                
                        
                    else:
                        inentries=allstatements.incomestatements.query("stock == '"+stockname+"'")
                        if inentries.empty:
                            print("Sorry "+newstock.name+"'s statement isn't in the dataset try again")
                            newstock.name=''
                            continue
                    incomevisualisation(newstock.incomedf)
            case 't'|'T':
                while True:
                    decision=input('Are you sure you want to delete the previous stock data (Y/N): ')
                    match decision:
                        case 'y' | 'Y':
                            print("Deleting Stock Data")
                            newstock.name = ''
                            newstock.stockdf = pd.DataFrame()
                            newstock.incomedf = pd.DataFrame()
                            break
                        case 'n' | 'N':
                            break
                        case _: # error handling
                            print("Please use a character within the scope")
            case 'e'|'E':
                break
            case _: # error handling
                print("Please use a character within the scope")
if __name__ == "__main__":
    #Tensor flow compatibility fix
    data_adapter._is_distributed_dataset = _is_distributed_dataset
    allstatements = incomestatements(pd.read_csv('CaseStudy2/incomeStatementHistory_annually.csv').fillna(0))
    # Make call to polygon
    client = RESTClient('GFgcUeXub0aA_nl3siNyAXRz1GiuLPZa')
    decisions()
