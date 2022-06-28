import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import json
import pickle 
from st_aggrid import AgGrid
import pandas as pd
from PIL import Image

from scipy import stats
from scipy.stats import norm, skew, kurtosis
import random
from statistics import mean, mode, median
import yfinance as yf
import warnings

warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
colours = ['#1F77B4', '#FF7F0E', '#2CA02C', '#DB2728', '#9467BD', '#8C564B', '#E377C2','#7F7F7F', '#BCBD22', '#17BECF','#67E568','#257F27','#08420D','#FFF000','#FFB62B','#E56124','#E53E30','#7F2353','#F911FF','#9F8CA6']

sns.set_palette(colours)
# %matplotlib inline
# import quantstats as qs
plt.rcParams['figure.figsize'] = (9, 6)
sns.set_style('darkgrid')
from stockdata import getDailysentiment,getStockSentiment
# import plotly.io as pio
# import plotly.express as px

# pio.templates.default = "ggplot2"

import snscrape.modules.twitter as sntwitter
from textblob import TextBlob

st.set_page_config(layout="wide")
import streamlit.components.v1 as components

##TOP PAGE
# st.title("Azure Cledo")
st.sidebar.image(Image.open('AzureCledo.png'))
st.subheader("Sentiment Based - Stock Predictor")
st.markdown('<style>h1{color:dark-grey;font-size:62px}</style>',unsafe_allow_html=True)
# st.sidebar.image(Image.open('text.png'))

@st.cache
def getData(start,end,ticker):
    stock = ticker
    data = yf.download(stock, start, end)
    data = data.reset_index(level=0)
    # data = data.reset_index(inplace=True)
    data['date'] = data['Date'].dt.date
    for i in range(10):
        data["price lag-" + str(i+1)]=data["Close"].shift(i+1)
    data['MA5'] = data["Close"].rolling(5, closed='left').mean()
    data['MA10'] = data["Close"].rolling(10, closed='left').mean()
    data['MA20'] = data["Close"].rolling(20, closed='left').mean()
    data['MA60'] = data["Close"].rolling(60, closed='left').mean()
    data = data.dropna(inplace=True)
    return data
# def getData(file):
#     df = pd.read_csv(file)
#     return df

from predict import makeModel,predPlot
import datetime
# import predict

menu = ["TWEET SCRAPER","STOCK ANALYSIS","STOCK PREDICTION"]
choice = st.sidebar.selectbox("Select Menu", menu)
if choice =="TWEET SCRAPER":
    # st.subheader('Custom Tweet Scraping')
    custom_search = st.expander(label='Search Parameters')
    with custom_search:
        search_term = st.text_input("Stock Name")
        from_date = st.date_input("from date",datetime.date(2022, 1, 1))
        end_date = st.date_input("until date")
        if st.button("Run Scraping"):
            data= getStockSentiment(search_term,from_date,end_date)
            AgGrid(data)
        # st.dataframe(df)
elif choice =="STOCK ANALYSIS":
    stocks = st.selectbox("Select Stock", ['ADRO.JK','PTBA.JK','TSLA'])
    start = st.date_input("from date",datetime.date(2022, 1, 1))
    end = st.date_input("until date")
    # df =getData(stocks+'.csv')
    df = yf.download(stocks, start, end)
    # df = getData(start,end,stocks)
    # st.write(end_date)
    # st.dataframe(df)
    data = df.reset_index(level=0)
    # data = data.reset_index(inplace=True)
    # data['date'] = data['Date'].dt.date
    for i in range(10):
        data["price lag-" + str(i+1)]=data["Close"].shift(i+1)
    data['MA5'] = data["Close"].rolling(5, closed='left').mean()
    data['MA10'] = data["Close"].rolling(10, closed='left').mean()
    data['MA20'] = data["Close"].rolling(20, closed='left').mean()
    data['MA60'] = data["Close"].rolling(60, closed='left').mean()
    # data = data.dropna(inplace=True)
    st.dataframe(data)
    data['price'] = data.Close
    # df3 =getData(stocks+'_tw.csv')
    df2 = getStockSentiment(stocks,start,end)
    # st.dataframe(df_sent)
    df3 = getDailysentiment(df2)
    fig = plt.figure(figsize=(10, 4))
    # sns.distplot(df['price'] , fit=norm);
    sns.distplot(data['price'] , fit=norm)
    # Get the fitted parameters used by the function
    (mu, sigma) = norm.fit(data['price'])
    print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

    #Now plot the distribution
    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
                loc='best')
    plt.ylabel('Frequency')
    plt.title('Price Distribution')
    st.pyplot(fig)

    c1,c2 = st.columns((1,1))
    with c1:
        from nltk.corpus import stopwords
        sw = stopwords.words('english')
        cek = df2['Text'].tolist()
        wordcloud = WordCloud (
                    background_color = 'white',
                    width = 650,
                    stopwords =set(sw+[stocks,'https','http','co','PT']),
                    height = 400
                        ).generate(' '.join(cek))
        fig0 = px.imshow(wordcloud,title=f'Wordcloud of {stocks} Tweets')
        st.plotly_chart(fig0)
    with c2:
        import plotly.graph_objects as go
        colors = ['gold', 'mediumturquoise', 'darkorange']
        neu = df2[df2['Sentiment']=='Neutral']
        pos = df2[df2['Sentiment']=='Positive']
        neg = df2[df2['Sentiment']=='Negative']

        fig2 = go.Figure(data=[go.Pie(labels=['Neutral','Positive','Negative'],
                                    values=[neu['Sentiment'].count(),pos['Sentiment'].count(),neg['Sentiment'].count()])])
        fig2.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,
                        marker=dict(colors=colors, line=dict(color='#000000', width=2)))
        fig2.update_layout(title='Tweet Sentiment Proportions')
        st.plotly_chart(fig2)

    # qs.extend_pandas()
    # stock = qs.utils.download_returns('ADRO.JK')
    # components.html(stock.plot_snapshot(title='Facebook Performance'))
    # st.write(qs.reports.html(stock, "ADRO.JK"))

elif choice == "STOCK PREDICTION":
    from datetime import datetime, timedelta, date
    # stocks = st.selectbox("Select Stock", ['adaro','ptba'])
    stocks = st.selectbox("Select Stock", ['ADRO.JK','PTBA.JK','TSLA'])
    # start = st.date_input("from date",datetime.date(2022, 1, 1))
    start = st.date_input("from date",date.today() - timedelta(days=550))
    end = st.date_input("until date")
    # df =getData(stocks+'.csv')
    df = yf.download(stocks, start, end)
    # df = getData(start,end,stocks)
    # st.write(end_date)
    # st.dataframe(df)
    data = df.reset_index(level=0)
    # data = data.reset_index(inplace=True)
    # data['date'] = data['Date'].dt.date
    for i in range(10):
        data["price lag-" + str(i+1)]=data["Close"].shift(i+1)
    data['MA5'] = data["Close"].rolling(5, closed='left').mean()
    data['MA10'] = data["Close"].rolling(10, closed='left').mean()
    data['MA20'] = data["Close"].rolling(20, closed='left').mean()
    data['MA60'] = data["Close"].rolling(60, closed='left').mean()
    # data = data.dropna(inplace=True)
    st.dataframe(data)
    data['price'] = data.Close
    data['date'] = data.Date
    # df = data[['Date','Close','price lag-1','price lag-2','price lag-3','price lag-4','price lag-5','price lag-6','price lag-7','price lag-8','price lag-9','price lag-10','MA5','MA10','MA20','MA60']]
    # df = df.dropna(inplace=False)
    # df =getData(stocks+'.csv')
    df = data.iloc[60:]
    # df2 =getData(stocks+'_fc.csv')
    # df2a = df2[df2['Type']=='Actual']
    # df2b = df2[df2['Type']=='Forecast']
    # df2b = df2[df2['Date']<'2022-06-25']
    pr = makeModel(df)
    st.dataframe(df)
    # st.pyplot(predPlot(df,pr))
    k1,k2 = st.columns((1,1))
    with k1:
        fig = go.Figure()
        # fig.add_trace(go.Scatter(x=df2a['Date'], y=df2a['Price_sentiment_positive'],
        #                     mode='lines+markers',
        #                     name='actual'))
        # fig.add_trace(go.Scatter(x=df2b['Date'], y=df2b['Price_sentiment_negative'],
        #                     mode='lines',
        #                     name='forecast min'))
        # fig.add_trace(go.Scatter(x=df2b['Date'], y=df2b['Price_sentiment_positive'],
        #                     mode='lines',
        #                     name='forecast max'))
        
        st.plotly_chart(fig)
    with k2:
        # size = 200*df['sentiment'].tolist()
        st.write('test')
        # size = [int(abs(x)*25) for x in df['sentiment'].tolist()]
        # fig4 = go.Figure()
        # fig4.add_trace(go.Scatter(x=df['date'], y=df['sentiment_positive'],
        #                 mode='markers',marker_size=size,
        #                 name='positive sentiment spot'))
        # fig4.add_trace(go.Scatter(x=df['date'], y=df['sentiment_negative'],
        #                 mode='markers', marker_size=size,
        #                 name='negative sentiment spot'))
        # st.plotly_chart(fig4)


    


    # components.html('''
    #     <div class='tableauPlaceholder' id='viz1650513950841' style='position: relative'><noscript><a href='#'><img alt='Geographic-Specific Risk Profile ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Pa&#47;PalmOilIndustry-RiskAnalysis&#47;Geographic-SpecificRiskProfile&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='PalmOilIndustry-RiskAnalysis&#47;Geographic-SpecificRiskProfile' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Pa&#47;PalmOilIndustry-RiskAnalysis&#47;Geographic-SpecificRiskProfile&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1650513950841');                    var vizElement = divElement.getElementsByTagName('object')[0];                    if ( divElement.offsetWidth > 800 ) { vizElement.style.width='1024px';vizElement.style.height='795px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.width='1024px';vizElement.style.height='795px';} else { vizElement.style.width='100%';vizElement.style.height='1877px';}                     var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>
    #     ''',height=795,
    #         width=1280)
    # maps_exp = st.expander('Palm oil mills maps')
    # with maps_exp:
    #     components.html('''
    #     <style>
    #     #wrap { width: 1020px; height: 900px; padding: 0; border: 0px solid grey; overflow: hidden; }
    #     #frame { width: 1680px; height: 900px;padding: 0; margin-top: -56px; border: 0px solid grey; overflow: hidden;}
    #     </style>
    #     <iframe id="frame" scrolling="no" class="wrapped-iframe" gesture="media"  allow="encrypted-media" allowfullscreen = "True"
    #     name="Framename" sandbox="allow-same-origin allow-scripts allow-popups allow-forms" 
    #     src="https://www.arcgis.com/home/webmap/viewer.html?useExisting=1&layers=3b28b8bcc5144cb685eb397979ea602f"
    #     style="width: 100%;">
    #     </iframe>
    #     '''
    #     ,height=795,
    #     width=1150)
    
