#!/usr/bin/env python
# coding: utf-8

# In[1]:


import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import pandas as pd
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from jupyter_dash import JupyterDash

import calendar

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import plotly.offline as offline
import plotly.graph_objs as go
offline.init_notebook_mode(connected = True)

df_holi = pd.read_csv('holidays_events.csv')
df_oil = pd.read_csv('oil.csv')
df_stores = pd.read_csv('stores.csv')
df_trans = pd.read_csv('transactions.csv')
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')


stores_num = df_stores.shape[0]
type_store_num = len(df_stores.groupby('type').size())
product_num = len(df_train.groupby('family').size())
cities_num = len(df_stores.groupby('city').size())
state_num = len(df_stores.groupby('state').size())

#df_train['date'] = pd.to_datetime(df_train['date'])
df_train.sort_values(by=['date'], inplace=True, ascending = True)
first_date=(df_train["date"].iloc[0])#.strftime("%Y-%m-%d")
last_date=(df_train["date"].iloc[-1])#.strftime("%Y-%m-%d")

fig0=go.Figure()
fig0.add_trace(go.Scatter(
    x=[0,1,2,3.2,4.5,5.5],
    y=[1.7, 1.7, 1.7, 1.7, 1.7, 1.7],
    mode="text",
    text=["Mağaza","Şehir","Eyalet","Mağaza Tipi", "Ürün Ailesi", "Küme"],
    textposition="bottom center"
))
fig0.add_trace(go.Scatter(
    x=[0,1,2,3.2,4.5,5.5],
    y=[1.1, 1.1, 1.1, 1.1, 1.1, 1.1],
    mode="text", 
    text=["<span style='font-size:24px'><b>"+ str(stores_num) +"</b></span>", 
          "<span style='font-size:24px'><b>"+ str(cities_num) +"</b></span>",
          "<span style='font-size:24px'><b>"+ str(state_num) + "</b></span>",
          "<span style='font-size:24px'><b>"+ str(type_store_num) + "</b></span>",
          "<span style='font-size:24px'><b>"+ str(product_num) + "</b></span>",
          "<span style='font-size:24px'><b>17</b></span>"],
    textposition="bottom center"
))
fig0.add_hline(y=2.2, line_width=5, line_color='orange')
fig0.add_hline(y=0.3, line_width=3, line_color='orange')
fig0.add_trace(go.Scatter(
    x=[2.5],
    y=[-0.2],
    mode="text",
    text=["<span style='font-size:18px'><b>         Veri tarih aralığı " + first_date + " ile " + last_date + " arasındadır."+"</b></span>"],
    textposition="bottom center"
))


fig0.update_yaxes(visible=False)
fig0.update_xaxes(visible=False)
fig0.update_layout(showlegend=False, height=300, width=800, 
                  title='Mağazaların Satış Özeti', title_x=0.5, title_y=0.9,
                  xaxis_range=[-0.5,6.6], yaxis_range=[-1.2,2.2],
                  plot_bgcolor='#fafafa', paper_bgcolor='#fafafa',
                  font=dict(size=20, color='#323232'),
                  title_font=dict(size=28, color='#222'),
                  margin=dict(t=90,l=70,b=0,r=70), 
    )


# In[2]:


colors={}
def colorFader(c1,c2,mix=0): 
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)
c1='#FAA831' 
c2='#9A4800' 
n=9
for x in range(n+1):
    colors['level'+ str(n-x+1)] = colorFader(c1,c2,x/n) 
colors['background'] = '#232425'
colors['text'] = '#fff'


# In[3]:


colors


# In[4]:


# copying of train data and merging other data
df_train1 = df_train.merge(df_holi, on = 'date', how='left')
df_train1 = df_train1.merge(df_oil, on = 'date', how='left')
df_train1 = df_train1.merge(df_stores, on = 'store_nbr', how='left')
df_train1 = df_train1.merge(df_trans, on = ['date', 'store_nbr'], how='left')
df_train1 = df_train1.rename(columns = {"type_x" : "holiday_type", "type_y" : "store_type"})

df_train1['date'] = pd.to_datetime(df_train1['date'])
df_train1['year'] = df_train1['date'].dt.year
df_train1['month'] = df_train1['date'].dt.month
df_train1['week'] = df_train1['date'].dt.isocalendar().week
df_train1['quarter'] = df_train1['date'].dt.quarter
df_train1['day_of_week'] = df_train1['date'].dt.day_name()
df_train1.sample(n=4)


# In[5]:


df_fa_sa = df_train1.groupby('family').agg({"sales" : "mean"}).reset_index().sort_values(by='sales', ascending=False)[:10]
df_fa_sa['color'] = colors['level10']
df_fa_sa['color'][:1] = colors['level1']
df_fa_sa['color'][1:2] = colors['level2']
df_fa_sa['color'][2:3] = colors['level3']
df_fa_sa['color'][3:4] = colors['level4']
df_fa_sa['color'][4:5] = colors['level5']

fig1 = go.Figure(data=[go.Bar(x=df_fa_sa['sales'],
                             y=df_fa_sa['family'], 
                             marker=dict(color= df_fa_sa['color']),
                             name='Family', orientation='h',
                             text=df_fa_sa['sales'].astype(int),
                             textposition='auto',
                             hoverinfo='text',
                             hovertext=
                            '<b>Ürün Ailesi</b>:'+ df_fa_sa['family'] +'<br>' +
                            '<b>Satış Adedi</b>:'+ df_fa_sa['sales'].astype(int).astype(str) +'<br>' ,
                            # hovertemplate='Family: %{y}'+'<br>Sales: $%{x:.0f}'
                            )])
fig1.update_layout(title_text='En Çok Satılan 10 Ürün Ailesi',paper_bgcolor=colors['background'],plot_bgcolor=colors['background'],
                font=dict(
                size=14,
                color='white'))

fig1.update_yaxes(showgrid=False, categoryorder='total ascending')


# In[6]:


df_st_sa = df_train1.groupby('store_type').agg({"sales" : "mean"}).reset_index().sort_values(by='sales', ascending=False)
fig2=go.Figure(data=[go.Pie(values=df_st_sa['sales'], labels=df_st_sa['store_type'], name='Store type',
                     marker=dict(colors=[colors['level1'],colors['level3'],colors['level5'],colors['level7'],colors['level9']]), hole=0.7,
                     hoverinfo='label+percent+value', textinfo='label'
                    )])
fig2.update_layout(title_text='Ortalama Satışlar ve Mağaza Türlerinin Karşılaştırılması',paper_bgcolor="#000000",plot_bgcolor='#1f2c56',
                font=dict(
                size=14,
                color='white'))
fig2.update_yaxes(showgrid=False, categoryorder='total ascending')


# In[7]:


df_cl_sa = df_train1.groupby('cluster').agg({"sales" : "mean"}).reset_index().sort_values(by='sales', ascending=False)
df_cl_sa['color'] = colors['level10']
df_cl_sa['color'][:1] = colors['level1']
df_cl_sa['color'][1:2] = colors['level2']
df_cl_sa['color'][2:3] = colors['level3']
df_cl_sa['color'][3:4] = colors['level4']
df_cl_sa['color'][4:5] = colors['level5']
fig3 = go.Figure(data=[go.Bar(y=df_cl_sa['sales'],
                             x=df_cl_sa['cluster'], 
                             marker=dict(color= df_cl_sa['color']),
                             name='Cluster',
                             text=df_cl_sa['sales'].astype(int),
                             textposition='auto',
                             hoverinfo='text',
                             hovertext=
                            '<b>Küme</b>:'+ df_cl_sa['cluster'].astype(str) +'<br>' +
                            '<b>Satış Adedi</b>:'+ df_cl_sa['sales'].astype(int).astype(str) +'<br>' ,
                            # hovertemplate='Family: %{y}'+'<br>Sales: $%{x:.0f}'
                            )])
fig3.update_layout(title_text='Kümelerine Göre Satış Adetleri',paper_bgcolor=colors['background'],plot_bgcolor=colors['background'],
                font=dict(
                size=14,
                color='white'))

fig3.update_xaxes(tickmode = 'array', tickvals=df_cl_sa.cluster)
fig3.update_yaxes(showgrid=False)


# In[8]:


df_city_sa = df_train1.groupby('city').agg({"sales" : "mean"}).reset_index().sort_values(by='sales', ascending=False)
df_city_sa['color'] = colors['level10']
df_city_sa['color'][:1] = colors['level1']
df_city_sa['color'][1:2] = colors['level2']
df_city_sa['color'][2:3] = colors['level3']
df_city_sa['color'][3:4] = colors['level4']
df_city_sa['color'][4:5] = colors['level5']

fig4 = go.Figure(data=[go.Bar(y=df_city_sa['sales'],
                             x=df_city_sa['city'], 
                             marker=dict(color= df_city_sa['color']),
                             name='State',
                             text=df_city_sa['sales'].astype(int),
                             textposition='auto',
                             hoverinfo='text',
                             hovertext=
                            '<b>Şehir</b>:'+ df_city_sa['city'] +'<br>' +
                            '<b>Satış Adedi</b>:'+ df_city_sa['sales'].astype(int).astype(str) +'<br>' ,
                            # hovertemplate='Family: %{y}'+'<br>Sales: $%{x:.0f}'
                            )])
fig4.update_layout(title_text='Şehir Bazlı Ortalama Satış Adedi',paper_bgcolor=colors['background'],plot_bgcolor=colors['background'],
                font=dict(
                size=14,
                color='white'))

fig4.update_yaxes(showgrid=False, categoryorder='total ascending')


# In[9]:


df_state_sa = df_train1.groupby('state').agg({"sales" : "mean"}).reset_index().sort_values(by='sales', ascending=False)
df_state_sa['color'] = colors['level10']
df_state_sa['color'][:1] = colors['level1']
df_state_sa['color'][1:2] = colors['level2']
df_state_sa['color'][2:3] = colors['level3']
df_state_sa['color'][3:4] = colors['level4']
df_state_sa['color'][4:5] = colors['level5']
df_state_sa
fig5 = go.Figure(data=[go.Bar(y=df_state_sa['sales'],
                             x=df_state_sa['state'], 
                             marker=dict(color= df_state_sa['color']),
                             name='State',
                             text=df_state_sa['sales'].astype(int),
                             textposition='auto',
                             hoverinfo='text',
                             hovertext=
                            '<b>Eyalet</b>:'+ df_state_sa['state'] +'<br>' +
                            '<b>Satış Adedi</b>:'+ df_state_sa['sales'].astype(int).astype(str) +'<br>' ,
                            # hovertemplate='Family: %{y}'+'<br>Sales: $%{x:.0f}'
                            )])
fig5.update_layout(title_text='Eyalet Bazlı Ortalama Satış Adedi',paper_bgcolor=colors['background'],plot_bgcolor=colors['background'],
                font=dict(
                size=14,
                color='white'))

fig5.update_yaxes(showgrid=False, categoryorder='total ascending')


# In[10]:


df_day_sa = df_train1.groupby('date').agg({"sales" : "mean"}).reset_index()
fig6 = go.Figure(data=[go.Scatter(x=df_day_sa['date'], y=df_day_sa['sales'], fill='tozeroy', fillcolor='#FAA831', line_color='#bA6800'                                 )])
fig6.update_layout(title_text='Günlük Ortalama Satış Adedi',height=300,paper_bgcolor='#232425',plot_bgcolor='#232425',
                font=dict(
                size=12,
                color='white'))
fig6.update_xaxes(showgrid=False)
fig6.update_yaxes(showgrid=False)


# In[11]:


df_w_sa = df_train1.groupby('week').agg({"sales" : "mean"}).reset_index()
fig7 = go.Figure(data=[go.Scatter(x=df_w_sa['week'], y=df_w_sa['sales'], fill='tozeroy', fillcolor='#FAA831', line_color='#bA6800'
                                  ,mode='lines+markers')])


fig7.update_layout(title_text='Ortalama Haftalık Satış Adedi',height=300,paper_bgcolor='#232425',plot_bgcolor='#232425',
                font=dict(
                size=12,
                color='white'))
fig7.update_yaxes(showgrid=False)
fig7.update_xaxes(showgrid=False,tickmode = 'array', tickvals=df_w_sa.week, ticktext=[i for i in range(1,52)])


# In[12]:


df_mon_sa = df_train1.groupby('month').agg({"sales" : "mean"}).reset_index()
fig8 = go.Figure(data=[go.Scatter(x=df_mon_sa['month'], y=df_mon_sa['sales'], fill='tozeroy', fillcolor='#FAA831', line_color='#bA6800'
                                  ,mode='lines+markers')])


fig8.update_layout(title_text='Aylık Ortalama Satış Adedi',height=300,paper_bgcolor='#232425',plot_bgcolor='#232425',
                font=dict(
                size=12,
                color='white'))
fig8.update_yaxes(showgrid=False)
fig8.update_xaxes(showgrid=False,tickmode = 'array', tickvals=df_mon_sa.month)


# In[13]:


df_qu_sa = df_train1.groupby('quarter').agg({"sales" : "mean"}).reset_index()
fig9 = go.Figure(data=[go.Scatter(x=df_qu_sa['quarter'], y=df_mon_sa['sales'], fill='tozeroy', fillcolor='#FAA831', line_color='#bA6800'
                                  ,mode='lines+markers')])


fig9.update_layout(title_text='Ortalama 3 Aylık Satış Adetleri (Quarter)',height=300,paper_bgcolor='#232425',plot_bgcolor='#232425',
                font=dict(
                size=12,
                color='white'))
fig9.update_yaxes(showgrid=False)
fig9.update_xaxes(showgrid=False,tickmode = 'array', tickvals=df_qu_sa.quarter)


# In[14]:


df_y_sa = df_train1.groupby('year').agg({"sales" : "mean"}).reset_index()
fig10= go.Figure(data=[go.Scatter(x=df_y_sa['year'], y=df_y_sa['sales'], fill='tozeroy', fillcolor='#FAA831', line_color='#bA6800'
                                  ,mode='lines+markers')])


fig10.update_layout(title_text='Ortalama Yıllık Satış Adetleri',height=300,paper_bgcolor='#232425',plot_bgcolor='#232425',
                font=dict(
                size=12,
                color='white'))
fig10.update_yaxes(showgrid=False)
fig10.update_xaxes(showgrid=False,tickmode = 'array', tickvals=df_y_sa.year)


# In[15]:


df_c_s_sa = df_train1.groupby(['state','city']).agg({"sales" : "mean"}).reset_index()
fig11 = px.sunburst(df_c_s_sa, path=['state', 'city' ], 
                    values='sales',color='sales',
                    color_continuous_scale=[colors['level1'], colors['level10']])

fig11.update_layout(title_text='Eyaletler & Şehirleri',width = 700,paper_bgcolor='#232425',plot_bgcolor='#232425',font=dict(color=colors['text']))
fig11.show()


# In[16]:


import xgboost as xg


# In[17]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

oil_df = pd.read_csv("oil.csv")
holidays_df = pd.read_csv("holidays_events.csv")
stores_df = pd.read_csv("stores.csv")
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
transactions_df = pd.read_csv("transactions.csv")


# In[18]:


train_df.head()


# In[19]:


test_df.head()


# In[20]:


oil_df.head()


# In[21]:


holidays_df.head()


# In[22]:


stores_df.head()


# In[23]:


transactions_df.head()


# In[24]:


train_df.isnull().sum()


# In[25]:


train_df['date'] = pd.to_datetime(train_df['date'],  errors='coerce')


# In[26]:


plt.figure(figsize=(15,8))
plt.plot(train_df.date, train_df.sales)
plt.show()


# In[27]:


months_sales = train_df.groupby(train_df['date'].dt.strftime('%B'))['sales'].sum().sort_values()


# In[28]:


months_pormotions = train_df.groupby(train_df['date'].dt.strftime('%B'))['onpromotion'].sum().sort_values()


# In[29]:


stores_sales = train_df.groupby('store_nbr')['sales'].sum()


# In[30]:


stores_sales.sort_values(inplace=True)


# In[31]:


round(stores_sales, 2)


# In[32]:


train_df.date = pd.to_numeric(train_df.date)


# In[33]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder


# In[34]:


le = LabelEncoder()
train_df.family = le.fit_transform(train_df.family)


# In[35]:


train_df.head(2)


# In[36]:


X = train_df.drop(['sales', 'id', 'date'], axis = 1).values
y = train_df.sales.values


# In[37]:


X.shape


# In[38]:


y.shape


# In[39]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)


# In[40]:


regressor = DecisionTreeRegressor(max_depth=5, max_leaf_nodes=12, random_state=42)


# In[41]:


regressor.fit(X_train, y_train)


# In[42]:


regressor.score(X_train, y_train)


# In[43]:


regressor.score(X_test, y_test)


# In[44]:


test_df.head(20)


# In[45]:


test_df.family = le.fit_transform(test_df.family)


# In[46]:


test_df.head(20)


# In[47]:


X = test_df.drop(["id", "date"], axis=1).values
X.shape


# In[48]:


X[0]


# In[49]:


predictions = regressor.predict(X)


# In[50]:


test_df['sales'] = predictions


# In[51]:


test_df.head(28512)


# In[52]:


test_df.to_csv('DecisionTreeSubmission.csv', index=False)
print("submission successed")


# In[53]:


import xgboost as xg


# In[54]:


xgb_r = xg.XGBRegressor(objective ='reg:linear',
                  n_estimators = 10, seed = 123, max_depth=10)


# In[55]:


xgb_r.fit(X_train, y_train)


# In[56]:


xgb_r.score(X_test, y_test)


# In[57]:


xgb_r.score(X_train, y_train)


# In[58]:


test_df.family = le.fit_transform(test_df.family)


# In[59]:


test_df.head(20)


# In[60]:


X = test_df.drop(["id", "date"], axis=1).values
X.shape


# In[61]:


X[0]


# In[62]:


test_df['sales'] = predictions


# In[63]:


test_df.head(200)


# In[64]:


test_df.to_csv('XGBOOSTsubmission.csv', index=False)
print("submission successed")


# In[65]:


df_mon_sa = test_df.groupby('family').agg({"sales" : "mean"}).reset_index()
fig12 = go.Figure(data=[go.Scatter(x=df_mon_sa['family'], y=df_mon_sa['sales'], fill='tozeroy', fillcolor='#FAA831', line_color='#bA6800'
                                  ,mode='lines+markers')])


fig12.update_layout(title_text='XGBOOST Algoritmasına göre -Ürün Kategorisine Göre Ortalama Satış Adedi',height=300,paper_bgcolor='#232425',plot_bgcolor='#232425',
                font=dict(
                size=12,
                color='white'))
fig12.update_yaxes(showgrid=False)
fig12.update_xaxes(showgrid=False,tickmode = 'array', tickvals=df_mon_sa.family)


# In[66]:


test_df['date'] = pd.to_datetime(test_df['date'])
test_df['year'] = test_df['date'].dt.year
test_df['month'] = test_df['date'].dt.month
test_df['week'] = test_df['date'].dt.isocalendar().week
test_df['quarter'] = test_df['date'].dt.quarter
test_df['day_of_week'] = test_df['date'].dt.day_name()
test_df['day'] = test_df['date'].dt.day_name()


# In[67]:


df_mon_sa = test_df.groupby('week').agg({"sales" : "sum"}).reset_index()
fig13 = go.Figure(data=[go.Scatter(x=df_mon_sa['week'], y=df_mon_sa['sales'], fill='tozeroy', fillcolor='#FAA831', line_color='#bA6800'
                                  ,mode='lines+markers')])


fig13.update_layout(title_text='XGBOOST Algoritmasına göre -Hafta bazlı Toplam Satış Adedi',height=300,paper_bgcolor='#232425',plot_bgcolor='#232425',
                font=dict(
                size=12,
                color='white'))
fig13.update_yaxes(showgrid=False)
fig13.update_xaxes(showgrid=False,tickmode = 'array', tickvals=df_mon_sa.week)


# In[68]:


df_mon_sa = test_df.groupby('day').agg({"sales" : "sum"}).reset_index()
fig14 = go.Figure(data=[go.Scatter(x=df_mon_sa['day'], y=df_mon_sa['sales'], fill='tozeroy', fillcolor='#FAA831', line_color='#bA6800'
                                  ,mode='lines+markers')])


fig14.update_layout(title_text='XGBOOST Algoritmasına göre - Gün bazlı Toplam Satış Adedi',height=300,paper_bgcolor='#232425',plot_bgcolor='#232425',
                font=dict(
                size=12,
                color='white'))
fig14.update_yaxes(showgrid=False)
fig14.update_xaxes(showgrid=False,tickmode = 'array', tickvals=df_mon_sa.day)


# In[72]:


app = dash.Dash()

app.layout = html.Div(
    
      [html.H1('FAVORITA MAĞAZA - SATIŞ TAHMİNLEME & İSTATİSTİKSEL VERİ ANALİZİ- DASBOARD',   ### page header
        style={'textAlign': 'center', 'margin-bottom': 0.3}),
    
    html.Div([ dcc.Graph(id='graph1',figure=fig0, style={'textAlign': 'center', 'margin-bottom': 0.3}),
              
                      
                      dcc.Graph(id='graph2',figure=fig1) 
                       ,
                      
                     
                       dcc.Graph(id='graph3',figure=fig2) ,

                        dcc.Graph(id='graph4',figure=fig3) ,
                    
                       dcc.Graph(id='graph5',figure=fig4) ,

                      dcc.Graph(id='graph6',figure=fig5) ,
                     dcc.Graph(id='graph7',figure=fig6) ,

                     dcc.Graph(id='graph8',figure=fig7) ,
                     dcc.Graph(id='graph9',figure=fig8) ,
                     dcc.Graph(id='graph10',figure=fig9) ,
                     dcc.Graph(id='graph11',figure=fig10) ,
                     dcc.Graph(id='graph12',figure=fig11) ,
                        dcc.Graph(id='graph13',figure=fig12),
                                  dcc.Graph(id='graph14',figure=fig13)
                                  ,
                                  dcc.Graph(id='graph15',figure=fig14)
                           ], style= {'width': '75%', 'display': 'inline-block', 'textAlign':'center'

})

                       
                       
                       
                       
                      ])


# In[ ]:


if __name__ == "__main__":
    app.run_server()


# In[ ]:




