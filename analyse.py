from PA_new import Portfolio_Analysis
import pandas as pd
from result_plt import *

data_path = "E:\第五学期\公司金融\data\\"
data_path2 = "E:\第五学期\公司金融\data_processed\\"

# panel = pd.read_csv(data_path+'step1.csv',index_col=0)

panel = pd.read_csv(data_path+'v4.csv',index_col=0)

# cycle5_dict = {1 : 'Introduction',
#                2 : 'Growth',
#                3 : 'Mature',
#                4 : 'Shake-Out',
#                5 : 'Shake-Out',
#                6 : 'Shake-Out',
#                7 : 'Decline',
#                8 : 'Decline'}

# panel['cycle5'] = [cycle5_dict.get(x, np.nan) for x in panel['cycle']]

# panel = panel[panel['year']>=2000]





#%%单变量is_dividend

sorts = ["roa","eps","lnAsset","depreciation_asset_ratio"]
i = 1
for sort in sorts:
    key_var="is_dividend"
    sort_vars = sort
    date_var = 'year'
    quantiles = [0.005,0.2,0.4,0.6,0.8,0.995]
    weight_var = False
    how = False
    NW_lag = 4
    
    model = Portfolio_Analysis(panel, sort_vars, quantiles)            
    model.Port_Analyse(panel, key_var, sort_vars, date_var, quantiles,weight_var=weight_var,how=how,NW_lag=NW_lag)            
    section = model.sectional_result
    avg = model.average_result
    t_df = model.t_result
    
    result = pd.concat([avg,t_df.T],axis=1)
    result.columns = [sort_vars, 't-stat']
    if i == 1:
        result1 = result.T
    else:
        result1 = pd.concat([result1,result.T],axis=0)     
    i += 1
    
    
# col_isdi = pd.DataFrame({'control':['roa','roa']},index=['AVG','t-stat'])
# result1 = pd.concat([result.T,col_isdi],axis=1)

#%%单变量dividend_rate

sorts = ["roa","eps","lnAsset","depreciation_asset_ratio"]
i = 1
for sort in sorts:
    key_var="dividend_rate"
    sort_vars = sort
    date_var = 'year'
    quantiles = [0.005,0.2,0.4,0.6,0.8,0.995]
    weight_var = False
    how = False
    NW_lag = 4
    
    model = Portfolio_Analysis(panel, sort_vars, quantiles)            
    model.Port_Analyse(panel, key_var, sort_vars, date_var, quantiles,weight_var=weight_var,how=how,NW_lag=NW_lag)            
    section = model.sectional_result
    avg = model.average_result
    t_df = model.t_result
    
    result = pd.concat([avg,t_df.T],axis=1)
    result.columns = [sort_vars, 't-stat']
    if i == 1:
        result2 = result.T
    else:
        result2 = pd.concat([result2,result.T],axis=0)     
    i += 1

#%%indus_isd
keys = ["is_dividend","dividend_rate"]
i = 1
for key in keys:
    key_var=key
    sort_vars = "sector_code_name"
    date_var = 'year'
    quantiles = False
    weight_var = False
    how = False
    NW_lag = 4
    
    model = Portfolio_Analysis(panel, sort_vars, quantiles)            
    model.Port_Analyse(panel, key_var, sort_vars, date_var, quantiles,weight_var=weight_var,how=how,NW_lag=NW_lag)            
    section = model.sectional_result
    avg = model.average_result
    t_df = model.t_result
    cls_map = model.cls_map
    sectors = list(cls_map[0].iloc[:,0])+['HML']
    
    result = pd.concat([avg,t_df.T],axis=1)
    result.columns = [key, 't-stat']
    result.index = sectors
    result = result.iloc[:-1,:]

    if i == 1:
        result3 = result.T
    else:
        result3 = pd.concat([result3,result.T],axis=0)     
    i += 1

    

#%%company_type
keys = ["is_dividend","dividend_rate"]
i = 1
for key in keys:
    key_var=key
    sort_vars = "company_type"
    date_var = 'year'
    quantiles = False
    weight_var = False
    how = False
    NW_lag = 4
    
    model = Portfolio_Analysis(panel, sort_vars, quantiles)            
    model.Port_Analyse(panel, key_var, sort_vars, date_var, quantiles,weight_var=weight_var,how=how,NW_lag=NW_lag)            
    section = model.sectional_result
    avg = model.average_result
    t_df = model.t_result
    cls_map = model.cls_map
    sectors = list(cls_map[0].iloc[:,0])+['HML']
    
    result = pd.concat([avg,t_df.T],axis=1)
    result.columns = [key, 't-stat']
    result.index = sectors
    result = result.iloc[:-1,:]

    if i == 1:
        result4 = result.T
    else:
        result4 = pd.concat([result4,result.T],axis=0)     
    i += 1


#%%cycle
keys = ["is_dividend","dividend_rate"]
i = 1
for key in keys:
    key_var=key
    sort_vars = "cycle"
    date_var = 'year'
    quantiles = False
    weight_var = False
    how = False
    NW_lag = 4
    
    model = Portfolio_Analysis(panel, sort_vars, quantiles)            
    model.Port_Analyse(panel, key_var, sort_vars, date_var, quantiles,weight_var=weight_var,how=how,NW_lag=NW_lag)            
    section = model.sectional_result
    avg = model.average_result
    t_df = model.t_result
    cls_map = model.cls_map
    sectors = list(cls_map[0].iloc[:,0])+['HML']
    
    result = pd.concat([avg,t_df.T],axis=1)
    result.columns = [key, 't-stat']
    result.index = sectors
    result = result.iloc[:-1,:].sort_index()
    
    if i == 1:
        result5 = result.T
    else:
        result5 = pd.concat([result5,result.T],axis=0)     
    i += 1

#%%cycle5
keys = ["is_dividend","dividend_rate"]
i = 1
for key in keys:
    key_var=key
    sort_vars = "cycle5"
    date_var = 'year'
    quantiles = False
    weight_var = False
    how = False
    NW_lag = 4
    
    model = Portfolio_Analysis(panel, sort_vars, quantiles)            
    model.Port_Analyse(panel, key_var, sort_vars, date_var, quantiles,weight_var=weight_var,how=how,NW_lag=NW_lag)            
    section = model.sectional_result
    avg = model.average_result
    t_df = model.t_result
    cls_map = model.cls_map
    sectors = list(cls_map[0].iloc[:,0])+['HML']
    
    result = pd.concat([avg,t_df.T],axis=1)
    result.columns = [key, 't-stat']
    result.index = sectors
    result = result.iloc[:-1,:].sort_index()
    
    if i == 1:
        result6 = result.T
    else:
        result6 = pd.concat([result6,result.T],axis=0)     
    i += 1


#%%board_type
keys = ["is_dividend","dividend_rate"]
i = 1
for key in keys:
    key_var=key
    sort_vars = "board_type"
    date_var = 'year'
    quantiles = False
    weight_var = False
    how = False
    NW_lag = 4
    
    model = Portfolio_Analysis(panel, sort_vars, quantiles)            
    model.Port_Analyse(panel, key_var, sort_vars, date_var, quantiles,weight_var=weight_var,how=how,NW_lag=NW_lag)            
    section = model.sectional_result
    avg = model.average_result
    t_df = model.t_result
    cls_map = model.cls_map
    sectors = list(cls_map[0].iloc[:,0])+['HML']

    
    result = pd.concat([avg,t_df.T],axis=1)
    result.columns = [key, 't-stat']
    result.index = sectors
    
    if i == 1:
        result6 = result.T
    else:
        result6 = pd.concat([result6,result.T],axis=0)     
    i += 1
    
    result7.to_excel(data_path2+'board_type.xlsx')
#%%bivar_

# panel = panel[panel['year']>=2021]

key_var="dividend_rate"
sort_vars = ["lnAsset","roa"]
date_var = 'year'
quantiles = [[0.005,0.2,0.4,0.6,0.8,0.995],[0.005,0.2,0.4,0.6,0.8,0.995]]
weight_var = False
how = "dependent"
NW_lag = 4

model = Portfolio_Analysis(panel, sort_vars, quantiles)            
model.Port_Analyse(panel, key_var, sort_vars, date_var, quantiles,weight_var=weight_var,how=how,NW_lag=NW_lag)            
section = model.sectional_result
avg = model.average_result
t_df = model.t_result

result8 = avg.iloc[:-2,:-2]
plt_3d_bar(result8, index_name='lnAsset', col_name='roa', key_name='dividend rate')

#%% trasition_matrix

