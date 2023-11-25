import pandas as pd 
import pyreadstat
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_hac


# data_path = "E:\第五学期\金融计量学\大作业\data\\"

# data_path2 = "E:\第五学期\金融计量学\大作业\data_processed\\"

# panel = pd.read_csv(data_path2+"panel.csv",index_col=0)

class Portfolio_Analysis():
    '''
    结果主要有三个，
    sectional_result:横截面组合分析结果
    average_result:时间序列均值组合分析结果
    t_result:时间序列均值显著性结果
    number_stats:横截面各组合样本数统计
    '''
    
    def __init__(self, data, sort_vars, quantiles = [0, 0.2, 0.5, 0.8, 1]):
        
        self.data = data
        self.sort_vars = sort_vars
        self.quantiles = quantiles

        self.sectional_result = None
        self.sections_stocks = None
        self.section_stocks_recent = {}
        
        self.average_result = None
        self.t_result = None
        self.number_stats = None
        self.average_num_stats = None
        self.number_ratio_stats = None
        self.average_number_ratio_stats = None
        self.alpha_result = None
        self.transition_matrix = None
        
        self.cls_map = []
        self.map_flag = 0
        self.var_flag = []
        
        if type(self.quantiles) == bool: 
            self.data = self.data.dropna(subset=sort_vars)
            self.map_flag = 1
            self.get_cls_map(self.data[self.sort_vars])

            
        elif (len(self.quantiles) == 2) and (any(isinstance(quantile, bool) for quantile in self.quantiles)):
            self.data = self.data.dropna(subset=sort_vars)
            i = 0
            for quantile in self.quantiles:
                if type(quantile) == bool:
                    self.var_flag.append(0)
                    self.map_flag += 1
                    self.get_cls_map(self.data[self.sort_vars[i]])
                else:
                    self.var_flag.append(1)
                    
                i += 1
        
        elif (len(self.quantiles) == 2) and (not (any(isinstance(quantile, bool) for quantile in self.quantiles))):
            self.var_flag = [1,1]
    
        
    def get_cls_map(self, ser):

            all_cls = ser.iloc[:].unique()
            cls_code = list(range(1, len(all_cls)+1))
            map_df = pd.concat([pd.Series(all_cls), pd.Series(cls_code)],axis=1)
            self.cls_map.append(map_df)
        
    
    def cls_to_int(self, x, flag = 0):
        
        cls_map = dict(zip(self.cls_map[flag][0], self.cls_map[flag][1]))
        
        return cls_map[x]
        
    
    def Sectional_PA(self, data, key_var, sort_vars, quantiles , weight_var = False, how = False, comp_var = False):           
        
        if weight_var == False:            
            data = data.replace([np.inf,-np.inf],np.nan)
            data = data.dropna(subset=key_var)
            data = data.dropna(subset=sort_vars)

        else:            
            data = data.replace([np.inf,-np.inf],np.nan)
            data.dropna(subset=[key_var,weight_var],inplace=True)
            data.dropna(subset=sort_vars,inplace=True)
        
        
        if type(sort_vars) == str : 
            
            if quantiles != False:
                label = [i for i in range(1,len(quantiles))]
                data['group_label'] = pd.qcut(data[sort_vars],quantiles,labels=label)
            
            else:
                label = list(self.cls_map[0].iloc[:,1])
                data['group_label'] = data[sort_vars].apply(self.cls_to_int)         
            
            if comp_var != False:
                self.section_stocks_recent = {}
                for lab in label:
                    self.section_stocks_recent[lab] = data[data['group_label'] == lab][comp_var].to_list()
            
            if weight_var == False:
                num_ser = data['group_label'].value_counts()
                groups = data.groupby('group_label')[key_var]
                ser = groups.mean()         
            
            else:
                num_ser = data['group_label'].value_counts()
                data['key_var_new'] = data[key_var]*data[weight_var]
                groups = data.groupby('group_label')[['key_var_new', weight_var]]
                result = groups.agg({'key_var_new' : 'sum', weight_var : 'sum'})                
                ser = result['key_var_new']/result[weight_var]
            
            index = list(ser.index)
            mismatch_flag = set(label)-set(index)
            
            if len(mismatch_flag) != 0:
                ser_nan = pd.Series([np.nan]*len(label),index=label)
                ser = ser_nan.combine_first(ser)

            HML = ser.iloc[-1] - ser.iloc[0]
            
            ser = pd.concat([ser, pd.Series([HML])], axis=0)
            col_names = ["Port{}".format(i) for i in label]+["HML"]
            ser.index = col_names
            
            final_result = [ser, num_ser]
            
            return final_result
        
        
        elif type(sort_vars) == list :
            
            if len(quantiles) == 1:
                quantiles = [quantiles, quantiles]
            
            labels_list = []
            flag = 1
            for quantile in quantiles:
                if type(quantile) == bool:
                    if flag == 1 or self.var_flag[0] == 1:
                        labels_list.append(list(self.cls_map[0].iloc[:,1]))   
                    else :
                        labels_list.append(list(self.cls_map[1].iloc[:,1]))
                    
                else:
                    labels_list.append([i for i in range(1,len(quantiles[flag-1]))])
                    
                flag += 1
            
            label1 = labels_list[0]
            label2 = labels_list[-1]
            
            
            if weight_var == False:
                weight_var = 'weight_var'
                data[weight_var] = np.ones([data.shape[0]])
            
            data['key_var_new'] = data[key_var]*data[weight_var]
            
            if how == "independent":
                flag2 = 0
                for i in [0,1]:
                    if self.var_flag[i] == 1:
                        data['group_label{}'.format(i+1)] = pd.qcut(data[sort_vars[i]],quantiles[i], labels=labels_list[i])
                    else:
                        data['group_label{}'.format(i+1)] = data[sort_vars[i]].apply(self.cls_to_int, args=[flag2])
                        flag2 += 1      
                        
                groups = data[['group_label1','group_label2','key_var_new',weight_var]].set_index(['group_label1','group_label2']).sort_index()
                
                
            else: 
                if self.var_flag[0] == 1:
                    data['group_label1'] = pd.qcut(data[sort_vars[0]],quantiles[0], labels=label1)
                    data = data.dropna(subset=['group_label1'])
                    
                else:
                    
                    data['group_label1'] = data[sort_vars[0]].apply(self.cls_to_int)
                
                groups1 = data.groupby(['group_label1'])[["key_var_new",sort_vars[-1],weight_var,'group_label1']]
                
                init_flag = 1
                for label, group in groups1:
                    
                    if self.var_flag[1] == 1:
                        group['group_label2'] = pd.qcut(data[sort_vars[-1]],quantiles[-1], labels=label2)
                        group = group.dropna(subset=['group_label2'])
                        
                    else:
                        
                        group['group_label2'] = group[sort_vars[-1]].apply(self.cls_to_int, args=[1])
                    
                    if init_flag == 1:
                        groups_gross = group
                        init_flag += 1
                    
                    else:    
                        groups_gross = pd.concat([groups_gross,group], axis = 0)
                        
                
                groups = groups_gross[['group_label1','group_label2',"key_var_new",weight_var]].set_index(['group_label1','group_label2']).sort_index()
            
            gps = groups.reset_index() 
            num_tab = pd.crosstab(gps['group_label1'], gps['group_label2'])   
            rows, cols =np.where(num_tab==0)
            zero_flag = list(zip(rows,cols))
            
            #判断分类变量在不同截面上是否都有
            index1 = groups.index.get_level_values('group_label1')
            index2 = groups.index.get_level_values('group_label2')
            mismatch_flag1 = set([])
            mismatch_flag2 = set([])
            i = 1
            for var in self.var_flag:
                if i == 1 and var == 0:
                    mismatch_flag1 = set(label1) - set(index1)
                
                elif i == 2 and var == 0:                         
                    mismatch_flag2 = set(label2) - set(index2)
                i += 1
            
            key_matrix = np.ones([len(label1)+2,len(label2)+2])
            
            for i in range(len(label1)+2):
                for j in range(len(label2)+2):  
                    
                    if (i<=len(label1)-1) and (j<=len(label2)-1) :

                        if ((i,j) not in zero_flag) and ((i+1) not in mismatch_flag1) and ((j+1) not in mismatch_flag2) :
                            group = groups.loc[(i+1,j+1)]          
                            key_matrix[i,j] = group['key_var_new'].sum()/group[weight_var].sum()
                        else:
                            key_matrix[i,j] = np.nan
                        
                    elif i == len(label1) and (j<=len(label2)-1):
                        if ((i+1) not in mismatch_flag1) and ((j+1) not in mismatch_flag2) :
                            group = groups.xs(j+1, level=1)
                            key_matrix[i,j] = group['key_var_new'].sum()/group[weight_var].sum()
                        else:
                            key_matrix[i,j] = np.nan
                            
                    elif i == len(label1) and (j==len(label2)+1):
                        if ((i+1) not in mismatch_flag1) and ((j+1) not in mismatch_flag2) :
                            h_group = groups.xs(len(label2), level=1)
                            m_group = groups.xs(1, level=1)
                            key_matrix[i,j] = (h_group['key_var_new'].sum() - m_group['key_var_new'].sum())/(h_group[weight_var].sum()+m_group[weight_var].sum())                        
                        else:
                            key_matrix[i,j] = np.nan
                            
                    elif j == len(label2) and (i==len(label2)+1):
                        if ((i+1) not in mismatch_flag1) and ((j+1) not in mismatch_flag2) :    
                            h_group = groups.xs(len(label1), level=0)
                            m_group = groups.xs(1, level=0)
                            key_matrix[i,j] = (h_group['key_var_new'].sum() - m_group['key_var_new'].sum())/(h_group[weight_var].sum()+m_group[weight_var].sum())                        
                        else:
                            key_matrix[i,j] = np.nan
                            
                    elif j == len(label2) and (i<=len(label1)-1):
                        if ((i+1) not in mismatch_flag1) and ((j+1) not in mismatch_flag2) :
                            group = groups.xs(i+1, level=0)
                            key_matrix[i,j] = group['key_var_new'].sum()/group[weight_var].sum()
                        else:
                            key_matrix[i,j] = np.nan
                            
                    elif (i == len(label1)+1) and (j<=len(label2)-1):
                        HML = key_matrix[i-2,j] - key_matrix[0,j]
                        key_matrix[i,j] = HML
                      
                    elif (j == len(label2)+1) and (i<=len(label1)-1):
                        HML = key_matrix[i,j-2] - key_matrix[i,0]
                        key_matrix[i,j] = HML
                        
            
            index_names = ["Port{}".format(i) for i in label1]+["AVG", "HML"]
            col_names = ["Port{}".format(i) for i in label2]+["AVG", "HML"]
            
            df = pd.DataFrame(key_matrix,index=index_names,columns=col_names)
            
            if len(mismatch_flag1) == 0 and len(mismatch_flag2) == 0: 
                num_tab.index = index_names[0:-2]
                num_tab.columns = col_names[0:-2]
            elif len(mismatch_flag1) != 0:
                for _ in range(len(mismatch_flag1)):
                    nan_row = pd.DataFrame(np.nan,index=list(mismatch_flag1), columns=label2)
                    num_tab = pd.concat([num_tab,nan_row],axis=0)
                mismatch_flag2 = []
            elif len(mismatch_flag2) != 0:
                for i in range(len(mismatch_flag2)):
                    nan_col = mismatch_flag2[i]
                    num_tab[nan_col] = np.nan
                
            final_result = [df, num_tab]
            
            return final_result
            
        else:
            raise ValueError("请设置匹配的quantiles 和 sort_vars")
           
    
    
    def cal_NW_t(self, ser, lag):
        
        ser = pd.Series(ser)
        data = np.array(ser.dropna())
        X = np.ones_like(data)  # 设计矩阵只有截距项
        model = sm.OLS(data, X)
        results = model.fit()
        
        nw_cov = cov_hac(results, nlags=lag)  
        nw_se = np.sqrt(np.diag(nw_cov))

        t_stat_nw = results.params / nw_se

        return t_stat_nw
        
        # data = np.array(ser)
        # mean_data = np.mean(data)
        # deviations = data - mean_data
    
        # def autocovariance(data, k):
        #     n = len(data)
        #     mean = np.mean(data)
        #     return np.sum((data[:n-k] - mean) * (data[k:] - mean)) / n
            
        # lag_length = lag 
        # autocovariances = [autocovariance(deviations, k) for k in range(lag_length + 1)]
        # nw_variance = autocovariances[0] + 2 * sum((1 - (k / (lag_length + 1))) * autocovariances[k] for k in range(1, lag_length + 1))
        # nw_se = np.sqrt(nw_variance) / len(data)
        
        # t_stat_nw = (mean_data - 0) / nw_se
        
        # return t_stat_nw
        
        
        
    def Port_Analyse(self, data, key_var, sort_vars, date_var, quantiles = [0, 0.2, 0.5, 0.8, 1], weight_var = False, how = False, NW_lag = 4, comp_var = False):
        '''
        data中应包含key_var/sort_vars/mkt(如果需要加权的)
        key_var是要研究的关键变量
        sort_vars是分组变量，当sort_vars有两个时（列表形式），会计算HML和AVG.当为dependent时，会先根据第一个sort_var分组，再在组内根据第二个sort_var分组
        quantiles是分组的分位数，要包含0和1，比如[0, 0.2, 0.5, 0.8, 1]
        how有两个取值，一个是dependent,一个是independent,如果进行双变量组合分析，需要指定how
        weight是所用权重，默认等权，如果使用自定义加权,需指定加权指标的字段名
        
        '''                
        if how == False:
            groups = data.groupby(date_var)
            sectional_result={}
            sectional_num_result = {}
            sections_stocks = {}
            
            for date, group in groups: 
                ser = self.Sectional_PA(group, key_var=key_var, sort_vars=sort_vars, quantiles=quantiles, weight_var = weight_var, how = how, comp_var=comp_var)
                sectional_result[date] = ser[0]
                sectional_num_result[date] = ser[1]
                sections_stocks[date] = self.section_stocks_recent
            
            sectional_result_df = pd.DataFrame(sectional_result).T
            sectional_num_result_df = pd.DataFrame(sectional_num_result).T

            self.sectional_result = sectional_result_df
            self.sections_stocks = sections_stocks
            self.number_stats = sectional_num_result_df

            average_df = sectional_result_df.mean(axis=0)
            average_num_df = sectional_num_result_df.mean(axis = 0)
            
            t_df = sectional_result_df.apply(self.cal_NW_t, args = [NW_lag])
            self.average_result = average_df
            self.average_num_stats = average_num_df
            self.t_result = t_df
            
        else:
            groups = data.groupby(date_var)
            sectional_result = {}
            sectional_num_result = {}
            for date, group in groups: 
                # print(date)
                df = self.Sectional_PA(group, key_var=key_var, sort_vars=sort_vars, quantiles=quantiles, weight_var = weight_var, how = how)
                index_names = df[0].index
                col_names = df[0].columns
                sectional_result[date] = df[0]
                sectional_num_result[date] = df[1]

                
            self.sectional_result = sectional_result
            self.number_stats = sectional_num_result
            
            key_3darr = np.array([df.to_numpy() for df in sectional_result.values()])            
            num_3darr = np.array([df.to_numpy() for df in sectional_num_result.values()])            
            
            shape = key_3darr[0,:,:].shape
            t_matrix = np.ones(shape)
            avg_matrix = np.ones(shape)
            avg_num_matrix = np.ones([shape[0]-2,shape[1]-2])
            
            for i in range(shape[0]):
                for j in range(shape[1]):
                    
                    # if not ((i >= shape[0] - 2) and (j >= shape[1] - 2)):
                    ser = key_3darr[:,i,j]
                    t_nw = self.cal_NW_t(ser=ser, lag=NW_lag)
                    t_matrix[i,j] = t_nw
                    avg_matrix[i,j] = np.nanmean(ser)
                     
                    if ((i < shape[0] - 2) and (j < shape[1] - 2)):
                        num_ser = num_3darr[:,i,j]
                        avg_num_matrix[i,j] = np.nanmean(num_ser)

                     
            self.average_result = pd.DataFrame(avg_matrix, index=index_names, columns=col_names)
            self.average_num_stats = pd.DataFrame(avg_num_matrix, index=index_names[0:-2], columns=col_names[0:-2])
            self.t_result = pd.DataFrame(t_matrix, index=index_names, columns=col_names)

        
    
    def cal_trans_matrix(self, period = 12):
        
        stocks = self.sections_stocks
        dates = list(stocks.keys())
        
        samples = []
        for i in range(len(dates)-period):
            key1 = dates[i]
            key2 = dates[i+period]
            origin = stocks[key1]
            end = stocks[key2]
            
            ports = list(origin.keys())
            num = len(ports)
            
            origin_all = []
            end_all = []
            for port in ports:
                origin_all = origin_all + origin[port]
                end_all = end_all + end[port]
            
            common_stocks = set(origin_all).intersection(set(end_all))
            
            matrix = np.ones([num,num])
            
            for j in range(num):        #origin
                for k in range(num):        #end
                    
                    ratio = len(set(origin[ports[j]]).intersection(end[ports[k]]))/len(set(common_stocks).intersection(set(origin[ports[j]])))
                    matrix[j,k] = ratio
            
            samples.append(matrix)
            
        arr3d = np.array(samples)
        df = pd.DataFrame(np.mean(arr3d,axis=0))
        
        names = ["Port{}".format(port) for port in ports]
        df.index = names
        df.columns = names
        
        self.transition_matrix = df
            
        
    def cal_alpha(self, factor_data, factor_vars, NW_lag = 4, weight_f = False, date_se = False):
        
        if date_se == False:  #暂时未开发设置日期的模式
            if len(self.var_flag) == 0 :
                dates = self.sectional_result.index
            else:
                dates = list(self.sectional_result.keys())
                
        data = factor_data.loc[dates,factor_vars]
        
        if len(self.var_flag) == 0 :                    
            cols = list(self.sectional_result.columns)
        else :
            cols = list(self.average_result.columns[:-2])+[(self.average_result.columns[-1])]
            
        alphas = []
        t_values = []
        
        for i in range(len(cols)):
            
            if self.var_flag == [] :
                ser = self.sectional_result.loc[:,cols[i]]
                
            else:
                arr3d = np.array(list(self.sectional_result.values()))
                if i <= len(cols)-2:
                    ser = arr3d[:,-2,i]
                    
                elif i == len(cols)-1:
                    ser = arr3d[:,-2,-1]

            X = sm.add_constant(data)
            model = sm.OLS(ser, X)
            results = model.fit(cov_type='HAC',cov_kwds={'maxlags':NW_lag})
            
            alpha = results.params[0]
            alphas.append(alpha)
            
            nw_se = results.bse[0]
            t_value = alpha / nw_se
            t_values.append(t_value)
            
        data2 =  {"alpha" : alphas,
                  "t-stat" : t_values}
        result = pd.DataFrame(data2).T
        result.columns = cols
        
        self.alpha_result = result
            

    # def summary(self,avg=True,t=True, num=False):
        
        
        