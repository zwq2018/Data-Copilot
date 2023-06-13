import tushare as ts
import matplotlib.pyplot as plt
import pandas as pd
import os
import random
from matplotlib.ticker import MaxNLocator
import matplotlib.font_manager as fm
from prettytable import PrettyTable
from blessed import Terminal
import time
from datetime import datetime, timedelta
import numpy as np
import mplfinance as mpf
from prettytable import PrettyTable
from typing import Optional
import matplotlib.font_manager as fm
from matplotlib.lines import Line2D
from typing import Union, Any
import json

token=os.getenv("TUSHARE_TOKEN")
pro = ts.pro_api(token)
all_atomic_api = {}


#######################################################################################################################
df = pro.fina_indicator(ts_code = "600000.SH",start_date = "20200104",end_date = "20220104",fields=["ts_code","end_date","eps","current_ratio","quick_ratio","inv_turn","netprofit_margin","grossprofit_margin","roe","roa","roic","debt_to_assets","netprofit_yoy","dt_netprofit_yoy"])
# 取出第一行和最后一行的数据
df_sample_str = df.iloc[[0,-1],:].to_string(header=False, index=False)


columns = df.columns.tolist()
columns_means= ['股票代码','报告期','每股收益','流动比率','速动比率','存货周转率','销售净利率','销售毛利率','净资产收益率','总资产净利率','投入资本回报率','资产负债率','净利润同比增长率','扣非净利润同比增长率']

columns_dict = dict(zip(columns, columns_means))

atomic_api = {"func_desc":None,"Column_name":None,"example_input":None,"output_first_and_last_row":None}
atomic_api["func_desc"] = "财务指标"
atomic_api["Column_name"] = columns_dict
atomic_api["example_input"] = "pro.fina_indicator(ts_code = \"600000.SH\",start_date = \"20200104\",end_date = \"20220104\",fields=[\"ts_code\",\"end_date\",\"eps\",\"current_ratio\",\"quick_ratio\",\"inv_turn\",\"netprofit_margin\",\"grossprofit_margin\",\"roe\",\"roa\",\"roic\",\"debt_to_assets\",\"netprofit_yoy\",\"dt_netprofit_yoy\"])"

atomic_api["output_first_and_last_row"] = df_sample_str

all_atomic_api["pro.fina_indicator"] = atomic_api

#######################################################################################################################

df = pro.stock_company(ts_code = '600230.SH', fields=[
        "ts_code","exchange","chairman", "manager","secretary", "reg_capital","setup_date", "province","city","introduction",
        "website", "email","office","employees","main_business","business_scope"])

df_sample_str = df.iloc[0, :].to_string(header=False, index=False)
columns = df.columns.tolist()
columns_means = ['股票代码','交易所代码','法人代表','总经理','董秘','注册资本','注册日期','所在省份','所在城市','公司介绍','公司主页','电子邮件','办公室地址','员工人数','主要业务及产品','经营范围'  ]
columns_dict = dict(zip(columns, columns_means))
atomic_api = {"func_desc":None,"Column_name":None,"example_input":None,"output_first_and_last_row":None}
atomic_api["func_desc"] = "获取上市公司基本信息如公司业务描述,员工人数等基本信息"
atomic_api["Column_name"] = columns_dict
atomic_api["example_input"] = "pro.stock_company(ts_code = '600230.SH', fields=[\"ts_code\",\"exchange\",\"chairman\", \"manager\",\"secretary\", \"reg_capital\",\"setup_date\", \"province\",\"city\",\"introduction\",\"website\", \"email\",\"office\",\"employees\",\"main_business\",\"business_scope\"])"
atomic_api["output_first_and_last_row"] = df_sample_str

all_atomic_api["pro.stock_company"] = atomic_api


#######################################################################################################################

df = pro.daily_basic(ts_code = "600230.SH",start_date = "20180726",end_date = "20200726", fields=[
    "ts_code", "trade_date","turnover_rate","turnover_rate_f","volume_ratio",
    "pe_ttm","pb","ps_ttm","dv_ttm","total_share",
    "float_share","free_share","total_mv","circ_mv"])

df_sample_str = df.iloc[[0, -1], :].to_string(header=False, index=False)
columns = df.columns.tolist()
columns_means = ['股票代码','交易日期','换手率(总股本)','换手率(自由流通股本)','量比','市盈率(动态)','市净率','市销率(动态)','股息率(动态)','总股本','流通股本','自由流通股本','总市值','流通市值'  ]

columns_dict = dict(zip(columns, columns_means))
atomic_api = {"func_desc":None,"Column_name":None,"example_input":None,"output_first_and_last_row":None}
atomic_api["func_desc"] = "获取股票每日基本指标例如换手率,市盈率市净率股息率等"
atomic_api["Column_name"] = columns_dict
atomic_api["example_input"] = "pro.daily_basic(ts_code = \"600230.SH\",start_date = \"20180726\",end_date = \"20200726\", fields=[\"ts_code\", \"trade_date\",\"turnover_rate\",\"turnover_rate_f\",\"volume_ratio\",\"pe_ttm\",\"pb\",\"ps_ttm\",\"dv_ttm\",\"total_share\",\"float_share\",\"free_share\",\"total_mv\",\"circ_mv\"])"

atomic_api["output_first_and_last_row"] = df_sample_str

all_atomic_api["pro.daily_basic"] = atomic_api

#######################################################################################################################

df = pro.stk_factor(ts_code="600000.SH",start_date= "20220520",end_date= "20230520",
                    fields=["ts_code","trade_date","close","macd_dif","macd_dea","macd","kdj_k","kdj_d","kdj_j",
                            "rsi_6","rsi_12","rsi_24","boll_upper","boll_mid","boll_lower","cci"])
df_sample_str = df.iloc[[0, -1], :].to_string(header=False, index=False)
columns = df.columns.tolist()
columns_means = ['股票代码','交易日期','收盘价','MACD_DIF','MACD_DEA','MACD','KDJ_K','KDJ_D','KDJ_J','RSI_6','RSI_12','RSI_24','BOLL_UPPER','BOLL_MID','BOLL_LOWER','CCI'  ]

columns_dict = dict(zip(columns, columns_means))

atomic_api = {"func_desc":None,"Column_name":None,"example_input":None,"output_first_and_last_row":None}

atomic_api["func_desc"] = "获取股票的每日的技术指标数据"
atomic_api["example_input"] = "pro.stk_factor(ts_code=stock_code,start_date= start_date,end_date= end_date,fields=[\"ts_code\",\"trade_date\",\"close\",\"macd_dif\",\"macd_dea\",\"macd\",\"kdj_k\",\"kdj_d\",\"kdj_j\",\"rsi_6\",\"rsi_12\",\"rsi_24\",\"boll_upper\",\"boll_mid\",\"boll_lower\",\"cci\"])"
atomic_api["output_first_and_last_row"] = df_sample_str
atomic_api["Column_name"] = str(columns_dict)

all_atomic_api["pro.stk_factor"] = atomic_api

#######################################################################################################################

df = pro.moneyflow_hsgt(start_date="20220101", end_date="20230101", fields=["trade_date","ggt_ss","ggt_sz","hgt","sgt","north_money","south_money"])
df_sample_str = df.iloc[[0, -1], :].to_string(header=False, index=False)
columns = df.columns.tolist()
columns_means = ['交易日期','港股通（上海）','港股通（深圳）','沪股通（百万元）','深股通（百万元）','北向资金（百万元）','南向资金（百万元）'  ]

columns_dict = dict(zip(columns, columns_means))

atomic_api = {"func_desc":None,"Column_name":None,"example_input":None,"output_first_and_last_row":None}

atomic_api["func_desc"] = "获取沪深港通资金每日的资金流向数据"
atomic_api["example_input"] = "pro.moneyflow_hsgt(start_date=\"20220101\", end_date=\"20230101\", fields=[\"trade_date\",\"ggt_ss\",\"ggt_sz\",\"hgt\",\"sgt\",\"north_money\",\"south_money\"])"
atomic_api["output_first_and_last_row"] = df_sample_str
atomic_api["Column_name"] = str(columns_dict)

all_atomic_api["pro.moneyflow_hsgt"] = atomic_api



#######################################################################################################################

df = pro.index_member(index_code= "850531.SI" , fields=["index_code","index_name","con_code","con_name","in_date","out_date","is_new"])
df_sample_str = df.iloc[[0, -1], :].to_string(header=False, index=False)
columns = df.columns.tolist()
columns_means = ['指数代码','指数名称', '成分股代码','成分股名称', '纳入日期', '剔除日期', '是否最新'  ]
columns_dict = dict(zip(columns, columns_means))

atomic_api = {"func_desc":None,"Column_name":None,"example_input":None,"output_first_and_last_row":None}

atomic_api["func_desc"] = "获取申万行业指数的成分股信息"
atomic_api["example_input"] = "pro.index_member(index_code= \"850531.SI \", fields=[\"index_code\",\"con_code\",\"in_date\",\"out_date\",\"is_new\",\"index_name\",\"con_name\"])"
atomic_api["output_first_and_last_row"] = df_sample_str
atomic_api["Column_name"] = str(columns_dict)


all_atomic_api["pro.index_member"] = atomic_api




#######################################################################################################################
atomic_api = {"func_desc":None,"Column_name":None,"example_input":None,"output_first_and_last_row":None}

df = pro.index_classify(level='L1', src='SW2021',fields=["index_code","industry_name","level"])


df_sample_str = df.iloc[[0, -1], :].to_string(header=False, index=False)
columns = df.columns.tolist()
columns_means = ['行业代码', '行业名称', '行业级别']
columns_dict = dict(zip(columns, columns_means))

atomic_api["func_desc"] = "获取申万一级(L1),二级(L2),三级(L3)的行业信息"
atomic_api["example_input"] = "pro.index_classify(level='L1', src='SW2021',filter=[\"index_code\",\"industry_name\",\"level\"])"
atomic_api["output_first_and_last_row"] = df_sample_str
atomic_api["Column_name"] = str(columns_dict)

all_atomic_api["pro.index_classify"] = atomic_api

###############################################################


atomic_api = {"func_desc":None,"Column_name":None,"example_input":None,"output_first_and_last_row":None}
df_stock_basic = pro.stock_basic(ts_code="", name="贵州茅台", fields = ["ts_code","name","area","industry","market","list_date"])
df_stock_basic_sample_str = df_stock_basic.iloc[0, :].to_string(header=False, index=False)
columns = df_stock_basic.columns.tolist()
columns_means = ['股票代码', '股票名称', '地域',  '所属行业', '市场类型',  '上市日期']
columns_dict = dict(zip(columns, columns_means))

atomic_api["func_desc"] = "按照股票名称或者股票代码查询股票相关信息"
atomic_api["example_input"] = "pro.stock_basic(ts_code=\"\", name=\"贵州茅台\", fields = [\"ts_code\",\"name\",\"area\",\"industry\",\"market\",\"list_date\"])"
atomic_api["output_first_and_last_row"] = df_stock_basic_sample_str
atomic_api["Column_name"] = str(columns_dict)


all_atomic_api["pro.stock_basic"] = atomic_api



###############################################################
atomic_api = {"func_desc":None,"Column_name":None,"example_input":None,"output_first_and_last_row":None}

df_adj_factor = pro.adj_factor(ts_code='000001.SZ',start_date='20180718',end_date='20190718', fields=["ts_code","trade_date","adj_factor"])
df_adj_factor_sample_str = df_adj_factor.iloc[[0, -1], :].to_string(header=False, index=False)
columns = df_adj_factor.columns.tolist()
columns_means = ['股票代码', '交易日期', '复权因子']
columns_dict = dict(zip(columns, columns_means))

atomic_api["func_desc"] = "获取股票每日的复权因子"
atomic_api["example_input"] = "pro.adj_factor(ts_code='000001.SZ',start_date='20180718',end_date='20190718', fields=[\"ts_code\",\"trade_date\",\"adj_factor\"])"
atomic_api["output_first_and_last_row"] = df_adj_factor_sample_str
atomic_api["Column_name"] = str(columns_dict)

all_atomic_api["pro.adj_factor"] = atomic_api
###################################################################################################################
atomic_api = {"func_desc":None,"Column_name":None,"example_input":None,"output_first_and_last_row":None}
df_daily = pro.daily(ts_code='000001.SZ', start_date='20180701', end_date='20180718')
# 第一行第二行和最后一行
df_daily_sample_str = df_daily.iloc[[0, -1], :].to_string(header=False, index=False)
# 整个表格转换成字符串
# 获取列名
columns = df_daily.columns.tolist()
columns_means = ['股票代码', '交易日期', '开盘价', '最高价', '最低价', '收盘价', '昨收价', '涨跌额', '涨跌幅%', '成交量', '成交额']
columns_dict = dict(zip(columns, columns_means))


# 获取函数描述
atomic_api["func_desc"] = "获取股票每日的行情数据"
# 获取函数输入样例
atomic_api["example_input"] = "pro.daily(ts_code='000001.SZ', start_date='20180701', end_date='20180718')"
# 获取函数输出样例
atomic_api["output_first_and_last_row"] = df_daily_sample_str
# 获取列名
atomic_api["Column_name"] = str(columns_dict)

all_atomic_api["pro.daily"] = atomic_api
#######################################################################################################################

atomic_api = {"func_desc":None,"Column_name":None,"example_input":None,"output_first_and_last_row":None}

df_weekly = pro.weekly(ts_code='000001.SZ', start_date='20180101', end_date='20181101')
df_weekly_sample_str = df_weekly.iloc[[0, -1], :].to_string(header=False, index=False)
columns = df_weekly.columns.tolist()
columns_means = ['股票代码', '交易日期','周收盘价' ,'周开盘价', '周最高价', '周最低价', '上一周收盘价','周涨跌额', '周涨跌幅%', '周成交量', '周成交额']
columns_dict = dict(zip(columns, columns_means))

atomic_api["func_desc"] = "获取股票每周的行情数据"
atomic_api["example_input"] = "pro.weekly(ts_code='000001.SZ', start_date='20180101', end_date='20181101')"
atomic_api["output_first_and_last_row"] = df_weekly_sample_str
atomic_api["Column_name"] = str(columns_dict)


all_atomic_api["pro.weekly"] = atomic_api

#######################################################################################################################

atomic_api = {"func_desc":None,"Column_name":None,"example_input":None,"output_first_and_last_row":None}

df_monthly = pro.monthly(ts_code='000001.SZ', start_date='20180101', end_date='20181101')
df_monthly_sample_str = df_monthly.iloc[[0, -1], :].to_string(header=False, index=False)
columns = df_monthly.columns.tolist()
# columns 转换成dict
columns_means = ['股票代码', '交易日期','月收盘价' ,'月开盘价', '月最高价', '月最低价', '上一月收盘价','月涨跌额', '月涨跌幅%', '月成交量', '月成交额']
columns_dict = dict(zip(columns, columns_means))

atomic_api["func_desc"] = "获取股票每月的行情数据"
atomic_api["example_input"] = "pro.monthly(ts_code='000001.SZ', start_date='20180101', end_date='20181101')"
atomic_api["output_first_and_last_row"] = df_monthly_sample_str
atomic_api["Column_name"] = str(columns_dict)

all_atomic_api["pro.monthly"] = atomic_api

#######################################################################################################################

atomic_api = {"func_desc":None, "Column_name":None, "example_input":None, "output_first_and_last_row":None}
df = pro.index_daily(ts_code='399300.SZ', start_date='20180101', end_date='20191010')
df_sample_str = df.iloc[[0, -1], :].to_string(header=False, index=False)
columns = df.columns.tolist()
columns_means = ['指数代码', '交易日期', '收盘点位', '开盘点位', '最高点位', '最低点位', '昨收盘点位', '涨跌点', '涨跌幅%', '成交量', '成交额']
columns_dict = dict(zip(columns, columns_means))

atomic_api["func_desc"] = "获取指数每日的行情数据"
atomic_api["example_input"] = "pro.index_daily(ts_code='399300.SZ', start_date='20180101', end_date='20191010')"
atomic_api["output_first_and_last_row"] = df_sample_str
atomic_api["Column_name"] = str(columns_dict)

all_atomic_api["pro.index_daily"] = atomic_api

#######################################################################################################################
atomic_api = {"func_desc":None, "Column_name":None, "example_input":None, "output_first_and_last_row":None}
df = pro.index_weekly(ts_code='399300.SZ', start_date='20180101', end_date='20191010')
df_sample_str = df.iloc[[0, -1], :].to_string(header=False, index=False)
columns = df.columns.tolist()
columns_means = ['指数代码', '交易日期', '周收盘点位', '周开盘点位', '周最高点位', '周最低点位', '上周收盘点位', '周涨跌点', '周涨跌幅%', '周成交量', '周成交额']
columns_dict = dict(zip(columns, columns_means))

atomic_api["func_desc"] = "获取指数每周的行情数据"
atomic_api["example_input"] = "pro.index_weekly(ts_code='399300.SZ', start_date='20180101', end_date='20191010')"
atomic_api["output_first_and_last_row"] = df_sample_str
atomic_api["Column_name"] = str(columns_dict)

all_atomic_api["pro.index_weekly"] = atomic_api

#######################################################################################################################
atomic_api = {"func_desc":None, "Column_name":None, "example_input":None, "output_first_and_last_row":None}
df = pro.index_monthly(ts_code='399300.SZ', start_date='20180101', end_date='20191010')
df_sample_str = df.iloc[[0, -1], :].to_string(header=False, index=False)
columns = df.columns.tolist()
columns_means = ['指数代码', '交易日期', '月收盘点位', '月开盘点位', '月最高点位', '月最低点位', '上月收盘点位', '月涨跌点', '月涨跌幅%', '月成交量', '月成交额']
columns_dict = dict(zip(columns, columns_means))

atomic_api["func_desc"] = "获取指数每月的行情数据"
atomic_api["example_input"] = "pro.index_monthly(ts_code='399300.SZ', start_date='20180101', end_date='20191010')"
atomic_api["output_first_and_last_row"] = df_sample_str
atomic_api["Column_name"] = str(columns_dict)

all_atomic_api["pro.index_monthly"] = atomic_api

#######################################################################################################################
atomic_api = {"func_desc":None, "Column_name":None, "example_input":None, "output_first_and_last_row":None}

df = pro.index_basic(name = "沪深300", fields=["ts_code","name"])
df_sample_str = df.iloc[[0, -1], :].to_string(header=False, index=False)
columns = df.columns.tolist()
columns_means = ['指数代码', '指数名称']
columns_dict = dict(zip(columns, columns_means))

atomic_api["func_desc"] = "获取指数基本信息"
atomic_api["example_input"] = "pro.index_basic(name = '沪深300', fields=['ts_code','name'])"
atomic_api["output_first_and_last_row"] = df_sample_str
atomic_api["Column_name"] = str(columns_dict)

all_atomic_api["pro.index_basic"] = atomic_api


#######################################################################################################################
atomic_api = {"func_desc":None, "Column_name":None, "example_input":None, "output_first_and_last_row":None}

df = pro.index_weight(index_code="399300.SZ",start_date="20210101",end_date="20220101", fields=["index_code","con_code","trade_date","weight"])
df_sample_str = df.iloc[[0, -1], :].to_string(header=False, index=False)
columns = df.columns.tolist()
columns_means = ['指数代码', '成分股代码', '交易日期', '成分股权重']
columns_dict = dict(zip(columns, columns_means))

atomic_api["func_desc"] = "获取指数的成分股和成分股的权重"
atomic_api["example_input"] = "pro.index_weight(index_code='399300.SZ',start_date='20210101',end_date='20220101', fields=['index_code','con_code','trade_date','weight'])"
atomic_api["output_first_and_last_row"] = df_sample_str
atomic_api["Column_name"] = str(columns_dict)

all_atomic_api["pro.index_weight"] = atomic_api




#######################################################################################################################



#######################################################################################################################

# 创建文件夹如果不存在
if not os.path.exists(""):
    os.mkdir("")

with open("all_atomic_api.json", "w") as f:
    json.dump(all_atomic_api, f, ensure_ascii=False, indent=4)









