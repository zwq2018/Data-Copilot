import tushare as ts
import matplotlib.pyplot as plt
import pandas as pd
import os
import random
from matplotlib.ticker import MaxNLocator
#from prettytable import PrettyTable
#from blessed import Terminal
import time
from datetime import datetime, timedelta
import numpy as np
import mplfinance as mpf

from typing import Optional
import matplotlib.font_manager as fm
from matplotlib.lines import Line2D
from typing import Union, Any
from sklearn.linear_model import LinearRegression


# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
# plt.rcParams['axes.unicode_minus'] = False


font_path = './fonts/SimHei.ttf'
font_prop = fm.FontProperties(fname=font_path)


tushare_token = os.getenv('TUSHARE_TOKEN')
pro = ts.pro_api(tushare_token)

# def last_month_end(date_str:str=''):
#     date_obj = datetime.strptime(date_str, '%Y%m%d')
#     current_month = date_obj.month
#     current_year = date_obj.year
#
#     if current_month == 1:
#         last_month = 12
#         last_year = current_year - 1
#     else:
#         last_month = current_month - 1
#         last_year = current_year
#
#     if date_obj.month != (date_obj + timedelta(days=1)).month:
#         last_month_end_date = date_obj
#     else:
#         last_day_of_last_month = (date_obj.replace(day=1) - timedelta(days=1)).day
#         last_month_end_date = datetime(last_year, last_month, last_day_of_last_month)
#
#     return last_month_end_date.strftime('%Y%m%d')



def get_last_year_date(date_str: str = '') -> str:
    """
        This function takes a date string in the format YYYYMMDD and returns the date string one year prior to the input date.

        Args:
        - date_str: string, the input date in the format YYYYMMDD

        Returns:
        - string, the date one year prior to the input date in the format YYYYMMDD
        """
    dt = datetime.strptime(date_str, '%Y%m%d')
    # To calculate the date one year ago
    one_year_ago = dt - timedelta(days=365)

    # To format the date as a string
    one_year_ago_str = one_year_ago.strftime('%Y%m%d')

    return one_year_ago_str


def get_adj_factor(stock_code: str = '', start_date: str = '', end_date: str = '') -> pd.DataFrame:
    # Get stock price adjustment factors. Retrieve the stock price adjustment factors for a single stock's entire historical data or for all stocks on a single trading day.
    # The input includes the stock code, start date, end date, and trading date, all in string format with the date in the YYYYMMDD format
    # The return value is a dataframe containing the stock code, trading date, and adjustment factor
    # ts_code	str	股票代码
    # adj_factor	float	复权因子
    """
       This function retrieves the adjusted stock prices for a given stock code and date range.

       Args:
       - stock_code: string, the stock code to retrieve data for
       - start_date: string, the start date in the format YYYYMMDD
       - end_date: string, the end date in the format YYYYMMDD

       Returns:
       - dataframe, a dataframe containing the stock code, trade date, and adjusted factor

       This will retrieve the adjusted stock prices for the stock with code '000001.SZ' between the dates '20220101' and '20220501'.
       """
    df = pro.adj_factor(**{
        "ts_code": stock_code,
        "trade_date": "",
        "start_date": start_date,
        "end_date": end_date,
        "limit": "",
        "offset": ""
    }, fields=[
        "ts_code",
        "trade_date",
        "adj_factor"
    ])

    return df

def get_stock_code(stock_name: str) -> str:
    # Retrieve the stock code of a given stock name. If we call get_stock_code('贵州茅台'), it will return '600519.SH'.


    df = pd.read_csv('tushare_stock_basic_20230421210721.csv')
    try:
        code = df.loc[df.name==stock_name].ts_code.iloc[0]
        return code
    except:
        return None




def get_stock_name_from_code(stock_code: str) -> str:
    """
        Reads a local file to retrieve the stock name from a given stock code.

        Args:
        - stock_code (str): The code of the stock.

        Returns:
        - str: The stock name of the given stock code.
        """
    # For example,if we call get_stock_name_from_code('600519.SH'), it will return '贵州茅台'.


    df = pd.read_csv('tushare_stock_basic_20230421210721.csv')
    name = df.loc[df.ts_code == stock_code].name.iloc[0]

    return name

def get_stock_prices_data(stock_name: str='', start_date: str='', end_date: str='', freq:str='daily') -> pd.DataFrame:
    """
        Retrieves the daily/weekly/monthly price data for a given stock code during a specific time period. get_stock_prices_data('贵州茅台','20200120','20220222','daily')

        Args:
        - stock_name (str)
        - start_date (str): The start date in the format 'YYYYMMDD'.
        - end_date (str): The end date in 'YYYYMMDD'.
        - freq (str): The frequency of the price data, can be 'daily', 'weekly', or 'monthly'.

        Returns:
        - pd.DataFrame: A dataframe that contains the daily/weekly/monthly data. The output columns contain stock_code, trade_date, open, high, low, close, pre_close(昨天收盘价), change(涨跌额), pct_chg(涨跌幅),vol(成交量),amount(成交额)
        """

    stock_code = get_stock_code(stock_name)

    if freq == 'daily':
        stock_data = pro.daily(**{
            "ts_code": stock_code,
            "trade_date": '',
            "start_date": start_date,
            "end_date": end_date,
            "offset": "",
            "limit": ""
        }, fields=[
            "ts_code",
            "trade_date",
            "open",
            "high",
            "low",
            "close",
            "pre_close",
            "change",
            "pct_chg",
            "vol",
            "amount"
        ])

    elif freq == 'weekly':
        stock_data = pro.weekly(**{
            "ts_code": stock_code,
            "trade_date": '',
            "start_date": start_date,
            "end_date": end_date,
            "limit": "",
            "offset": ""
        }, fields=[
            "ts_code",
            "trade_date",
            "close",
            "open",
            "high",
            "low",
            "pre_close",
            "change",
            "pct_chg",
            "vol",
            "amount"
        ])
    elif freq == 'monthly':
        stock_data = pro.monthly(**{
            "ts_code": stock_code,
            "trade_date": '',
            "start_date": start_date,
            "end_date": end_date,
            "limit": "",
            "offset": ""
        }, fields=[
            "ts_code",
            "trade_date",
            "close",
            "open",
            "high",
            "low",
            "pre_close",
            "change",
            "pct_chg",
            "vol",
            "amount"
        ])


    adj_f = get_adj_factor(stock_code, start_date, end_date)
    stock_data = pd.merge(stock_data, adj_f, on=['ts_code', 'trade_date'])
    # Multiply the values of open, high, low, and close by their corresponding adjustment factors.
    # To obtain the adjusted close price
    stock_data[['open', 'high', 'low', 'close']] *= stock_data['adj_factor'].values.reshape(-1, 1)

    #stock_data.rename(columns={'vol': 'volume'}, inplace=True)
    df = pd.read_csv('tushare_stock_basic_20230421210721.csv')
    stock_data_merged = pd.merge(stock_data, df, on='ts_code')
    stock_data_merged.rename(columns={'ts_code': 'stock_code'}, inplace=True)
    stock_data_merged.rename(columns={'name': 'stock_name'}, inplace=True)
    stock_data_merged = stock_data_merged.sort_values(by='trade_date', ascending=True)  # To sort the DataFrame by date in ascending order
    return stock_data_merged



def get_stock_technical_data(stock_name: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
        Retrieves the daily technical data of a stock including macd turnover rate, volume, PE ratio, etc. Those technical indicators are usually plotted as subplots in a k-line chart.

        Args:
            stock_name (str):
            start_date (str): Start date "YYYYMMDD"
            end_date (str): End date "YYYYMMDD"

        Returns:
            pd.DataFrame: A DataFrame containing the technical data of the stock,
            including various indicators such as ts_code, trade_date, close, macd_dif, macd_dea, macd, kdj_k, kdj_d, kdj_j, rsi_6, rsi_12, boll_upper, boll_mid, boll_lower, cci, turnover_rate, turnover_rate_f, volume_ratio, pe_ttm(市盈率), pb(市净率), ps_ttm, dv_ttm, total_share, float_share, free_share, total_mv, circ_mv

    """

    # Technical factors
    stock_code = get_stock_code(stock_name)
    stock_data1 = pro.stk_factor(**{
        "ts_code": stock_code,
        "start_date": start_date,
        "end_date": end_date,
        "trade_date": '',
        "limit": "",
        "offset": ""
    }, fields=[
        "ts_code",
        "trade_date",
        "close",
        "macd_dif",
        "macd_dea",
        "macd",
        "kdj_k",
        "kdj_d",
        "kdj_j",
        "rsi_6",
        "rsi_12",
        "rsi_24",
        "boll_upper",
        "boll_mid",
        "boll_lower",
        "cci"
    ])
    # Trading factors
    stock_data2 = pro.daily_basic(**{
        "ts_code": stock_code,
        "trade_date": '',
        "start_date": start_date,
        "end_date": end_date,
        "limit": "",
        "offset": ""
    }, fields=[
        "ts_code",  #
        "trade_date",
        "turnover_rate",
        "turnover_rate_f",
        "volume_ratio",
        "pe_ttm",
        "pb",
        "ps_ttm",
        "dv_ttm",
        "total_share",
        "float_share",
        "free_share",
        "total_mv",
        "circ_mv"
    ])

    #
    stock_data = pd.merge(stock_data1, stock_data2, on=['ts_code', 'trade_date'])
    df = pd.read_csv('tushare_stock_basic_20230421210721.csv')
    stock_data_merged = pd.merge(stock_data, df, on='ts_code')
    stock_data_merged = stock_data_merged.sort_values(by='trade_date', ascending=True)

    stock_data_merged.drop(['symbol'], axis=1, inplace=True)

    stock_data_merged.rename(columns={'ts_code': 'stock_code'}, inplace=True)
    stock_data_merged.rename(columns={'name': 'stock_name'}, inplace=True)

    return stock_data_merged


def plot_stock_data(stock_data: pd.DataFrame, ax: Optional[plt.Axes] = None, figure_type: str = 'line', title_name: str ='') -> plt.Axes:

    """
    This function plots stock data.

    Args:
    - stock_data: pandas DataFrame, the stock data to plot. The DataFrame should contain three columns:
        - Column 1: trade date in 'YYYYMMDD'
        - Column 2: Stock name or code (string format)
        - Column 3: Index value (numeric format)
        The DataFrame can be time series data or cross-sectional data. If it is time-series data, the first column represents different trade time, the second column represents the same name. For cross-sectional data, the first column is the same, the second column contains different stocks.

    - ax: matplotlib Axes object, the axes to plot the data on
    - figure_type: the type of figure (either 'line' or 'bar')
    - title_name

    Returns:
    - matplotlib Axes object, the axes containing the plot
    """

    index_name = stock_data.columns[2]
    name_list = stock_data.iloc[:,1]
    date_list = stock_data.iloc[:,0]
    if name_list.nunique() == 1 and date_list.nunique() != 1:
        # Time Series Data
        unchanged_var = name_list.iloc[0]   # stock name
        x_dim = date_list                   # tradingdate
        x_name = stock_data.columns[0]

    elif name_list.nunique() != 1 and date_list.nunique() == 1:
        # Cross-sectional Data
        unchanged_var = date_list.iloc[0]    # tradingdate
        x_dim = name_list                    # stock name
        x_name = stock_data.columns[1]

        data_size = x_dim.shape[0]



    start_x_dim, end_x_dim = x_dim.iloc[0], x_dim.iloc[-1]

    start_y = stock_data.iloc[0, 2]
    end_y = stock_data.iloc[-1, 2]


    def generate_random_color():
        r = random.randint(0, 255)/ 255.0
        g = random.randint(0, 100)/ 255.0
        b = random.randint(0, 255)/ 255.0
        return (r, g, b)

    color = generate_random_color()
    if ax is None:
        _, ax = plt.subplots()

    if figure_type =='line':
        #

        ax.plot(x_dim, stock_data.iloc[:, 2], label = unchanged_var+'_' + index_name, color=color,linewidth=3)
        #
        plt.scatter(x_dim, stock_data.iloc[:, 2], color=color,s=3)  # Add markers to the data points

        #
        #ax.scatter(x_dim, stock_data.iloc[:, 2],label = unchanged_var+'_' + index_name, color=color, s=3)
        #

        ax.annotate(unchanged_var + ':' + str(round(start_y, 2)) + ' @' + start_x_dim, xy=(start_x_dim, start_y),
                    xytext=(start_x_dim, start_y),
                    textcoords='data', fontsize=14,color=color, horizontalalignment='right',fontproperties=font_prop)

        ax.annotate(unchanged_var + ':' + str(round(end_y, 2)) +' @' + end_x_dim, xy=(end_x_dim, end_y),
                    xytext=(end_x_dim, end_y),
                    textcoords='data', fontsize=14, color=color, horizontalalignment='left',fontproperties=font_prop)


    elif figure_type == 'bar':
        ax.bar(x_dim, stock_data.iloc[:, 2], label = unchanged_var + '_' + index_name, width=0.3, color=color)
        ax.annotate(unchanged_var + ':' + str(round(start_y, 2)) + ' @' + start_x_dim, xy=(start_x_dim, start_y),
                    xytext=(start_x_dim, start_y),
                    textcoords='data', fontsize=14, color=color, horizontalalignment='right',fontproperties=font_prop)

        ax.annotate(unchanged_var + ':' + str(round(end_y, 2)) + ' @' + end_x_dim, xy=(end_x_dim, end_y),
                    xytext=(end_x_dim, end_y),
                    textcoords='data', fontsize=14, color=color, horizontalalignment='left',fontproperties=font_prop)

    plt.xticks(x_dim,rotation=45)                                                  #
    ax.xaxis.set_major_locator(MaxNLocator( integer=True, prune=None, nbins=100))  #


    plt.xlabel(x_name, fontproperties=font_prop,fontsize=18)
    plt.ylabel(f'{index_name}', fontproperties=font_prop,fontsize=16)
    ax.set_title(title_name , fontproperties=font_prop,fontsize=16)
    plt.legend(prop=font_prop)  # 显示图例
    fig = plt.gcf()
    fig.set_size_inches(18, 12)

    return ax


def query_fund_Manager(Manager_name: str) -> pd.DataFrame:
    # 代码fund_code,公告日期ann_date,基金经理名字name,性别gender,出生年份birth_year,学历edu,国籍nationality,开始管理日期begin_date,结束日期end_date,简历resume
    """
        Retrieves information about a fund manager.

        Args:
            Manager_name (str): The name of the fund manager.

        Returns:
            df (DataFrame): A DataFrame containing the fund manager's information, including the fund codes, announcement dates,
                            manager's name, gender, birth year, education, nationality, start and end dates of managing funds,
                            and the manager's resume.
    """

    df = pro.fund_manager(**{
        "ts_code": "",
        "ann_date": "",
        "name": Manager_name,
        "offset": "",
        "limit": ""
    }, fields=[
        "ts_code",
        "ann_date",
        "name",
        "gender",
        "birth_year",
        "edu",
        "nationality",
        "begin_date",
        "end_date",
        "resume"
    ])
    #
    df.rename(columns={'ts_code': 'fund_code'}, inplace=True)
    # To query the fund name based on the fund code and store it in a new column called fund_name, while removing the rows where the fund name is not found
    df['fund_name'] = df['fund_code'].apply(lambda x: query_fund_name_or_code('', x))
    df.dropna(subset=['fund_name'], inplace=True)
    df.rename(columns={'name': 'manager_name'}, inplace=True)
    #
    df_out = df[['fund_name','fund_code','ann_date','manager_name','begin_date','end_date']]

    return df_out


# def save_stock_prices_to_csv(stock_prices: pd.DataFrame, stock_name: str, file_path: str) -> None:
#
#     """
#         Saves the price data of a specific stock symbol during a specific time period to a local CSV file.
#
#         Args:
#         - stock_prices (pd.DataFrame): A pandas dataframe that contains the daily price data for the given stock symbol during the specified time period.
#         - stock_name (str): The name of the stock.
#         - file_path (str): The file path where the CSV file will be saved.
#
#         Returns:
#         - None: The function only saves the CSV file to the specified file path.
#     """
#     # The function checks if the directory to save the CSV file exists and creates it if it does not exist.
#     # The function then saves the price data of the specified stock symbol during the specified time period to a local CSV file with the name {stock_name}_price_data.csv in the specified file path.
#
#
#     if not os.path.exists(file_path):
#         os.makedirs(file_path)
#
#
#     file_path = f"{file_path}{stock_name}_stock_prices.csv"
#     stock_prices.to_csv(file_path, index_label='Date')
#     print(f"Stock prices for {stock_name} saved to {file_path}")


def calculate_stock_index(stock_data: pd.DataFrame, index:str='close') -> pd.DataFrame:
    """
        Calculate a specific index of a stock based on its price information.

        Args:
            stock_data (pd.DataFrame): DataFrame containing the stock's price information.
            index (str, optional): The index to calculate. The available options depend on the column names in the
                input stock price data. Additionally, there are two special indices: 'candle_K' and 'Cumulative_Earnings_Rate'.

        Returns:
            DataFrame containing the corresponding index data of the stock. In general, it includes three columns: 'trade_date', 'name', and the corresponding index value.
            Besides, if index is 'candle_K', the function returns the DataFrame containing 'trade_date', 'Open', 'High', 'Low', 'Close', 'Volume','name' column.
            If index is a technical index such as 'macd' or a trading index likes 'pe_ttm', the function returns the DataFrame with corresponding columns.
        """


    if 'stock_name' not in  stock_data.columns and 'index_name' in stock_data.columns:
        stock_data.rename(columns={'index_name': 'stock_name'}, inplace=True)
    #
    index = index.lower()
    if index=='Cumulative_Earnings_Rate' or index =='Cumulative_Earnings_Rate'.lower() :
        stock_data[index] = (1 + stock_data['pct_chg'] / 100.).cumprod() - 1.
        stock_data[index] = stock_data[index] * 100.
        if 'stock_name' in stock_data.columns :
           selected_index = stock_data[['trade_date', 'stock_name', index]].copy()
        #
        if 'fund_name' in stock_data.columns:
            selected_index = stock_data[['trade_date', 'fund_name', index]].copy()
        return selected_index

    elif index == 'candle_K' or index == 'candle_K'.lower():
        #tech_df = tech_df.drop(['name', 'symbol', 'industry', 'area','market','list_date','ts_code','close'], axis=1)
        # Merge two DataFrames based on the 'trade_date' column.

        stock_data = stock_data.rename(
            columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close',
                     'vol': 'Volume'})
        selected_index = stock_data[['trade_date', 'Open', 'High', 'Low', 'Close', 'Volume','stock_name']].copy()
        return selected_index

    elif index =='macd':
        selected_index = stock_data[['trade_date','macd','macd_dea','macd_dif']].copy()
        return selected_index

    elif index =='rsi':
        selected_index = stock_data[['trade_date','rsi_6','rsi_12']].copy()
        return selected_index

    elif index =='boll':
        selected_index = stock_data[['trade_date', 'boll_upper', 'boll_lower','boll_mid']].copy()
        return selected_index

    elif index =='kdj':
        selected_index = stock_data[['trade_date', 'kdj_k', 'kdj_d','kdj_j']].copy()
        return selected_index

    elif index =='cci':
        selected_index = stock_data[['trade_date', 'cci']].copy()
        return selected_index

    elif index == '换手率':
        selected_index = stock_data[['trade_date', 'turnover_rate','turnover_rate_f']].copy()
        return selected_index

    elif index == '市值':
        selected_index = stock_data[['trade_date', 'total_mv','circ_mv']].copy()
        return selected_index


    elif index in stock_data.columns:
        stock_data = stock_data

        if 'stock_name' in stock_data.columns :
           selected_index = stock_data[['trade_date', 'stock_name', index]].copy()

        if 'fund_name' in stock_data.columns:
            selected_index = stock_data[['trade_date', 'fund_name', index]].copy()
        # Except for candlestick chart and technical indicators, the remaining outputs consist of three columns: date, name, and indicator.
        return selected_index



def rank_index_cross_section(stock_data: pd.DataFrame, Top_k: int = -1, ascending: bool = False) -> pd.DataFrame:
    """
        Sort the cross-sectional data based on the given index.

        Args:
            stock_data : DataFrame containing the cross-sectional data. It should have three columns, and the last column represents the variable to be sorted.
            Top_k : The number of data points to retain after sorting. (Default: -1, which retains all data points)
            ascending: Whether to sort the data in ascending order or not. (Default: False)

        Returns:
            stock_data_selected : DataFrame containing the sorted data. It has the same structure as the input DataFrame.
        """

    index = stock_data.columns[-1]
    stock_data = stock_data.sort_values(by=index, ascending=ascending)
    #stock_data_selected = stock_data[['trade_date','stock_name', index]].copy()
    stock_data_selected = stock_data[:Top_k]
    stock_data_selected = stock_data_selected.drop_duplicates(subset=['stock_name'], keep='first')
    return stock_data_selected


def get_company_info(stock_name: str='') -> pd.DataFrame:
    # ts_code: str	股票代码,  exchange:str	交易所代码SSE上交所 SZSE深交所, chairman:str 法人代表, manager:str 总经理, secretary:str	董秘 # reg_capital:float	注册资本, setup_date:str 注册日期, province:str 所在省份 ,city:str 所在城市
    # introduction:str 公司介绍, website:str 公司主页 , email:str	电子邮件, office:str 办公室 # ann_date: str 公告日期, business_scope:str 经营范围, employees:int	员工人数, main_business:str 主要业务及产品
    """
            This function retrieves company information including stock code, exchange, chairman, manager, secretary,
            registered capital, setup date, province, city, website, email, employees, business scope, main business,
            introduction, office, and announcement date.

            Args:
            - stock_name (str): The name of the stock.

            Returns:
            - pd.DataFrame: A DataFrame that contains the company information.
    """

    stock_code = get_stock_code(stock_name)
    df = pro.stock_company(**{
        "ts_code": stock_code,"exchange": "","status": "", "limit": "","offset": ""
    }, fields=[
        "ts_code","exchange","chairman", "manager","secretary", "reg_capital","setup_date", "province","city",
        "website", "email","employees","business_scope","main_business","introduction","office", "ann_date"
    ])


    en_to_cn = {
        'ts_code': '股票代码',
        'exchange': '交易所代码',
        'chairman': '法人代表',
        'manager': '总经理',
        'secretary': '董秘',
        'reg_capital': '注册资本',
        'setup_date': '注册日期',
        'province': '所在省份',
        'city': '所在城市',
        'introduction': '公司介绍',
        'website': '公司主页',
        'email': '电子邮件',
        'office': '办公室',
        'ann_date': '公告日期',
        'business_scope': '经营范围',
        'employees': '员工人数',
        'main_business': '主要业务及产品'
    }

    df.rename(columns=en_to_cn, inplace=True)
    df.insert(0, '股票名称', stock_name)
    # for column in df.columns:
    #     print(f"[{column}]: {df[column].values[0]}")


    return df





# def get_Financial_data(stock_code: str, report_date: str,  financial_index: str = '' ) -> pd.DataFrame:
#     # report_date的格式为"YYYYMMDD",包括"yyyy0331"为一季报,"yyyy0630"为半年报,"yyyy0930"为三季报,"yyyy1231"为年报
#     # index包含: # current_ratio	流动比率 # quick_ratio	速动比率 # netprofit_margin	销售净利率 # grossprofit_margin	销售毛利率 # roe	净资产收益率 # roe_dt	净资产收益率(扣除非经常损益)
#     # roa	总资产报酬率 # debt_to_assets 资产负债率 # roa_yearly	年化总资产净利率  # q_dtprofit	扣除非经常损益后的单季度净利润 # q_eps	每股收益(单季度)
#     # q_netprofit_margin	销售净利率(单季度) # q_gsprofit_margin	销售毛利率(单季度) # basic_eps_yoy 基本每股收益同比增长率(%) # netprofit_yoy	归属母公司股东的净利润同比增长率(%)   # q_netprofit_yoy	归属母公司股东的净利润同比增长率(%)(单季度) # q_netprofit_qoq	归属母公司股东的净利润环比增长率(%)(单季度) # equity_yoy	净资产同比增长率
#     """
#         Retrieves financial data for a specific stock within a given date range.
#
#         Args:
#             stock_code (str): The stock code or symbol of the company for which financial data is requested.
#             report_date (str): The report date in the format "YYYYMMDD" .
#             financial_index (str, optional): The financial indicator to be queried. If not specified, all available financial
#                                               indicators will be included.
#
#         Returns:
#             pd.DataFrame: A DataFrame containing the financial data for the specified stock and date range. The DataFrame
#                           consists of the following columns: "stock_name",
#                           "trade_date" (reporting period), and the requested financial indicator(s).
#
#         """
#     stock_data = pro.fina_indicator(**{
#         "ts_code": stock_code,
#         "ann_date": "",
#         "start_date": '',
#         "end_date": '',
#         "period": report_date,
#         "update_flag": "1",
#         "limit": "",
#         "offset": ""
#     }, fields=["ts_code","end_date", financial_index])
#
#     stock_name = get_stock_name_from_code(stock_code)
#     stock_data['stock_name'] = stock_name
#     stock_data = stock_data.sort_values(by='end_date', ascending=True)  # 按照日期升序排列
#     # 把end_data列改名为trade_date
#     stock_data.rename(columns={'end_date': 'trade_date'}, inplace=True)
#     stock_financial_data = stock_data[['stock_name', 'trade_date', financial_index]]
#     return stock_financial_data


def get_Financial_data_from_time_range(stock_name:str, start_date:str, end_date:str, financial_index:str='') -> pd.DataFrame:
    # start_date='20190101',end_date='20221231',financial_index='roe', The returned data consists of the ROE values for the entire three-year period from 2019 to 2022.
    # To query quarterly or annual financial report data for a specific moment, "yyyy0331"为一季报,"yyyy0630"为半年报,"yyyy0930"为三季报,"yyyy1231"为年报,例如get_Financial_data_from_time_range("600519.SH", "20190331", "20190331", "roe") means to query the return on equity (ROE) data from the first quarter of 2019,
    #  # current_ratio	流动比率 # quick_ratio	速动比率 # netprofit_margin	销售净利率 # grossprofit_margin	销售毛利率 # roe	净资产收益率 # roe_dt	净资产收益率(扣除非经常损益)
    # roa	总资产报酬率 # debt_to_assets 资产负债率 # roa_yearly	年化总资产净利率  # q_dtprofit	扣除非经常损益后的单季度净利润 # q_eps	每股收益(单季度)
    # q_netprofit_margin	销售净利率(单季度) # q_gsprofit_margin	销售毛利率(单季度) # basic_eps_yoy 基本每股收益同比增长率(%) # netprofit_yoy	归属母公司股东的净利润同比增长率(%)   # q_netprofit_yoy	归属母公司股东的净利润同比增长率(%)(单季度) # q_netprofit_qoq	归属母公司股东的净利润环比增长率(%)(单季度) # equity_yoy	净资产同比增长率
    """
        Retrieves the financial data for a given stock within a specified date range.

        Args:
            stock_name (str): The stock code.
            start_date (str): The start date of the data range in the format "YYYYMMDD".
            end_date (str): The end date of the data range in the format "YYYYMMDD".
            financial_index (str, optional): The financial indicator to be queried.

        Returns:
            pd.DataFrame: A DataFrame containin financial data for the specified stock and date range.

"""
    stock_code = get_stock_code(stock_name)
    stock_data = pro.fina_indicator(**{
        "ts_code": stock_code,
        "ann_date": "",
        "start_date": start_date,
        "end_date": end_date,
        "period": '',
        "update_flag": "1",
        "limit": "",
        "offset": ""
    }, fields=["ts_code", "end_date", financial_index])

    #stock_name = get_stock_name_from_code(stock_code)
    stock_data['stock_name'] = stock_name
    stock_data = stock_data.sort_values(by='end_date', ascending=True)  # 按照日期升序排列
    # 把end_data列改名为trade_date
    stock_data.rename(columns={'end_date': 'trade_date'}, inplace=True)
    stock_financial_data = stock_data[['stock_name', 'trade_date', financial_index]]
    return stock_financial_data


def get_GDP_data(start_quarter:str='', end_quarter:str='', index:str='gdp_yoy') -> pd.DataFrame:
    # The available indicators for query include the following 9 categories: # gdp GDP累计值（亿元）# gdp_yoy 当季同比增速（%）# pi 第一产业累计值（亿元）# pi_yoy 第一产业同比增速（%）# si 第二产业累计值（亿元）# si_yoy 第二产业同比增速（%）# ti 第三产业累计值（亿元） # ti_yoy 第三产业同比增速（%）
    """
        Retrieves GDP data for the chosen index and specified time period.

        Args:
        - start_quarter (str): The start quarter of the query, in YYYYMMDD format.
        - end_quarter (str): The end quarter, in YYYYMMDD format.
        - index (str): The specific GDP index to retrieve. Default is `gdp_yoy`.

        Returns:
        - pd.DataFrame: A pandas DataFrame with three columns: `quarter`, `country`, and the selected `index`.
        """

    # The output is a DataFrame with three columns:
    # the first column represents the quarter (quarter), the second column represents the country (country), and the third column represents the index (index).
    df = pro.cn_gdp(**{
        "q":'',
        "start_q": start_quarter,
        "end_q": end_quarter,
        "limit": "",
        "offset": ""
    }, fields=[
        "quarter",
        "gdp",
        "gdp_yoy",
        "pi",
        "pi_yoy",
        "si",
        "si_yoy",
        "ti",
        "ti_yoy"
    ])
    df = df.sort_values(by='quarter', ascending=True)  #
    df['country'] = 'China'
    df = df[['quarter', 'country', index]].copy()


    return df

def get_cpi_ppi_currency_supply_data(start_month: str = '', end_month: str = '', type: str = 'cpi', index: str = '') -> pd.DataFrame:
    # The query types (type) include three categories: CPI, PPI, and currency supply. Each type corresponds to different indices.
    # Specifically, CPI has 12 indices, PPI has 30 indices, and currency supply has 9 indices.
    # The output is a DataFrame table with three columns: the first column represents the month (month), the second column represents the country (country), and the third column represents the index (index).

    # type='cpi',monthly CPI data include the following 12 categories:
    # nt_val	全国当月值 # nt_yoy	全国同比（%）# nt_mom	全国环比（%）# nt_accu	全国累计值# town_val	城市当月值# town_yoy	城市同比（%）# town_mom	城市环比（%）# town_accu	城市累计值# cnt_val	农村当月值# cnt_yoy	农村同比（%）# cnt_mom	农村环比（%）# cnt_accu	农村累计值

    # type = 'ppi', monthly PPI data include the following 30 categories:
    # ppi_yoy	PPI：全部工业品：当月同比
    # ppi_mp_yoy    PPI：生产资料：当月同比
    # ppi_mp_qm_yoy	PPI：生产资料：采掘业：当月同比
    # ppi_mp_rm_yoy	PPI：生产资料：原料业：当月同比
    # ppi_mp_p_yoy	PPI：生产资料：加工业：当月同比
    # ppi_cg_yoy	PPI：生活资料：当月同比
    # ppi_cg_f_yoy	PPI：生活资料：食品类：当月同比
    # ppi_cg_c_yoy	PPI：生活资料：衣着类：当月同比
    # ppi_cg_adu_yoy	PPI：生活资料：一般日用品类：当月同比
    # ppi_cg_dcg_yoy	PPI：生活资料：耐用消费品类：当月同比
    # ppi_mom	PPI：全部工业品：环比
    # ppi_mp_mom	PPI：生产资料：环比
    # ppi_mp_qm_mom	PPI：生产资料：采掘业：环比
    # ppi_mp_rm_mom	PPI：生产资料：原料业：环比
    # ppi_mp_p_mom	PPI：生产资料：加工业：环比
    # ppi_cg_mom	PPI：生活资料：环比
    # ppi_cg_f_mom	PPI：生活资料：食品类：环比
    # ppi_cg_c_mom	PPI：生活资料：衣着类：环比
    # ppi_cg_adu_mom	PPI：生活资料：一般日用品类：环比
    # ppi_cg_dcg_mom		PPI：生活资料：耐用消费品类：环比
    # ppi_accu		PPI：全部工业品：累计同比
    # ppi_mp_accu		PPI：生产资料：累计同比
    # ppi_mp_qm_accu		PPI：生产资料：采掘业：累计同比
    # ppi_mp_rm_accu		PPI：生产资料：原料业：累计同比
    # ppi_mp_p_accu	    PPI：生产资料：加工业：累计同比
    # ppi_cg_accu	PPI：生活资料：累计同比
    # ppi_cg_f_accu		PPI：生活资料：食品类：累计同比
    # ppi_cg_c_accu		PPI：生活资料：衣着类：累计同比
    # ppi_cg_adu_accu	PPI：生活资料：一般日用品类：累计同比
    # ppi_cg_dcg_accu	PPI：生活资料：耐用消费品类：累计同比

    # type = 'currency_supply', monthly currency supply data include the following 9 categories:
    # m0  M0（亿元）# m0_yoy  M0同比（%）# m0_mom  M0环比（%）# m1  M1（亿元）# m1_yoy  M1同比（%）# m1_mom  M1环比（%）# m2  M2（亿元）# m2_yoy  M2同比（%）# m2_mom  M2环比（%）

    """
        This function is used to retrieve China's monthly CPI (Consumer Price Index), PPI (Producer Price Index),
        and monetary supply data published by the National Bureau of Statistics,
        and return a DataFrame table containing month, country, and index values.
        The function parameters include start month, end month, query type, and query index.
        For query indexes that are not within the query range, the default index for the corresponding type is returned.

        Args:
        - start_month (str): start month of the query, in the format of YYYYMMDD.
        - end_month (str):end month in YYYYMMDD
        - type (str): required parameter, query type, including three types: cpi, ppi, and currency_supply.
        - index (str): optional parameter, query index, the specific index depends on the query type.
        If the query index is not within the range, the default index for the corresponding type is returned.

        Returns:
        - pd.DataFrame: DataFrame type, including three columns: month, country, and index value.
        """

    if type == 'cpi':

        df = pro.cn_cpi(**{
            "m": '',
            "start_m": start_month,
            "end_m": end_month,
            "limit": "",
            "offset": ""
        }, fields=[
            "month", "nt_val","nt_yoy", "nt_mom","nt_accu", "town_val", "town_yoy",  "town_mom",
            "town_accu", "cnt_val", "cnt_yoy", "cnt_mom", "cnt_accu"])
        # If the index is not within the aforementioned range, the index is set as "nt_yoy".
        if index not in df.columns:
            index = 'nt_yoy'


    elif type == 'ppi':
        df = pro.cn_ppi(**{
            "m": '',
            "start_m": start_month,
            "end_m": end_month,
            "limit": "",
            "offset": ""
        }, fields=[
            "month", "ppi_yoy", "ppi_mp_yoy", "ppi_mp_qm_yoy", "ppi_mp_rm_yoy", "ppi_mp_p_yoy", "ppi_cg_yoy",
            "ppi_cg_f_yoy", "ppi_cg_c_yoy", "ppi_cg_adu_yoy", "ppi_cg_dcg_yoy",
            "ppi_mom", "ppi_mp_mom", "ppi_mp_qm_mom", "ppi_mp_rm_mom", "ppi_mp_p_mom", "ppi_cg_mom", "ppi_cg_f_mom",
            "ppi_cg_c_mom", "ppi_cg_adu_mom", "ppi_cg_dcg_mom",
            "ppi_accu", "ppi_mp_accu", "ppi_mp_qm_accu", "ppi_mp_rm_accu", "ppi_mp_p_accu", "ppi_cg_accu",
            "ppi_cg_f_accu", "ppi_cg_c_accu", "ppi_cg_adu_accu", "ppi_cg_dcg_accu"
        ])
        if index not in df.columns:
            index = 'ppi_yoy'

    elif type == 'currency_supply':
        df = pro.cn_m(**{
            "m": '',
            "start_m": start_month,
            "end_m": end_month,
            "limit": "",
            "offset": ""
        }, fields=[
            "month", "m0",  "m0_yoy","m0_mom", "m1",
            "m1_yoy",  "m1_mom", "m2", "m2_yoy", "m2_mom"])
        if index not in df.columns:
            index = 'm2_yoy'


    df = df.sort_values(by='month', ascending=True)  #
    df['country'] = 'China'
    df = df[['month', 'country', index]].copy()
    return df

def predict_next_value(df: pd.DataFrame, pred_index: str = 'nt_yoy', pred_num:int = 1. ) -> pd.DataFrame:
    """
    Predict the next n values of a specific column in the DataFrame using linear regression.

        Parameters:
            df (pandas.DataFrame): The input DataFrame.
            pred_index (str): The name of the column to predict.
            pred_num (int): The number of future values to predict.

        Returns:
        pandas.DataFrame: The DataFrame with the predicted values appended to the specified column
                          and other columns filled as pred+index.
        """
    input_array = df[pred_index].values

    # Convert the input array into the desired format.
    x = np.array(range(len(input_array))).reshape(-1, 1)
    y = input_array.reshape(-1, 1)

    # Train a linear regression model.
    model = LinearRegression()
    model.fit(x, y)

    # Predict the future n values.
    next_indices = np.array(range(len(input_array), len(input_array) + pred_num)).reshape(-1, 1)
    predicted_values = model.predict(next_indices).flatten()

    for i, value in enumerate(predicted_values, 1):
        row_data = {pred_index: value}
        for other_col in df.columns:
            if other_col != pred_index:
                row_data[other_col] = 'pred' + str(i)
        df = df.append(row_data, ignore_index=True)

        # Return the updated DataFrame
    return df






def get_latest_new_from_web(src: str = 'sina') -> pd.DataFrame:

    # 新浪财经	sina	获取新浪财经实时资讯
    # 同花顺	    10jqka	同花顺财经新闻
    # 东方财富	eastmoney	东方财富财经新闻
    # 云财经	    yuncaijing	云财经新闻
    """
    Retrieves the latest news data from major news websites, including Sina Finance, 10jqka, Eastmoney, and Yuncaijing.

    Args:
        src (str): The name of the news website. Default is 'sina'. Optional parameters include: 'sina' for Sina Finance,
        '10jqka' for 10jqka, 'eastmoney' for Eastmoney, and 'yuncaijing' for Yuncaijing.

    Returns:
        pd.DataFrame: A DataFrame containing the news data, including two columns for date/time and content.
    """

    df = pro.news(**{
        "start_date": '',
        "end_date": '',
        "src": src,
        "limit": "",
        "offset": ""
    }, fields=[
        "datetime",
        "content",
    ])
    df = df.apply(lambda x: '[' + x.name + ']' + ': ' + x.astype(str))
    return df


# def show_dynamic_table(df: pd.DataFrame) -> None:
#     '''
#      This function displays a dynamic table in the terminal window, where each row of the input DataFrame is shown one by one.
#      Arguments:
#         df: A Pandas DataFrame containing the data to be displayed in the dynamic table.
#
#      Returns: None. This function does not return anything.
#
#     '''
#
#     return df
#     # table = PrettyTable(df.columns.tolist(),align='l')
#
#     # 将 DataFrame 的数据添加到表格中
#     # for row in df.itertuples(index=False):
#     #     table.add_row(row)
#
#     # 初始化终端
#     # term = Terminal()
#     #
#     # # 在终端窗口中滚动显示表格
#     # with term.fullscreen():
#     #     with term.cbreak():
#     #         print(term.clear())
#     #         with term.location(0, 0):
#     #             # 将表格分解为多行，并遍历每一行
#     #             lines = str(table).split('\n')
#     #             for i, line in enumerate(lines):
#     #                 with term.location(0, i):
#     #                     print(line)
#     #                     time.sleep(1)
#     #
#     #             while True:
#     #                 # 读取输入
#     #                 key = term.inkey(timeout=0.1)
#     #
#     #                 # 如果收到q键，则退出
#     #                 if key.lower() == 'q':
#     #                     break


def get_index_constituent(index_name: str = '', start_date:str ='', end_date:str ='') -> pd.DataFrame:
    """
        Query the constituent stocks of basic index (中证500) or a specified SW (申万) industry index

        args:
             index_name: the name of the index.
             start_date: the start date in "YYYYMMDD".
             end_date:  the end date in "YYYYMMDD".

        return:
            A pandas DataFrame containing the following columns:
            index_code
            index_name
            stock_code: the code of the constituent stock.
            stock_name:  the name of the constituent stock.
            weight: the weight of the constituent stock.
    """

    if '申万' in index_name:
        if '申万一级行业' in index_name:
            # index_name取后面的名字
            index_name = index_name[6:]
            df1 = pd.read_csv('SW2021_industry_L1.csv')
            index_code = df1[df1['industry_name'] == index_name]['index_code'].iloc[0]
        elif '申万二级行业' in index_name:
            index_name = index_name[6:]
            df1 = pd.read_csv('SW2021_industry_L2.csv')
            index_code = df1[df1['industry_name'] == index_name]['index_code'].iloc[0]
        elif '申万三级行业' in index_name:
            index_name = index_name[6:]
            df1 = pd.read_csv('SW2021_industry_L3.csv')
            index_code = df1[df1['industry_name'] == index_name]['index_code'].iloc[0]

        print('The industry code for ', index_name, ' is: ', index_code)

        # 拉取数据
        df = pro.index_member(**{
            "index_code": index_code ,  #'851251.SI'
            "is_new": "",
            "ts_code": "",
            "limit": "",
            "offset": ""
        }, fields=[
            "index_code",
            "con_code",
            "in_date",
            "out_date",
            "is_new",
            "index_name",
            "con_name"
        ])
        #
        # For each stock, filter the start_date and end_date that are between in_date and out_date.
        df = df[(df['in_date'] <= start_date)]
        df = df[(df['out_date'] >= end_date) | (df['out_date'].isnull())]



        df.rename(columns={'con_code': 'stock_code'}, inplace=True)

        df.rename(columns={'con_name': 'stock_name'}, inplace=True)
        #
        df['weight'] = np.nan

        df = df[['index_code', "index_name", 'stock_code', 'stock_name','weight']]

    else: # 宽基指数
        df1 = pro.index_basic(**{
            "ts_code": "",
            "market": "",
            "publisher": "",
            "category": "",
            "name": index_name,
            "limit": "",
            "offset": ""
        }, fields=[
            "ts_code",
            "name",
        ])

        index_code = df1["ts_code"][0]
        print(f'index_code for basic index {index_name} is {index_code}')


        # Step 2: Retrieve the constituents of an index based on the index code and given date.
        df = pro.index_weight(**{
            "index_code": index_code,
            "trade_date": '',
            "start_date": start_date,
            "end_date": end_date,
            "limit": "",
            "offset": ""
        }, fields=[
            "index_code",
            "con_code",
            "trade_date",
            "weight"
        ])
        # df = df.sort_values(by='trade_date', ascending=True)  #
        df['index_name'] = index_name
        last_day = df['trade_date'][0]
        #  for the last trading day
        df = df[df['trade_date'] == last_day]
        df_stock = pd.read_csv('tushare_stock_basic_20230421210721.csv')
        # Merge based on the stock code.
        df = pd.merge(df, df_stock, how='left', left_on='con_code', right_on='ts_code')
        # df.rename(columns={'name_y': 'name'}, inplace=True)
        df = df.drop(columns=['symbol', 'area', 'con_code'])
        df.sort_values(by='weight', ascending=False, inplace=True)
        df.rename(columns={'name': 'stock_name'}, inplace=True)
        df.rename(columns={'ts_code': 'stock_code'}, inplace=True)
        df.dropna(axis=0, how='any', inplace=True)
        #
        df = df[['index_code', "index_name", 'stock_code', 'stock_name', 'weight']]

    return df

# Determine whether the given name is a stock or a fund.,
def is_fund(ts_name: str = '') -> bool:
    # call  get_stock_code()和query_fund_name_or_code()
    if get_stock_code(ts_name) is not None and query_fund_name_or_code(ts_name) is None:
        return False
    elif get_stock_code(ts_name) is None and query_fund_name_or_code(ts_name) is not None:
        return True




def calculate_earning_between_two_time(stock_name: str = '', start_date: str = '', end_date: str = '', index: str = 'close') -> float:
    """
        Calculates the rate of return for a specified stock/fund between two dates.

        Args:
            stock_name: stock_name or fund_name
            start_date
            end_date
            index (str): The index used to calculate the stock return, including 'open' and 'close'.

        Returns:
            float: The rate of return for the specified stock between the two dates.
    """
    if is_fund(stock_name):
        fund_code = query_fund_name_or_code(stock_name)
        stock_data = query_fund_data(fund_code, start_date, end_date)
        if index =='':
            index = 'adj_nav'
    else:
        stock_data = get_stock_prices_data(stock_name, start_date, end_date,'daily')
    try:
        end_price = stock_data.iloc[-1][index]
        start_price = stock_data.iloc[0][index]
        earning = cal_dt(end_price, start_price)
        # earning = round((end_price - start_price) / start_price * 100, 2)
    except:
        print(ts_code,start_date,end_date)
        print('##################### 该股票没有数据 #####################')
        return None
    # percent = earning * 100
    # percent_str = '{:.2f}%'.format(percent)

    return  earning


def loop_rank(df: pd.DataFrame,  func: callable, *args, **kwargs) -> pd.DataFrame:
    """
        It iteratively applies the given function to each row and get a result using function. It then stores the calculated result in 'new_feature' column.

        Args:
        df: DataFrame with a single column
        func : The function to be applied to each row: func(row, *args, **kwargs)
        *args: Additional positional arguments for `func` function.
        **kwargs: Additional keyword arguments for `func` function.

        Returns:
        pd.DataFrame: A output DataFrame with three columns: the constant column, input column, and new_feature column.
                     The DataFrame is sorted based on the new_feature column in descending order.

        """
    df['new_feature'] = None
    loop_var = df.columns[0]
    for _, row in df.iterrows():
        res  = None
        var = row[loop_var]                                         #

        if var is not None:
            if loop_var == 'stock_name':
                stock_name = var
            elif loop_var == 'stock_code':
                stock_name = get_stock_name_from_code(var)
            elif loop_var == 'fund_name':
                stock_name = var
            elif loop_var == 'fund_code':
                stock_name = query_fund_name_or_code('',var)
            time.sleep(0.4)
            try:
                res = func(stock_name, *args, **kwargs)             #
            except:
                raise ValueError('#####################Error for func#####################')
            # res represents the result obtained for the variable. For example, if the variable is a stock name, res could be the return rate of that stock over a certain period or a specific feature value of that stock. Therefore, res should be a continuous value.
            # If the format of res is a float, then it can be used directly. However, if res is in DataFrame format, you can retrieve the value corresponding to the index.
            if isinstance(res, pd.DataFrame) and not res.empty:
                #
                try:
                    res = round(res.loc[:,args[-1]][0], 2)
                    df.loc[df[loop_var] == var, 'new_feature'] = res
                except:
                    raise ValueError('##################### Error ######################')
            elif isinstance(res, float): #
                res = res
                df.loc[df[loop_var] == var, 'new_feature'] = res
            print(var, res)


    # Remove the rows where the new_feature column is empty.
    df = df.dropna(subset=['new_feature'])
    stock_data = df.sort_values(by='new_feature', ascending=False)
    #
    stock_data.insert(0, 'unchanged', loop_var)
    stock_data = stock_data.loc[:,[stock_data.columns[0], loop_var, 'new_feature']]

    return stock_data

def output_mean_median_col(data: pd.DataFrame, col: str = 'new_feature') -> float:
    # It calculates the mean and median value for the specified column.

    mean = round(data[col].mean(), 2)
    median = round(data[col].median(), 2)
    #
    #print(title, mean)
    return (mean, median)


# def output_median_col(data: pd.DataFrame, col: str, title_name: str = '') -> float:
#     # It calculates the median value for the specified column and returns the median as a float value.
#
#     median = round(data[col].median(), 2)
#     #print(title_name, median)
#
#     return median


def output_weighted_mean_col(data: pd.DataFrame, col: str, weight_col: pd.Series) -> float:

    """
        Calculates the weighted mean of a column and returns the result as a float.

        Args:
            data (pd.DataFrame): The input cross-sectional or time-series data containing the feature columns.
            col (str): The name of the feature column to calculate the weighted mean for.
            weight_col (pd.Series): The weights used for the calculation, as a pandas Series.

        Returns:
            float: The weighted mean of the specified feature column.
        """

    weighted_mean = round(np.average(data[col], weights = weight_col)/100., 2)
    return weighted_mean



def get_index_data(index_name: str = '', start_date: str = '', end_date: str = '', freq: str = 'daily') -> pd.DataFrame:
    """
        This function retrieves daily, weekly, or monthly data for a given stock index.

        Arguments:
        - index_name: Name of the index
        - start_date: Start date in 'YYYYMMDD'
        - end_date: End date in 'YYYYMMDD'
        - freq: Frequency 'daily', 'weekly', or 'monthly'

        Returns:
        A DataFrame containing the following columns:
        trade_date, ts_code, close, open, high, low, pre_close: Previous day's closing price, change(涨跌额), pct_chg(涨跌幅), vol(成交量), amount(成交额), name: Index Name
        """
    df1 = pro.index_basic(**{
        "ts_code": "",
        "market": "",
        "publisher": "",
        "category": "",
        "name": index_name,
        "limit": "",
        "offset": ""
    }, fields=[
        "ts_code",
        "name",
    ])

    index_code = df1["ts_code"][0]
    print(f'index_code for index {index_name} is {index_code}')
    #
    if freq == 'daily':
        df = pro.index_daily(**{
            "ts_code": index_code,
            "trade_date": '',
            "start_date": start_date,
            "end_date": end_date,
            "limit": "",
            "offset": ""
        }, fields=[
            "trade_date",
            "ts_code",
            "close",
            "open",
            "high",
            "low",
            "pre_close",
            "change",
            "pct_chg",
            "vol",
            "amount"
        ])
    elif freq == 'weekly':
        df = pro.index_weekly(**{
            "ts_code": index_code,
            "trade_date": '',
            "start_date": start_date,
            "end_date": end_date,
            "limit": "",
            "offset": ""
        }, fields=[
            "trade_date",
            "ts_code",
            "close",
            "open",
            "high",
            "low",
            "pre_close",
            "change",
            "pct_chg",
            "vol",
            "amount"
        ])
    elif freq == 'monthly':
        df = pro.index_monthly(**{
            "ts_code": index_code,
            "trade_date": '',
            "start_date": start_date,
            "end_date": end_date,
            "limit": "",
            "offset": ""
        }, fields=[
            "trade_date",
            "ts_code",
            "close",
            "open",
            "high",
            "low",
            "pre_close",
            "change",
            "pct_chg",
            "vol",
            "amount"
        ])

    df = df.sort_values(by='trade_date', ascending=True)  #
    df['index_name'] = index_name
    return df





def get_north_south_money(start_date: str = '', end_date: str = '', trade_date: str = '') -> pd.DataFrame:
    #
    # trade_date: 交易日期
    # ggt_ss:	港股通（上海）
    # ggt_sz:	港股通（深圳）
    # hgt:	沪股通（亿元）
    # sgt:	深股通（亿元）
    # north_money:	北向资金（亿元）= hgt + sgt
    # south_money:	南向资金（亿元）= ggt_ss + ggt_sz
    # name:  固定为'A-H',代表A股和H股
    # accumulate_north_money: 累计北向资金流入
    # accumulate_south_money: 累计南向资金流入


    month_df = pro.moneyflow_hsgt(**{
        "trade_date": trade_date,
        "start_date": start_date,
        "end_date": end_date,
        "limit": "",
        "offset": ""
    }, fields=[
        "trade_date",
        "ggt_ss",
        "ggt_sz",
        "hgt",
        "sgt",
        "north_money",
        "south_money"
    ])

    month_df[['ggt_ss','ggt_sz','hgt','sgt','north_money','south_money']] = month_df[['ggt_ss','ggt_sz','hgt','sgt','north_money','south_money']]/100.0
    month_df = month_df.sort_values(by='trade_date', ascending=True)  #
    month_df['stock_name'] = 'A-H'
    month_df['accumulate_north_money'] = month_df['north_money'].cumsum()
    month_df['accumulate_south_money'] = month_df['south_money'].cumsum()
    return month_df



def plot_k_line(stock_data: pd.DataFrame, title: str = '') -> None:
    """
        Plots a K-line chart of stock price and volume.

        Args:
            stock_data : A pandas DataFrame containing the stock price information, in which each row
                represents a daily record. The DataFrame must contain the 'trade_date','open', 'close', 'high', 'low','volume', 'name' columns, which is used for k-line and volume.
                如果dataframe中还含有'macd'，'kdj', 'rsi', 'cci', 'boll','pe_ttm','turnover_rate'等列，则在k线图下方绘制这些指标的子图.
            title : The title of the K-line chart.

        Returns:
            None
    """

    #
    stock_data['trade_date'] = pd.to_datetime(stock_data['trade_date'], format='%Y%m%d')
    stock_data.set_index('trade_date', inplace=True)
    #
    custom_style = mpf.make_marketcolors(up='r', down='k', inherit=True)
    china_style = mpf.make_mpf_style(marketcolors=custom_style)

    # MACD
    # stock_data['macd1'] = stock_data['Close'].ewm(span=12).mean() - stock_data['Close'].ewm(span=26).mean()
    # stock_data['macd_signal1'] = stock_data['macd'].ewm(span=9).mean()

    #
    #mpf.plot(stock_data, type='candle', volume=True, title=title, mav=(5, 10, 20), style = china_style, addplot = macd)
    add_plot = []
    # The index column is located after the name column in the last few columns.
    # Retrieve the column names after the 'name' column.
    index_list = stock_data.columns[stock_data.columns.get_loc('stock_name')+1:]

    index_df = stock_data[index_list]

    color_list = ['green','blue','red','yellow','black','purple','orange','pink','brown','gray']
    custom_lines = []
    for i in range(len(index_list)):
        # If the column names contain 'boll', set panel to 0. Otherwise, set panel to 2.
        if 'boll' in index_list[i]:
            sub_plot = mpf.make_addplot(index_df[index_list[i]], panel=0, ylabel=index_list[i], color=color_list[i], type='line', secondary_y=True)
        elif  index_list[i] =='macd':
            sub_plot = mpf.make_addplot(index_df[index_list[i]], panel=2, ylabel=index_list[i], color=color_list[i], type='bar', secondary_y=False)

        else:
            sub_plot = mpf.make_addplot(index_df[index_list[i]], panel=2, ylabel=index_list[i], color=color_list[i], type='line', secondary_y=False)

        custom_line = Line2D([0], [0], color=color_list[i], lw=1, linestyle='dashed')


        add_plot.append(sub_plot)
        custom_lines.append(custom_line)

    mav_colors = ['red', 'green', 'blue']

    fig, axes = mpf.plot(stock_data, type='candle', volume=True, title=title, mav=(5, 10, 20), mavcolors=mav_colors, style=china_style, addplot=add_plot, returnfig=True)


    mav_labels = ['5-day MA', '10-day MA', '20-day MA']
    #
    legend_lines = [plt.Line2D([0], [0], color=color, lw=2) for color in mav_colors]

    #
    axes[0].legend(legend_lines, mav_labels)

    if len(index_list) ==1:
        label = index_list[0]
    elif len(index_list) > 1:
        label_list = [i.split('_')[0] for i in index_list]
        #
        label = list(set(label_list))[0]

    if len(index_list) >= 1:
        if 'boll' in label:
            axes[0].legend(custom_lines, index_list, loc='lower right')

        elif len(index_list) > 1:
            axes[-2].set_ylabel(label)
            axes[-2].legend(custom_lines, index_list,  loc='lower right')

    #
    fig.set_size_inches(20, 16)
    #
    for ax in axes:
        ax.grid(True)

    #fig.show()
    return axes


def cal_dt(num_at_time_2: float = 0.0, num_at_time_1: float = 0.0) -> float:
    """
        This function calculates the percentage change of a metric from one time to another.

        Args:
        - num_at_time_2: the metric value at time 2 (end time)
        - num_at_time_1: the metric value at time 1 (start time)

        Returns:
        - float: the percentage change of the metric from time 1 to time 2

        """
    if num_at_time_1 == 0:
        num_at_time_1 = 0.0000000001
    return round((num_at_time_2 - num_at_time_1) / num_at_time_1, 4)


def query_fund_info(fund_code: str = '') -> pd.DataFrame:
    #
    # fund_code	str	Y	基金代码 # fund_name	str	Y	简称 # management	str	Y	管理人 # custodian	str	Y	托管人 # fund_type	str	Y	投资类型 # found_date	str	Y	成立日期 # due_date	str	Y	到期日期 # list_date	str	Y	上市时间 # issue_date	str	Y	发行日期 # delist_date	str	Y	退市日期 # issue_amount	float	Y	发行份额(亿) # m_fee	float	Y	管理费 # c_fee	float	Y	托管费
    # duration_year	float	Y	存续期 # p_value	float	Y	面值 # min_amount	float	Y	起点金额(万元) # benchmark	str	Y	业绩比较基准 # status	str	Y	存续状态D摘牌 I发行 L已上市 # invest_type	str	Y	投资风格 # type	str	Y	基金类型 # purc_startdate	str	Y	日常申购起始日 # redm_startdate	str	Y	日常赎回起始日 # market	str	Y	E场内O场外
    """
        Retrieves information about a fund based on the fund code.

        Args:
            fund_code (str, optional): Fund code. Defaults to ''.

        Returns:
            df (DataFrame): A DataFrame containing various information about the fund, including fund code, fund name,
                            management company, custodian company, investment type, establishment date, maturity date,
                            listing date, issuance date, delisting date, issue amount, management fee, custodian fee,
                            fund duration, face value, minimum investment amount, benchmark, fund status, investment style,
                            fund type, start date for daily purchases, start date for daily redemptions, and market type.
                            The column 'ts_code' is renamed to 'fund_code', and 'name' is renamed to 'fund_name' in the DataFrame.
        """
    df = pro.fund_basic(**{
        "ts_code": fund_code,
        "market": "",
        "update_flag": "",
        "offset": "",
        "limit": "",
        "status": "",
        "name": ""
    }, fields=[
        "ts_code",
        "name",
        "management",
        "custodian",
        "fund_type",
        "found_date",
        "due_date",
        "list_date",
        "issue_date",
        "delist_date",
        "issue_amount",
        "m_fee",
        "c_fee",
        "duration_year",
        "p_value",
        "min_amount",
        "benchmark",
        "status",
        "invest_type",
        "type",
        "purc_startdate",
        "redm_startdate",
        "market"
    ])
    #
    df.rename(columns={'ts_code': 'fund_code'}, inplace=True)
    df.rename(columns={'name': 'fund_name'}, inplace=True)
    return df

def query_fund_data(fund_code: str = '', start_date: str = '', end_date: str = '') -> pd.DataFrame:
    #
    # ts_code	str	Y	TS代码 # ann_date	str	Y	公告日期 # nav_date	str	Y	净值日期 # unit_nav	float	Y	单位净值 # accum_nav	float	Y	累计净值
    # accum_div	float	Y	累计分红 # net_asset	float	Y	资产净值 # total_netasset	float	Y	合计资产净值 # adj_nav	float	Y	复权单位净值  pct_chg 每日涨跌幅
    """
        Retrieves fund data based on the fund code, start date, and end date.

        Args:
            fund_code (str, optional): Fund code. Defaults to ''.
            start_date (str, optional): Start date in YYYYMMDD format. Defaults to ''.
            end_date (str, optional): End date in YYYYMMDD format. Defaults to ''.

        Returns:
            df (DataFrame): A DataFrame containing fund data, including TS code, announcement date, net asset value date,
                            unit net asset value, accumulated net asset value, accumulated dividends, net asset value,
                            total net asset value, adjusted unit net asset value, and fund name. The 'ts_code' column is renamed
                            to 'fund_code', 'nav_date' is renamed to 'trade_date', and the DataFrame is sorted by the trade date
                            in ascending order. If the fund code does not exist, None is returned.
        """
    df = pro.fund_nav(**{
        "ts_code": fund_code,
        "nav_date": "",
        "offset": "",
        "limit": "",
        "market": "",
        "start_date": start_date,
        "end_date": end_date
    }, fields=[
        "ts_code",
        "ann_date",
        "nav_date",
        "unit_nav",
        "accum_nav",
        "accum_div",
        "net_asset",
        "total_netasset",
        "adj_nav",
        "update_flag"
    ])
    try:
        fund_name= query_fund_name_or_code(fund_code=fund_code)
        df['fund_name'] = fund_name
        #
        df.rename(columns={'ts_code': 'fund_code'}, inplace=True)
        df.rename(columns={'nav_date': 'trade_date'}, inplace=True)
        df.sort_values(by='trade_date', ascending=True, inplace=True)
    except:
        print(fund_code,'基金代码不存在')
        return None
    #
    df['pct_chg'] = df['adj_nav'].pct_change()
    #
    df.loc[0, 'pct_chg'] = 0.0


    return df

def query_fund_name_or_code(fund_name: str = '', fund_code: str = '') -> str:
    #
    """
        Retrieves the fund code based on the fund name or Retrieves the fund name based on the fund code.

        Args:
            fund_name (str, optional): Fund name. Defaults to ''.
            fund_code (str, optional): Fund code. Defaults to ''.

        Returns:
            code or name: Fund code if fund_name is provided and fund_code is empty. Fund name if fund_code is provided and fund_name is empty.
        """


    #df = pd.read_csv('./tushare_fund_basic_20230508193747.csv')
    # Query the fund code based on the fund name.
    if fund_name != '' and fund_code == '':
        #
        df = pd.read_csv('./tushare_fund_basic_all.csv')
        #
        # df = pro.fund_basic(**{
        #     "ts_code": "",
        #     "market": "",
        #     "update_flag": "",
        #     "offset": "",
        #     "limit": "",
        #     "status": "",
        #     "name": fund_name
        # }, fields=[
        #     "ts_code",
        #     "name"
        # ])
        try:
            #
            code = df[df['name'] == fund_name]['ts_code'].values[0]
        except:
            #print(fund_name,'基金名称不存在')
            return None
        return code
    # Query the fund name based on the fund code.
    if fund_code != '' and fund_name == '':
        df = pd.read_csv('./tushare_fund_basic_all.csv')
        try:
            name = df[df['ts_code'] == fund_code]['name'].values[0]
        except:
            #print(fund_code,'基金代码不存在')
            return None
        return name



def print_save_table(df: pd.DataFrame, title_name: str, save:bool = False ,file_path: str = './output/') -> None:
    """
        It prints the dataframe as a formatted table using the PrettyTable library and saves it to a CSV file at the specified file path.

        Args:
        - df: the dataframe to be printed and saved to a CSV file
        - title_name: the name of the table to be printed and saved
        - save: whether to save the table to a CSV file
        - file_path:  the file path where the CSV file should be saved.

        Returns: None
    """

    # 创建表格table.max_width = 20

    # table = PrettyTable(df.columns.tolist())
    # table.align = 'l'
    # table.max_width = 40
    #
    # #
    # for row in df.itertuples(index=False):
    #     table.add_row(row)

    #print(table)


    if not os.path.exists(file_path):
        os.makedirs(file_path)

    if file_path is not None and save == True:
        file_path = file_path + title_name + '.csv'
        df.to_csv(file_path, index=False)
    return df



#
def merge_indicator_for_same_stock(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
        Merges two DataFrames (two indicators of the same stock) based on common names for same stock. Data from two different stocks cannot be merged

        Args:
            df1: DataFrame contains some indicators for stock A.
            df2: DataFrame contains other indicators for stock A.

        Returns:
            pd.DataFrame: The merged DataFrame contains two different indicators.
    """
    if len(set(df1.columns).intersection(set(df2.columns))) > 0:
        # If there are identical column names, merge the two DataFrames based on the matching column names.
        #
        common_cols = list(set(df1.columns).intersection(set(df2.columns)))
        #
        df = pd.merge(df1, df2, on=common_cols)
        return  df
    else:
        #
        raise ValueError('The two dataframes have no columns in common.')

def select_value_by_column(df1:pd.DataFrame, col_name: str = '', row_index: int = -1) -> Union[pd.DataFrame, Any]:
    """
        Selects a specific column or a specific value within a DataFrame.

        Args:
            df1: The input DataFrame.
            col_name: The name of the column to be selected.
            row_index: The index of the row to be selected.

        Returns:
            Union[pd.DataFrame, Any]. row_index=-1: df1[col_name].to_frame() or df1[col_name][row_index]
    """
    if row_index == -1:
        #
        return df1[col_name].to_frame()
    else:
        #
        return df1[col_name][row_index]



if __name__ == "__main__":
    stock_name='成都银行'
    stock_name2='五粮液'
    stock_name3 = '宁德时代'
    start = '20230104'
    end = '20230504'
    fund_name = "华商优势行业" #'易方达蓝筹精选'

    start_quarter = '201001'
    end_quarter = '202303'
    title_name ='上证50成分股收益率'
    ax = None
    res = is_fund('易方达蓝筹精选')
    #_, ax = plt.subplots()
    # code = query_fund_name_or_code('华商优势行业')
    # ------------step1 数据查询层  获取股票代码
    # start_last_year = get_last_year_date(start)
    # end_last_year = get_last_year_date(end)
    stock_code = get_stock_code(stock_name)
    # name = get_stock_name_from_code(stock_code)
    # print(name)
    # print(stock_code)
    # stock_code2 = get_stock_code(stock_name2)
    # stock_code3 = get_stock_code(stock_name3)
    # stock_technical_data = get_Financial_data(stock_code, start, end)
    # macrodata = get_ppi_data('', start_quarter, end_quarter, 'ppi_yoy')
    # index_daily = get_index_data('沪深300',start,end,'daily')
    # index_daily2 = get_index_data('中证500',start,end,'daily')
    # index_daily3 = get_index_data('中证1000',start,end,'daily')
    # index_daily4 = get_index_data('创业板指',start,end,'daily')
    #stock_data = get_index_constituent('上证50','20230101','20230508')
    # money = get_north_south_money('20230425', '20230426')
    # stock_data = get_stock_prices_data(stock_code, start, end)

    # stock_data = get_stock_monthly_prices_data("","", "",'20230331')
    # stock_data = get_stock_prices_data('', start, end, 'daily')
    # fund_df = query_fund_Manager('周海栋')
    #
    # fund_code  = select_value_by_column(fund_df, 'fund_code', -1)
    # res_earning = loop_rank(fund_code, calculate_earning_between_two_time, start, end, 'adj_nav')
    # print(res_earning)
    #fund_code = query_fund_name_or_code(fund_name,'')


    #fund_data = query_fund_data(fund_code, start, end)
    #df_daily = get_daily_trading_data(stock_code,'20200101', '20230526')
    # stock_data2 = get_stock_prices_data(stock_code2, start, end,'daily')
    # stock_data3 = get_stock_prices_data(stock_code3, start, end,'daily')

    # dynamic_new = get_latest_new_from_web('sina')
    #stock_df = get_sw_industry_stock('城商行Ⅱ','L2')
    # df_macro = get_cpi_ppi_currency_supply_data('200101','202304','cpi','nt_yoy')
    # df_macro = get_cpi_ppi_currency_supply_data('200101','202304','ppi','ppi_yoy')
    # df_macro = get_cpi_ppi_currency_supply_data('200101','202304','currency_supply','m2_yoy')
    # df_gdp = get_GDP_data('2001Q1','2023Q1','gdp_yoy')
    # df_gdp = predict_next_value(df_gdp, 'gdp_yoy', 4)
    #company_df = get_company_info('贵州茅台')
    #print_save_table(company_df, '贵州茅台公司信息')
    #fin_df = get_Financial_data_from_time_range(stock_code, '20200101', '20230526','roe')

    #tech_df = get_stock_technical_data(stock_code, start, end)



    # ----------------------------------step2 数据处理层  在截面或者时序数据-------------------------------------------------------
    # 提取相应指标, 数据处理, 排序,提取,求差,加工..,
    # fund_info = query_fund_info('005827.OF')
    # value = select_value_by_column(fund_info, 'fund_name', 0)
    #fund_index = calculate_stock_index(fund_data,'adj_nav')
    #stock_index = rank_index_cross_section(stock_data, 'pct_chg', -1, False)
    #stock_index = calculate_stock_index(stock_data, 'pct_chg')
    #stock_index_each_day = calculate_stock_index(money, 'north_money')
    #stock_index = calculate_stock_index(fin_df, 'roe')
    # stock_index2 = calculate_stock_index(stock_data2, 'Cumulative_Earnings_Rate')
    # stock_index3 = calculate_stock_index(stock_data3, 'Cumulative_Earnings_Rate')
    # stock_index4 = calculate_stock_index(index_daily4, 'Cumulative_Earnings_Rate')
    # stock_index2 = calculate_stock_index(stock_data2, 'Cumulative_Earnings_Rate')
    #stock_index = calculate_stock_index(stock_data1, 'close')
    #stock_index2 = calculate_stock_index(tech_df, 'macd')
    #stock_index1 = calculate_stock_index(stock_data, 'candle_K')
    #stock_index2 = calculate_stock_index(df_daily, 'pe_ttm')
    #merge_df = merge_data(stock_index1, stock_index2)
    #res_earning = loop_rank(stock_data, 'stock_name', calculate_earning_between_two_time, start, end)
    # index_profit_yoy = loop_rank(stock_data, 'stock_name', get_Financial_data, start, end, 'profit_dedt')
    # index_profit_yoy = loop_rank(stock_data, 'stock_name', get_Financial_data, start, end, 'netprofit_yoy')

    #res_earning_top_n = rank_index_cross_section(stock_index, 10, False)
    #index_profit_yoy_last = loop_rank(stock_data, 'stock_name', get_Financial_data, start_last_year, end_last_year, 'profit_dedt')
    # profit_yoy = calculate_stock_index(stock_technical_data, 'dt_netprofit_yoy')
    # accumulate_north_month = calculate_stock_index(money, 'accumulate_south_money')
    # accumulate_north_month = calculate_stock_index(res_earning, 'accumulate_south_money')
    # stock_code = get_stock_code(stock_name)
    # fin_df1 = get_Financial_data_from_time_range(stock_code, '20150101', '20230526', 'roa')
    # fin_df2 = get_Financial_data_from_time_range(stock_code, '20150101', '20230526', 'roa')
    # ax = plot_stock_data(fin_df1, ax, 'line', title_name)
    # ax = plot_stock_data(fin_df2, ax, 'line', title_name)
    #stock_data = get_index_constituent('上证50','20220105', '20230505')
    # stock_data = get_index_constituent('申万二级行业城商行Ⅱ','20220105', '20220505')
    # #stock_list = select_value_by_column(stock_data, 'stock_name', -1)
    #
    # index_profit_yoy = loop_rank(stock_list, get_Financial_data, start, 'netprofit_yoy')
    # median = output_median_col(index_profit_yoy, 'new_feature')
    # ax = plot_stock_data(index_profit_yoy, ax, 'bar', '上证50的最近季度归母净利润同比增长率')






    # ----------------------------------step3 可视化层：文字，图片，表格等多种模态数据输出-------------------------------------------------------
    #ax = plot_stock_data(stock_index, ax, 'line', title_name)
    #ax = plot_stock_data(stock_index_each_day, ax, 'bar', title_name)
    #print_save_table(fund_info, title_name)

    #_, sum_new = output_mean_sum_col(index_profit_yoy,'new_feature')
    #_, sum_old = output_mean_sum_col(index_profit_yoy_last,'new_feature')


    #print('科创50成分股的最近季度归母净利润同比增长率中位数%：', median)
    #dt = cal_dt(sum_new, sum_old)
    #print('上证50成分股的最近季度归母净利润同比增长率：',dt)

    #plot_k_line(merge_df, title_name)
    # ax = plot_stock_data(index_profit_yoy, ax, 'bar', '上证50成分股的最近季度归母净利润同比增长率')
    #ax = plot_stock_data(accumulate_north_month, ax, 'line', '2023年1月至4月南向资金累计流向')

    # ax2 = plot_stock_data(stock_index2, ax1, 'line', '贵州茅台VS五粮液近十年收益率对比图')
    # ax = plot_stock_data(stock_index, ax,'line', title_name)
    # ax = plot_stock_data(stock_index2, ax,'line', title_name)
    # ax = plot_stock_data(stock_index3, ax,'line', title_name)
    # ax = plot_stock_data(stock_index4, ax,'line', title_name)

    #ax = plot_stock_data(df_gdp, ax, 'line','2010-2022年国内每季度gdp增速同比')
    print_save_table(df_gdp,'GDP预测',True)

    # show_dynamic_table(dynamic_new)


    # ax = plot_stock_data(res_earning, None, 'bar', '张坤管理各个基金收益率')
    # stock_data = get_index_constituent('上证50', '20230101', '20230508')
    # stock_list = select_value_by_column(stock_data, 'stock_name', -1)
    # res_earning = loop_rank(stock_list, calculate_earning_between_two_time, start, end)
    # res_earning_top_n = rank_index_cross_section(res_earnng, 10, False)
    # ax = plot_stock_data(res_earning_top_n, ax, 'bar', title_name)

    # stock_data = get_index_constituent('上证50', '20230101', '20230508')
    # stock_list = select_value_by_column(stock_data, 'stock_name', -1)
    # res_earning = loop_rank(stock_list, calculate_earning_between_two_time, '20230101', '20230508')
    # res_earning_top_n = rank_index_cross_section(res_earning, 10, False)
    # ax = plot_stock_data(res_earning_top_n, ax, 'bar', title_name)

    # fund_code = query_fund_name_or_code(fund_name, '')
    # fund_data = query_fund_data(fund_code, start, end)
    # fund_index = calculate_stock_index(fund_data, 'adj_nav')
    # ax = plot_stock_data(fund_index, ax, 'line', title_name)
    # fund_df = query_fund_Manager('张坤')
    # fund_code = select_value_by_column(fund_df, 'fund_code', -1)
    # res_earning = loop_rank(fund_code, calculate_earning_between_two_time, start, end, 'adj_nav')
    # ax = plot_stock_data(res_earning, None, 'bar', '张坤管理各个基金收益率')
    # company_df = get_company_info('贵州茅台')
    # print_save_table(company_df,'gzmt', False)




    if ax is not None:
        plt.grid()
        plt.show()



# xxx基金经理管理的几只基金中，收益率最高的那只基金的规模是多少----找基金经理search，按收益率排序rank，找到收益率最高的那个select，显示基金信息 show
# 食品饮料行业中所有股票近十年涨幅最大的股票的信息----找行业search（行业分类--找到行业代码，根据行业代码找到股票成分), 收益率排序rank，找到涨幅最大的那个select，显示股票信息show












