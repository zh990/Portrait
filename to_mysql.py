from sqlalchemy import create_engine
import pandas as pd
from datetime import datetime

engine = create_engine('mysql+pymysql://root:密码@localhost:3306/test')
data = pd.read_csv('./LBCMP.csv', encoding='gb18030', header=0)
data["create_time"] = pd.date_range('01/03/2018 12:56:31',periods=len(data),freq='3H')
data.to_sql('vip_look', engine, chunksize=10, index=None, if_exists='replace')
print('批量存入成功！')