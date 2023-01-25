import matplotlib
import warnings
import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams.update({'font.size': 16})
plt.style.use('ggplot')
time_end = datetime(2018, 1, 3)

df_cum = pd.read_excel('./cum.xlsx')[:1000]
df_sale = pd.read_excel('./sale.xlsx')[:1000]

print("--------------会员卡cum表信息---------------")
print("处理前：\n", df_cum.info())
a = df_cum.shape[0]
df_cum = df_cum.sort_values("登记时间", ascending=True).drop_duplicates('会员卡号', keep='first', ignore_index=True)
a1 = df_cum.shape[0]
# 会员卡号没有缺失值
print('step1：会员卡号共去重{}条记录，并保留登记时间最早的一条记录'.format(a - a1))

print("--------------会员消费sale表信息---------------")
# 该表只有会员卡和对应的会员积分缺失
df_sale_clean = df_sale.drop(['收银机号', '柜组编码', '柜组名称'], axis=1)
print("处理前：\n", df_sale_clean.info())
df_sale_clean = df_sale_clean.drop_duplicates()
index1 = df_sale_clean['消费金额'] > 0
index2 = df_sale_clean['此次消费的会员积分'] > 0
index3 = df_sale_clean['销售数量'] > 0
index4 = df_sale_clean['消费金额'] == df_sale_clean['此次消费的会员积分']
index5 = pd.to_datetime(df_sale_clean['消费产生的时间']) < time_end
df_sale_clean = df_sale_clean.loc[index1 & index2 & index3 & index4 & index5, :]
df_sale_clean.index = range(df_sale_clean.shape[0])
print("消费表处理后：\n", df_sale_clean.info())

# 根据会员卡号有无划分sale表
sale_without_card = df_sale_clean[df_sale_clean["会员卡号"].isnull()]
sale_with_card = df_sale_clean[df_sale_clean["会员卡号"].notnull()]
df_sale_wait = sale_with_card.sort_values('消费产生的时间', ascending=True).drop_duplicates('会员卡号', keep='first',
                                                                                     ignore_index=True)

# 找到登记时间缺失但具有消费记录的会员卡号，将最早消费记录作为登记时间；并合并到会员表中
df2 = pd.merge(df_sale_wait, df_cum, on='会员卡号')  # 内连接，保留主键相同的两个表的信息

for index, row in df2.iterrows():
    if pd.isna(df2.loc[index, '登记时间']) and pd.notna(df2.loc[index, '消费产生的时间']):
        df2.loc[index, '登记时间'] = df2.loc[index, '消费产生的时间']
# df2['登记时间2'] = df2['登记时间'].apply(lambda x: x if x != None else df2['消费产生的时间'])
df2 = pd.concat([df2['会员卡号'], df2['登记时间']], axis=1)
df2.columns = ['会员卡号', '登记时间2']
df_cum = pd.merge(df_cum, df2, on='会员卡号', how='outer')
for index, row in df_cum.iterrows():
    if pd.isna(df_cum.loc[index, '登记时间']) and pd.notna(df_cum.loc[index, '登记时间2']):
        df_cum.loc[index, '登记时间'] = df_cum.loc[index, '登记时间2']
# df_cum['登记时间'] = df_cum['登记时间'].apply(lambda x: x if x != None else df_cum['登记时间2'])
df_cum = df_cum.drop(['登记时间2'], axis=1)

# 获得补充登记时间的会员表后，填补性别列
df_cum['性别'].fillna(df_cum['性别'].mode().values[0], inplace=True)
index6 = df_cum['登记时间'] > df_cum['出生日期']  # 出生日期和登记时间的比较
index7 = pd.to_datetime(df_cum['登记时间']) < time_end  # 将超过收集数据2018-1-3的登记时间当作异常值进行剔除
df_cum = df_cum.loc[index6 & index7, :]
df_cum.index = range(df_cum.shape[0])
print("会员表处理后：", df_cum.info())
# df_cum.to_excel("会员信息.xlsx", index=False)


# df为有消费记录的会员消费表
df = pd.merge(sale_with_card, df_cum, on='会员卡号', how='left').reindex()
# print("表合并后：", df.info())

# df1所有消费记录
df1 = pd.concat([df, sale_without_card]).reindex()
df1['会员'] = 1
df1.loc[df1['性别'].isnull(), '会员'] = 0

# 分析会员情况
# L = pd.read_excel("会员信息.xlsx", header=0)
L = df_cum
# L2 = pd.DataFrame(L.loc[L["出生日期"].notnull(), ["出生日期"]]).reindex()
# L3 = pd.DataFrame(L.loc[L["登记时间"].notnull(), ["会员卡号", "登记时间"]]).reindex()
# # 处理男女比例这一列，女表示0，男表示1
# L['性别'] = L['性别'].apply(lambda x: '男' if x == 1 else '女')
# sex_sort = L['性别'].value_counts()


# 可以将年龄划分为老年（1920-1950）、中年（1960-1990）、青少年（1990-2010），再重新绘制一个饼图，
def get_age(L2):
    L2['年龄'] = L2['出生日期'].astype(str).apply(lambda x: x[:3] + '0')
    L2['年龄'] = L2['年龄'].astype(int)
    condition = "年龄 >= 1920 and 年龄 <= 2010"
    L2 = L2.query(condition)
    L2.index = range(L2.shape[0])
    L2['年龄段'] = '中年'
    L2.loc[L2['年龄'] <= 1950, '年龄段'] = '老年'
    L2.loc[L2['年龄'] >= 1990, '年龄段'] = '青少年'


# get_age(L2)
# age_count = L2['年龄段'].value_counts()
#
# # 自定义一个函数来实现两列数据时间相减
# def time_minus(df, end_time):
#     """
#     df: 为DataFrame形式，有列数据，第一列为“会员卡号”，第二列为被减的时间
#     end_time: 结束时间
#     """
#     df.columns = ['A', 'B']
#     df['C'] = end_time
#     l = pd.to_datetime(df['C']) - pd.to_datetime(df['B'])
#     l = l.apply(lambda x: str(x).split(' ')[0])
#     l = l.astype(int) / 30
#     return l
#
# # 入会程度
# end_time = '2018-1-3'
# LL = time_minus(L3, end_time)
# L3['入会程度'] = LL.apply(lambda x: '老用户' if int(x) >= 13 else '中等用户' if int(x) >= 4 else '新用户')
# time_count = L3['入会程度'].value_counts()
#
# # 绘图分析
# # 使用上述预处理后的数据集L，包含两个字段，分别是“年龄”和“性别”，先画出年龄的条形图
# fig, axs = plt.subplots(1, 3, figsize=(16, 7), dpi=100)
# # 绘制条形图
# ax = sns.countplot(x='年龄', data=L2, ax=axs[0])
# # 设置数字标签
# for p in ax.patches:
#     height = p.get_height()
#     ax.text(x=p.get_x() + (p.get_width() / 2), y=height + 500, s='{:.0f}'.format(height), ha='center')
# axs[0].set_title('会员的出生年代')
# # 绘制饼图
# axs[1].pie(sex_sort, labels=sex_sort.index, wedgeprops={'width': 0.4}, counterclock=False, autopct='%.2f%%',
#            pctdistance=0.8)
# axs[1].set_title('会员的男女比例')
#
# axs[2].pie(time_count, labels=time_count.index, wedgeprops={'width': 0.4}, counterclock=False, autopct='%.2f%%',
#            pctdistance=0.8)
# axs[2].set_title('会员的入会程度')
# plt.show()
# plt.savefig('./会员的基本情况.png')

# 分析所有会员与非会员的销售情况
# fig, axs = plt.subplots(1, 2, figsize=(12, 7), dpi=100)
# # 订单以消费产生的时间为准
# axs[0].pie([len(df1.loc[df1['会员'] == 1, '消费产生的时间'].unique()), len(df1.loc[df1['会员'] == 0, '消费产生的时间'].unique())],
#            labels=['会员', '非会员'], wedgeprops={'width': 0.4}, counterclock=False, autopct='%.2f%%', pctdistance=0.8)
# axs[0].set_title('总订单占比')
# axs[1].pie([df1.loc[df1['会员'] == 1, '消费金额'].sum(), df1.loc[df1['会员'] == 0, '消费金额'].sum()],
#            labels=['会员', '非会员'], wedgeprops={'width': 0.4}, counterclock=False, autopct='%.2f%%', pctdistance=0.8)
# axs[1].set_title('总消费金额占比')
# plt.show()
# plt.savefig('./总订单和总消费占比情况.png')

# 总体分季度描述
df1['消费产生的时间'] = pd.to_datetime(df1['消费产生的时间'])
# 新增四列数据，季度、天、年份和月份的字段
df1['年份'] = df1['消费产生的时间'].dt.year
df1['月份'] = df1['消费产生的时间'].dt.month
df1['季度'] = df1['消费产生的时间'].dt.quarter
df1['天'] = df1['消费产生的时间'].dt.day
df1.to_excel("所有消费情况解析表.xlsx", index=False)


# 自定义一个函数来计算每个季度和每天的消费订单均数
def orders(df, label, div):
    '''
    df: 对应的数据集
    label: 为对应的列标签
    div: 为被除数
    '''
    x_list = range(div)
    order_nums = []
    for i in range(len(x_list)):
        # 该季度消费订单均数 = 该季度消费订单总数/4
        order_nums.append(int(len(df.loc[df[label] == x_list[i], '消费产生的时间'].unique()) / div))
    return x_list, order_nums


# 前提假设：消费者偏好在时间上不会发生太大的变化（均值），消费偏好——>以不同时间的订单数来衡量
quarters_list, quarters_order = orders(df1, '季度', 4)
month_list, month_order = orders(df1, '月份', 12)
days_list, days_order = orders(df1, '天', 12)
time_list = [quarters_list, month_list, days_list]
order_list = [quarters_order, month_order, days_order]
maxindex_list = [quarters_order.index(max(quarters_order)), month_order.index(max(month_order)),
                 days_order.index(max(days_order))]
fig, axs = plt.subplots(1, 3, figsize=(18, 7), dpi=100)
colors = np.random.choice(['r', 'g', 'b', 'orange', 'y'], replace=False, size=len(axs))
titles = ['季度的均值消费偏好', '月份的均值消费偏好', '天数的均值消费偏好']
labels = ['季度', '月份', '天数']
for i in range(len(axs)):
    ax = axs[i]
    ax.plot(time_list[i], order_list[i], linestyle='-.', c=colors[i], marker='o', alpha=0.85)
    ax.axvline(x=time_list[i][maxindex_list[i]], linestyle='--', c='k', alpha=0.8)
    ax.set_title(titles[i])
    ax.set_xlabel(labels[i])
    ax.set_ylabel('均值消费订单数')
    print(f'{titles[i]}最优的时间为: {time_list[i][maxindex_list[i]]}\t 对应的均值消费订单数为: {order_list[i][maxindex_list[i]]}')
plt.savefig('./季度月份天数的均值消费偏好情况.png')


# 自定义函数来绘制不同年份之间的的季度或天数的消费订单差异
def plot_qd(df, label_y, label_m, nrow, ncol):
    """
    df: 为DataFrame的数据集
    label_y: 为年份的字段标签
    label_m: 为标签的一个列表
    n_row: 图的行数
    n_col: 图的列数
    """
    # 必须去掉最后一年的数据，只能对2015-2017之间的数据进行分析
    y_list = np.sort(df[label_y].unique().tolist())[:-1]
    colors = np.random.choice(['r', 'g', 'b', 'orange', 'y', 'k', 'c', 'm'], replace=False, size=len(y_list))
    markers = ['o', '^', 'v']
    plt.figure(figsize=(8, 6), dpi=100)
    fig, axs = plt.subplots(nrow, ncol, figsize=(16, 7), dpi=100)
    for k in range(len(label_m)):
        m_list = np.sort(df[label_m[k]].unique().tolist())
        for i in range(len(y_list)):
            order_m = []
            index1 = df[label_y] == y_list[i]
            for j in range(len(m_list)):
                index2 = df[label_m[k]] == m_list[j]
                order_m.append(len(df.loc[index1 & index2, '消费产生的时间'].unique()))
            axs[k].plot(m_list, order_m, linestyle='-.', c=colors[i], alpha=0.8, marker=markers[i], label=y_list[i],
                        markersize=4)
        axs[k].set_xlabel(f'{label_m[k]}')
        axs[k].set_ylabel('消费订单数')
        axs[k].set_title(f'2015-2017年会员的{label_m[k]}消费订单差异')
        axs[k].legend()
    plt.savefig(f'./2015-2017年会员的{"和".join(label_m)}消费订单差异.png')


plot_qd(df1, '年份', ['季度', '月份', '天'], 1, 3)

# 构建会员用户特征
df1 = pd.read_excel("所有消费情况解析表.xlsx", header=0)
df_vip = df1.loc[df1["会员"] == 1, :].drop(["会员"], axis=1).dropna().reindex()
get_age(df_vip)
df_vip = df_vip.drop(["出生日期"], axis=1)
df_vip['时间'] = df_vip['消费产生的时间'].dt.hour

# 自定义一个函数来实现两列数据时间相减
def time_minus(df, end_time):
    """
    df: 为DataFrame形式，有列数据，第一列为“会员卡号”，第二列为被减的时间
    end_time: 结束时间
    """
    df.columns = ['A', 'B']
    df['C'] = end_time
    l = pd.to_datetime(df['C']) - pd.to_datetime(df['B'])
    l = l.apply(lambda x: str(x).split(' ')[0])
    l = l.astype(int) / 30
    return l

# 开始登记的时间
df_L = df_vip.groupby('会员卡号')['登记时间'].agg(lambda x: x.values[-1]).reset_index()
# 最后一次消费的时间
df_B = df_vip.groupby('会员卡号')['消费产生的时间'].agg(lambda x: x.values[-1]).reset_index()

# 调用函数，end_time为“2018-1-3”
end_time = '2018-1-3'
L = time_minus(df_L, end_time)
B = time_minus(df_B, end_time)
# 会员消费的总次数
C = df_vip.groupby('会员卡号')['消费产生的时间'].agg(lambda x: len(np.unique(x.values))).reset_index(drop=True)
# 会员消费的总金额
M = df_vip.groupby('会员卡号')['消费金额'].agg(lambda x: np.sum(x.values)).reset_index(drop=True)
# 会员的积分总数
P = df_vip.groupby('会员卡号')['此次消费的会员积分'].agg(lambda x: np.sum(x.values)).reset_index(drop=True)
# 创造一列特征字段“消费时间偏好”（凌晨、上午、中午、下午、晚上）
"""
凌晨：0-5点
上午：6-10点
中午：11-13点
下午：14-17点
晚上：18-23点
"""
# df_vip['消费时间偏好'] = df_vip['时间'].apply(lambda x: '晚上' if x >= 18 else '下午' if x >= 14 else '中午'
# if x >= 11 else '上午' if x >= 6 else '凌晨')
# 开始构建对应的特征标签
df_i = pd.Series(df_vip['会员卡号'].unique())
df_LBCMP = pd.concat([df_i, L, B, C, M, P], axis=1)
df_LBCMP.columns = ['id', 'L', 'B', 'C', 'M', 'P']
# 保存数据
df_LBCMP.to_csv('./LBCMP.csv', encoding='gb18030', index=None)

# df_LBCMP = pd.read_csv('./LBCMP.csv', encoding='gbk')
print(df_LBCMP.describe())  # 对每列描述性统计

# 构建描述表
"""
L（入会等级）：3个月以下为新用户，4-12个月为中等用户，13个月以上为老用户
B（最近购买时间）：购买日期
C（消费次数）：次数20次以上的为高频消费，6-19次为中频消费，5次以下为低频消费
M（消费金额）：10万以上为高等消费，1万-10万为中等消费，1万以下为低等消费
P（消费积分）：10万以上为高等积分用户，1万-10万为中等积分用户，1万以下为低等积分用户
"""
# df_profile = pd.DataFrame()
# df_profile['会员卡号'] = df_LBCMP['id']
# df_profile['入会程度'] = df_LBCMP['L'].apply(lambda x: '老用户' if int(x) >= 13 else '中等用户' if int(x) >= 4 else '新用户')
# df_profile['最近购买的时间'] = df_LBCMP['B'].apply(lambda x: '您最近' + str(int(x) * 30) + '天前进行过一次购物')
# df_profile['消费频次'] = df_LBCMP['C'].apply(lambda x: '高频消费' if x >= 20 else '中频消费' if x >= 6 else '低频消费')
# df_profile['消费金额'] = df_LBCMP['M'].apply(lambda x: '高等消费用户' if int(x) >= 1e+05 else '中等消费用户' if int(x) >= 1e+04 else '低等消费用户')
# df_profile['消费积分'] = df_LBCMP['P'].apply(lambda x: '高等积分用户' if int(x) >= 1e+05 else '中等积分用户' if int(x) >= 1e+04 else '低等积分用户')

from wordcloud import WordCloud
# 绘制词云来描述会员特征
def wc_plot(df, id_label = None):
    """
    df: 为DataFrame的数据集
    id_label: 为输入用户的会员卡号，默认为随机取一个会员进行展示
    """
    myfont = 'C:/Windows/Fonts/simkai.ttf'
    if id_label == None:
        id_label = df.loc[np.random.choice(range(df.shape[0])), '会员卡号']
    text = df[df['会员卡号'] == id_label].T.iloc[:, 0].values.tolist()
    plt.figure(dpi = 100)
    wc = WordCloud(font_path = myfont, background_color = 'white', width = 500, height = 400).generate_from_text(' '.join(text))
    plt.imshow(wc)
    plt.axis('off')
    plt.savefig(f'./会员卡号为{id_label}的用户画像.png')
    plt.show()

# 随机查找一个会员来绘制用户画像
# wc_plot(df_profile)
