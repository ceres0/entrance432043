import pandas as pd

# 1. 用户行为数据预处理
# 1.1 读取用户行为数据
user_data = pd.read_csv('tianchi_mobile_recommend_train_user.txt', sep=',')

# 1.2 时间处理
user_data['time'] = pd.to_datetime(user_data['time'])
# 筛选出训练时间范围内（2014-11-18至2014-12-18）的数据
start_date = pd.to_datetime('2014-11-18')
end_date = pd.to_datetime('2014-12-18')
user_data = user_data[(user_data['time'] >= start_date) & (user_data['time'] <= end_date)]

# 1.3 缺失值处理
# 对于user_geohash字段，这里简单示例采用删除包含缺失值的行（可根据实际情况调整策略）
user_data = user_data.dropna(subset=['user_geohash'])

# 1.4 数据编码
# 对behavior_type字段进行标签编码（这里使用简单的映射，也可以用sklearn的LabelEncoder等更规范方式）
behavior_mapping = {1: '浏览', 2: '收藏', 3: '加购物车', 4: '购买'}
user_data['behavior_type'] = user_data['behavior_type'].map(behavior_mapping)

# 1.5 其他可能的预处理（比如去重等，根据实际需求添加）
user_data = user_data.drop_duplicates()

# 2. 商品数据预处理
# 2.1 读取商品数据
item_data = pd.read_csv('tianchi_mobile_recommend_train_item.txt', sep=',')

# 2.2 缺失值处理
# 假设商品数据中如果有缺失值，可根据字段特点选择合适的处理方式，这里简单示例删除包含缺失值的行
item_data = item_data.dropna()

# 2.3 数据编码（如果有分类字段需要编码，根据实际情况进行）
# 例如对item_category字段进行独热编码（示例使用pandas的get_dummies方法）
item_data = pd.get_dummies(item_data, columns=['item_category'])

# 2.4 其他可能的预处理（比如对商品位置相关字段进行标准化等，依实际情况而定）
# 这里暂未添加额外处理，如有需要可自行补充

print("预处理后的用户行为数据：")
print(user_data.head())
print("预处理后的商品数据：")
print(item_data.head())