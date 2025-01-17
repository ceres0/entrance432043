# 阿里移动推荐算法挑战赛

<https://tianchi.aliyun.com/competition/entrance/532043/>

## 数据分析

### 商品数据

- 总数：$6,781,009$
- 去除浏览数为0的商品后，商品数：$6,560,901$
- 有购买记录：$1,482,657$（$21.9\%$）
- item_geohash缺失数：$595,7739$（$87.9\%$）
- >10：$144,239$
- >100：$6,079$

### 用户行为数据

- 总数：$1,165,522,826$,
- user_geohash缺失数：$795,826,525$（$68.3\%$）
- 每日购买量：332,478(301,913 ~ 365,579)

### 用户数据

- 用户数：$1,000,000$
- 去除浏览数为0的用户后，用户数：$999,958$
- >10：$353,598$
- >100：$3,928$

### 数据范围

- 设定观察日时间范围为：$2014-12-04$至$2014-12-18$

## 特征工程

### 可能的特征

- 用户特征
  - 用户浏览次数
  - 用户收藏次数
  - 用户加购物车次数
  - 用户购买次数
  - 浏览购买比
- 商品特征
  - 商品被浏览次数
  - 商品被收藏次数
  - 商品被加购物车次数
  - 商品被购买次数
  - 浏览购买比
- 交叉特征
  - 用户1、3、7、15日内对该商品交互（浏览、收藏、加购物车、购买）的次数
  - 用户对该商品总交互（浏览、收藏、加购物车、购买）的次数

### 数据集

- 正负样本比例：1:1060

## 输出结果

- 所有用户与商品的映射，二分类是否购买
- 购买则保存在输出结果中，否则不保存
