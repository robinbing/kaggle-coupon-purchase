__author__ = 'admin'
# using logistic regression
import os
import pandas as pd
import numpy as np
import time

# load data
os.chdir('D:/pycode/coupon/data')
start = time.time()
cpatr = pd.read_csv('coupon_area_train.csv')
cpdtr = pd.read_csv('coupon_detail_train.csv')
cpltr = pd.read_csv('coupon_list_train.csv')
cpvtr = pd.read_csv('coupon_visit_train.csv')
ulist = pd.read_csv('user_list.csv')                  # 22873
cplte = pd.read_csv('coupon_list_test.csv')
print time.time() - start

# item profiling
# data process and feature engineering
orin_name = [u'PRICE_RATE', u'CATALOG_PRICE', u'DISCOUNT_PRICE', u'DISPPERIOD', u'VALIDPERIOD', u'USABLE_DATE_MON',
             u'USABLE_DATE_TUE', u'USABLE_DATE_WED', u'USABLE_DATE_THU', u'USABLE_DATE_FRI', u'USABLE_DATE_SAT',
             u'USABLE_DATE_SUN', u'USABLE_DATE_HOLIDAY', u'USABLE_DATE_BEFORE_HOLIDAY']
date_name = [u'DISPFROM', u'DISPEND', u'VALIDFROM', u'VALIDEND']
dummy_name = [u'CAPSULE_TEXT', u'GENRE_NAME', u'large_area_name', u'ken_name', u'small_area_name']

# creat coupon vector


def scale(data):
    mean = np.mean(data)
    std = np.std(data)
    sca = [(x - mean) / std for x in data]
    return sca


def trans2vect(data):
    item_vec = data.reindex(columns=orin_name)
    # dummy
    capsule = pd.get_dummies(data.CAPSULE_TEXT, prefix='cap_')
    genre = pd.get_dummies(data.GENRE_NAME, prefix='gen_')
    large_area = pd.get_dummies(data.large_area_name, prefix='larg_area_')
    ken_name = pd.get_dummies(data.ken_name, prefix='ken_')
    small_name = pd.get_dummies(data.small_area_name, prefix='small_area_')
    # time
    dispfrom = pd.to_datetime(data.DISPFROM)
    item_vec['dispfrom'] = [x.dayofyear for x in dispfrom]
    dispend = pd.to_datetime(data.DISPEND)
    item_vec['dispend'] = [x.dayofyear for x in dispend]
    validfrom = pd.to_datetime(data.VALIDFROM)
    item_vec['validfrom'] = [x.dayofyear for x in validfrom]
    validend = pd.to_datetime(data.VALIDEND)
    item_vec['validend'] = [x.dayofyear for x in validend]
    # join
    item_vec = item_vec.join([capsule, genre, large_area, ken_name, small_name])
    item_vec.index = data.COUPON_ID_hash
    item_vec = item_vec.fillna(0)
    # feature engineering
    item_vec.DISCOUNT_PRICE = 1 / np.log10(item_vec.DISCOUNT_PRICE)
    item_vec.CATALOG_PRICE = 1 / np.log10(item_vec.CATALOG_PRICE)
    item_vec.PRICE_RATE = (item_vec.PRICE_RATE ** 2) / (100 * 100)
    scale_name = [u'DISPPERIOD', u'VALIDPERIOD',u'dispfrom', u'dispend', u'validfrom', u'validend']
    for i in scale_name:
        item_vec[i] = scale(item_vec[i])
    return item_vec

# get coupon vector
cpvec_tr = trans2vect(cpltr)
cpvec_te = trans2vect(cplte)

no_name = list(set(cpvec_tr.columns) - set(cpvec_te.columns))
for i in no_name:
    cpvec_te[i] = 0
cpvec_te = cpvec_te.reindex(columns=cpvec_tr.columns)

tr_vec = cpvec_tr.values   #19413*163
te_vec = cpvec_te.values   #310*163

# cosine distance
temp_tr = cpvec_tr.apply(lambda x:
                         np.sqrt(sum(x * x)), axis=1).values  # 19413
temp_te = cpvec_te.apply(lambda x:
                         np.sqrt(sum(x * x)), axis=1).values  # 310

start = time.time()
dist = np.zeros([len(tr_vec), len(te_vec)])
for i in range(len(tr_vec)):
    if i % 5000 == 0:
        print i
    for j in range(len(te_vec)):
        dist[i][j] = np.dot(tr_vec[i], te_vec[j]) / (temp_tr[i] * temp_te[j])
print 'eplise time:%d' % (time.time() - start)

dist = pd.DataFrame(dist, columns=cpvec_te.index, index=cpvec_tr.index)
dist = dist.T  # 310 * 19413

# user item (consider all the items which were browsed, may consider purchased only later)
start = time.time()
vis_item = cpvtr.reindex(columns=['USER_ID_hash', 'VIEW_COUPON_ID_hash'])
grouped = vis_item.groupby('USER_ID_hash')
users_item = {}
for user, item in grouped:
    users_item[user] = item
print 'eplise time:%d' % (time.time() - start)

# find how a user like items(users_item and dist)
# extract items and users from users_item
items = []
users = []
for i in users_item.iteritems():
    items.append(list(i[1]['VIEW_COUPON_ID_hash']))
    users.append(i[0])

# define function to find top k similar items


def topK(data, k):

    # k is number of top items
    d = dict()
    for i in data.index:
        d[i] = data[i]
    d = sorted(d.iteritems(), key=lambda x:x[1], reverse=True)
    topk_items = dict(d[:k])
    return topk_items

# obtain the total scores for users each new item
start = time.time()
k = 20
scores = np.zeros([len(users), len(dist)])
for i in range(len(dist)):
    if i % 50 == 0:
        print i
    item_dist = dist.ix[i]
    top = topK(item_dist, k)
    for j in range(len(users)):
        total = 0
        browse_item = items[j]
        for name in browse_item:
            if top.has_key(name):
                total += top[name]
        scores[j][i] = total
print 'eplise time:%d' % (time.time() - start)

df = pd.DataFrame(scores)
df.columns = dist.index
df.index = users

# get the top 10(or less) items users will purchase
start = time.time()
rc_user = dict()
for i in range(len(df)):
    temp = dict(df.ix[i])
    temp = sorted(temp.iteritems(), key=lambda x: x[1], reverse=True)
    temp = temp[:10]
    rc_user[users[i]] = []
    for x in temp:
        if x[1] != 0:
            rc_user[users[i]].append(x[0])
print 'eplise time:%d' % (time.time() - start)

# some users don't have browsing records
all_user = ulist.USER_ID_hash
for name in all_user:
    if rc_user.has_key(name) == 0:
        rc_user[name] = []

# obtain submission
start = time.time()
pieces = []
for name in all_user:
    temp = []
    if rc_user[name] == []:
        temp.append(name)
    else:
        for i in range(len(rc_user[name])):
            temp.append(name)
    df = pd.DataFrame([temp, rc_user[name]], index=['USER_ID_hash', 'PURCHASED_COUPONS']).T
    pieces.append(df)
print 'eplise time:%d' % (time.time() - start)

result = pd.concat(pieces)
result = result.fillna(' ')

def top_merge(df):
    return " ".join(df['PURCHASED_COUPONS'])

grouped = result.groupby('USER_ID_hash').apply(top_merge)
grouped.name = "PURCHASED_COUPONS"
grouped.to_csv("submission.csv", header=True)