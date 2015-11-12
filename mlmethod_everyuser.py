import os
import pandas as pd
import numpy as np
import time
from numpy import inf

# load data
#os.chdir('D:/pycode/coupon/data')
os.chdir('/home/ybluo/coupon/data')
start = time.time()
cpatr = pd.read_csv('coupon_area_train.csv')
cpdtr = pd.read_csv('coupon_detail_train.csv')
cpltr = pd.read_csv('coupon_list_train.csv')
cpvtr = pd.read_csv('coupon_visit_train.csv')
ulist = pd.read_csv('user_list.csv')                  # 22873
cplte = pd.read_csv('coupon_list_test.csv')
print time.time() - start

# Create train matrix. Combine cpltr(coupon_ID), (View_coupon_id_hash)cpvtr(user_id_hash),
# and ulist(User_id_hash), cpdtr

# 1. process cpltr
# item profiling
# data process and feature engineering
orin_name = [u'PRICE_RATE', u'CATALOG_PRICE', u'DISCOUNT_PRICE', u'DISPPERIOD', u'VALIDPERIOD', u'USABLE_DATE_MON',
             u'USABLE_DATE_TUE', u'USABLE_DATE_WED', u'USABLE_DATE_THU', u'USABLE_DATE_FRI', u'USABLE_DATE_SAT',
             u'USABLE_DATE_SUN', u'USABLE_DATE_HOLIDAY', u'USABLE_DATE_BEFORE_HOLIDAY', u'COUPON_ID_hash']
date_name = [u'DISPFROM', u'DISPEND', u'VALIDFROM', u'VALIDEND']
dummy_name = [u'CAPSULE_TEXT', u'GENRE_NAME', u'large_area_name', u'ken_name', u'small_area_name']


def trans2vect(data):
    item_vec = data.reindex(columns=orin_name)
    item_vec = item_vec.fillna(0)
    # dummy
    capsule = pd.get_dummies(data.CAPSULE_TEXT, prefix='cap_')
    genre = pd.get_dummies(data.GENRE_NAME, prefix='gen_')
    large_area = pd.get_dummies(data.large_area_name, prefix='larg_area_')
    ken_name = pd.get_dummies(data.ken_name, prefix='ken_')
    small_name = pd.get_dummies(data.small_area_name, prefix='small_area_')
    # join
    item_vec = item_vec.join([capsule, genre, large_area, ken_name, small_name])
    item_vec.index = data.COUPON_ID_hash
    # feature engineering
    item_vec.PRICE_RATE = (item_vec.PRICE_RATE ** 2) / (100 * 100)
    return item_vec

# 1. cpltr
cpvec_tr = trans2vect(cpltr)
cpvec_te = trans2vect(cplte)
name_tr = cpvec_tr.columns
name_te = cpvec_te.columns
for i in range(len(name_tr)):
    if name_tr[i] not in name_te:
        cpvec_te[name_tr[i]] = 0

# 2. cpvtr  (2833180, 3)
cpvtr_trans = cpvtr.reindex(columns=['PURCHASE_FLG', 'VIEW_COUPON_ID_hash', 'USER_ID_hash'])
cpvtr_trans.columns = ['PURCHASE_FLG', 'COUPON_ID_hash', 'USER_ID_hash']

# 3. ulist
ulist.PREF_NAME[ulist.PREF_NAME.isnull()] = 'NULL'
pref_name = pd.get_dummies(ulist.PREF_NAME, prefix='user_')
gender = pd.get_dummies(ulist.SEX_ID, prefix='sex_')
ulist_trans = ulist.reindex(columns=['USER_ID_hash', 'AGE'])
ulist_trans = ulist_trans.join([pref_name, gender])

# combine
# train
start = time.time()
temp = pd.merge(cpvtr_trans, cpvec_tr, on='COUPON_ID_hash',how='inner')
train_df = pd.merge(temp, ulist_trans, on='USER_ID_hash', how='inner')
print 'eplise time:%d' % (time.time() - start)

#test: ulist_trans, cpvec_te
ulist_trans['cross'] = 1
cpvec_te['cross'] = 1
test_df = pd.merge(ulist_trans, cpvec_te, on='cross')

# Seperate X and Y
start = time.time()
Y = train_df.PURCHASE_FLG
del train_df['PURCHASE_FLG']
del train_df['USER_ID_hash']
del train_df['COUPON_ID_hash']
X = train_df.values


user_id = test_df['USER_ID_hash']
coupon_id = test_df['COUPON_ID_hash']
del test_df['USER_ID_hash']
del test_df['COUPON_ID_hash']
del test_df['cross']
# from numpy import float32
# new_data = data.astype(float32)
test_df = test_df.reindex(columns=train_df.columns)
print 'eplise time:%d' % (time.time() - start)

# Random Forest Tree
start = time.time()
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_jobs=20, n_estimators=300)
clf.fit(X, Y)
print 'eplise time:%d' % (time.time() - start)

newX = test_df.values
start = time.time()
newY = clf.predict_proba(newX)
print 'eplise time:%d' % (time.time() - start)


pos_idx = np.where(clf.classes_ == True)[0][0]
test_df['predict'] = newY[:, pos_idx]
test_df['USER_ID_hash'] = user_id
test_df['COUPON_ID_hash'] = coupon_id


def top_merge(df, n=10, column="predict", merge_column="COUPON_ID_hash"):
    return " ".join(df.sort_index(by=column)[-n:][merge_column])

top10_coupon = test_df.groupby("USER_ID_hash").apply(top_merge)
top10_coupon.name = "PURCHASED_COUPONS"
top10_coupon.to_csv("submission.csv", header=True)