__author__ = 'admin'
import os
import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier

# load data
#os.chdir('D:/pycode/coupon/data')
os.chdir('/home/ybluo/data')
start = time.time()
cpatr = pd.read_csv('coupon_area_train.csv')
cpdtr = pd.read_csv('coupon_detail_train.csv')
cpltr = pd.read_csv('coupon_list_train.csv')
cpvtr = pd.read_csv('coupon_visit_train.csv')
ulist = pd.read_csv('user_list.csv')                  # 22873
cplte = pd.read_csv('coupon_list_test.csv')
print time.time() - start

# data processing:
# 1. cpltr
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

cpvec_tr = trans2vect(cpltr)
cpvec_te = trans2vect(cplte)
name_tr = cpvec_tr.columns
name_te = cpvec_te.columns
for i in range(len(name_tr)):
    if name_tr[i] not in name_te:
        cpvec_te[name_tr[i]] = 0

cpvec_te = cpvec_te.reindex(columns=cpvec_tr.columns)

# 2. cpvtr  (2833180, 3)
cpvtr_trans = cpvtr.reindex(columns=['PURCHASE_FLG', 'VIEW_COUPON_ID_hash', 'USER_ID_hash'])
cpvtr_trans.columns = ['PURCHASE_FLG', 'COUPON_ID_hash', 'USER_ID_hash']

# combine
# train
start = time.time()
train_df = pd.merge(cpvtr_trans, cpvec_tr, on='COUPON_ID_hash',how='inner')
print 'eplise time:%d' % (time.time() - start)

# test
test_df = cpvec_te.copy()
coupon_id = test_df.COUPON_ID_hash
del test_df['COUPON_ID_hash']


def modeling(df):
    Y = df.PURCHASE_FLG
    if np.sum(Y) == 0:
        return " "
    del df['PURCHASE_FLG']
    del df['USER_ID_hash']
    del df['COUPON_ID_hash']
    X = df.values
    #
    clf = RandomForestClassifier(n_jobs=30, n_estimators=900)
    clf.fit(X, Y)
    temp = test_df.copy()
    newY = clf.predict_proba(temp)
    pos_idx = np.where(clf.classes_ == True)[0][0]
    temp['predict'] = newY[:, pos_idx]
    temp['COUPON_ID_hash'] = coupon_id
    return " ".join(temp.sort_index(by='predict')[-10:]['COUPON_ID_hash'])


d = dict()
k = 1
for name in train_df.USER_ID_hash:
    if d.has_key(name) == False:
        d[name] = modeling(train_df[train_df.USER_ID_hash == name])
    k = k + 1
    if k%1000==0: print k

pd.Series(d).to_csv('aaaaa.csv')




