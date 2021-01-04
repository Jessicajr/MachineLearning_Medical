from dataloader import *
import numpy as np
import pandas as pd
import json
import sys
def mean_Y(data_Y):
    return np.mean(data_Y)

def var(data_Y):
    return np.var(data_Y)*len(data_Y)


def group(orgGroup, label, feature=None):
    '''

    :param orgGroup: [{},{}]
    :param label: 'age','sex',...
    :param feature: decimal or (0,1),(0,2),(0,3)
    :return: [{},{}],[{},{}],avg1,avg2
    '''
    tag = is_Discrete(label)
    group_list = []
    if tag == 0:
        group1 = [x for x in orgGroup if x[label] <= feature]
        group2 = [x for x in orgGroup if x[label] > feature]
        group_list=[group1, group2]
    elif tag==2:
        assert feature is None
        group1 = [x for x in orgGroup if x[label] == 0]
        group2 = [x for x in orgGroup if x[label] == 1]
        group_list = [group1, group2]
    else:
        for i in range(4):
            group_ = [x for x in orgGroup if x[label]==i]
            group_list.append(group_)
    # if label=='region':
    #     print(tag)
    #     s = set()
    #     for item in orgGroup:
    #         s.add(item['region'])
    #     print(s)
    #     print(len(group_list))
    #     sys.exit()
    var_list = []
    for group_ in group_list:
        var_ = var([x['charges'] for x in group_])
        var_list.append(var_)

    return group_list, var_list

def chooseBestSplit(orgGroup, tolS=1, tolN=4):
    '''

    :param orgGroup:
    :param tolS: 切分前后误差小于tolS则不切分
    :param tolN: 切分后组中元素小于tolN则不切分
    :return: group, feature, feature_value
    '''
    y_data = [x['charges'] for x in orgGroup]
    if np.unique(y_data).size == 1:           # Y全部一样
        return orgGroup, None, mean_Y(y_data)
    TosErr = var(y_data)
    bestErr = np.inf
    bestLabel = 'age'
    bestFeaVal = 0
    bestGroup=[]
    feature_list = ['age','sex','bmi','children','smoker','region']
    k_choice = random.sample(feature_list, 4)
    for label in k_choice:
        tag = is_Discrete(label)
        if tag == 0:                         #连续型
            label_feature = set([x[label] for x in orgGroup])
            label_feature = list(label_feature)
            label_feature.sort()
            for value in label_feature[:-1]:
                grouplist, varlist = group(orgGroup, label, value)
                lens = [len(item) for item in grouplist]
                if min(lens)<tolN:
                    continue
                if bestErr > sum(varlist):
                    bestLabel=label
                    bestFeaVal = value
                    bestErr =  sum(varlist)
                    bestGroup = grouplist
        else:
            grouplist, varlist = group(orgGroup, label)
            lens = [len(item) for item in grouplist]
            if min(lens) < tolN:
                continue
            if bestErr > sum(varlist):
                bestLabel = label
                bestFeaVal = None
                bestErr = sum(varlist)
                bestGroup = grouplist
    if (TosErr - bestErr) < tolS:
        return orgGroup, None, mean_Y(y_data)
    return bestGroup, bestLabel, bestFeaVal

def createTree(group,stopN=4,stopErr=1):
    '''

    :param group:
    :return: dic
    '''
    ngroup, label, feature = chooseBestSplit(group, stopErr, stopN)
    if label is None:
        return feature
    tree = {}
    tree['label'] = label
    tree['feature'] = feature
    tree['child']=[]
    # if label=='region':
        #print(len(ngroup))
    for sub in ngroup:
        tree['child'].append(createTree(sub,stopErr,stopN))
    # tree['left'] = createTree(left,stopErr, stopN)
    # tree['right'] = createTree(right, stopErr, stopN)
    return tree


def forward(file_path, data_path,err,n,n_fold,fold):
    json_f = open(file_path,'w')
    dl = Dataloader(data_path,bagging=fold,fold=n_fold)
    forest=[]
    for item in dl.data:
        orgGroup = item
        tree_dic = createTree(orgGroup,err,n)
        forest.append(tree_dic)
    json.dump(forest, json_f, indent=4)
    json_f.close()

def predict(item, tree_dic):
    if type(tree_dic) is not dict:
        return tree_dic
    else:
        label = tree_dic['label']
        feature = tree_dic['feature']
        subTree = tree_dic['child']
        if feature is None:
            return predict(item,subTree[item[label]])
        else:
            value = item[label]
            if value<=feature:
                return predict(item, subTree[0])
            else:
                return predict(item,subTree[1])

def test(tree_dict_path, test_file_path, has_y=True):
    json_r = open(tree_dict_path,'r')
    forest = json.load(json_r)
    n_fold=len(forest)
    dl = Dataloader(test_file_path)
    data = dl.data
    true_y = []
    predict_y = []
    for item in data:
        sum=0
        for tree_dic in forest:
            charge = predict(item,tree_dic)
            # print(charge)
            sum+=charge
        charges = sum/n_fold
        if has_y:
            true_y.append(item['charges'])
        predict_y.append(charges)
    if has_y:
        y_true = np.array(true_y)
        y_pred = np.array(predict_y)
        r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
        print('r_score:{}'.format(r2))
    return data, predict_y

def write_answer(path,test_data,predict_y):
    sex_dic = { 0:'male', 1:'female'}
    smoker_dic = { 0:'no', 1:'yes'}
    reigon_dic = {0:'northwest', 1:'northeast', 2: 'southwest', 3:'southeast'}
    fw = open(path, 'w', newline="", encoding='utf-8')
    csv_write = csv.writer(fw)
    csv_write.writerow(['age','sex','bmi','children','smoker','region','charges'])
    for item,y in zip(test_data,predict_y):
        csv_write.writerow([item['age'],sex_dic[item['sex']],item['bmi'],item['children'],smoker_dic[item['smoker']],reigon_dic[item['region']],y])
    fw.close()


if __name__=="__main__":
    n_fold=10
    fold=True
    #forward('tree_0.5_6_fold_10_k_4_all.json', 'train.csv', err=0.5, n=6,n_fold=n_fold,fold=fold)
    #test('tree_0.5_6_fold_10_k_4_2.json','s_test.csv')
    # data, predict_y = test('tree_0.1_1_all.json','public_dataset/test_sample.csv',False)
    data, predict_y = test('tree_0.5_6_fold_10_k_4_1.json','test_sample.csv',False)
    write_answer('submission.csv',data,predict_y)