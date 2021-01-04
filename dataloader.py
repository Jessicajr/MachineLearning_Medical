import csv
import random
from random import randrange,randint
def is_Discrete(label_name):
    if label_name=='sex':
        return 2
    if label_name=='smoker':
        return 2
    if label_name=='region':
        return 4
    return 0

class Atom:
    def __init__(self):
        self.sex_dic = {'male': 0, 'female': 1}
        self.smoker_dic = {'no': 0, 'yes': 1}
        self.reigon_dic = {'northwest': 0,'northeast': 1,'southwest': 2,'southeast': 3}
    def get_item(self,csv_line):
        return {'age': int(csv_line[0]), 'sex': self.sex_dic.get(csv_line[1]), 'bmi':float(csv_line[2]), 'children': int(csv_line[3]), 'smoker': self.smoker_dic.get(csv_line[4]),'region':self.reigon_dic.get(csv_line[5]),'charges':float(csv_line[6])}


class Dataloader:
    def __init__(self, path,bagging=False,fold=5):
        fr_data = open(path, encoding='utf-8')
        self.reader = csv.reader(fr_data)
        self.atom = Atom()
        self.data = []
        self.bagging = bagging
        self.n_fold = fold
        self.init_module()
    def init_module(self):
        all_data = []
        groups=[]
        for i,item in enumerate(self.reader):
            if i==0:
                continue
            all_data.append(self.atom.get_item(item))
        if self.bagging:
            fold_size = len(all_data) / self.n_fold
            for i in range(self.n_fold):
                fold = list()
                while len(fold) < fold_size:
                    index = randrange(len(all_data))
                    fold.append(all_data[index])
                groups.append(fold)
            self.data=groups
        else:
            self.data = all_data

def splitData(pathw1,pathw2,pathr):
    f_r = open(pathr,'r')
    fw_train =open(pathw1,'w',newline="")
    fw_test = open(pathw2,'w',newline="")
    csv_reader = csv.reader(f_r)
    csv_writer_train = csv.writer(fw_train)
    csv_writer_test = csv.writer(fw_test)
    csv_writer_train.writerow(['age','sex','bmi','children','smoker','region','charges'])
    csv_writer_test.writerow(['age', 'sex', 'bmi', 'children', 'smoker', 'region', 'charges'])
    for i,item in enumerate(csv_reader):
        if i==0:
            continue
        seed =random.randint(1,10)
        if seed==5:
            csv_writer_test.writerow([item[i] for i in range(7)])
        else:
            csv_writer_train.writerow([item[i] for i in range(7)])
    fw_test.close()
    fw_train.close()
    f_r.close()
if __name__=="__main__":
    splitData('s_train.csv','s_test.csv','train.csv')
