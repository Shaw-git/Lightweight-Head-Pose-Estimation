import os
import numpy as np
import argparse
from prettytable import  PrettyTable

def load_train_file(path):
    dicts={}
    parameters=[]
    results=[[],[],[],[],[]]
    with  open(path,'r') as fin:
        for k, line in enumerate(fin):
            if k==0:
                items = line.strip()
                items=items if len(items)<8 else items[0:8]
                dicts.update({'Remarks': items})
                continue
            if  not "Roll" in line:
                line = line.replace("[", " ").replace("]", "")
                items = line.strip().split(":")
                key=items[0].strip()
                value = ""
                for i in range(len(items)-1):
                    value=value + " %s "%( items[i+1].strip())
                dicts.update({key: value})
            else:
                line = line.replace(":", " ")
                items = line.strip().split()
                items=items[3: ]
                for i in range(5):
                    if len(parameters)<5 :
                        parameters.append(items[i*2])
                    results[i].append(float(items[i*2+1]))

        if len(parameters)!=0:
            idx_min = np.argmin(results[3])
            for i in range(len(parameters)):
                dicts.update({parameters[i] : results[i][idx_min]})
    return dicts

if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--dataset', default='BIWI')
    argparse.add_argument('--sort',default=1,type=int)
    arg=argparse.parse_args()

    log_path="./log"
    TestOn=arg.dataset

    Table=[#"Train Dataset",
              "Feature Extractor",
              "Dir",
              "Remarks",
              #"Alpha",
              "Beta",
              "DSFD",
              'Batch Size',
              'Rotation',
              #'Flip',
              #'Weight Decay',
              #'Milestones',
              #'CE Weight Rate',
              'Bin Revise',
              'Roll',
              'Yaw',
              'Pitch',
              'MAE',
              'SUM']

    x= PrettyTable(Table)
    for extractor in os.listdir(log_path):
        extractor_path=os.path.join(log_path,extractor)
        for train_path in os.listdir(extractor_path):
            dir = {"Dir": train_path}
            train_path=os.path.join(extractor_path, train_path)
            info=load_train_file(os.path.join(train_path,TestOn+".txt"))
            info.update(dir)
            row=[]
            for key in Table:
                if key in info.keys():
                    row.append(info[key])
                else:
                    if key in Table[-5:]:
                        row.append(-1)
                    else:
                        row.append('null')
            x.add_row(row)
    results=x.get_string(sortby=Table[int(arg.sort)], reversesort=True)
    print(results)
    file=open(TestOn+".txt",'w')
    file.write(results)
    file.close()








    # print(os.listdir(path))










