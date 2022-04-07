#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys 
import os 
import string 
import collections    
import json 


# In[2]:



#root_direct: train_data 
def getpath(root_path):
    global neg_path, pos_path, dec_path, tru_path
    for root, dirs, _ in os.walk(root_path,topdown = False):
        tmp = root.split('\\')
        if len(tmp) >=4 and tmp[-1]!='.ipynb_checkpoints':
            if tmp[-3].split("_")[0]=='negative':
                neg_path.append(root)
                if tmp[-2].split("_")[0] == "deceptive":
                    dec_path.append(root)

                else :
                    tru_path.append(root)

            else:
                pos_path.append(root)
                if tmp[-2].split("_")[0]=="deceptive":
                    dec_path.append(root)

                else:
                    tru_path.append(root)             
    return



def build_dict(np, target_path,neg,dec):
    global word_count_np, word_count_dt, bad_dic
    if np == True:
        target_dict = word_count_np
        flag = neg
    else:
        target_dict = word_count_dt
        flag = dec
    for path in target_path:
        for root, dirs, files in os.walk(path,topdown= False):
            for file in files:
                commt_path = root+"/"+file
                with open(commt_path,'r') as f:
                    for line in f:
                        line = line.strip("\n")
                        line = ''.join([i for i in line if i not in string.punctuation])
                        line = line.split(" ")
                        for word in line:
                            std_word = word.lower()
                            if std_word not in bad_dic and std_word and not std_word[0].isdigit() and std_word!="":
                                if flag == True:
                                    if std_word not in target_dict:
                                        target_dict[std_word] = [0,0]
                                        target_dict[std_word][1] += 1 
                                    else:
                                        target_dict[std_word][1] += 1 
                                else:
                                    if std_word not in target_dict:
                                        target_dict[std_word] = [0,0]
                                        target_dict[std_word][0] += 1 
                                    else:
                                        target_dict[std_word][0] += 1 
                



# In[3]:



def NB_classifier_np():
    global word_count_np,np_count,neg_count,pos_count,model_dic
    length_np = len(word_count_np)
    for key, val in word_count_np.items():
        np_count+= val[0]
        np_count+= val[1]
        neg_count += val[1]
        pos_count += val[0]
    model_dic['np'] = {'negative': neg_count/np_count, 'positive':pos_count/np_count}
    
    for key, val in word_count_np.items():
        if 'pos' not in model_dic['np']:
            model_dic['np']['pos'] = dict()
        model_dic['np']['pos'][key] = (val[0]+1)/(pos_count+length_np)
        if 'neg' not in model_dic['np']:
            model_dic['np']['neg'] = dict()
        model_dic['np']['neg'][key] = (val[1]+1)/(neg_count+length_np)
def NB_classifier_dt():
    global word_count_dt,dt_count,dec_count,tru_count,model_dic
    length_dt = len(word_count_dt)
    for key, val in word_count_dt.items():
        dt_count+= val[0]
        dt_count+= val[1]
        dec_count += val[1]
        tru_count += val[0]
    model_dic['dt'] = {'deceptive': dec_count/dt_count, 'truth':tru_count/dt_count}
    
    for key, val in word_count_dt.items():
        if 'tru' not in model_dic['dt']:
            model_dic['dt']['tru'] = dict()
        model_dic['dt']['tru'][key] = (val[0]+1)/(tru_count+length_dt)
        if 'dec' not in model_dic['dt']:
            model_dic['dt']['dec'] = dict()
        model_dic['dt']['dec'][key] = (val[1]+1)/(dec_count+length_dt)

        
    
    


# In[4]:


root_path = sys.argv[1]
word_count_np = collections.defaultdict(list)
word_count_dt = collections.defaultdict(list)
neg_path = []
pos_path = []
dec_path = []
tru_path = []
getpath(root_path)
print(neg_path)
print(pos_path)
print(dec_path)
print(tru_path)
bad_dic = set()
with open('stopword.txt','r') as f:
    for line in f:
        bad_dic.add(line.strip("\n"))
target_path = dec_path       
build_dict(False, target_path, None,True)
target_path = tru_path
build_dict(False, target_path, None,False)
target_path = pos_path
build_dict(True, target_path, False,None)
target_path = neg_path
build_dict(True, target_path, True, None)
np_count = 0 
dt_count = 0 
neg_count = 0 
pos_count = 0
dec_count = 0
tru_count = 0
model_dic = dict()
NB_classifier_dt()
NB_classifier_np()
f = open('nbmodel.txt', 'w')
jsondata = json.dumps(model_dic,indent=4,separators=(',', ': '))
f.write(jsondata)
f.close()


# In[ ]:




