#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys 
import os 
import string 
import collections    
import json 
import math 


# In[2]:



def getpath(root_path):
    global paths
    for root, dirs, _ in os.walk(root_path,topdown = False):
        tmp = root.split('\\')
        if len(tmp) >=2 and tmp[-1]!='.ipynb_checkpoints':
            paths.append(root)      
    return


# In[4]:


bad_dic = set()
with open('stopword.txt','r') as f:
    for line in f:
        bad_dic.add(line.strip("\n"))

with open("nbmodel.txt",'r', encoding='UTF-8') as f:
     model = json.load(f)
result = []
def Classifier():
    global bad_dic, paths, model, result
    for path in paths:
        for root, dirs, files in os.walk(path,topdown= False):
            for file in files:
                commt_path = root+"\\"+file
                prob_dec = math.log(model['dt']['deceptive'],10)
                prob_tru = math.log(model['dt']['truth'],10)
                prob_pos = math.log(model['np']['positive'],10) 
                prob_neg = math.log(model['np']['negative'],10)
                with open(commt_path,'r') as f:
                    for line in f:
                        line = line.strip("\n")
                        line = ''.join([i for i in line if i not in string.punctuation])
                        line = line.split(" ")
                        
                        for word in line:
                            std_word = word.lower()
                            if std_word not in bad_dic:
                                #print(prob_dec)
                                if std_word in model['dt']['tru']:
                                    prob_tru += math.log(model['dt']['tru'][std_word],10)
                               
                                if std_word in model['dt']['dec']:
                                    prob_dec += math.log(model['dt']['dec'][std_word],10)
                  
                                    
                                if std_word in model['np']['pos']:
                                    prob_pos += math.log(model['np']['pos'][std_word],10)
                    
                                if std_word in model['np']['neg']:
                                    prob_neg += math.log(model['np']['neg'][std_word],10)
                        lb_1 = ""
                        lb_2 = ""
                        if prob_tru > prob_dec:
                            lb_1 = "truthful"
                        else:
                            lb_1 = "deceptive"
                        if prob_pos > prob_neg:
                            lb_2 = "positive"
                        else:
                            lb_2 = "negative"
                        result.append(lb_1 +" "+ lb_2+" "+commt_path)
                        
                            
root_path = 'dev_data'
paths = []

getpath(root_path)
print(paths)
Classifier()

out = open('nboutput.txt','w')
for ele in result:
    out.writelines(ele)
    out.write('\n')
    
out.close()
                            
                                    
                    
                        
                            
    


# In[ ]:





# In[ ]:




