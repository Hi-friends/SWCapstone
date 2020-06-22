import numpy as np
import os
import re
import pandas as pd
import pickle
from sklearn.externals import joblib
from keras.models import load_model
import operator

print('==================== Start Working ====================')
#load 68 landmark test file, affine_mask*.npy
#file_lst = os.listdir('./stranger/')
file_lst = os.listdir('./affine_landmark/test/')
file_list_npy = [f for f in file_lst if f.endswith(".npy")]
data = {}
for file in file_list_npy :
    #print(file)
    TEMPLATE = np.load('./affine_landmark/test/{0}'.format(file),allow_pickle=True)
    lst = []
    for x,y in TEMPLATE:
        lst.append(x)
        lst.append(y)
    filename = file.split('_')[1]
    filename = filename.split('.')[0]
    data[filename] = lst

# 68 landmark (x,y) separation
df = pd.DataFrame(data)
df = df.transpose()
df.rename(columns = lambda x: 'x'+str(int(x/2)+1) if(x%2 == 0) else 'y'+str(int(x/2)+1), inplace = True)
df = df.sort_index()


# distance and slope
dataset = df.loc[:, ['x1','y1','x17','y17','x18','y18','x19','y19','x20','y20','x21','y21','x22','y22','x23','y23',
                     'x24','y24','x25','y25','x26','y26','x27','y27','x28','y28','x37','y37','x38','y38','x39','y39',
                     'x40','y40','x41','y41','x42','y42','x43','y43','x44','y44','x45','y45','x46','y46','x47','y47',
                     'x48','y48']]
dat = df.loc[:, ['x1','y1','x37','y37','x38','y38','x39','y39','x40','y40','x41','y41','x42','y42','x43','y43','x44','y44',
                 'x45','y45','x46','y46','x47','y47','x48','y48','x17','y17']]
dat.rename(columns = {"x37":"x2", "y37":"y2", "x40":"x4", "y40":"y4", "x43":"x5", "y43":"y5",
                      "x46":"x7", "y46":"y7", "x17":"x8","y17":"y8"}, inplace=True)
dat['x3'] = (dat['x38'] + dat['x39'] + dat['x41'] + dat['x42'])/4
dat['y3'] = (dat['y38'] + dat['y39'] + dat['y41'] + dat['y42'])/4
dat['x6'] = (dat['x44'] + dat['x45'] + dat['x47'] + dat['x48'])/4
dat['y6'] = (dat['y44'] + dat['y45'] + dat['y47'] + dat['y48'])/4
dat = dat.loc[:, ['x1','y1','x2','y2','x3','y3','x4','y4','x5','y5','x6','y6','x7','y7','x8','y8']]
dat['1'] = ((dat['x2'] - dat['x1'])**2 + (dat['y2'] - dat['y1'])**2)**1/2
dat['2'] = (dat['y2'] - dat['y1'])/(dat['x2'] - dat['x1'])
dat['3'] = ((dat['x3'] - dat['x2'])**2 + (dat['y3'] - dat['y2'])**2)**1/2
dat['4'] = (dat['y3'] - dat['y2'])/(dat['x3'] - dat['x2'])
dat['5'] = ((dat['x4'] - dat['x3'])**2 + (dat['y4'] - dat['y3'])**2)**1/2
dat['6'] = (dat['y4'] - dat['y3'])/(dat['x4'] - dat['x3'])
dat['7'] = ((dat['x5'] - dat['x4'])**2 + (dat['y5'] - dat['y4'])**2)**1/2
dat['8'] = (dat['y5'] - dat['y4'])/(dat['x5'] - dat['x4'])
dat['9'] = ((dat['x6'] - dat['x5'])**2 + (dat['y6'] - dat['y5'])**2)**1/2
dat['10'] = (dat['y6'] - dat['y5'])/(dat['x6'] - dat['x5'])
dat['11'] = ((dat['x7'] - dat['x6'])**2 + (dat['y7'] - dat['y6'])**2)**1/2
dat['12'] = (dat['y7'] - dat['y6'])/(dat['x7'] - dat['x6'])
dat['13'] = ((dat['x8'] - dat['x7'])**2 + (dat['y8'] - dat['y7'])**2)**1/2
dat['14'] = (dat['y8'] - dat['y7'])/(dat['x8'] - dat['x7'])
dat['15'] = ((dat['x8'] - dat['x1'])**2 + (dat['y8'] - dat['y1'])**2)**1/2
dat['16'] = (dat['y8'] - dat['y1'])/(dat['x8'] - dat['x1'])
dat['17'] = ((dat['x6'] - dat['x3'])**2 + (dat['y6'] - dat['y3'])**2)**1/2
dat['18'] = (dat['y6'] - dat['y3'])/(dat['x6'] - dat['x3'])
data1 = dat.loc[:,'1':'18']
data1 = data1.sort_index()

# 128 embedding
emb = pd.read_csv("./reps.csv", header=None)
emb.rename(columns = lambda x: str(int(x)+1), inplace = True)

# testset people's name
dict = {'sdr':0, 'phj':1, 'yjy':2, 'jh':3, 'irene':4, 'rjy':5, 'v':6, 'skj':7, 'ysh':8, 'iu':9}
n_dict = {}
for k,v in dict.items() :
    n_dict[v] = k

df2 = pd.read_csv("./labels.csv", header=None)
df2 = list(df2[1])
#target = []
test = []
idx = []
for i in range(len(df2)):
    #target.append(df2[i].split('/')[2])
    if '-' not in (df2[i].split('/')[3]):
        test.append(i)
        idx.append(df2[i].split('/')[3].split('_')[0])
#for i in range(len(target)):
#    target[i] = dict[target[i]]
#emb['target'] = target
emb = emb.iloc[test]
emb['idx'] = idx
emb = emb.set_index("idx")
emb = emb.sort_index()

test1 = np.array(data1.loc[:, '1':'18'])
test2 = np.array(df.loc[:, 'x1':'y68'])
test34 = np.array(dataset.loc[:, 'x1':'y48'])
test56 = np.array(emb.loc[:, '1':'128'])
test_y = np.array(emb.index.tolist())
print(test1.shape, test2.shape, test34.shape, test56.shape, test_y.shape)

fin_pred = []
softmax = []  
for i in range(test1.shape[0]) :
    softmax.append(0)
    fin_pred.append('Stranger')

model1 = load_model('./model/best_1model.h5')
print('model 1 loaded.')
model2 = load_model('./model/best_2model.h5')
print('model 2 loaded.')
model3 = load_model('./model/best_3model.h5')
print('model 3 loaded.')
model4 = load_model('./model/best_4model.h5')
print('model 4 loaded.')
model5 = load_model('./model/best_5model.h5')
print('model 5 loaded.')
model6 = load_model('./model/best_6model.h5')
print('model 6 loaded.')
clf = joblib.load('./model/svm.pkl')
print('model 7 loaded.')


for i in range(len(data)) :
    predict_per = []
    pred_ans = []
    real = []
    
    t1 = test1[i].reshape(1,18)
    t2 = test2[i].reshape(1,136)
    t34 = test34[i].reshape(1,50)
    t56 = test56[i].reshape(1,128)
    
    #predict_per(softmax), predict_ans(answer)
    predict1 = model1.predict(t1)
    predict_per.append(predict1[0])
    pred_ans.append(np.argmax(predict_per[0]))
    
    predict2 = model2.predict(t2)
    predict_per.append(predict2[0])
    pred_ans.append(np.argmax(predict_per[1]))
    
    predict3 = model3.predict(t34)
    predict_per.append(predict3[0])
    pred_ans.append(np.argmax(predict_per[2]))
    
    predict4 = model4.predict(t34)
    predict_per.append(predict4[0])
    pred_ans.append(np.argmax(predict_per[3]))
    
    predict5 = model5.predict(t56)
    predict_per.append(predict5[0])
    pred_ans.append(np.argmax(predict_per[4]))
    
    predict6 = model6.predict(t56)
    predict_per.append(predict6[0])
    pred_ans.append(np.argmax(predict_per[5]))
    
    predict7 = clf.predict(test34[i].reshape(1,-1))[0]
    pred_ans.append(predict7)

    
    pred_ans = np.array(pred_ans) #해당 인물이 몇번 인물인지
    count = np.bincount(pred_ans)
    val = count.max() # 7개 중 가장 많이 뽑힌 모델이 몇 번 뽑혔는지
    #print(test_y[i], val)
    
    if val > 3 :
        fin_pred[i] = n_dict[np.argmax(count)] # 최종 예측값
    else :
        ans = np.zeros(10)
        for j in range(len(predict_per)) :
            arr = predict_per[j]
            ans = ans + arr
        softmax[i] = ans/6 # softmax 평균, shape:(1,6)
        
        if np.max(softmax[i]) > 0.5 :
            fin_pred[i] = n_dict[val]

            #print(compare_softmax)
        #compare_softmax = np.array(compare_softmax)
        #fin_pred[i] = n_dict[num_count[np.argmax(compare_softmax)]]

print('==================== Results ====================')
for i in range(len(fin_pred)) :
    ans = int(re.findall("\d+",test_y[i])[0])-1
    print('actual :', test_y[i],'->', fin_pred[i])
    #print('actual :',n_dict[ans%10],'->', fin_pred[i])
