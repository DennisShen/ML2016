###############################################################
#  Author : Dennis Shen
#  Date   : 2016.10.11
#  Title  : Linear Regression for PM2.5 prediction
###############################################################

# import
import csv
import numpy as np

print 'read training data...'
# read training data
data = np.empty((12, 20*24, 18))
with open('data/train.csv', 'r') as f:
  next(f)
  for idx, row in enumerate(csv.reader(f)):
    for i in range(24):
      if row[i+3] == 'NR':
        data[idx/360][((idx%360)/18)*24 + i][idx%18] = 0
      else:
        data[idx/360][((idx%360)/18)*24 + i][idx%18] = row[i+3]


print 'feature scaling...'
# feature scaling
mean = np.empty(18)
for k in range(18):
  mean[k] = np.mean(data[:,:,k])

std = np.empty(18)
for k in range(18):
  std[k] = np.std(data[:,:,k])
  
mean_s = np.empty(18)
for k in range(18):
  mean_s[k] = np.mean(data[:,:,k]**2)
  
std_s = np.empty(18)
for k in range(18):
  std_s[k] = np.std(data[:,:,k]**2)

  
print 'start training data...'
# train data
select_list    = [2, 3, 4, 5, 7, 9, 12]
w              = np.zeros((2, len(select_list), 9))
w_s            = np.zeros((2, len(select_list), 9))
b              = np.zeros(2)
rate           = np.zeros((2, len(select_list), 9)) + 1
rate[:,5,:]    = 10
rate_s         = np.zeros((2, len(select_list), 9)) + 0.1
rate_s[:,5,:]  = 1
rate_b         = np.zeros(2) + 10
iteration_time = 2000                                     
G_w            = np.zeros((2, len(select_list), 9))
G_w_s          = np.zeros((2, len(select_list), 9))     
G_b            = np.zeros(2)
smooth         = 1e-8
lamda          = 0
printError     = False

for it in range(iteration_time):
  w_d   = np.zeros((2, len(select_list), 9))
  w_d_s = np.zeros((2, len(select_list), 9))
  b_d   = np.zeros(2)
  
  err = np.empty((12, 471))
  # traverse all data 
  for i in range(12):
    for j in range(471):
    
      # check if rain
      b_hasRain = False
      if np.count_nonzero(data[i,j:j+9,10]) != 0:
        b_hasRain = True

      # prediction answer
        # order 0
      y_pred  = b[0] * (b_hasRain) + b[1] * (not b_hasRain) 
        # order 1
      y_pred += np.sum(     b_hasRain *w[0]*((data[i,j:j+9,select_list] - mean[select_list,None])/std[select_list,None]))
      y_pred += np.sum((not b_hasRain)*w[1]*((data[i,j:j+9,select_list] - mean[select_list,None])/std[select_list,None]))
        # order 2
      y_pred += np.sum(     b_hasRain *w_s[0]*((data[i,j:j+9,select_list]**2 - mean_s[select_list,None])/std_s[select_list,None]))
      y_pred += np.sum((not b_hasRain)*w_s[1]*((data[i,j:j+9,select_list]**2 - mean_s[select_list,None])/std_s[select_list,None]))      

      err[i][j] = (data[i][j+9][9] - y_pred)**2    
 
      # gradient   
        # order 0
      b_d[0] += ((    b_hasRain)*(-2)*(data[i][j+9][9] - y_pred) + 2*lamda*b[0])
      b_d[1] += ((not b_hasRain)*(-2)*(data[i][j+9][9] - y_pred) + 2*lamda*b[1])  
        # order 1
      w_d[0] += ((    b_hasRain)*(-2)*(data[i][j+9][9] - y_pred)*((data[i,j:j+9,select_list] - mean[select_list,None])/std[select_list,None]) + 2*lamda*w[0])
      w_d[1] += ((not b_hasRain)*(-2)*(data[i][j+9][9] - y_pred)*((data[i,j:j+9,select_list] - mean[select_list,None])/std[select_list,None]) + 2*lamda*w[1])
        # order 2
      w_d_s[0] += ((    b_hasRain)*(-2)*(data[i][j+9][9] - y_pred)*((data[i,j:j+9,select_list]**2 - mean_s[select_list,None])/std_s[select_list,None]) + 2*lamda*w_s[0])
      w_d_s[1] += ((not b_hasRain)*(-2)*(data[i][j+9][9] - y_pred)*((data[i,j:j+9,select_list]**2 - mean_s[select_list,None])/std_s[select_list,None]) + 2*lamda*w_s[1])
  
  if printError:
    print str(it) + ':' + str((np.average(err))**0.5)

  # update
  G_w   += w_d**2
  G_w_s += w_d_s**2
  G_b   += b_d**2
  w   = w   - (rate   /np.sqrt(G_w  +smooth))*w_d
  w_s = w_s - (rate_s /np.sqrt(G_w_s+smooth))*w_d_s
  b   = b   - (rate_b /np.sqrt(G_b  +smooth))*b_d

print w
print w_s
print b


print 'make prediction...'
# prediction
test = np.empty((240, 9, 18))
with open('data/test_X.csv', 'r') as f:
  for idx, row in enumerate(csv.reader(f)):
    for i in range(9):
      if row[i+2] == 'NR':
        test[idx/18][i][idx%18] = 0        
      else:
        test[idx/18][i][idx%18] = row[i+2]

ans = np.empty(240)
for i in range(240):
  
  # check if rain
  b_hasRain = False
  if np.count_nonzero(test[i,0:9,10]) != 0:
    b_hasRain = True

  # order 0
  ans[i]  = b[0] * (b_hasRain) + b[1] * (not b_hasRain)
  
  # order 1
  ans[i] += np.sum(     b_hasRain *  w[0]*((test[i,0:9,select_list]    - mean[select_list,None])  /std[select_list,None]))
  ans[i] += np.sum((not b_hasRain)*  w[1]*((test[i,0:9,select_list]    - mean[select_list,None])  /std[select_list,None]))
  
  # order 2
  ans[i] += np.sum(     b_hasRain *w_s[0]*((test[i,0:9,select_list]**2 - mean_s[select_list,None])/std_s[select_list,None]))
  ans[i] += np.sum((not b_hasRain)*w_s[1]*((test[i,0:9,select_list]**2 - mean_s[select_list,None])/std_s[select_list,None]))

  
print 'write file...'
with open('linear_regression.csv', 'w') as f:
  f.write('id,value\n')
  for i in range(240):
    f.write('id_' + str(i) + ',' + str(ans[i]) + '\n')
