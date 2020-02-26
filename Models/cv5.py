import time as t
import numpy as np
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

class fold_5_cv:
    
    def __init__(self, data, model, name):
        
        self.data=data
        self.model=model
        self.name=name
        self.fold_size=len(self.data)//5
        
        self.idx=[i for i in range(len(self.data))]
        np.random.shuffle(self.data)
        
        return
    
    def split_data_train_test(self, i):        
        
        self.test_idx=self.idx[(i-1)*self.fold_size : i*self.fold_size]
        self.train_idx=np.delete(self.idx, self.test_idx).tolist()
        
        self.test_data=self.data[self.test_idx]
        self.train_data=self.data[self.train_idx]
        
        self.x_train=self.train_data[:, 0:22500] ; self.y_train=self.train_data[:, 22500]
        self.x_test=self.test_data[:, 0:22500] ; self.y_test=self.test_data[:, 22500]
        self.check=self.test_data[:, 22501]
        
        return (self.x_train, self.y_train, self.x_test, self.y_test, self.check)
            
    def train_model(self):
        
        for i in range(1,6):
            self.data_set=self.split_data_train_test(i)
            
            self.model=self.model.fit(self.data_set[0], self.data_set[1])
            self.pred=self.model.predict(self.data_set[2])
            
            self.cal_OriNG_accuracy(self.pred, self.data_set[3], self.data_set[4], i)
        
        return
    
    def cal_OriNG_accuracy(self, pred, target, check, th):
        
        self.total_GAN_NG_cnt=0 ; self.total_Ori_NG_cnt=0 ; self.total_OK_cnt=0
        self.GAN_NG_correct_cnt=0 ; self.Ori_NG_correct_cnt=0 ; self.OK_correct_cnt=0
        
        mark=t.asctime(t.localtime(t.time()))
        f=open(mark.replace(':', '_')+' '+str(th)+'-fold_result.txt', 'w')
        
        for prd, ans, che in zip(pred, target, check):
            
            res=str(prd)+'  ||  '+str(ans)
            judge=' => Correct' if prd==ans else ' => Wrong'
            f.write(res+judge+'\n')
            
            if ans==0 and che==0 :
                self.total_GAN_NG_cnt+=1
                if ans==prd :
                    self.GAN_NG_correct_cnt+=1
            elif ans==0 and che==1 :
                self.total_Ori_NG_cnt+=1
                if ans==prd :
                    self.Ori_NG_correct_cnt+=1
            elif ans==1 and che==1 :
                self.total_OK_cnt+=1
                if ans==prd :
                    self.OK_correct_cnt+=1
        
        f.close()
        
        total_cnt=[self.total_GAN_NG_cnt, self.total_Ori_NG_cnt, self.total_OK_cnt]
        correct=[self.GAN_NG_correct_cnt, self.Ori_NG_correct_cnt, self.OK_correct_cnt]
        
        print(str(th)+'th test-fold', self.name)
        
        print('Total num of Ori_NG :',self.total_Ori_NG_cnt)
        print('Total num of GAN_NG :',self.total_GAN_NG_cnt)
        print('Total num of Ori_OK :',self.total_OK_cnt)
        print('\n')
        
        try :
            print('Ori_NG Accuracy Rate : %.2f%%' % (self.Ori_NG_correct_cnt/self.total_Ori_NG_cnt*100))
            print('scikit_NG Accuracy Rate : %.2f%%' % (self.GAN_NG_correct_cnt/self.total_GAN_NG_cnt*100))
            print('OK Accuracy Rate : %.2f%%' % (self.OK_correct_cnt/self.total_OK_cnt*100))
            print('Whole Accuracy Rate : %.2f%%\n\n' % (sum(correct)/sum(total_cnt)*100))
            
        except ZeroDivisionError as e :
            print(e)
        
        return
