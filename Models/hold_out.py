import time as t
import numpy as np
from lightgbm import LGBMClassifier

class hold_out :

    def __init__(self, data, model, name):

        self.data=data
        self.model=model
        self.name=name

        self.train_size=int(len(self.data)*0.7)

        np.random.shuffle(self.data)
        return

    def split_data_train_test(self) :

        self.train=self.data[0:self.train_size]
        self.test=self.data[self.train_size:len(self.data)]

        self.x_train=self.train[:, 0:22500] ; self.y_train=self.train[:, 22500]
        self.x_test=self.test[:, 0:22500] ; self.y_test=self.test[:, 22500]
        self.check=self.test[:, 22501]

        return (self.x_train, self.y_train, self.x_test, self.y_test, self.check)

    def train_model(self):

        self.data_set=self.split_data_train_test()
        
        self.model=self.model.fit(self.data_set[0], self.data_set[1])
        self.pred=self.model.predict(self.data_set[2])

        self.cal_OriNG_accuracy(self.pred, self.data_set[3], self.data_set[4])

        return

    def cal_OriNG_accuracy(self, pred, target, check):
        
        self.total_Aug_NG_cnt=0 ; self.total_Ori_NG_cnt=0 ; self.total_OK_cnt=0
        self.Aug_NG_correct_cnt=0 ; self.Ori_NG_correct_cnt=0 ; self.OK_correct_cnt=0

        mark=t.asctime(t.localtime(t.time()))
        f=open(mark.replace(':', '_')+' '+'Result.txt', 'w')
        
        for prd, ans, che in zip(pred, target, check):
            
            res=str(prd)+'  ||  '+str(ans)
            judge=' => Correct' if prd==ans else ' => Wrong'
            f.write(res+judge+'\n')
            
            if ans==0 and che==0 :
                self.total_Aug_NG_cnt+=1
                if ans==prd :
                    self.Aug_NG_correct_cnt+=1
            elif ans==0 and che==1 :
                self.total_Ori_NG_cnt+=1
                if ans==prd :
                    self.Ori_NG_correct_cnt+=1
            elif ans==1 and che==1 :
                self.total_OK_cnt+=1
                if ans==prd :
                    self.OK_correct_cnt+=1
        
        f.close()
        
        total_cnt=[self.total_Aug_NG_cnt, self.total_Ori_NG_cnt, self.total_OK_cnt]
        correct=[self.Aug_NG_correct_cnt, self.Ori_NG_correct_cnt, self.OK_correct_cnt]
                
        print('Total num of Ori_NG :',self.total_Ori_NG_cnt)
        print('Total num of Aug_NG :',self.total_Aug_NG_cnt)
        print('Total num of Ori_OK :',self.total_OK_cnt)
        print('\n')
        
        try :
            print('Ori_NG Accuracy Rate : %.2f%%' % (self.Ori_NG_correct_cnt/self.total_Ori_NG_cnt*100))
            print('Scikit_NG Accuracy Rate : %.2f%%' % (self.Aug_NG_correct_cnt/self.total_Aug_NG_cnt*100))
            print('OK Accuracy Rate : %.2f%%' % (self.OK_correct_cnt/self.total_OK_cnt*100))
            print('Whole Accuracy Rate : %.2f%%\n\n' % (sum(correct)/sum(total_cnt)*100))
            
        except ZeroDivisionError as e :
            print(e)
        
        return