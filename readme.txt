---流程处理---
      |----数据收集
                |------scrapy
      |----数据预处理
                |------去重(df.drop_duplicates)
                |------去空(df.dropna,df.fillna)
                |------Normalize
                |------数据增强
                |------
      |----特诊工程
      |----建模
      |----调参/调优
                |------蒸馏
                |------剪枝
      |----部署
      |----迭代
               |------增加数据量


---数学部分---
      |----线性回归                    #处理预测问题
                |------nn.Linear(input,output)--Pytorch
                |------tf.matmul(X,W)+b--Tensorflow
                |------sklearn.linear_model.LinearRegression()--SKlearn
                |------np.dot(X,W.T)+b -- numpy
      |----逻辑回归
                |------sklearn.linear_model.LogisticRegression -- SKlearn
                |------nn.Sequetial(nn.Linear,nn.Sigmoid())    -- Pytorch #可以更复杂，处理分类问题
                |------model_output = tf.add(tf.matmul(x_data, A), b) --Tensorflow
      |----SVM
      |----决策树
      |----随机森林
      |----XGboost


---激活函数---
      |------ReLu
              |------f(x)=max(0,x)
                       |------(0,1)
                                |------f(x)=max(0,x)
      |------Sigmoid
               |------f(x) = 1/(1+exp**-x)
                       |------ y(1-y)
                                |------f(x)=max(0,1)
      |------tanh
               |------f(x) = (e**x - e**-x) / (e**x + e**-x)
                       |------1 - tanh(x)**2      或者 1 - y**2
                                |------(-1,1)

---损失函数---
      |------CrossEntropyLoss
      |------MSE


---优化器(optimizer)---
      |------Adam
      |------SGD


---
   |__CNN__CV
            |__New Model
            |__Resnet,Lenet,VGG,inception,GoogleNet,DenseNet
   |__RNN__NLP
   |__GNN



---数据库相关---
____
____---<Scrapy>---
   |
   |---<Spark> ---
   |
   |---<Hadoop><Redis><MySQL>---