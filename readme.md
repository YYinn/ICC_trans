# 1.data

之前一共有三批数据：
1. train 365  包括N开头和无前缀的
   val 101    包括N开头和无前缀的 还有几个外部测试集，原来的前缀是 z和x，现在改成valz和valx
2. internaltest (5月)  145  原来没有前缀，为了和其他区分，现在加上前缀G
3. add 338   前缀X
   external 40  外部测试集名字作为前缀， 包括sx,T，xm,xu,z

现在重新进行划分，label在/media/yinn147/Data/ICC_transformer/Data中。

其中：
- train val共683个，包括X开头的338个，G开头的6个，n和无前缀的共339个
- internal test共225个，其中n和无前缀的共386个，G开头的139个
- external test共82个，其中40个就是最后新增的external数据，其余的43个为之前val中的部分数据。

## 1. preprocess

/media/yinn147/Data/ICC_transformer/preprocess/preprocess.ipynb
保存到/media/yinn147/Data/ICC_transformer/Preprocessed_data

## 2. json

/media/yinn147/Data/ICC_transformer/preprocess/json/json_create.ipynb
保存到/media/yinn147/Data/ICC_transformer/preprocess/json