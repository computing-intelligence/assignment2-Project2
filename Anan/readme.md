## 豆瓣影评情感分析

## data source
+ https://github.com/aakaking/get_douban_comment

## data processing
+ clean data
+ Select the same amount of data from each category
+ comment to id
+ pad_comment

## model

+ Embedding
+ CNN
+ Bi-LSTM
+ Dropout
+ Fully connected

## Result

+ 豆瓣1星2星评论作为好评，5星评论作为差评，验证集和测试集准确度为80%（约3w条评论）
+ 豆瓣1星2星评论作为好评，5星评论作为差评，3星评论作为中评，验证集和测试集准确度为60%（约4w5条评论）
+ 按星数做5分类，验证集和测试集准确度为40%（约3w条数据）

## Summary

+ benchmark——从每个类别中选取相同数量的数据。
+ pretrained embedding——在该项目中并未提高准确率，使用wikipedia corpus训练的的word2vec时，有很多词没有出现，准确率下降，使用影评corpus训练的word2vec，可能因为句子不够丰富，并没有体现出word2vec的优势，准确度没有提高，所以最后选择了embedding layer。
+ 关于模型单层双向LSTM结果优于多层单向LSTM。
+ 观察训练集和测试集loss发现过拟合，可以使用早停法，减小epoch，或者增加数据量。
+ 关于结果精度，数据方面，清洗过后有些评论只剩一个词，这部分并未去除，另外打分很主观，相邻分数差别模糊；模型方面还可以调整参数，增加Attention机制。
