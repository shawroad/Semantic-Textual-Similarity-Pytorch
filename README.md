# 中文文本语义匹配模型集锦
## 数据说明
|  | 训练集(数量) | 验证集(数量) | 测试集(数量) | 
| :-: | :-: | :-: | :-: | 
| ATEC | 62477 | 20000 | 20000 | 
| BQ |  100000 | 10000 | 10000 |   
| LCQMC | 238766 | 8802 | 12500 | 
| PAWSX |  49401 | 2000 | 2000 | 
| STS-B |  5231 | 1458 | 1361 |

## 评价指标的说明
- **皮尔逊系数(pearsonr)**: 是衡量两个连续型变量的线性相关关系。 
- **斯皮尔曼相关系数(spearmanr)**: 是衡量两变量之间的单调关系，两个变量同时变化，但是并非同样速率变化，即并非一定是线性关系。

## 实验结果: 
没有专门去调参。 无监督的模型从训练集中随机采样了10000条数据。下面是在测试集上的结果。对最终结果影响比较大的就是学习率。尽可能的小就行。

### 斯皮尔曼系数(spearmanr)对比:
|  | ATEC | BQ | LCQMC | PAWSX | STS-B |  Avg |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | 
| SimCSE (_unsup_) | 30.8634 | **49.1813** | **68.9802** | 9.5895 | 71.3976 | 46.0024 |  
| PromptBERT (_unsup_) | **34.9434** | 48.7067 | 67.7634 | **14.3244** | 71.4191 | **47.4314** | 
| GS-infoNCE (_unsup_) | 28.9731 | 46.3247 | 67.3204 | 11.2317 | 73.2998 | 45.4299 |   
| ESimCSE (_unsup_) |  31.8443 | 48.0718 | 66.8673 | 9.1819 | 65.1843 | 44.2299 |
| ConSERT (_unsup_) |  29.7437 | 46.7806 | 67.5121 | 8.1442 | **74.1097** | 45.2580 |  
| SentenceBert (_sup_)| 48.5157 | 67.8545 | **79.6023** | 60.1675 | ** | ** | 
| CoSENT (_sup_) | **50.5969** | **72.5191** | 79.3777 | **60.5475** | 80.4344 | 68.69512 |  
| SimCSE (_sup_) |  ** | ** | ** | ** | ** | ** |  


### 皮尔逊相关系数(pearsonr)对比:
|  | ATEC | BQ | LCQMC | PAWSX | STS-B |  Avg |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | 
| SimCSE (_unsup_) |  33.1678 | **49.0413** | 57.5075 | 9.9956 | 72.8918 | 44.5207 | 
| PromptBERT (_unsup_) | **35.6218** | 48.6450 | 59.8181 | **13.5495** | 71.7247 | **45.8718** | 
| GS-infoNCE (_unsup_)| 30.3781 | 46.2700 | 57.2458 | 10.3298 | 74.4048 | 43.7257 |   
| ESimCSE (_unsup_) | 32.6815 | 47.9271 | 52.8407 | 10.5426 | 65.2000 | 41.8383 |   
| ConSERT (_unsup_) | 31.1873 | 46.6954 | **60.7141** | 8.2408 | **75.3964** | 44.4468 |  
| SentenceBert (_sup_) | 45.4922 | 66.3670 | 75.2732 | **57.7105** | ** | ** |  
| CoSENT (_sup_)| **50.4301** | **72.5830** | **77.6607** | 57.6305 | 78.5165 | 67.36416 |  
| SimCSE (_sup_) |  ** | ** | ** | ** | ** | ** |   

 

