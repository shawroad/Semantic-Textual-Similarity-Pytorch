# 中文文本语义匹配模型集锦
## 数据说明

### ATEC
dfafd
### BQ
dafdaga
### LCQMC
fdafa
### PAWSX
dfadf

### STS-B
dfadf


## 实验结果: 

### 斯皮尔曼系数(spearmanr)对比:

|  | ATEC | BQ | LCQMC | PAWSX | STS-B |  Avg |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | 
| SentenceBert | ** | ** | ** | ** | ** | ** | 
| SimCSE |  ** | ** | ** | ** | ** | ** |   
| CoSENT | 50.6160 | 72.8400 | ** | ** | ** | ** |   
| GS-infoNCE |  ** | ** | ** | ** | ** | ** |   
| ESimCSE |  ** | ** | ** | ** | ** | ** |   


### 皮尔逊相关系数(pearsonr)对比:

|  | ATEC | BQ | LCQMC | PAWSX | STS-B |  Avg |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | 
| SentenceBert |  ** | ** | ** | ** | ** | ** |   
| SimCSE |  ** | ** | ** | ** | ** | ** |   
| CoSENT | 49.8967 | 73.1022 | ** | ** | ** | ** |  
| GS-infoNCE | ** | ** | ** | ** | ** | ** |   
| ESimCSE | ** | ** | ** | ** | ** | ** |   

