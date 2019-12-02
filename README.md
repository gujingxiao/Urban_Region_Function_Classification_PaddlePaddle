# Urban_Region_Function_Classification_飞桨基线挑战赛
百度AiStudio飞桨基线挑战赛，第九名解决方案，使用PaddlePaddle，ID为Seigato

## 综述
  本次比赛沿用了国际大数据竞赛用题，给定区域遥感图像及访问信息来进行区域功能分类
  
## 引用
  由于之前的比赛很多大佬已经开源方案，所以这次我并没有任何创新点，只不过按照比赛要求，使用大佬们的特征提取方法，加上纯PaddlePaddle来参赛。引用方案如下：  
  https://github.com/destiny19960207/URFC-top4  
  https://aistudio.baidu.com/aistudio/projectDetail/176495?_=1575249248130  

## 方案说明
本次比赛共提取了7组特征，然后进行融合：		
1. 使用Images + Visit，采用SE-ResNeXt50和Resnet20_vd模型进行训练	
2. 使用Images + Visit，采用Resnet50_vd和DPN26模型进行训练	
3. 使用Images + Visit，采用Densenet121和ResNeXt20模型进行训练	
4. 仅使用Visit，采用全连接FC1024模型进行训练	
5. 使用Visit数据手动提取特征，根据每天流量进行提取	
6. 使用Visit数据手动提取特征，基于严格高可信用户的规则	
7. 使用Visit数据手动提取特征，基于广义高可信用户的规则	

## 单模型脚本运行		
1. 数据处理采用与官方baseline一致的方法，再data_process中先运行makelist.py生成不同的list，然后进行convert.py即可，此过程较慢，预计需要3-4个小时
2. 使用cnn_combine下seresnext50_resnet20文件夹中的train_seresnext50_resnet20.py进行训练，infer_seresnext50_resnet20.py进行输出，将其中的数据集路径配置好即可	
3. 使用cnn_combine下resnet50_dpn26文件夹中的train_resnet50_dpn26.py进行训练，infer_resnet50_dpn26.py进行输出，将其中的数据集路径配置好即可	
4. 使用cnn_combine下densenet121_resnext20文件夹中的train_densenet121_resnext20.py进行训练，infer_densenet121_resnext20.py进行输出，将其中的数据集路径配置好即可	
5. 使用cnn_txt下的train_cnn_txt.py进行训练，infer_cnn_txt.py进行输出，将其中的数据集路径配置好即可	
6. 使用rule_feature下cxq_limit/cxq_limit.py提取特征，并生成probs文件用于融合	
7. 使用rule_feature下cxq_wide/cxq_wide.py提取特征，并生成probs文件用于融合	
8. 使用rule_feature下day_rule/day.py提取特征，并生成probs文件用于融合	

## 特征融合脚本运行		
1. 使用feature_stacking下的feature_stacking_probs.py脚本进行所有特征probs的融合
2. 使用feature_stacking/vector_stacking下的train_feature_stacking.py脚本进行4-fold训练，其中需要手动修改每个fold的配置，infer_feature_stacking.py进行probs输出，将其中的数据集路径配置好即可	
3. 使用feature_stacking/matrix_stacking下的train_matrix_stacking.py脚本进行4-fold训练，其中需要手动修改每个fold的配置，infer_matrix_stacking.py进行probs输出，将其中的数据集路径配置好即可	
4. 最后使用feature_stacking下的vector_matrix_ensemble.py进行加权平均，输出最终结果		

## 成绩记录
### 1. CNN Images + Visit
|Index|Image Model|Visit Model|Local|LeaderBoard|
|:---|:---|:---|:---|:---|
|1|SE-ResNeXt50|Resnet20_vd|0.66302|0.65452|
|2|Resnet50_vd|DPN26|0.62633|0.62104|
|3|Densenet121|ResNeXt20|0.68885|0.68259|
|4|---|FC1024|0.60223|0.59467|

### 2. Visit Rule Feature Engineering
|Index|Rule|Memory|Time|Local|LeaderBoard|
|:---|:---|:---|:---|:---|:---|
|1|day(del 80%)|29.0GB|4 hours|0.70971|0.7041|
|2|cxq_limit|31.0GB|10 hours|0.75484|0.7562|
|3|cxq_wide|35.0GB|14 hours|0.75409|0.7531|

### 3. PADDLE CNN STACKING - 4-Fold Cross Validation
|Index|Features|Function|Loss|Local|
|:---|:---|:---|:---|:---|
|1|cxq_limit cxq_wide sernxt50-res20 txt day resn50-dpn26 dns121-rnxt20|FC Vector|0.8458|0.46891|
|2|cxq_limit cxq_wide sernxt50-res20 txt day resn50-dpn26 dns121-rnxt20|FC Vector|0.848|0.46801|
|3|cxq_limit cxq_wide sernxt50-res20 txt day resn50-dpn26 dns121-rnxt20|FC Vector|0.848|0.46036|
|4|cxq_limit cxq_wide sernxt50-res20 txt day resn50-dpn26 dns121-rnxt20|FC Vector|0.851|0.45664|
|1|cxq_limit cxq_wide sernxt50-res20 txt day resn50-dpn26 dns121-rnxt20|Matrix|0.8492|0.44923|
|2|cxq_limit cxq_wide sernxt50-res20 txt day resn50-dpn26 dns121-rnxt20|Matrix|0.848|0.45876|
|3|cxq_limit cxq_wide sernxt50-res20 txt day resn50-dpn26 dns121-rnxt20|Matrix|0.8501|0.4512|
|4|cxq_limit cxq_wide sernxt50-res20 txt day resn50-dpn26 dns121-rnxt20|Matrix|0.8523|0.45444|

### 4. Ensemble
|Index|Features|Function|LeaderBoard|
|:---|:---|:---|:---|
|1|cxq_limit cxq_wide sernxt50-res20 txt day resn50-dpn26 dns121-rnxt20|FC Vector + Matrix|0.84717|
