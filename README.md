# Urban_Region_Function_Classification_飞桨基线挑战赛
百度AiStudio飞桨基线挑战赛，第九名解决方案，使用PaddlePaddle，ID为Seigato

## 综述
  本次比赛沿用了国际大数据竞赛用题，给定区域遥感图像及访问信息来进行区域功能分类
  
## 引用
  由于之前的比赛很多大佬已经开源方案，所以这次我并没有任何创新点，只不过按照比赛要求，使用大佬们的特征提取方法，加上纯PaddlePaddle来参赛。引用方案如下：  
  https://github.com/destiny19960207/URFC-top4  
  https://aistudio.baidu.com/aistudio/projectDetail/176495?_=1575249248130  

## 方案说明
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
