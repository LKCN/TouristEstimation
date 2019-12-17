# 绍兴智慧文创旅游大数据分析<img src="https://github.com/LKCN/TouristEstimation/tree/master/img/logo.png" style="zoom:50%;" />

## 浙江工业大学之江学院<img src="https://github.com/LKCN/TouristEstimation/tree/master/img/xiaohui.png" style="zoom:50%;" />

### 一、项目简介

旅游业在绍兴2018年GDP中占比22%，是绍兴的支柱产业之一，却面临着“旅游景点人满为患体验差、千篇一律无差异回头客少”等痛点。本项目利用了绍兴市公共数据开放平台中的景区、交通和天气等数据集，研发了基于神经网络的预测算法，提供了对不同人群的旅游建议、景点的差异化建议等服务，帮助绍兴从传统旅游发展成为智慧文创旅游。

<img src="https://github.com/LKCN/TouristEstimation/tree/master/img/bg.png" style="zoom:50%;"/>

### 二、绍兴传统旅游业的痛点

- 旅游景点人满为患体验差

- 千篇一律无差异回头客少

<img src="https://github.com/LKCN/TouristEstimation/tree/master/img/td.png" style="zoom:50%;"/>

### 三、本项目创新点与目标

- 本项目**创新点**
  **回归分析**和**神经网络**：绍兴市公共数据开放平台，大数据分析预测交通流量、旅客流量等
  智慧旅游服务：**不同类别游客的旅游建议**、**景点不同时段的差异化服务**

- 本项目目标
  **提升游客旅游体验舒适度**
  **促进绍兴旅游业创收增长**

### 四、数据集信息

- 数据集1：[省内越城信息](https://data.sx.zjzwfw.gov.cn/kf/open/table/detail/7003)

  数据提供方：市交通运输局

- 数据集2：[省外越城信息](https://data.sx.zjzwfw.gov.cn/kf/open/table/detail/7005)

  数据提供方：市交通运输局

- 数据集3：[天气预报-3到10天预报信息](https://data.sx.zjzwfw.gov.cn/kf/open/table/detail/7113)

  数据提供方：市气象局

- 数据集4：[天气预报-短期预报信息](https://data.sx.zjzwfw.gov.cn/kf/open/table/detail/7127)

  数据提供方：市气象局 

- 数据集5：[天气实况-日资料信息](https://data.sx.zjzwfw.gov.cn/kf/open/table/detail/7119)

  数据提供方：市气象局

- 数据集6：[鲁迅故里景区接待信息](https://data.sx.zjzwfw.gov.cn/kf/open/table/detail/6889)

  数据提供方：市文旅集团

### 五、算法简介

神经网络

PyTorch框架，LSTM（长短时记忆网络）作为Backbone；数据按9：1分原始数据集为训练集、测试集；损失函数使用MSEloss（均方误差）；优化函数使用Adam优化算法，公式如下：

<img src="https://github.com/LKCN/TouristEstimation/tree/master/img/adam.png" style="zoom:50%;" />

神经网络结构图如下：

<img src="https://github.com/LKCN/TouristEstimation/tree/master/img/LSTM.png" style="zoom:33%;" />

### 六、训练及测试结果

范例：

<img src="https://github.com/LKCN/TouristEstimation/tree/master/img/test.png" style="zoom:50%;" />

<img src="https://github.com/LKCN/TouristEstimation/tree/master/img/result.png" style="zoom:50%;" />

#### 团队成员：张智、刘子瑜、王剑锋

##### 联系方式：

<img src="https://github.com/LKCN/TouristEstimation/tree/master/img/lianxi.png" style="zoom:50%;" />

