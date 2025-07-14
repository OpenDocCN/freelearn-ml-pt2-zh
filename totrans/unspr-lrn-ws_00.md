# The

# 无监督学习

# 研讨会

开始使用无监督学习算法，并简化你的未整理数据，以帮助进行未来的预测

Aaron Jones, Christopher Kruger, 和 Benjamin Johnston

# 无监督学习研讨会

版权 © 2020 Packt Publishing

版权所有。未经出版商事先书面许可，本课程的任何部分不得以任何形式或任何手段复制、存储于检索系统中或传播，但在关键文章或评论中嵌入简短的引用除外。

本课程在准备过程中已尽力确保所提供信息的准确性。然而，本课程所包含的信息是按“原样”销售的，不提供任何形式的明示或暗示的担保。无论是作者、Packt Publishing 还是其经销商和分销商，都不对因本课程直接或间接引起的或被声称引起的任何损害负责。

Packt Publishing 力求通过恰当使用大写字母提供课程中提及的所有公司和产品的商标信息。然而，Packt Publishing 无法保证这些信息的准确性。

**作者：** Aaron Jones, Christopher Kruger, 和 Benjamin Johnston

**审阅人：** Richard Brooker, John Wesley Doyle, Priyanjit Ghosh, Sani Kamal, Ashish Pratik Patil, Geetank Raipuria, 和 Ratan Singh

**执行编辑：** Rutuja Yerunkar

**采购编辑：** Manuraj Nair, Royluis Rodrigues, Anindya Sil, 和 Karan Wadekar

**生产编辑：** Salma Patel

**编辑委员会：** Megan Carlisle, Samuel Christa, Mahesh Dhyani, Heather Gopsill, Manasa Kumar, Alex Mazonowicz, Monesh Mirpuri, Bridget Neale, Dominic Pereira, Shiny Poojary, Abhishek Rane, Brendan Rodrigues, Erol Staveley, Ankita Thakur, Nitesh Thakur, 和 Jonathan Wray

初版：2020年7月

生产参考：1280720

ISBN：978-1-80020-070-8

由 Packt Publishing Ltd. 出版

Livery Place, 35 Livery Street

英国伯明翰 B3 2PB

# 目录

## [前言    i](B15923_Preface_Final_NT_ePub.xhtml#_idTextAnchor001)

## [1. 聚类简介    1](B15923_01_Final_RK_ePub.xhtml#_idTextAnchor022)

### [简介    2](B15923_01_Final_RK_ePub.xhtml#_idTextAnchor023)

### [无监督学习与有监督学习   2](B15923_01_Final_RK_ePub.xhtml#_idTextAnchor024)

### [聚类    4](B15923_01_Final_RK_ePub.xhtml#_idTextAnchor026)

### [识别聚类   5](B15923_01_Final_RK_ePub.xhtml#_idTextAnchor027)

### [二维数据   6](B15923_01_Final_RK_ePub.xhtml#_idTextAnchor028)

### [练习 1.01：数据中的聚类识别   7](B15923_01_Final_RK_ePub.xhtml#_idTextAnchor029)

### [k-means 聚类简介   11](B15923_01_Final_RK_ePub.xhtml#_idTextAnchor030)

### [无数学 k-means 步骤详解   11](B15923_01_Final_RK_ePub.xhtml#_idTextAnchor031)

### [k-means 聚类深入讲解    13](B15923_01_Final_RK_ePub.xhtml#_idTextAnchor032)

### [替代距离度量 – 曼哈顿距离 14](B15923_01_Final_RK_ePub.xhtml#_idTextAnchor033)

### [更深的维度 15](B15923_01_Final_RK_ePub.xhtml#_idTextAnchor034)

### [练习 1.02：在 Python 中计算欧氏距离 16](B15923_01_Final_RK_ePub.xhtml#_idTextAnchor035)

### [练习 1.03：通过距离概念形成聚类 18](B15923_01_Final_RK_ePub.xhtml#_idTextAnchor036)

### [练习 1.04：从零开始的 k-means – 第 1 部分：数据生成 20](B15923_01_Final_RK_ePub.xhtml#_idTextAnchor037)

### [练习 1.05：从零开始的 k-means – 第 2 部分：实现 k-means 24](B15923_01_Final_RK_ePub.xhtml#_idTextAnchor038)

### [聚类性能 – 轮廓系数 29](B15923_01_Final_RK_ePub.xhtml#_idTextAnchor039)

### [练习 1.06：计算轮廓系数 31](B15923_01_Final_RK_ePub.xhtml#_idTextAnchor040)

### [活动 1.01：实现 k-means 聚类 33](B15923_01_Final_RK_ePub.xhtml#_idTextAnchor041)

### [总结 35](B15923_01_Final_RK_ePub.xhtml#_idTextAnchor042)

## [2. 层次聚类 37](B15923_02_Final_SMP_ePub.xhtml#_idTextAnchor043)

### [简介 38](B15923_02_Final_SMP_ePub.xhtml#_idTextAnchor044)

### [聚类复习 38](B15923_02_Final_SMP_ePub.xhtml#_idTextAnchor045)

### [k-means 复习 39](B15923_02_Final_SMP_ePub.xhtml#_idTextAnchor046)

### [层次结构的组织 39](B15923_02_Final_SMP_ePub.xhtml#_idTextAnchor047)

### [层次聚类简介 41](B15923_02_Final_SMP_ePub.xhtml#_idTextAnchor048)

### [层次聚类步骤 43](B15923_02_Final_SMP_ePub.xhtml#_idTextAnchor049)

### [层次聚类示例演练 43](B15923_02_Final_SMP_ePub.xhtml#_idTextAnchor050)

### [练习 2.01：构建层次结构 47](B15923_02_Final_SMP_ePub.xhtml#_idTextAnchor051)

### [连接 52](B15923_02_Final_SMP_ePub.xhtml#_idTextAnchor052)

### [练习 2.02：应用连接标准 53](B15923_02_Final_SMP_ePub.xhtml#_idTextAnchor053)

### [凝聚型与分裂型聚类 58](B15923_02_Final_SMP_ePub.xhtml#_idTextAnchor054)

### [练习 2.03：使用 scikit-learn 实现凝聚层次聚类 60](B15923_02_Final_SMP_ePub.xhtml#_idTextAnchor055)

### [活动 2.01：比较 k-means 和层次聚类 64](B15923_02_Final_SMP_ePub.xhtml#_idTextAnchor056)

### [k-means 与层次聚类 68](B15923_02_Final_SMP_ePub.xhtml#_idTextAnchor057)

### [总结 69](B15923_02_Final_SMP_ePub.xhtml#_idTextAnchor058)

## [3. 邻域方法与 DBSCAN 71](B15923_03_Final_NT_ePub.xhtml#_idTextAnchor059)

### [简介 72](B15923_03_Final_NT_ePub.xhtml#_idTextAnchor060)

### [聚类作为邻域 73](B15923_03_Final_NT_ePub.xhtml#_idTextAnchor061)

### [DBSCAN 简介 75](B15923_03_Final_NT_ePub.xhtml#_idTextAnchor062)

### [DBSCAN 详细解析 76](B15923_03_Final_NT_ePub.xhtml#_idTextAnchor063)

### [DBSCAN 算法演练 77](B15923_03_Final_NT_ePub.xhtml#_idTextAnchor064)

### [练习 3.01：评估邻域半径大小的影响 80](B15923_03_Final_NT_ePub.xhtml#_idTextAnchor065)

### [DBSCAN 属性 - 邻域半径 84](B15923_03_Final_NT_ePub.xhtml#_idTextAnchor066)

### [活动 3.01：从头实现 DBSCAN 86](B15923_03_Final_NT_ePub.xhtml#_idTextAnchor067)

### [DBSCAN 属性 - 最小点数 88](B15923_03_Final_NT_ePub.xhtml#_idTextAnchor068)

### [练习 3.02：评估最小点数阈值的影响 89](B15923_03_Final_NT_ePub.xhtml#_idTextAnchor069)

### [活动 3.02：将 DBSCAN 与 k-means 和层次聚类进行比较 93](B15923_03_Final_NT_ePub.xhtml#_idTextAnchor070)

### [DBSCAN 与 k-means 和层次聚类的对比 95](B15923_03_Final_NT_ePub.xhtml#_idTextAnchor071)

### [总结 96](B15923_03_Final_NT_ePub.xhtml#_idTextAnchor072)

## [4. 主成分分析与降维技术 99](B15923_04_Final_RK_ePub.xhtml#_idTextAnchor074)

### [介绍 100](B15923_04_Final_RK_ePub.xhtml#_idTextAnchor075)

### [什么是降维？ 100](B15923_04_Final_RK_ePub.xhtml#_idTextAnchor076)

### [降维技术的应用 102](B15923_04_Final_RK_ePub.xhtml#_idTextAnchor077)

### [维度灾难 104](B15923_04_Final_RK_ePub.xhtml#_idTextAnchor078)

### [降维技术概览 106](B15923_04_Final_RK_ePub.xhtml#_idTextAnchor079)

### [降维 108](B15923_04_Final_RK_ePub.xhtml#_idTextAnchor080)

### [主成分分析 109](B15923_04_Final_RK_ePub.xhtml#_idTextAnchor081)

### [均值 109](B15923_04_Final_RK_ePub.xhtml#_idTextAnchor082)

### [标准差 109](B15923_04_Final_RK_ePub.xhtml#_idTextAnchor083)

### [协方差 110](B15923_04_Final_RK_ePub.xhtml#_idTextAnchor084)

### [协方差矩阵 110](B15923_04_Final_RK_ePub.xhtml#_idTextAnchor085)

### [练习 4.01：使用 pandas 库计算均值、标准差和方差 111](B15923_04_Final_RK_ePub.xhtml#_idTextAnchor086)

### [特征值与特征向量 116](B15923_04_Final_RK_ePub.xhtml#_idTextAnchor087)

### [练习 4.02：计算特征值和特征向量 117](B15923_04_Final_RK_ePub.xhtml#_idTextAnchor088)

### [PCA 的过程 121](B15923_04_Final_RK_ePub.xhtml#_idTextAnchor089)

### [练习 4.03：手动执行 PCA 123](B15923_04_Final_RK_ePub.xhtml#_idTextAnchor090)

### [练习 4.04：使用 scikit-learn 进行 PCA 128](B15923_04_Final_RK_ePub.xhtml#_idTextAnchor091)

### [活动 4.01：手动 PCA 与 scikit-learn 对比 133](B15923_04_Final_RK_ePub.xhtml#_idTextAnchor092)

### [恢复压缩后的数据集 136](B15923_04_Final_RK_ePub.xhtml#_idTextAnchor093)

### [练习 4.05：使用手动 PCA 可视化方差减少 136](B15923_04_Final_RK_ePub.xhtml#_idTextAnchor094)

### [练习 4.06：使用 scikit-learn 可视化方差减少 143](B15923_04_Final_RK_ePub.xhtml#_idTextAnchor095)

### [练习 4.07：在 Matplotlib 中绘制 3D 图 147](B15923_04_Final_RK_ePub.xhtml#_idTextAnchor096)

### [活动 4.02: 使用扩展的种子数据集进行 PCA 150](B15923_04_Final_RK_ePub.xhtml#_idTextAnchor097)

### [总结 153](B15923_04_Final_RK_ePub.xhtml#_idTextAnchor098)

## [5\. 自编码器 155](B15923_05_Final_RK_ePub.xhtml#_idTextAnchor099)

### [简介 156](B15923_05_Final_RK_ePub.xhtml#_idTextAnchor100)

### [人工神经网络基础 157](B15923_05_Final_RK_ePub.xhtml#_idTextAnchor101)

### [神经元 159](B15923_05_Final_RK_ePub.xhtml#_idTextAnchor102)

### [Sigmoid 函数 160](B15923_05_Final_RK_ePub.xhtml#_idTextAnchor104)

### [修正线性单元 (ReLU) 161](B15923_05_Final_RK_ePub.xhtml#_idTextAnchor105)

### [练习 5.01: 建模人工神经网络的神经元 161](B15923_05_Final_RK_ePub.xhtml#_idTextAnchor106)

### [练习 5.02: 使用 ReLU 激活函数建模神经元 165](B15923_05_Final_RK_ePub.xhtml#_idTextAnchor107)

### [神经网络: 架构定义 169](B15923_05_Final_RK_ePub.xhtml#_idTextAnchor108)

### [练习 5.03: 定义一个 Keras 模型 171](B15923_05_Final_RK_ePub.xhtml#_idTextAnchor109)

### [神经网络: 训练 173](B15923_05_Final_RK_ePub.xhtml#_idTextAnchor110)

### [练习 5.04: 训练一个 Keras 神经网络模型 175](B15923_05_Final_RK_ePub.xhtml#_idTextAnchor111)

### [活动 5.01: MNIST 神经网络 185](B15923_05_Final_RK_ePub.xhtml#_idTextAnchor112)

### [自编码器 187](B15923_05_Final_RK_ePub.xhtml#_idTextAnchor113)

### [练习 5.05: 简单自编码器 188](B15923_05_Final_RK_ePub.xhtml#_idTextAnchor114)

### [活动 5.02: 简单 MNIST 自编码器 193](B15923_05_Final_RK_ePub.xhtml#_idTextAnchor115)

### [练习 5.06: 多层自编码器 194](B15923_05_Final_RK_ePub.xhtml#_idTextAnchor116)

### [卷积神经网络 199](B15923_05_Final_RK_ePub.xhtml#_idTextAnchor117)

### [练习 5.07: 卷积自编码器 200](B15923_05_Final_RK_ePub.xhtml#_idTextAnchor118)

### [活动 5.03: MNIST 卷积自编码器 205](B15923_05_Final_RK_ePub.xhtml#_idTextAnchor119)

### [总结 207](B15923_05_Final_RK_ePub.xhtml#_idTextAnchor120)

## [6\. t-分布随机邻居嵌入 209](B15923_06_Final_RK_ePub.xhtml#_idTextAnchor121)

### [简介 210](B15923_06_Final_RK_ePub.xhtml#_idTextAnchor123)

### [MNIST 数据集 210](B15923_06_Final_RK_ePub.xhtml#_idTextAnchor124)

### [随机邻居嵌入 (SNE) 212](B15923_06_Final_RK_ePub.xhtml#_idTextAnchor125)

### [t-分布 SNE 213](B15923_06_Final_RK_ePub.xhtml#_idTextAnchor126)

### [练习 6.01: t-SNE MNIST 214](B15923_06_Final_RK_ePub.xhtml#_idTextAnchor127)

### [活动 6.01: 葡萄酒 t-SNE 227](B15923_06_Final_RK_ePub.xhtml#_idTextAnchor128)

### [解释 t-SNE 图 229](B15923_06_Final_RK_ePub.xhtml#_idTextAnchor129)

### [困惑度 230](B15923_06_Final_RK_ePub.xhtml#_idTextAnchor130)

### [练习 6.02: t-SNE MNIST 和困惑度 230](B15923_06_Final_RK_ePub.xhtml#_idTextAnchor131)

### [活动 6.02: t-SNE 葡萄酒与困惑度 235](B15923_06_Final_RK_ePub.xhtml#_idTextAnchor132)

### [迭代   236](B15923_06_Final_RK_ePub.xhtml#_idTextAnchor133)

### [练习 6.03：t-SNE MNIST 和迭代   237](B15923_06_Final_RK_ePub.xhtml#_idTextAnchor134)

### [活动 6.03：t-SNE 葡萄酒和迭代   242](B15923_06_Final_RK_ePub.xhtml#_idTextAnchor135)

### [关于可视化的最终思考   243](B15923_06_Final_RK_ePub.xhtml#_idTextAnchor136)

### [总结   243](B15923_06_Final_RK_ePub.xhtml#_idTextAnchor137)

## [7. 主题建模   245](B15923_07_Final_RK_ePub.xhtml#_idTextAnchor138)

### [简介   246](B15923_07_Final_RK_ePub.xhtml#_idTextAnchor139)

### [主题模型   247](B15923_07_Final_RK_ePub.xhtml#_idTextAnchor140)

### [练习 7.01：设置环境   249](B15923_07_Final_RK_ePub.xhtml#_idTextAnchor141)

### [主题模型的高级概览   250](B15923_07_Final_RK_ePub.xhtml#_idTextAnchor142)

### [商业应用   254](B15923_07_Final_RK_ePub.xhtml#_idTextAnchor143)

### [练习 7.02：数据加载   256](B15923_07_Final_RK_ePub.xhtml#_idTextAnchor144)

### [清理文本数据   259](B15923_07_Final_RK_ePub.xhtml#_idTextAnchor145)

### [数据清理技术   260](B15923_07_Final_RK_ePub.xhtml#_idTextAnchor146)

### [练习 7.03：逐步清理数据   261](B15923_07_Final_RK_ePub.xhtml#_idTextAnchor147)

### [练习 7.04：完整数据清理   266](B15923_07_Final_RK_ePub.xhtml#_idTextAnchor148)

### [活动 7.01：加载和清理 Twitter 数据   268](B15923_07_Final_RK_ePub.xhtml#_idTextAnchor149)

### [潜在狄利克雷分配   270](B15923_07_Final_RK_ePub.xhtml#_idTextAnchor150)

### [变分推理   272](B15923_07_Final_RK_ePub.xhtml#_idTextAnchor151)

### [词袋模型   275](B15923_07_Final_RK_ePub.xhtml#_idTextAnchor152)

### [练习 7.05：使用计数向量化器创建词袋模型   276](B15923_07_Final_RK_ePub.xhtml#_idTextAnchor153)

### [困惑度   277](B15923_07_Final_RK_ePub.xhtml#_idTextAnchor154)

### [练习 7.06：选择主题数量   279](B15923_07_Final_RK_ePub.xhtml#_idTextAnchor155)

### [练习 7.07：运行 LDA   281](B15923_07_Final_RK_ePub.xhtml#_idTextAnchor156)

### [可视化   286](B15923_07_Final_RK_ePub.xhtml#_idTextAnchor157)

### [练习 7.08：可视化 LDA   287](B15923_07_Final_RK_ePub.xhtml#_idTextAnchor158)

### [练习 7.09：尝试四个主题   291](B15923_07_Final_RK_ePub.xhtml#_idTextAnchor159)

### [活动 7.02：LDA 和健康推文   296](B15923_07_Final_RK_ePub.xhtml#_idTextAnchor160)

### [练习 7.10：使用 TF-IDF 创建词袋模型   298](B15923_07_Final_RK_ePub.xhtml#_idTextAnchor161)

### [非负矩阵分解   299](B15923_07_Final_RK_ePub.xhtml#_idTextAnchor162)

### [弗罗贝纽斯范数   301](B15923_07_Final_RK_ePub.xhtml#_idTextAnchor163)

### [乘法更新算法   301](B15923_07_Final_RK_ePub.xhtml#_idTextAnchor164)

### [练习 7.11：非负矩阵分解   302](B15923_07_Final_RK_ePub.xhtml#_idTextAnchor165)

### [练习 7.12：可视化 NMF   306](B15923_07_Final_RK_ePub.xhtml#_idTextAnchor166)

### [活动 7.03：非负矩阵分解 309](B15923_07_Final_RK_ePub.xhtml#_idTextAnchor167)

### [总结 310](B15923_07_Final_RK_ePub.xhtml#_idTextAnchor168)

## [市场购物篮分析 313](B15923_08_Final_RK_ePub.xhtml#_idTextAnchor169)

### [介绍 314](B15923_08_Final_RK_ePub.xhtml#_idTextAnchor170)

### [市场购物篮分析 314](B15923_08_Final_RK_ePub.xhtml#_idTextAnchor171)

### [应用案例 317](B15923_08_Final_RK_ePub.xhtml#_idTextAnchor172)

### [重要的概率性指标 318](B15923_08_Final_RK_ePub.xhtml#_idTextAnchor173)

### [练习 8.01：创建样本事务数据 319](B15923_08_Final_RK_ePub.xhtml#_idTextAnchor174)

### [支持度 321](B15923_08_Final_RK_ePub.xhtml#_idTextAnchor175)

### [置信度 322](B15923_08_Final_RK_ePub.xhtml#_idTextAnchor176)

### [提升度和杠杆度 323](B15923_08_Final_RK_ePub.xhtml#_idTextAnchor177)

### [信念度 324](B15923_08_Final_RK_ePub.xhtml#_idTextAnchor178)

### [练习 8.02：计算指标 325](B15923_08_Final_RK_ePub.xhtml#_idTextAnchor179)

### [事务数据的特征 328](B15923_08_Final_RK_ePub.xhtml#_idTextAnchor180)

### [练习 8.03：加载数据 329](B15923_08_Final_RK_ePub.xhtml#_idTextAnchor181)

### [数据清理和格式化 333](B15923_08_Final_RK_ePub.xhtml#_idTextAnchor182)

### [练习 8.04：数据清理和格式化 334](B15923_08_Final_RK_ePub.xhtml#_idTextAnchor183)

### [数据编码 339](B15923_08_Final_RK_ePub.xhtml#_idTextAnchor184)

### [练习 8.05：数据编码 341](B15923_08_Final_RK_ePub.xhtml#_idTextAnchor185)

### [活动 8.01：加载和准备完整的在线零售数据 343](B15923_08_Final_RK_ePub.xhtml#_idTextAnchor186)

### [Apriori算法 344](B15923_08_Final_RK_ePub.xhtml#_idTextAnchor187)

### [计算修正 347](B15923_08_Final_RK_ePub.xhtml#_idTextAnchor188)

### [练习 8.06：执行Apriori算法 348](B15923_08_Final_RK_ePub.xhtml#_idTextAnchor189)

### [活动 8.02：在完整的在线零售数据集上运行Apriori算法 354](B15923_08_Final_RK_ePub.xhtml#_idTextAnchor190)

### [关联规则 356](B15923_08_Final_RK_ePub.xhtml#_idTextAnchor191)

### [练习 8.07：推导关联规则 358](B15923_08_Final_RK_ePub.xhtml#_idTextAnchor192)

### [活动 8.03：在完整的在线零售数据集上找出关联规则 365](B15923_08_Final_RK_ePub.xhtml#_idTextAnchor193)

### [总结 367](B15923_08_Final_RK_ePub.xhtml#_idTextAnchor194)

## [热点分析 369](B15923_09_Final_RK_ePub.xhtml#_idTextAnchor195)

### [介绍 370](B15923_09_Final_RK_ePub.xhtml#_idTextAnchor197)

### [空间统计 371](B15923_09_Final_RK_ePub.xhtml#_idTextAnchor198)

### [概率密度函数 372](B15923_09_Final_RK_ePub.xhtml#_idTextAnchor199)

### [在商业中使用热点分析 374](B15923_09_Final_RK_ePub.xhtml#_idTextAnchor200)

### [核密度估计 375](B15923_09_Final_RK_ePub.xhtml#_idTextAnchor201)

### [带宽值 376](B15923_09_Final_RK_ePub.xhtml#_idTextAnchor202)

### [练习 9.01：带宽值的影响 376](B15923_09_Final_RK_ePub.xhtml#_idTextAnchor203)

### [选择最优带宽 380](B15923_09_Final_RK_ePub.xhtml#_idTextAnchor204)

### [练习 9.02：使用网格搜索选择最优带宽 381](B15923_09_Final_RK_ePub.xhtml#_idTextAnchor205)

### [核函数 384](B15923_09_Final_RK_ePub.xhtml#_idTextAnchor206)

### [练习 9.03：核函数的影响 387](B15923_09_Final_RK_ePub.xhtml#_idTextAnchor207)

### [核密度估计推导 389](B15923_09_Final_RK_ePub.xhtml#_idTextAnchor208)

### [练习 9.04：模拟核密度估计推导 389](B15923_09_Final_RK_ePub.xhtml#_idTextAnchor209)

### [活动 9.01：一维密度估计 393](B15923_09_Final_RK_ePub.xhtml#_idTextAnchor210)

### [热点分析 394](B15923_09_Final_RK_ePub.xhtml#_idTextAnchor211)

### [练习 9.05：使用 Seaborn 加载数据和建模 396](B15923_09_Final_RK_ePub.xhtml#_idTextAnchor212)

### [练习 9.06：与底图一起工作 404](B15923_09_Final_RK_ePub.xhtml#_idTextAnchor213)

### [活动 9.02：分析伦敦犯罪 411](B15923_09_Final_RK_ePub.xhtml#_idTextAnchor214)

### [总结 414](B15923_09_Final_RK_ePub.xhtml#_idTextAnchor215)

## [附录 417](B15923_Solution_Final_RK_ePub.xhtml#_idTextAnchor216)
