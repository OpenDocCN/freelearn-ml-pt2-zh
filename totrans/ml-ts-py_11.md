# 12

# 多变量预测

如果你一直在关注本书，你可能已经注意到，过去十年时间序列领域取得了很多进展。许多扩展和新技术出现，用于将机器学习应用于时间序列。在每一章中，我们涵盖了许多关于预测、异常和漂移检测、回归和分类以及包括传统方法、梯度提升机器学习和其他方法、强化学习、在线学习、深度学习和概率模型的不同问题。

在本章中，我们将更深入地实践一些内容。到目前为止，我们主要涵盖了单变量时间序列，但在本章中，我们将应用预测到能源需求的一个案例。鉴于世界各地持续存在能源或供应危机，这是一个非常及时的主题。我们将使用多变量时间序列，并使用不同的方法进行多步预测。

我们将涵盖以下主题：

+   多变量时间序列预测

+   时间序列的下一步是什么？

第二部分将展望时间序列应用和研究的未来。但首先让我们讨论多变量序列。然后我们将应用几个模型来进行能源需求预测。

# 多变量时间序列预测

时间序列预测是学术界的一个活跃研究课题。预测长期趋势不仅是一项有趣的挑战，而且对于战略规划和运营研究在实际应用中具有重要的影响，例如IT运营管理、制造业和网络安全。

多变量时间序列具有多个因变量。这意味着每个因变量不仅依赖于其自身的过去值，还可能依赖于其他变量的过去值。这引入了复杂性，如共线性，其中因变量不是独立的，而是相关的。共线性违反了许多线性模型的假设，因此更有吸引力的是使用能够捕获特征交互作用的模型。

此图显示了一个多变量时间序列的示例，即各国COVID-19死亡情况（来自维基百科关于COVID-19大流行的英文文章）：

![](img/B17577_12_01.png)

图12.1：COVID-19每10万人口死亡人数作为多变量时间序列的示例。

各国之间的COVID死亡人数存在相关性，尽管它们可能会有所偏移，或者可能属于不同的群体。

我们在*第5章*、*时间序列机器学习简介*中提到过Makridakis竞赛。作为主办方的Spyros Makridakis是尼科西亚大学的教授，专门研究时间序列预测。这些竞赛作为最佳算法的基准，研究人员和实践者相互竞争，争夺现金奖励。这项竞赛的希望是能够激发并促进机器学习的发展，并为未来的工作开辟方向。

M4竞赛使用了来自ForeDeCk数据库的100,000个多变量时间序列，涵盖了不同的应用领域和时间尺度，结果于2020年发布。49个参赛者或团队提交了点预测，测试了主要的机器学习和统计方法的准确性。

M4的组织者Spyros Makridakis、Evangelos Spiliotis和Vassilios Assimakopoulos在（"*M4竞赛：100,000个时间序列和61种预测方法*"，2020年）中观察到，主要由成熟的统计方法组合（混合或集成）往往比纯统计方法或纯机器学习方法更为准确，后者的表现相对较差，通常位于参赛队伍的后半部分。尽管机器学习方法在解决预测问题中的应用日益增加，但统计方法依然强大，尤其在处理低粒度数据时。不过需要注意的是，数据集未包含外生变量或时间戳。深度学习和其他机器学习方法可能在处理高维数据时表现更好，特别是在共线性存在的情况下，因此这些额外的信息可能会提升这些模型的表现。

然而，来自Uber Technologies的Slawek Smyl以第一名的成绩获得了9000欧元，他的模型结合了递归神经网络和统计时间序列模型（Holt-Winters指数平滑）。这两个组件是通过梯度下降法同时拟合的。作为一位经验丰富的时间序列专家，Smyl曾在*2016年计算智能预测国际时间序列竞赛*中使用递归神经网络获胜。可以说，这个结果表明，机器学习（及深度学习作为一种扩展）与实用主义相结合可以带来成功。

经济学家们长期以来在预测中使用混合模型，例如高斯混合模型或GARCH模型的混合。`Skaters`库提供了各种集成功能，也支持ARMA及类似模型的集成。你可以在微预测时间序列排行榜上找到不同集成模型的概述：[https://microprediction.github.io/timeseries-elo-ratings/html_leaderboards/overall.html](https://microprediction.github.io/timeseries-elo-ratings/html_leaderboards/overall.html)

在机器学习方面，集成方法，特别是在集成学习中的一种常见方法是训练多个模型，并根据它们的性能对预测结果进行加权。集成学习通过带放回采样来创建训练样本，进而拟合基础模型。袋外（OOB）误差是指在未参与训练集的训练样本上的模型预测误差的均值。

集成模型还可以由不同类型的基础模型组成，这被称为异质集成。Scikit-learn为回归和分类提供了堆叠方法，最终模型可以根据基础模型的预测，找到加权系数来合并基础模型的预测结果。

在行业中，时间序列分析工作流仍然存在许多痛点。最主要的一个问题是，没有很多软件库支持多变量预测。

截至2021年9月，尽管它在开发路线图上，Kats库尚不支持多变量预测（尽管已支持多变量分类）。`statsmodels`库中有`VAR`和`VARMAX`模型；然而，目前没有对多变量时间序列进行季节性去除的支持。

Salesforce的Merlion库声称支持多变量预测，但似乎不在当前功能中。`Darts`库提供了几种适用于多变量预测的模型。

神经网络和集成方法，如随机森林或提升决策树，支持对多变量时间序列进行训练。在*第7章*，*时间序列的机器学习模型*中，我们使用XGBoost创建了一个时间序列预测的集成模型。在本书附带的GitHub代码库中，我附上了一个笔记本，展示了如何将scikit-learn管道和多输出回归器应用于多变量预测。然而，在这一章中，我们将重点介绍深度学习模型。

英国诺里奇东安格利亚大学的Alejandro Pasos Ruiz及其同事在他们的论文《*伟大的多变量时间序列分类大比拼：近期算法进展的回顾与实验评估*》（2020年）中指出，尽管有很多关于单变量数据集的建模，然而多变量应用却被忽视了。这不仅体现在软件解决方案的可用性上，也体现在数据集、以往的竞赛以及研究中。

他们对UEA数据集中的30个多变量时间序列进行了时间序列分类的基准测试。结果发现，三个分类器比动态时间规整算法（DTW）准确度高得多：HIVE-COTE、CIF和ROCKET（有关这些方法的详细信息，请参考*第4章*，*时间序列机器学习简介*）；然而，深度学习方法ResNet的表现与这些领先者差距不大。

在 *Hassan Ismail Fawaz* 等人（2019）发表的论文《*用于时间序列分类的深度学习：综述*》中，其中一项基准测试的发现是，一些深度神经网络可以与其他方法竞争。随后他们进一步展示了神经网络集成在相同数据集上的表现与 HIVE-COTE 不相上下（《*深度神经网络集成用于时间序列分类*》，2019）。

*Pedro Lara-Benítez* 等人（2021）在他们的论文《*时间序列预测的深度学习架构实验评审*》中做了另一次比较。他们运行了一个回声状态网络（ESN）、卷积神经网络（CNN）、时间卷积网络（TCN）、一个全连接的前馈网络（MLP），以及几个递归架构，如 Elman 递归网络、门控递归单元（GRU）网络和长短期记忆（LSTM）网络。

从统计学角度来看，基于平均排名，CNN、MLP、LSTM、TCN、GRU 和 ESN 没有显著区别。

总体而言，深度学习模型非常有前景，且由于其灵活性，它们能够填补多变量预测中的空白。我希望在本章中展示它们的实用性。

在本章中，我们将应用以下模型：

+   N-BEATS

+   亚马逊的 DeepAR

+   递归神经网络（LSTM）

+   Transformer

+   时间卷积网络（TCN）

+   高斯过程

我们在 *第10章* 《*时间序列的深度学习*》中已经详细介绍了大部分方法，但我将简要概述每个方法的主要特点。

**可解释时间序列预测的神经基础扩展分析**（**N-BEATS**），该模型在2020年ICLR大会上展示，相较于M4竞赛的冠军模型，提升了3%的预测精度。作者展示了一种纯深度学习方法，且没有任何时间序列特定组件，能够超越统计方法处理像 M3 和 M4 竞赛数据集及 TOURISM 数据集等具有挑战性的数据集。此方法的另一个优势是其可解释性（尽管我们在本章中不会重点讨论这一点）。

**DeepAR** 是一个来自亚马逊德国研究院的概率自回归递归网络模型。他们比较了三个不同数据集的分位数预测的准确性，并仅与一种因式分解技术（MatFact）在两个数据集（交通和电力）上进行准确性比较。

**长短期记忆网络**（**LSTM**）用于序列建模。像 LSTM 这样的递归神经网络的一个重要优势是它们可以学习长时间序列的数据点。

**Transformer** 是基于注意力机制的神经网络，最初在 2017 年的论文 "*Attention Is All You Need*" 中提出。它们的关键特点是与特征数量呈线性复杂度，并具备长时记忆能力，使我们能够直接访问序列中的任意点。Transformer 相较于循环神经网络的优势在于，它们是并行执行的，而不是按顺序执行，因此在训练和预测中运行速度更快。

Transformer 是为了解决 **自然语言处理**（**NLP**）任务中的序列问题而设计的；然而，它们同样可以应用于时间序列问题，包括预测，尽管这种应用不使用诸如位置编码等更特定于句子的特征。

**时间卷积网络**（**TCN**）由膨胀的、因果的 1D 卷积层组成，具有相同的输入和输出长度。我们使用的是包含残差块的实现，该实现由 Shaojie Bai 等人（2018）提出。

这些方法中的最后一个，**高斯过程**，不能被 convincingly 地归类为深度学习模型；然而，它们等同于一个具有独立同分布先验参数的单层全连接神经网络。它们可以被视为多元正态分布的无限维推广。

一个有趣的附加方面——尽管我们在这里不深入探讨——是许多这些方法允许使用额外的解释性（外生）变量。

我们将使用一个包含不同州的 10 维时间序列能源需求的数据集。该数据集来自 2017 年全球能源预测竞赛（GEFCom2017）。

每个变量记录特定区域的能源使用情况。这突出了长时记忆的问题——为了突出这一点，我们将进行多步预测。

你可以在我为演示目的创建的 GitHub 仓库中找到 `tensorflow/keras` 的模型实现及其数据的工具函数：[https://github.com/benman1/time-series](https://github.com/benman1/time-series)。

让我们直接进入正题。

## Python 实践

我们将加载能源需求数据集，并应用几种预测方法。我们正在使用一个大数据集，并且这些模型有些相当复杂，因此训练可能需要较长时间。我建议你使用 Google Colab 并启用 GPU 支持，或者减少迭代次数或数据集的大小。待会儿在相关时，我会提到性能优化。

首先从上述提到的 GitHub 仓库中安装库：

[PRE0]

这不应该花费太长时间。由于需求包括 `tensorflow` 和 `numpy`，我建议将它们安装到虚拟环境中。

然后，我们将使用库中的一个工具方法加载数据集，并将其包装在一个 `TrainingDataSet` 类中：

[PRE1]

如果你想加速训练，你可以减少训练样本的数量。例如，代替前面的那一行，你可以写：`tds = TrainingDataSet(train_df.head(500))`。

我们稍后会对 `GaussianProcess` 进行操作，它无法处理完整数据集。

对于这些大多数模型，我们将使用 TensorFlow 图模型，这些模型依赖于非 eager 执行。我们必须显式地禁用 eager 执行。此外，对于其中一个模型，我们需要设置中间输出以避免 TensorFlow 问题：`Connecting to invalid output X of source node Y which has Z outputs`：

[PRE2]

我已经设置了我们将用于所有生成的预测的指标和绘图方法。我们可以直接从时间序列库加载它们：

[PRE3]

我们还将训练中的 epoch 数设置为 `100` ——每个模型都相同：

[PRE4]

如果你发现训练时间过长，你可以将此值设置为更小的值，以便训练更早结束。

我们将依次介绍不同的预测方法，首先是`DeepAR`：

[PRE5]

我们将看到模型的总结，然后是训练误差随时间的变化（此处省略）：

![/var/folders/80/g9sqgdws2rn0yc3rd5y3nd340000gp/T/TemporaryItems/NSIRD_screencaptureui_4ElEIb/Screenshot 2021-10-04 at 22.37.08.png](img/B17577_12_02.png)

图 12.2：DeepAR 模型参数。

这个模型相对简单，正如我们所见：只有 `360` 个参数。显然，我们可以调整这些参数并添加更多。

然后我们将在测试数据集上生成预测：

[PRE6]

我们将查看误差——首先是总体误差，然后是每个 `10` 维度的误差：

[PRE7]

我们将看到前`10`个时间步的图表：

![](img/B17577_12_03.png)

图 12.3：DeepAR 对 10 个时间步的预测。

让我们继续下一个方法：N-BEATS：

[PRE8]

N-BEATS 训练两个网络。前向网络有 `1,217,024` 个参数。

让我们看看预测结果：

![](img/B17577_12_04.png)

图 12.4：N-BEATS 预测。

接下来是 LSTM：

[PRE9]

这个模型比 DeepAR 需要更多的参数：

![/var/folders/80/g9sqgdws2rn0yc3rd5y3nd340000gp/T/TemporaryItems/NSIRD_screencaptureui_0VYxgy/Screenshot 2021-10-04 at 22.45.24.png](img/B17577_12_05.png)

图 12.5：LSTM 模型参数。

`45,000` 个参数——这意味着训练时间比 `DeepAR` 更长。

在这里我们再次看到预测：

![](img/B17577_12_06.png)

图 12.6：LSTM 预测。

让我们做一下 Transformer：

[PRE10]

这是预测图：

![forecast_transformer.png](img/B17577_12_07.png)

图 12.7：Transformer 预测。

这个模型训练非常长，且性能是所有模型中最差的。

我们的最后一个深度学习模型是 TCN：

[PRE11]

预测结果如下：

![](img/B17577_12_08.png)

图 12.8：TCN 预测。

不幸的是，高斯过程无法处理我们的数据集——因此，我们只加载了一小部分。高斯过程还依赖于即时执行，因此我们需要重新启动内核，重新导入库，然后执行这段代码。如果你不确定如何操作，请查看本书GitHub代码库中的`gaussian_process`笔记本。

继续往下看：

[PRE12]

预测结果如下：

![forecast_gp.png](img/B17577_12_09.png)

图12.9：高斯过程预测。

所有算法（除了高斯过程）都是在`99336`个数据点上训练的。如前所述，我们将训练轮次设为`100`，但是有一个早停规则，如果训练损失在`5`次迭代内没有变化，训练就会停止。

这些模型是在测试集上验证的。

让我们来查看统计数据：

|  | 参数 | MSE（测试） | 轮次 |
| --- | --- | --- | --- |
| DeepAR | 360 | 0.4338 | 100 |
| N-BEATS | 1,217,024 | 0.1016 | 100 |
| LSTM | 45,410 | 0.1569 | 100 |
| Transformer | 51,702 | 0.9314 | 55 |
| TCN | 145,060 | 0.0638 | 100 |
| 高斯过程 | 8 | 0.4221 | 100 |
| ES | 1 | 11.28 | - |

鉴于深度学习方法之间存在巨大的误差差异，可能是变换器的实现出了问题——我将在某个时刻尝试修复它。

我已经将一个基准方法——**指数平滑法**（**ES**）加入了模型中。你可以在时间序列代码库中找到这部分代码。

这为本章和整本书画上了句号。如果你想更好地理解背后的原理，可以查看代码库，你也可以调整模型参数。

# 时间序列的未来是什么？

在本书中，我们已探讨了时间序列的许多方面。如果你能读到这里，你应该已经学会了如何分析时间序列，以及如何应用传统的时间序列预测方法。这通常是市场上其他书籍的主要内容；然而，我们超越了这些。

我们探讨了与机器学习相关的时间序列预处理和转换方法。我们还查看了许多应用机器学习的实例，包括无监督和有监督的时间序列预测、异常检测、漂移检测和变更点检测。我们深入研究了在线学习、强化学习、概率模型和深度学习等技术。

在每一章中，我们都在探讨最重要的库，有时甚至是前沿的技术，最后，我们还涉及了广泛的工业应用。我们探讨了最先进的模型，如HIVE-COTE、预处理方法如ROCKET、适应漂移的模型（自适应在线模型），并回顾了多种异常检测方法。

我们甚至探讨了使用多臂赌博机在时间序列模型之间切换的场景，或者通过反事实进行因果分析的情景。

由于其普遍性，时间序列建模和预测在多个领域至关重要，并具有很大的经济意义。尽管传统和成熟的方式一直占主导地位，但时间序列的机器学习仍是一个相对较新的研究领域，刚刚走出其初期阶段，深度学习正处于这一革命的最前沿。

对于优秀模型的寻找将持续进行，并扩展到更大的新挑战。正如我在本章前面的部分希望展示的那样，其中一个挑战就是使多变量方法更具实用性。

下一届Makridakis竞赛M5，聚焦沃尔玛提供的层次时间序列（42,000个时间序列）。最终结果将在2022年发布。机器学习模型在时间序列的层次回归上表现出色，超越了一些文献中的成熟模型，正如*Mahdi Abolghasemi*等人（"*机器学习在时间序列层次预测中的应用*," 2019）在一个包含61组具有不同波动性的时间序列的基准测试中所展示的那样。混合效应模型（应用于组和层次）在时间序列预测中也是一个活跃的研究领域。

M6比赛涉及实时财务预测，包括S&P500美国股票和国际ETF。未来的比赛可能会聚焦于非线性问题，如黑天鹅事件、具有厚尾的时间序列，以及对风险管理和决策至关重要的分布。

![](img/Image21868.png)

[packt.com](http://packt.com)

订阅我们的在线数字图书馆，全面访问超过7000本书籍和视频，以及帮助你规划个人发展并推进职业生涯的行业领先工具。欲了解更多信息，请访问我们的网站。

# 为什么要订阅？

+   利用来自4000多名行业专家的实用电子书和视频，减少学习时间，增加编码时间。

+   通过特别为你制定的技能计划学习得更好

+   每月免费获得一本电子书或视频

+   完全可搜索，便于快速访问关键信息

+   复制和粘贴、打印及收藏内容

你知道Packt为每本出版的书提供电子书版本，并且有PDF和ePub文件可供选择吗？你可以在 [www.Packt.com](http://www.Packt.com) 升级到电子书版本，作为纸质书客户，你有权获得电子书折扣。详情请通过 [customercare@packtpub.com](http://customercare@packtpub.com) 联系我们。

在 [www.Packt.com](http://www.Packt.com)，你还可以阅读一系列免费的技术文章，注册各种免费的电子通讯，并获得Packt图书和电子书的独家折扣和优惠。

# 你可能喜欢的其他书籍

如果你喜欢这本书，可能对Packt出版的这些其他书籍感兴趣：

[![](img/9781801815093.png)](https://www.packtpub.com/product/learn-python-programming-third-edition/9781801815093)

**学习Python编程（第三版）**

法布里齐奥·罗马诺

Heinrich Kruger

ISBN: 978-1-80181-509-3

+   在 Windows、Mac 和 Linux 上启动和运行 Python

+   在任何情况下编写优雅、可重用且高效的代码

+   避免常见的陷阱，如重复、复杂的设计和过度工程化

+   理解何时使用函数式编程或面向对象编程的方法

+   使用 FastAPI 构建简单的 API，并使用 Tkinter 编写 GUI 应用程序

+   了解更复杂的主题，如数据持久化和加密学的初步概述

+   获取、清洗和操作数据，高效利用 Python 的内建数据结构

[![](img/9781801077262.png)](https://www.packtpub.com/product/python-object-oriented-programming-fourth-edition/9781801077262)

**Python 面向对象编程 – 第四版**

Steven F. Lott

Dusty Phillips

ISBN: 978-1-80107-726-2

+   通过创建类并定义方法，在 Python 中实现对象

+   使用继承扩展类的功能

+   使用异常处理不寻常的情况，保持代码的清晰

+   理解何时使用面向对象的特性，更重要的是，何时不使用它们

+   探索几种广泛使用的设计模式及其在 Python 中的实现方式

+   揭开单元测试和集成测试的简单性，并理解它们为何如此重要

+   学会静态检查你的动态代码类型

+   理解使用 asyncio 处理并发性以及它如何加速程序

[![](img/9781801071109.png)](https://www.packtpub.com/product/expert-python-programming-fourth-edition/9781801071109)

**专家 Python 编程 – 第四版**

Michał Jaworski

Tarek Ziadé

ISBN: 978-1-80107-110-9

+   探索设置可重复且一致的 Python 开发环境的现代方法

+   高效地打包 Python 代码供社区和生产使用

+   学习 Python 编程的现代语法元素，如 f-strings、枚举和 lambda 函数

+   通过元类揭开 Python 元编程的神秘面纱

+   编写 Python 中的并发代码

+   使用 C 和 C++ 编写的代码扩展和集成 Python

# Packt 正在寻找像你这样的作者

如果你有兴趣成为 Packt 的作者，请访问 [authors.packtpub.com](http://authors.packtpub.com) 并立即申请。我们与成千上万的开发者和技术专业人士合作，帮助他们与全球技术社区分享见解。你可以提交一般申请，申请我们正在招聘作者的特定热门话题，或者提交自己的想法。

# 分享你的想法

现在你已经完成了 *《Python 时间序列机器学习》*，我们很想听听你的想法！如果你从 Amazon 购买了这本书，请 [点击这里直接进入 Amazon 的评论页面](https://packt.link/r/1801819629) 来分享你的反馈或留下评论。

你的评论对我们和技术社区非常重要，将帮助我们确保提供优质的内容。

索引

A

激活函数 [264](Chapter_10.xhtml#_idIndexMarker777)

激活函数 [266](Chapter_10.xhtml#_idIndexMarker790)

AdaBoost [101](Chapter_4.xhtml#_idIndexMarker282)

自适应学习 [222](Chapter_8.xhtml#_idIndexMarker672)

方法 [222](Chapter_8.xhtml#_idIndexMarker673)

自适应 XGBoost [222](Chapter_8.xhtml#_idIndexMarker674)

ADWIN（自适应窗口） [220](Chapter_8.xhtml#_idIndexMarker669)

智能体 [98](Chapter_4.xhtml#_idIndexMarker266)

赤池信息量准则（AIC） [140](Chapter_5.xhtml#_idIndexMarker437)

赤池信息量准则（AIC） [156](Chapter_5.xhtml#_idIndexMarker496)

AlexNet [265](Chapter_10.xhtml#_idIndexMarker786)

亚马逊 [169](Chapter_6.xhtml#_idIndexMarker520)

anaconda 文档

参考链接 [22](Chapter_1.xhtml#_idIndexMarker044)

年金 [7](Chapter_1.xhtml#_idIndexMarker015)

异常检测 [95](Chapter_4.xhtml#_idIndexMarker248), [164](Chapter_6.xhtml#_idIndexMarker507), [165](Chapter_6.xhtml#_idIndexMarker508), [166](Chapter_6.xhtml#_idIndexMarker511), [167](Chapter_6.xhtml#_idIndexMarker512), [168](Chapter_6.xhtml#_idIndexMarker513), [178](Chapter_6.xhtml#_idIndexMarker546), [179](Chapter_6.xhtml#_idIndexMarker548), [180](Chapter_6.xhtml#_idIndexMarker552)

亚马逊 [169](Chapter_6.xhtml#_idIndexMarker519)

Facebook [170](Chapter_6.xhtml#_idIndexMarker524)

Google Analytics [169](Chapter_6.xhtml#_idIndexMarker518)

实现 [170](Chapter_6.xhtml#_idIndexMarker531), [171](Chapter_6.xhtml#_idIndexMarker532), [172](Chapter_6.xhtml#_idIndexMarker533)

微软 [168](Chapter_6.xhtml#_idIndexMarker515), [169](Chapter_6.xhtml#_idIndexMarker516)

Twitter [170](Chapter_6.xhtml#_idIndexMarker526)

Anticipy [146](Chapter_5.xhtml#_idIndexMarker465)

应用统计学 [17](Chapter_1.xhtml#_idIndexMarker035)

ARCH（自回归条件异方差） [143](Chapter_5.xhtml#_idIndexMarker453)

曲线下面积 [115](Chapter_4.xhtml#_idIndexMarker339)

人工通用智能（AGI） [298](Chapter_11.xhtml#_idIndexMarker850)

天文学 [11](Chapter_1.xhtml#_idIndexMarker020), [12](Chapter_1.xhtml#_idIndexMarker021)

自相关 [58](Chapter_2.xhtml#_idIndexMarker143), [59](Chapter_2.xhtml#_idIndexMarker144)

自编码器（AEs） [272](Chapter_10.xhtml#_idIndexMarker808), [273](Chapter_10.xhtml#_idIndexMarker812)

自动特征提取 [88](Chapter_3.xhtml#_idIndexMarker233), [89](Chapter_3.xhtml#_idIndexMarker234)

自回归（AR） [135](Chapter_5.xhtml#_idIndexMarker411), [136](Chapter_5.xhtml#_idIndexMarker416)

自回归条件异方差（ARCH） [146](Chapter_5.xhtml#_idIndexMarker466)

自回归积分滑动平均（ARIMA）[129](Chapter_4.xhtml#_idIndexMarker399)

自回归积分滑动平均模型（ARIMA） [138](Chapter_5.xhtml#_idIndexMarker423)

自回归模型 [135](Chapter_5.xhtml#_idIndexMarker412)

自回归滑动平均（ARMA）[137](Chapter_5.xhtml#_idIndexMarker421)

B

反向预测 [95](Chapter_4.xhtml#_idIndexMarker245)

反向传播 [99](Chapter_4.xhtml#_idIndexMarker271)，[103](Chapter_4.xhtml#_idIndexMarker289)，[264](Chapter_10.xhtml#_idIndexMarker780)

包袋法 [100](Chapter_4.xhtml#_idIndexMarker279)，[101](Chapter_4.xhtml#_idIndexMarker283)

与提升方法相比 [102](Chapter_4.xhtml#_idIndexMarker286)

模式包（BoP）[122](Chapter_4.xhtml#_idIndexMarker371)

模式包（BOP）[123](Chapter_4.xhtml#_idIndexMarker378)

SFA符号包（BOSS）[122](Chapter_4.xhtml#_idIndexMarker370)

强盗算法 [302](Chapter_11.xhtml#_idIndexMarker861)，[303](Chapter_11.xhtml#_idIndexMarker863)

基础学习器 [100](Chapter_4.xhtml#_idIndexMarker280)

贝叶斯信息准则（BIC）[140](Chapter_5.xhtml#_idIndexMarker438)

贝叶斯结构时间序列（BSTS）模型 [236](Chapter_9.xhtml#_idIndexMarker697)，[242](Chapter_9.xhtml#_idIndexMarker726)，[243](Chapter_9.xhtml#_idIndexMarker727)，[244](Chapter_9.xhtml#_idIndexMarker728)

实现，使用 Python [256](Chapter_9.xhtml#_idIndexMarker758)，[257](Chapter_9.xhtml#_idIndexMarker761)，[259](Chapter_9.xhtml#_idIndexMarker764)

生物学 [10](Chapter_1.xhtml#_idIndexMarker018)

提升法 [100](Chapter_4.xhtml#_idIndexMarker278)

自举法 [101](Chapter_4.xhtml#_idIndexMarker284)

向量空间中的BOSS（BOSS VS）[122](Chapter_4.xhtml#_idIndexMarker372)

Box-Cox 变换 [72](Chapter_3.xhtml#_idIndexMarker178)，[81](Chapter_3.xhtml#_idIndexMarker217)，[82](Chapter_3.xhtml#_idIndexMarker219)

工作日

提取，按月[88](Chapter_3.xhtml#_idIndexMarker232)

C

C4.5算法 [100](Chapter_4.xhtml#_idIndexMarker275)

电缆理论 [263](Chapter_10.xhtml#_idIndexMarker772)

典型区间森林（CIF）[121](Chapter_4.xhtml#_idIndexMarker366)

CART算法（分类与回归树）[100](Chapter_4.xhtml#_idIndexMarker274)

因果滤波器 [74](Chapter_3.xhtml#_idIndexMarker190)

细胞 [265](Chapter_10.xhtml#_idIndexMarker781)

中心极限定理 [12](Chapter_1.xhtml#_idIndexMarker023)

应用数学中心（CMAP）[129](Chapter_4.xhtml#_idIndexMarker394)

变点检测（CPD）[172](Chapter_6.xhtml#_idIndexMarker534)，[173](Chapter_6.xhtml#_idIndexMarker535)，[174](Chapter_6.xhtml#_idIndexMarker537)，[175](Chapter_6.xhtml#_idIndexMarker539)，[176](Chapter_6.xhtml#_idIndexMarker540)，[180](Chapter_6.xhtml#_idIndexMarker554)，[181](Chapter_6.xhtml#_idIndexMarker555)，[182](Chapter_6.xhtml#_idIndexMarker558)

经典模型 [132](Chapter_5.xhtml#_idIndexMarker400)，[133](Chapter_5.xhtml#_idIndexMarker401)

ARCH（自回归条件异方差）[143](Chapter_5.xhtml#_idIndexMarker452)

自回归（AR）[134](Chapter_5.xhtml#_idIndexMarker405)，[135](Chapter_5.xhtml#_idIndexMarker410)，[136](Chapter_5.xhtml#_idIndexMarker415)

GARCH（广义ARCH）[144](Chapter_5.xhtml#_idIndexMarker457)

模型选择 [139](Chapter_5.xhtml#_idIndexMarker430)

移动平均（MA）[134](Chapter_5.xhtml#_idIndexMarker404)

顺序 [139](Chapter_5.xhtml#_idIndexMarker429), [140](Chapter_5.xhtml#_idIndexMarker436)

向量自回归模型 [144](Chapter_5.xhtml#_idIndexMarker458), [145](Chapter_5.xhtml#_idIndexMarker461)

分类 [95](Chapter_4.xhtml#_idIndexMarker243), [97](Chapter_4.xhtml#_idIndexMarker259), [113](Chapter_4.xhtml#_idIndexMarker325)

聚类 [95](Chapter_4.xhtml#_idIndexMarker246), [176](Chapter_6.xhtml#_idIndexMarker541), [177](Chapter_6.xhtml#_idIndexMarker542)

决定系数 [107](Chapter_4.xhtml#_idIndexMarker306), [108](Chapter_4.xhtml#_idIndexMarker307)

共线性 [50](Chapter_2.xhtml#_idIndexMarker115)

复杂细胞 [265](Chapter_10.xhtml#_idIndexMarker783)

概念漂移 [217](Chapter_8.xhtml#_idIndexMarker660)

conda [22](Chapter_1.xhtml#_idIndexMarker043)

置信区间 [45](Chapter_2.xhtml#_idIndexMarker101)

混淆矩阵 [114](Chapter_4.xhtml#_idIndexMarker326)

上下文博弈 [303](Chapter_11.xhtml#_idIndexMarker866)

列联表 [220](Chapter_8.xhtml#_idIndexMarker668)

连续时间马尔可夫链 (CTMC) [239](Chapter_9.xhtml#_idIndexMarker716)

ConvNets [276](Chapter_10.xhtml#_idIndexMarker823)

卷积神经网络 (CNN) [322](Chapter_12.xhtml#_idIndexMarker900)

卷积神经网络 (CNN) [77](Chapter_3.xhtml#_idIndexMarker206)

相关性热图 [53](Chapter_2.xhtml#_idIndexMarker124)

相关性矩阵 [52](Chapter_2.xhtml#_idIndexMarker123)

相关比率 [115](Chapter_4.xhtml#_idIndexMarker340), [116](Chapter_4.xhtml#_idIndexMarker341)

协变量漂移 [217](Chapter_8.xhtml#_idIndexMarker656)

临界差异 (CD) 图 [119](Chapter_4.xhtml#_idIndexMarker361)

交叉验证 [105](Chapter_4.xhtml#_idIndexMarker295)

交叉验证准确度加权概率集成 (CAWPE) [124](Chapter_4.xhtml#_idIndexMarker383)

曲线拟合 [94](Chapter_4.xhtml#_idIndexMarker241)

循环变化 [56](Chapter_2.xhtml#_idIndexMarker131)

D

DataFrame [30](Chapter_1.xhtml#_idIndexMarker059)

数据预处理

关于 [68](Chapter_3.xhtml#_idIndexMarker159), [69](Chapter_3.xhtml#_idIndexMarker164)

数据预处理，技术

特征工程 [68](Chapter_3.xhtml#_idIndexMarker162)

特征变换 [68](Chapter_3.xhtml#_idIndexMarker160)

数据集漂移 [216](Chapter_8.xhtml#_idIndexMarker652)

与日期和时间相关的特征 [75](Chapter_3.xhtml#_idIndexMarker193)

日期注释 [85](Chapter_3.xhtml#_idIndexMarker226), [86](Chapter_3.xhtml#_idIndexMarker227)

日期时间 [39](Chapter_2.xhtml#_idIndexMarker082), [40](Chapter_2.xhtml#_idIndexMarker085), [41](Chapter_2.xhtml#_idIndexMarker087)

决策树 [100](Chapter_4.xhtml#_idIndexMarker276)

解码器 [273](Chapter_10.xhtml#_idIndexMarker811)

DeepAR [236](Chapter_9.xhtml#_idIndexMarker693), [274](Chapter_10.xhtml#_idIndexMarker818)

DeepAR 模型 [325](Chapter_12.xhtml#_idIndexMarker908), [326](Chapter_12.xhtml#_idIndexMarker909)

深度学习 [261](Chapter_10.xhtml#_idIndexMarker765), [262](Chapter_10.xhtml#_idIndexMarker767)

深度学习方法

类型学 [268](Chapter_10.xhtml#_idIndexMarker792)

深度学习，应用于时间序列 [269](Chapter_10.xhtml#_idIndexMarker793), [270](Chapter_10.xhtml#_idIndexMarker800), [271](Chapter_10.xhtml#_idIndexMarker806)

深度Q学习 [303](Chapter_11.xhtml#_idIndexMarker869), [304](Chapter_11.xhtml#_idIndexMarker870), [305](Chapter_11.xhtml#_idIndexMarker873)

深度Q网络（DQN）[311](Chapter_11.xhtml#_idIndexMarker883)

深度强化学习（DRL）[301](Chapter_11.xhtml#_idIndexMarker858)

DeepState [236](Chapter_9.xhtml#_idIndexMarker694)

人口统计学 [6](Chapter_1.xhtml#_idIndexMarker012), [7](Chapter_1.xhtml#_idIndexMarker014), [8](Chapter_1.xhtml#_idIndexMarker016), [9](Chapter_1.xhtml#_idIndexMarker017)

树突 [263](Chapter_10.xhtml#_idIndexMarker773)

描述性分析 [36](Chapter_2.xhtml#_idIndexMarker071)

差分 [138](Chapter_5.xhtml#_idIndexMarker425)

膨胀因果卷积神经网络 [292](Chapter_10.xhtml#_idIndexMarker845), [293](Chapter_10.xhtml#_idIndexMarker846), [294](Chapter_10.xhtml#_idIndexMarker847), [295](Chapter_10.xhtml#_idIndexMarker848)

狄利克雷采样 [303](Chapter_11.xhtml#_idIndexMarker865)

离散时间马尔可夫链（DTMC）[239](Chapter_9.xhtml#_idIndexMarker715)

基于距离的方法 [118](Chapter_4.xhtml#_idIndexMarker353)

dl-4-tsc [271](Chapter_10.xhtml#_idIndexMarker805)

漂移 [216](Chapter_8.xhtml#_idIndexMarker651), [217](Chapter_8.xhtml#_idIndexMarker654), [218](Chapter_8.xhtml#_idIndexMarker663), [219](Chapter_8.xhtml#_idIndexMarker664)

概念漂移 [217](Chapter_8.xhtml#_idIndexMarker661)

协变量漂移 [217](Chapter_8.xhtml#_idIndexMarker655)

概率漂移 [217](Chapter_8.xhtml#_idIndexMarker659)

漂移检测 [224](Chapter_8.xhtml#_idIndexMarker677), [225](Chapter_8.xhtml#_idIndexMarker678)

方法 [219](Chapter_8.xhtml#_idIndexMarker665), [220](Chapter_8.xhtml#_idIndexMarker670), [222](Chapter_8.xhtml#_idIndexMarker671)

漂移检测方法（DDM）[220](Chapter_8.xhtml#_idIndexMarker666)

漂移转换 [216](Chapter_8.xhtml#_idIndexMarker653)

随机失活 [282](Chapter_10.xhtml#_idIndexMarker832)

动态时间规整

在K近邻算法中的使用 [189](Chapter_7.xhtml#_idIndexMarker567)

动态时间规整（DTW）[116](Chapter_4.xhtml#_idIndexMarker345)

动态时间规整（DTW）[118](Chapter_4.xhtml#_idIndexMarker354), [270](Chapter_10.xhtml#_idIndexMarker798)

动态时间规整

K近邻算法，使用Python [193](Chapter_7.xhtml#_idIndexMarker585), [194](Chapter_7.xhtml#_idIndexMarker588), [195](Chapter_7.xhtml#_idIndexMarker590)

E

提前停止 [282](Chapter_10.xhtml#_idIndexMarker833)

回声状态网络（ESN）[322](Chapter_12.xhtml#_idIndexMarker899)

ECL（电力消耗负载）[279](Chapter_10.xhtml#_idIndexMarker830)

经济学 [13](Chapter_1.xhtml#_idIndexMarker025), [14](Chapter_1.xhtml#_idIndexMarker026)

弹性集成（EE）[124](Chapter_4.xhtml#_idIndexMarker382)

心电图（ECG）[118](Chapter_4.xhtml#_idIndexMarker349)

脑电图（EEG）[118](Chapter_4.xhtml#_idIndexMarker350)

脑电图（EEG）[16](Chapter_1.xhtml#_idIndexMarker032), [17](Chapter_1.xhtml#_idIndexMarker034), [60](Chapter_2.xhtml#_idIndexMarker149)

电子数值积分器和计算机（ENIAC）[15](Chapter_1.xhtml#_idIndexMarker029), [16](Chapter_1.xhtml#_idIndexMarker030)

编码器 [273](Chapter_10.xhtml#_idIndexMarker810)

epsilon-greedy [301](Chapter_11.xhtml#_idIndexMarker857)

错误指标

时间序列 [106](Chapter_4.xhtml#_idIndexMarker299)

ETT（电力变压器温度）[279](Chapter_10.xhtml#_idIndexMarker829)

欧几里得距离 [116](Chapter_4.xhtml#_idIndexMarker343)

经验重放技术 [304](Chapter_11.xhtml#_idIndexMarker872)

探索与开发困境 [301](Chapter_11.xhtml#_idIndexMarker856)

探索性分析 [36](Chapter_2.xhtml#_idIndexMarker072)

探索性数据分析（EDA）[36](Chapter_2.xhtml#_idIndexMarker069)

指数平滑 [140](Chapter_5.xhtml#_idIndexMarker439), [141](Chapter_5.xhtml#_idIndexMarker441), [142](Chapter_5.xhtml#_idIndexMarker445)

指数平滑（ES）[269](Chapter_10.xhtml#_idIndexMarker795), [275](Chapter_10.xhtml#_idIndexMarker821), [335](Chapter_12.xhtml#_idIndexMarker919)

指数平滑模型 [157](Chapter_5.xhtml#_idIndexMarker499), [158](Chapter_5.xhtml#_idIndexMarker500)

用于创建预测 [157](Chapter_5.xhtml#_idIndexMarker498)

极端学生化偏差（ESD）[170](Chapter_6.xhtml#_idIndexMarker528)

F

Facebook [170](Chapter_6.xhtml#_idIndexMarker523)

虚警率 [115](Chapter_4.xhtml#_idIndexMarker338)

假阴性（FN）[115](Chapter_4.xhtml#_idIndexMarker333)

假阳性率（FPR）[115](Chapter_4.xhtml#_idIndexMarker337)

假阳性（FP）[115](Chapter_4.xhtml#_idIndexMarker332)

特征工程 [68](Chapter_3.xhtml#_idIndexMarker163)

关于 [74](Chapter_3.xhtml#_idIndexMarker188), [75](Chapter_3.xhtml#_idIndexMarker192)

日期和时间相关特征 [75](Chapter_3.xhtml#_idIndexMarker194)

ROCKET 特征 [76](Chapter_3.xhtml#_idIndexMarker196), [77](Chapter_3.xhtml#_idIndexMarker201)

形状特征 [77](Chapter_3.xhtml#_idIndexMarker207)

特征泄漏 [49](Chapter_2.xhtml#_idIndexMarker112)

特征变换 [68](Chapter_3.xhtml#_idIndexMarker161)

关于 [69](Chapter_3.xhtml#_idIndexMarker165)

填充 [73](Chapter_3.xhtml#_idIndexMarker184)

对数变换 [71](Chapter_3.xhtml#_idIndexMarker172)

幂变换 [71](Chapter_3.xhtml#_idIndexMarker174)

缩放 [70](Chapter_3.xhtml#_idIndexMarker167)

前馈传播 [264](Chapter_10.xhtml#_idIndexMarker778)

滤波器 [76](Chapter_3.xhtml#_idIndexMarker199)

预测

创建，使用指数平滑模型 [156](Chapter_5.xhtml#_idIndexMarker497)

预测误差 [107](Chapter_4.xhtml#_idIndexMarker304)

预测 [95](Chapter_4.xhtml#_idIndexMarker244)

预测 [6](Chapter_1.xhtml#_idIndexMarker011)

全连接前馈神经网络 [98](Chapter_4.xhtml#_idIndexMarker270)

全连接网络 [281](Chapter_10.xhtml#_idIndexMarker831), [282](Chapter_10.xhtml#_idIndexMarker835), [283](Chapter_10.xhtml#_idIndexMarker836), [284](Chapter_10.xhtml#_idIndexMarker837), [285](Chapter_10.xhtml#_idIndexMarker838), [286](Chapter_10.xhtml#_idIndexMarker839), [288](Chapter_10.xhtml#_idIndexMarker841)

全连接网络（FCNs） [273](Chapter_10.xhtml#_idIndexMarker815)

全卷积神经网络（FCN） [273](Chapter_10.xhtml#_idIndexMarker816)

模糊建模 [240](Chapter_9.xhtml#_idIndexMarker720), [241](Chapter_9.xhtml#_idIndexMarker723), [242](Chapter_9.xhtml#_idIndexMarker725)

模糊集理论 [240](Chapter_9.xhtml#_idIndexMarker721)

模糊时间序列

用Python实现 [252](Chapter_9.xhtml#_idIndexMarker748), [253](Chapter_9.xhtml#_idIndexMarker749), [254](Chapter_9.xhtml#_idIndexMarker752), [255](Chapter_9.xhtml#_idIndexMarker753), [256](Chapter_9.xhtml#_idIndexMarker756)

G

GARCH（广义ARCH） [144](Chapter_5.xhtml#_idIndexMarker456)

门控递归单元（GRU） [323](Chapter_12.xhtml#_idIndexMarker902)

高斯过程 [333](Chapter_12.xhtml#_idIndexMarker917), [334](Chapter_12.xhtml#_idIndexMarker918)

高斯过程（GP） [271](Chapter_10.xhtml#_idIndexMarker804)

广义加法模型（GAM） [129](Chapter_4.xhtml#_idIndexMarker395), [170](Chapter_6.xhtml#_idIndexMarker525), [238](Chapter_9.xhtml#_idIndexMarker702)

广义线性模型（GLM） [19](Chapter_1.xhtml#_idIndexMarker038)

广义线性模型（GLM） [129](Chapter_4.xhtml#_idIndexMarker398)

广义随机形状森林（gRFS） [119](Chapter_4.xhtml#_idIndexMarker358)

生成对抗网络（GANs） [261](Chapter_10.xhtml#_idIndexMarker766)

全局最大池化 [77](Chapter_3.xhtml#_idIndexMarker202)

全球温度时间序列

参考链接 [57](Chapter_2.xhtml#_idIndexMarker136)

Gluon-TS [271](Chapter_10.xhtml#_idIndexMarker802)

Google Analytics [169](Chapter_6.xhtml#_idIndexMarker517)

梯度提升回归树（GBRT） [191](Chapter_7.xhtml#_idIndexMarker573)

梯度提升树

实现 [102](Chapter_4.xhtml#_idIndexMarker288)

梯度提升 [102](Chapter_4.xhtml#_idIndexMarker287), [191](Chapter_7.xhtml#_idIndexMarker571), [192](Chapter_7.xhtml#_idIndexMarker576), [199](Chapter_7.xhtml#_idIndexMarker605), [200](Chapter_7.xhtml#_idIndexMarker606), [201](Chapter_7.xhtml#_idIndexMarker609), [202](Chapter_7.xhtml#_idIndexMarker611), [203](Chapter_7.xhtml#_idIndexMarker615), [204](Chapter_7.xhtml#_idIndexMarker617)

梯度提升机（GBM） [191](Chapter_7.xhtml#_idIndexMarker572)

格兰杰因果关系 [117](Chapter_4.xhtml#_idIndexMarker346)

图形处理单元（GPUs） [265](Chapter_10.xhtml#_idIndexMarker787)

H

异质集成 [321](Chapter_12.xhtml#_idIndexMarker897)

隐马尔可夫模型（HMM） [239](Chapter_9.xhtml#_idIndexMarker718)

基于变换的集成模型的层次投票集成（HIVE-COTE）[123](Chapter_4.xhtml#_idIndexMarker379)

HIVE-COTE（基于变换的集成模型的层次投票集成）[270](Chapter_10.xhtml#_idIndexMarker799)

Hoeffding 树 [215](Chapter_8.xhtml#_idIndexMarker647)

留出法 [212](Chapter_8.xhtml#_idIndexMarker635)

节假日特征 [83](Chapter_3.xhtml#_idIndexMarker223)，[84](Chapter_3.xhtml#_idIndexMarker224)，[85](Chapter_3.xhtml#_idIndexMarker225)

霍尔茨-温特斯法（Holtz-Winters method） [142](Chapter_5.xhtml#_idIndexMarker444)

I

识别函数 [266](Chapter_10.xhtml#_idIndexMarker791)

插补 [82](Chapter_3.xhtml#_idIndexMarker220)，[83](Chapter_3.xhtml#_idIndexMarker222)

插补技术 [73](Chapter_3.xhtml#_idIndexMarker185)

InceptionTime [273](Chapter_10.xhtml#_idIndexMarker813)，[274](Chapter_10.xhtml#_idIndexMarker817)

推理 [96](Chapter_4.xhtml#_idIndexMarker255)

Informer [278](Chapter_10.xhtml#_idIndexMarker827)，[279](Chapter_10.xhtml#_idIndexMarker828)

集成开发环境（IDE）[27](Chapter_1.xhtml#_idIndexMarker054)

集成 [138](Chapter_5.xhtml#_idIndexMarker424)

四分位间距 [45](Chapter_2.xhtml#_idIndexMarker104)

J

JupyterLab [26](Chapter_1.xhtml#_idIndexMarker052)，[27](Chapter_1.xhtml#_idIndexMarker053)

Jupyter Notebook [26](Chapter_1.xhtml#_idIndexMarker050)

K

K臂赌博机 [212](Chapter_8.xhtml#_idIndexMarker638)

Kats 安装 [205](Chapter_7.xhtml#_idIndexMarker619)，[206](Chapter_7.xhtml#_idIndexMarker621)，[207](Chapter_7.xhtml#_idIndexMarker623)

核函数 [76](Chapter_3.xhtml#_idIndexMarker198)

K近邻

使用动态时间规整 [189](Chapter_7.xhtml#_idIndexMarker566)

使用动态时间规整的 Python [193](Chapter_7.xhtml#_idIndexMarker584)，[194](Chapter_7.xhtml#_idIndexMarker587)，[195](Chapter_7.xhtml#_idIndexMarker591)

L

标签漂移 [217](Chapter_8.xhtml#_idIndexMarker662)

最小二乘算法 [144](Chapter_5.xhtml#_idIndexMarker455)

最小二乘法 [12](Chapter_1.xhtml#_idIndexMarker022)

赖斯法则（lex parsimoniae）[139](Chapter_5.xhtml#_idIndexMarker432)

库

安装 [22](Chapter_1.xhtml#_idIndexMarker042)，[23](Chapter_1.xhtml#_idIndexMarker045)，[25](Chapter_1.xhtml#_idIndexMarker048)

寿命表 [7](Chapter_1.xhtml#_idIndexMarker013)

Light Gradient Boosting Machine（LightGBM）[191](Chapter_7.xhtml#_idIndexMarker574)

线性四率 [220](Chapter_8.xhtml#_idIndexMarker667)

线性回归（LR）[238](Chapter_9.xhtml#_idIndexMarker705)

折线图 [51](Chapter_2.xhtml#_idIndexMarker119)

对数变换 [71](Chapter_3.xhtml#_idIndexMarker171)，[82](Chapter_3.xhtml#_idIndexMarker218)

对数变换 [78](Chapter_3.xhtml#_idIndexMarker210)，[79](Chapter_3.xhtml#_idIndexMarker212)，[81](Chapter_3.xhtml#_idIndexMarker214)

长短期记忆（LSTM）[103](Chapter_4.xhtml#_idIndexMarker290)，[265](Chapter_10.xhtml#_idIndexMarker785)

长短期记忆（LSTM）[323](Chapter_12.xhtml#_idIndexMarker903)，[329](Chapter_12.xhtml#_idIndexMarker911)

长短期记忆模型（LSTM）[269](Chapter_10.xhtml#_idIndexMarker796)

损失函数 [106](Chapter_4.xhtml#_idIndexMarker297)

M

机器学习 [93](Chapter_4.xhtml#_idIndexMarker238), [98](Chapter_4.xhtml#_idIndexMarker267)

历史 [98](Chapter_4.xhtml#_idIndexMarker268), [99](Chapter_4.xhtml#_idIndexMarker272)

时间序列 [94](Chapter_4.xhtml#_idIndexMarker239)

工作流 [103](Chapter_4.xhtml#_idIndexMarker292), [104](Chapter_4.xhtml#_idIndexMarker293), [105](Chapter_4.xhtml#_idIndexMarker294)

机器学习算法

时间序列 [117](Chapter_4.xhtml#_idIndexMarker347)

查询时间与准确度 [124](Chapter_4.xhtml#_idIndexMarker385), [125](Chapter_4.xhtml#_idIndexMarker386)

机器学习方法

时间序列 [186](Chapter_7.xhtml#_idIndexMarker559), [187](Chapter_7.xhtml#_idIndexMarker561)

脑磁图（MEG）[118](Chapter_4.xhtml#_idIndexMarker351)

马尔可夫假设 [239](Chapter_9.xhtml#_idIndexMarker714)

马尔可夫性 [239](Chapter_9.xhtml#_idIndexMarker712)

马尔可夫模型 [239](Chapter_9.xhtml#_idIndexMarker710)

隐马尔可夫模型（HMM）[239](Chapter_9.xhtml#_idIndexMarker717)

实现，使用 Python [251](Chapter_9.xhtml#_idIndexMarker743), [252](Chapter_9.xhtml#_idIndexMarker745)

马尔可夫过程 [239](Chapter_9.xhtml#_idIndexMarker713)

马尔可夫属性 [239](Chapter_9.xhtml#_idIndexMarker711)

马尔可夫切换模型

实现，使用 Python [248](Chapter_9.xhtml#_idIndexMarker737), [249](Chapter_9.xhtml#_idIndexMarker739), [250](Chapter_9.xhtml#_idIndexMarker741)

最大似然估计（MLE）[139](Chapter_5.xhtml#_idIndexMarker434)

最大池化 [77](Chapter_3.xhtml#_idIndexMarker203)

平均值 [44](Chapter_2.xhtml#_idIndexMarker097)

平均绝对误差（MAE）[109](Chapter_4.xhtml#_idIndexMarker310), [110](Chapter_4.xhtml#_idIndexMarker314), [228](Chapter_8.xhtml#_idIndexMarker682)

平均绝对百分比误差（MAPE）[238](Chapter_9.xhtml#_idIndexMarker709)

平均百分比误差（MAPE）[111](Chapter_4.xhtml#_idIndexMarker318)

平均相对绝对误差（MRAE）[108](Chapter_4.xhtml#_idIndexMarker309), [113](Chapter_4.xhtml#_idIndexMarker322)

均方误差（MSE）[109](Chapter_4.xhtml#_idIndexMarker311), [110](Chapter_4.xhtml#_idIndexMarker313)

均方误差（MSE）[229](Chapter_8.xhtml#_idIndexMarker685)

中位数 [45](Chapter_2.xhtml#_idIndexMarker102)

中位数绝对偏差（MAD）[165](Chapter_6.xhtml#_idIndexMarker510)

中位数绝对误差（MdAE）[111](Chapter_4.xhtml#_idIndexMarker317)

医学 [16](Chapter_1.xhtml#_idIndexMarker031), [17](Chapter_1.xhtml#_idIndexMarker033)

气象学 [14](Chapter_1.xhtml#_idIndexMarker027)

指标 [106](Chapter_4.xhtml#_idIndexMarker298)

微观预测时间序列排行榜

参考链接 [321](Chapter_12.xhtml#_idIndexMarker895)

微软 [168](Chapter_6.xhtml#_idIndexMarker514)

MINIROCKET [119](Chapter_4.xhtml#_idIndexMarker360), [120](Chapter_4.xhtml#_idIndexMarker362)

最小-最大缩放 [70](Chapter_3.xhtml#_idIndexMarker169)

基于模型的插补 [73](Chapter_3.xhtml#_idIndexMarker187)

建模

在 Python 中 [148](Chapter_5.xhtml#_idIndexMarker474), [149](Chapter_5.xhtml#_idIndexMarker479), [150](Chapter_5.xhtml#_idIndexMarker480), [151](Chapter_5.xhtml#_idIndexMarker485), [152](Chapter_5.xhtml#_idIndexMarker487), [153](Chapter_5.xhtml#_idIndexMarker489), [154](Chapter_5.xhtml#_idIndexMarker491), [155](Chapter_5.xhtml#_idIndexMarker492)

模型选择 [139](Chapter_5.xhtml#_idIndexMarker431), [230](Chapter_8.xhtml#_idIndexMarker687), [231](Chapter_8.xhtml#_idIndexMarker688), [232](Chapter_8.xhtml#_idIndexMarker689)

模型堆叠 [74](Chapter_3.xhtml#_idIndexMarker189)

单调性 [71](Chapter_3.xhtml#_idIndexMarker175)

移动平均 [134](Chapter_5.xhtml#_idIndexMarker406)

移动平均 (MA) [129](Chapter_4.xhtml#_idIndexMarker396), [238](Chapter_9.xhtml#_idIndexMarker704)

MrSEQL [123](Chapter_4.xhtml#_idIndexMarker374)

多臂老虎机 [212](Chapter_8.xhtml#_idIndexMarker637)

多臂老虎机 (MAB) [302](Chapter_11.xhtml#_idIndexMarker862)

多层感知器 (MLP) [238](Chapter_9.xhtml#_idIndexMarker706)

乘法季节性 [142](Chapter_5.xhtml#_idIndexMarker450)

多元分析 [38](Chapter_2.xhtml#_idIndexMarker079)

多元时间序列 [4](Chapter_1.xhtml#_idIndexMarker003)

多元时间序列

预测 [320](Chapter_12.xhtml#_idIndexMarker892), [321](Chapter_12.xhtml#_idIndexMarker894), [322](Chapter_12.xhtml#_idIndexMarker898), [323](Chapter_12.xhtml#_idIndexMarker904), [324](Chapter_12.xhtml#_idIndexMarker906)

多元时间序列分类

临界差异图 [127](Chapter_4.xhtml#_idIndexMarker391)

多元时间序列 (MTS) [273](Chapter_10.xhtml#_idIndexMarker814)

多元无监督符号与导数 [123](Chapter_4.xhtml#_idIndexMarker377)

N

自然语言处理 (NLP) [269](Chapter_10.xhtml#_idIndexMarker794)

N-BEATS [275](Chapter_10.xhtml#_idIndexMarker819)

最近邻算法 [99](Chapter_4.xhtml#_idIndexMarker273)

可解释时间序列预测的神经基础扩展分析 (N-BEATS) [323](Chapter_12.xhtml#_idIndexMarker905), [327](Chapter_12.xhtml#_idIndexMarker910)

神经突 [262](Chapter_10.xhtml#_idIndexMarker769)

神经元 [262](Chapter_10.xhtml#_idIndexMarker768), [263](Chapter_10.xhtml#_idIndexMarker770), [264](Chapter_10.xhtml#_idIndexMarker774)

非线性方法 [70](Chapter_3.xhtml#_idIndexMarker166)

标准化均方误差 (NMSE) [112](Chapter_4.xhtml#_idIndexMarker320)

标准化回归指标 [112](Chapter_4.xhtml#_idIndexMarker321)

NumPy [28](Chapter_1.xhtml#_idIndexMarker055), [29](Chapter_1.xhtml#_idIndexMarker057)

O

目标函数 [98](Chapter_4.xhtml#_idIndexMarker264)

离线学习 [210](Chapter_8.xhtml#_idIndexMarker628)

与在线学习 [210](Chapter_8.xhtml#_idIndexMarker632), [211](Chapter_8.xhtml#_idIndexMarker634) 相对

在线算法 [213](Chapter_8.xhtml#_idIndexMarker639), [214](Chapter_8.xhtml#_idIndexMarker644), [215](Chapter_8.xhtml#_idIndexMarker650)

在线学习 [210](Chapter_8.xhtml#_idIndexMarker629)

使用案例 [210](Chapter_8.xhtml#_idIndexMarker630)

对比离线学习 [210](Chapter_8.xhtml#_idIndexMarker631), [211](Chapter_8.xhtml#_idIndexMarker633)

在线均值 [213](Chapter_8.xhtml#_idIndexMarker641)

在线方差 [213](Chapter_8.xhtml#_idIndexMarker642)

我们的数据世界（OWID）[46](Chapter_2.xhtml#_idIndexMarker107)

异常检测 [95](Chapter_4.xhtml#_idIndexMarker249)

袋外（OOB）误差 [321](Chapter_12.xhtml#_idIndexMarker896)

样本外测试 [105](Chapter_4.xhtml#_idIndexMarker296)

P

pandas [30](Chapter_1.xhtml#_idIndexMarker058), [31](Chapter_1.xhtml#_idIndexMarker062), [41](Chapter_2.xhtml#_idIndexMarker089), [42](Chapter_2.xhtml#_idIndexMarker092), [43](Chapter_2.xhtml#_idIndexMarker094)

发薪日

获取 [86](Chapter_3.xhtml#_idIndexMarker228)

Pearson 相关系数 [50](Chapter_2.xhtml#_idIndexMarker116)

百分位数 [45](Chapter_2.xhtml#_idIndexMarker105)

感知机 [98](Chapter_4.xhtml#_idIndexMarker269), [264](Chapter_10.xhtml#_idIndexMarker779)

感知机模型 [264](Chapter_10.xhtml#_idIndexMarker775)

周期图 [64](Chapter_2.xhtml#_idIndexMarker158)

分段聚合近似（PAA）[121](Chapter_4.xhtml#_idIndexMarker368)

pip [25](Chapter_1.xhtml#_idIndexMarker047)

Pmdarima [146](Chapter_5.xhtml#_idIndexMarker464)

基于策略的学习 [300](Chapter_11.xhtml#_idIndexMarker854)

正比例值（PPV）[77](Chapter_3.xhtml#_idIndexMarker204)

幂函数 [71](Chapter_3.xhtml#_idIndexMarker176)

幂变换 [78](Chapter_3.xhtml#_idIndexMarker211), [79](Chapter_3.xhtml#_idIndexMarker213), [81](Chapter_3.xhtml#_idIndexMarker215)

Box-Cox 变换 [72](Chapter_3.xhtml#_idIndexMarker179)

Yeo-Johnson 变换 [72](Chapter_3.xhtml#_idIndexMarker181)

幂变换 [71](Chapter_3.xhtml#_idIndexMarker173)

精度 [114](Chapter_4.xhtml#_idIndexMarker331)

预测 [96](Chapter_4.xhtml#_idIndexMarker256)

预测误差 [107](Chapter_4.xhtml#_idIndexMarker302)

前瞻性评估 [212](Chapter_8.xhtml#_idIndexMarker636)

主成分分析（PCA）[272](Chapter_10.xhtml#_idIndexMarker809)

概率库 [237](Chapter_9.xhtml#_idIndexMarker698)

概率模型 [236](Chapter_9.xhtml#_idIndexMarker691)

时间序列 [236](Chapter_9.xhtml#_idIndexMarker692)

概率 [235](Chapter_9.xhtml#_idIndexMarker690)

概率漂移 [217](Chapter_8.xhtml#_idIndexMarker658)

概率排序原则（PRP）[303](Chapter_11.xhtml#_idIndexMarker868)

Prophet 模型 [236](Chapter_9.xhtml#_idIndexMarker695), [237](Chapter_9.xhtml#_idIndexMarker700)

预测模型 [237](Chapter_9.xhtml#_idIndexMarker701), [238](Chapter_9.xhtml#_idIndexMarker703)

实现，在 Python 中 [245](Chapter_9.xhtml#_idIndexMarker731), [246](Chapter_9.xhtml#_idIndexMarker733), [247](Chapter_9.xhtml#_idIndexMarker735)

邻近森林 (PF) [121](Chapter_4.xhtml#_idIndexMarker364)

修剪的精确线性时间 (Pelt) [174](Chapter_6.xhtml#_idIndexMarker536)

pytest 文档

参考链接 [33](Chapter_1.xhtml#_idIndexMarker067)

Python

最佳实践 [31](Chapter_1.xhtml#_idIndexMarker063), [32](Chapter_1.xhtml#_idIndexMarker064)

用于时间序列 [18](Chapter_1.xhtml#_idIndexMarker036), [19](Chapter_1.xhtml#_idIndexMarker037), [21](Chapter_1.xhtml#_idIndexMarker040), [22](Chapter_1.xhtml#_idIndexMarker041)

建模 [148](Chapter_5.xhtml#_idIndexMarker475), [149](Chapter_5.xhtml#_idIndexMarker478), [150](Chapter_5.xhtml#_idIndexMarker481), [151](Chapter_5.xhtml#_idIndexMarker484), [152](Chapter_5.xhtml#_idIndexMarker486), [153](Chapter_5.xhtml#_idIndexMarker488), [154](Chapter_5.xhtml#_idIndexMarker490), [155](Chapter_5.xhtml#_idIndexMarker493)

实践 [177](Chapter_6.xhtml#_idIndexMarker543)

Python 练习 [245](Chapter_9.xhtml#_idIndexMarker729)

关于 [192](Chapter_7.xhtml#_idIndexMarker577)

BSTS 模型，实现 [256](Chapter_9.xhtml#_idIndexMarker757), [257](Chapter_9.xhtml#_idIndexMarker760), [258](Chapter_9.xhtml#_idIndexMarker762), [259](Chapter_9.xhtml#_idIndexMarker763)

模糊时间序列模型，实现 [252](Chapter_9.xhtml#_idIndexMarker747), [253](Chapter_9.xhtml#_idIndexMarker750), [254](Chapter_9.xhtml#_idIndexMarker751), [255](Chapter_9.xhtml#_idIndexMarker754), [256](Chapter_9.xhtml#_idIndexMarker755)

梯度提升 [199](Chapter_7.xhtml#_idIndexMarker604), [200](Chapter_7.xhtml#_idIndexMarker607), [201](Chapter_7.xhtml#_idIndexMarker608), [202](Chapter_7.xhtml#_idIndexMarker610), [203](Chapter_7.xhtml#_idIndexMarker614), [204](Chapter_7.xhtml#_idIndexMarker616)

Kats 安装 [205](Chapter_7.xhtml#_idIndexMarker618), [206](Chapter_7.xhtml#_idIndexMarker620), [207](Chapter_7.xhtml#_idIndexMarker624)

K-近邻，带有动态时间规整 [193](Chapter_7.xhtml#_idIndexMarker583), [194](Chapter_7.xhtml#_idIndexMarker586), [195](Chapter_7.xhtml#_idIndexMarker589)

马尔可夫切换模型，实现 [248](Chapter_9.xhtml#_idIndexMarker736), [249](Chapter_9.xhtml#_idIndexMarker738), [250](Chapter_9.xhtml#_idIndexMarker740), [251](Chapter_9.xhtml#_idIndexMarker742), [252](Chapter_9.xhtml#_idIndexMarker744)

Prophet 模型，实现 [245](Chapter_9.xhtml#_idIndexMarker730), [246](Chapter_9.xhtml#_idIndexMarker732), [247](Chapter_9.xhtml#_idIndexMarker734)

Silverkite [195](Chapter_7.xhtml#_idIndexMarker592), [197](Chapter_7.xhtml#_idIndexMarker594), [198](Chapter_7.xhtml#_idIndexMarker598), [199](Chapter_7.xhtml#_idIndexMarker600)

虚拟环境 [192](Chapter_7.xhtml#_idIndexMarker579), [193](Chapter_7.xhtml#_idIndexMarker581)

Python 库 [145](Chapter_5.xhtml#_idIndexMarker462)

datetime [39](Chapter_2.xhtml#_idIndexMarker083), [40](Chapter_2.xhtml#_idIndexMarker084), [41](Chapter_2.xhtml#_idIndexMarker086)

pandas [41](Chapter_2.xhtml#_idIndexMarker090), [42](Chapter_2.xhtml#_idIndexMarker091), [43](Chapter_2.xhtml#_idIndexMarker093), [44](Chapter_2.xhtml#_idIndexMarker095)

要求 [39](Chapter_2.xhtml#_idIndexMarker081)

Statsmodels [146](Chapter_5.xhtml#_idIndexMarker467), [147](Chapter_5.xhtml#_idIndexMarker471)

Python实践

关于 [305](Chapter_11.xhtml#_idIndexMarker874)

异常检测 [178](Chapter_6.xhtml#_idIndexMarker545), [179](Chapter_6.xhtml#_idIndexMarker547), [180](Chapter_6.xhtml#_idIndexMarker551)

变化点检测 (CPD) [180](Chapter_6.xhtml#_idIndexMarker553), [181](Chapter_6.xhtml#_idIndexMarker556), [182](Chapter_6.xhtml#_idIndexMarker557)

推荐系统 [305](Chapter_11.xhtml#_idIndexMarker875), [306](Chapter_11.xhtml#_idIndexMarker876), [307](Chapter_11.xhtml#_idIndexMarker877), [308](Chapter_11.xhtml#_idIndexMarker878), [309](Chapter_11.xhtml#_idIndexMarker879), [310](Chapter_11.xhtml#_idIndexMarker880)

要求 [177](Chapter_6.xhtml#_idIndexMarker544)

使用DQN的交易 [310](Chapter_11.xhtml#_idIndexMarker882), [311](Chapter_11.xhtml#_idIndexMarker884), [312](Chapter_11.xhtml#_idIndexMarker885), [313](Chapter_11.xhtml#_idIndexMarker886), [314](Chapter_11.xhtml#_idIndexMarker887), [315](Chapter_11.xhtml#_idIndexMarker889), [316](Chapter_11.xhtml#_idIndexMarker890), [317](Chapter_11.xhtml#_idIndexMarker891)

Python实践 [223](Chapter_8.xhtml#_idIndexMarker676)

Pytorch-forecasting [271](Chapter_10.xhtml#_idIndexMarker807)

Q

分位数变换 [73](Chapter_3.xhtml#_idIndexMarker183)

四分位数 [45](Chapter_2.xhtml#_idIndexMarker103)

R

随机森林 [102](Chapter_4.xhtml#_idIndexMarker285)

随机区间特征 (RIF) [124](Chapter_4.xhtml#_idIndexMarker381)

随机区间谱集成 (RISE) [124](Chapter_4.xhtml#_idIndexMarker380)

召回 [114](Chapter_4.xhtml#_idIndexMarker327)

接收者操作特征曲线 (ROC) [115](Chapter_4.xhtml#_idIndexMarker335)

循环神经网络 [289](Chapter_10.xhtml#_idIndexMarker842), [290](Chapter_10.xhtml#_idIndexMarker843), [291](Chapter_10.xhtml#_idIndexMarker844)

循环神经网络 (RNNs) [275](Chapter_10.xhtml#_idIndexMarker820)

回归 [95](Chapter_4.xhtml#_idIndexMarker242), [97](Chapter_4.xhtml#_idIndexMarker258), [107](Chapter_4.xhtml#_idIndexMarker301), [225](Chapter_8.xhtml#_idIndexMarker679), [226](Chapter_8.xhtml#_idIndexMarker680), [228](Chapter_8.xhtml#_idIndexMarker683), [229](Chapter_8.xhtml#_idIndexMarker684)

正则化贪心森林 (RGF) [191](Chapter_7.xhtml#_idIndexMarker575)

强化学习 [95](Chapter_4.xhtml#_idIndexMarker250), [98](Chapter_4.xhtml#_idIndexMarker265)

强化学习 [96](Chapter_4.xhtml#_idIndexMarker253)

强化学习 (RL)

关于 [298](Chapter_11.xhtml#_idIndexMarker849), [299](Chapter_11.xhtml#_idIndexMarker851), [301](Chapter_11.xhtml#_idIndexMarker855)

对时间序列 [301](Chapter_11.xhtml#_idIndexMarker859)

R错误（RE） [108](Chapter_4.xhtml#_idIndexMarker308)

残差 [107](Chapter_4.xhtml#_idIndexMarker303)

残差平方和 [107](Chapter_4.xhtml#_idIndexMarker305)

ResNets [266](Chapter_10.xhtml#_idIndexMarker788)

River库 [214](Chapter_8.xhtml#_idIndexMarker645)

ROCKET [119](Chapter_4.xhtml#_idIndexMarker359)

ROCKET特征 [76](Chapter_3.xhtml#_idIndexMarker197), [90](Chapter_3.xhtml#_idIndexMarker235), [91](Chapter_3.xhtml#_idIndexMarker236)

均方根误差（RMSE） [109](Chapter_4.xhtml#_idIndexMarker312), [110](Chapter_4.xhtml#_idIndexMarker315), [238](Chapter_9.xhtml#_idIndexMarker708)

均方根偏差（RMSD） [110](Chapter_4.xhtml#_idIndexMarker316)

均方根对数误差（RMSLE） [113](Chapter_4.xhtml#_idIndexMarker323)

运行图 [51](Chapter_2.xhtml#_idIndexMarker120)

S

尺度不变特征（SIFT） [118](Chapter_4.xhtml#_idIndexMarker355)

缩放方法 [70](Chapter_3.xhtml#_idIndexMarker168)

散点图 [54](Chapter_2.xhtml#_idIndexMarker125)

scikit-learn [214](Chapter_8.xhtml#_idIndexMarker643)

scikit-learn项目

参考链接 [33](Chapter_1.xhtml#_idIndexMarker066)

SciPy [28](Chapter_1.xhtml#_idIndexMarker056)

季节

获取特定日期的数据 [87](Chapter_3.xhtml#_idIndexMarker229)

季节性ARIMA（SARIMA） [238](Chapter_9.xhtml#_idIndexMarker707)

季节性自回归（SAR） [138](Chapter_5.xhtml#_idIndexMarker427)

季节性自回归整合滑动平均模型（SARIMA） [138](Chapter_5.xhtml#_idIndexMarker426)

季节性 [56](Chapter_2.xhtml#_idIndexMarker129)

识别 [56](Chapter_2.xhtml#_idIndexMarker135), [57](Chapter_2.xhtml#_idIndexMarker138), [58](Chapter_2.xhtml#_idIndexMarker142), [59](Chapter_2.xhtml#_idIndexMarker146), [60](Chapter_2.xhtml#_idIndexMarker148), [61](Chapter_2.xhtml#_idIndexMarker153), [63](Chapter_2.xhtml#_idIndexMarker155), [64](Chapter_2.xhtml#_idIndexMarker157)

季节性滑动平均（SMA） [139](Chapter_5.xhtml#_idIndexMarker428)

分割 [95](Chapter_4.xhtml#_idIndexMarker247)

自组织映射（SOM）[242](Chapter_9.xhtml#_idIndexMarker724)

敏感度 [114](Chapter_4.xhtml#_idIndexMarker330)

SEQL [123](Chapter_4.xhtml#_idIndexMarker373)

形状元件 [78](Chapter_3.xhtml#_idIndexMarker208), [91](Chapter_3.xhtml#_idIndexMarker237)

优势 [78](Chapter_3.xhtml#_idIndexMarker209)

形状元件 [119](Chapter_4.xhtml#_idIndexMarker356)

形状元件变换分类器（STC） [119](Chapter_4.xhtml#_idIndexMarker357)

Silverkite [195](Chapter_7.xhtml#_idIndexMarker593), [197](Chapter_7.xhtml#_idIndexMarker595), [198](Chapter_7.xhtml#_idIndexMarker599), [199](Chapter_7.xhtml#_idIndexMarker601)

Silverkite算法 [190](Chapter_7.xhtml#_idIndexMarker569), [191](Chapter_7.xhtml#_idIndexMarker570), [236](Chapter_9.xhtml#_idIndexMarker696)

简单细胞 [265](Chapter_10.xhtml#_idIndexMarker782)

简单指数平滑（SES）[140](Chapter_5.xhtml#_idIndexMarker440), [141](Chapter_5.xhtml#_idIndexMarker443), [142](Chapter_5.xhtml#_idIndexMarker446), [143](Chapter_5.xhtml#_idIndexMarker451)

简单移动平均 [134](Chapter_5.xhtml#_idIndexMarker407)

跳跃连接 [266](Chapter_10.xhtml#_idIndexMarker789)

Sktime-DL [271](Chapter_10.xhtml#_idIndexMarker801)

斯皮尔曼等级相关系数 [56](Chapter_2.xhtml#_idIndexMarker127)

标准差 [44](Chapter_2.xhtml#_idIndexMarker098)

标准误差（SE）[45](Chapter_2.xhtml#_idIndexMarker099)

平稳性 [6](Chapter_1.xhtml#_idIndexMarker008), [56](Chapter_2.xhtml#_idIndexMarker133), [136](Chapter_5.xhtml#_idIndexMarker418), [137](Chapter_5.xhtml#_idIndexMarker419)

平稳过程 [136](Chapter_5.xhtml#_idIndexMarker417)

平稳过程 [56](Chapter_2.xhtml#_idIndexMarker134)

Statsmodels [146](Chapter_5.xhtml#_idIndexMarker463), [147](Chapter_5.xhtml#_idIndexMarker472)

Statsmodels 库

用于建模时 [147](Chapter_5.xhtml#_idIndexMarker473)

结构化查询语言（SQL）[30](Chapter_1.xhtml#_idIndexMarker060)

PEP 8 风格指南

参考链接 [32](Chapter_1.xhtml#_idIndexMarker065)

日照小时数

获取，针对特定日期 [87](Chapter_3.xhtml#_idIndexMarker231)

监督算法，用于回归和分类

实现 [128](Chapter_4.xhtml#_idIndexMarker392), [129](Chapter_4.xhtml#_idIndexMarker393)

监督学习 [96](Chapter_4.xhtml#_idIndexMarker251), [97](Chapter_4.xhtml#_idIndexMarker257)

支持向量机（SVMs）[103](Chapter_4.xhtml#_idIndexMarker291), [271](Chapter_10.xhtml#_idIndexMarker803)

悬浮颗粒物（SPM）[55](Chapter_2.xhtml#_idIndexMarker126)

符号聚合近似（SAX）[121](Chapter_4.xhtml#_idIndexMarker367)

符号傅里叶近似（SFA）[122](Chapter_4.xhtml#_idIndexMarker369)

对称平均绝对百分比误差（SMAPE）[111](Chapter_4.xhtml#_idIndexMarker319)

突触 [263](Chapter_10.xhtml#_idIndexMarker771)

T

时间卷积网络（TCN）[276](Chapter_10.xhtml#_idIndexMarker824), [322](Chapter_12.xhtml#_idIndexMarker901), [331](Chapter_12.xhtml#_idIndexMarker915), [333](Chapter_12.xhtml#_idIndexMarker916)

时间字典集成（TDE）[124](Chapter_4.xhtml#_idIndexMarker384)

时间差分（TD）学习 [299](Chapter_11.xhtml#_idIndexMarker852)

时间融合变换器（TFT）[278](Chapter_10.xhtml#_idIndexMarker826)

泰尔 U 指数 [113](Chapter_4.xhtml#_idIndexMarker324)

Theta 方法 [141](Chapter_5.xhtml#_idIndexMarker442)

汤普森采样 [303](Chapter_11.xhtml#_idIndexMarker864)

时间序列 [3](Chapter_1.xhtml#_idIndexMarker002)

特征 [4](Chapter_1.xhtml#_idIndexMarker005), [5](Chapter_1.xhtml#_idIndexMarker007)

比较 [116](Chapter_4.xhtml#_idIndexMarker342)

机器学习方法，使用 [186](Chapter_7.xhtml#_idIndexMarker560), [187](Chapter_7.xhtml#_idIndexMarker562)

使用 Python 时 [38](Chapter_2.xhtml#_idIndexMarker080)

时间序列 [335](Chapter_12.xhtml#_idIndexMarker920), [336](Chapter_12.xhtml#_idIndexMarker921)

参考链接 [324](Chapter_12.xhtml#_idIndexMarker907)

强化学习（RL）[301](Chapter_11.xhtml#_idIndexMarker860)

无监督方法 [162](Chapter_6.xhtml#_idIndexMarker502), [163](Chapter_6.xhtml#_idIndexMarker504), [164](Chapter_6.xhtml#_idIndexMarker505)

时间序列 [6](Chapter_1.xhtml#_idIndexMarker010)

离线学习 [210](Chapter_8.xhtml#_idIndexMarker627)

在线学习 [210](Chapter_8.xhtml#_idIndexMarker626)

时间序列分析 [6](Chapter_1.xhtml#_idIndexMarker009)

时间序列分析（TSA）[36](Chapter_2.xhtml#_idIndexMarker068), [37](Chapter_2.xhtml#_idIndexMarker074), [38](Chapter_2.xhtml#_idIndexMarker076)

时间序列分类算法

临界差异图 [126](Chapter_4.xhtml#_idIndexMarker389), [127](Chapter_4.xhtml#_idIndexMarker390)

异构与集成嵌入森林的时间序列组合（TS-CHIEF）[121](Chapter_4.xhtml#_idIndexMarker365)

时间序列数据

示例 [2](Chapter_1.xhtml#_idIndexMarker001)

时间序列数据 [2](Chapter_1.xhtml#_idIndexMarker000)

时间序列数据集 [94](Chapter_4.xhtml#_idIndexMarker240)

时间序列预测 [97](Chapter_4.xhtml#_idIndexMarker262)

时间序列森林（TSF）[120](Chapter_4.xhtml#_idIndexMarker363)

时间序列机器学习算法

详细分类法 [125](Chapter_4.xhtml#_idIndexMarker387), [126](Chapter_4.xhtml#_idIndexMarker388)

时间序列机器学习飞轮 [38](Chapter_2.xhtml#_idIndexMarker075)

时间序列回归 [107](Chapter_4.xhtml#_idIndexMarker300)

transformer [330](Chapter_12.xhtml#_idIndexMarker913), [331](Chapter_12.xhtml#_idIndexMarker914)

transformer 架构 [277](Chapter_10.xhtml#_idIndexMarker825)

趋势 [56](Chapter_2.xhtml#_idIndexMarker130)

识别 [56](Chapter_2.xhtml#_idIndexMarker132), [57](Chapter_2.xhtml#_idIndexMarker137), [58](Chapter_2.xhtml#_idIndexMarker140), [59](Chapter_2.xhtml#_idIndexMarker145), [60](Chapter_2.xhtml#_idIndexMarker147), [61](Chapter_2.xhtml#_idIndexMarker152), [62](Chapter_2.xhtml#_idIndexMarker154), [63](Chapter_2.xhtml#_idIndexMarker156)

三重指数平滑 [142](Chapter_5.xhtml#_idIndexMarker448)

真阳性率 [114](Chapter_4.xhtml#_idIndexMarker329)

真阳性率（TPR）[115](Chapter_4.xhtml#_idIndexMarker336)

真阳性（TP）[115](Chapter_4.xhtml#_idIndexMarker334)

Twitter [170](Chapter_6.xhtml#_idIndexMarker527)

U

东安格利亚大学（UAE）[118](Chapter_4.xhtml#_idIndexMarker352)

UCR（加利福尼亚大学河滨分校）[118](Chapter_4.xhtml#_idIndexMarker348)

单元插补 [73](Chapter_3.xhtml#_idIndexMarker186), [82](Chapter_3.xhtml#_idIndexMarker221)

单变量分析 [38](Chapter_2.xhtml#_idIndexMarker078)

单变量系列 [4](Chapter_1.xhtml#_idIndexMarker004)

米纳斯吉拉斯联邦大学（UFMG）[252](Chapter_9.xhtml#_idIndexMarker746)

无监督学习[96](Chapter_4.xhtml#_idIndexMarker252), [97](Chapter_4.xhtml#_idIndexMarker263)

无监督方法

时间序列[162](Chapter_6.xhtml#_idIndexMarker501), [163](Chapter_6.xhtml#_idIndexMarker503), [164](Chapter_6.xhtml#_idIndexMarker506)

V

验证[187](Chapter_7.xhtml#_idIndexMarker563), [188](Chapter_7.xhtml#_idIndexMarker565)

基于值的学习[300](Chapter_11.xhtml#_idIndexMarker853)

变量[44](Chapter_2.xhtml#_idIndexMarker096), [45](Chapter_2.xhtml#_idIndexMarker100), [46](Chapter_2.xhtml#_idIndexMarker108), [47](Chapter_2.xhtml#_idIndexMarker109), [48](Chapter_2.xhtml#_idIndexMarker110), [49](Chapter_2.xhtml#_idIndexMarker111)

关系[49](Chapter_2.xhtml#_idIndexMarker114), [50](Chapter_2.xhtml#_idIndexMarker118), [52](Chapter_2.xhtml#_idIndexMarker121)

向量自回归模型[144](Chapter_5.xhtml#_idIndexMarker459), [145](Chapter_5.xhtml#_idIndexMarker460)

向量自回归（VAR）[129](Chapter_4.xhtml#_idIndexMarker397)

向量自回归（VAR）[133](Chapter_5.xhtml#_idIndexMarker402)

非常快速决策树（VFDT）[215](Chapter_8.xhtml#_idIndexMarker648)

虚拟环境[193](Chapter_7.xhtml#_idIndexMarker580)

W

滚动前进验证[188](Chapter_7.xhtml#_idIndexMarker564)

弱学习器[100](Chapter_4.xhtml#_idIndexMarker281)

WEASEL+MUSE[123](Chapter_4.xhtml#_idIndexMarker375)

基于窗口的特征[75](Chapter_3.xhtml#_idIndexMarker191)

Wold的分解[137](Chapter_5.xhtml#_idIndexMarker420)

时间序列分类的词提取[123](Chapter_4.xhtml#_idIndexMarker376)

Y

Yeo-Johnson变换[72](Chapter_3.xhtml#_idIndexMarker182)

Z

Z-score标准化[70](Chapter_3.xhtml#_idIndexMarker170)

# 索引
