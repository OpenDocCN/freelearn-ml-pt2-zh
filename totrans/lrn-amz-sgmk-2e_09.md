# 第 7 章：使用内置框架扩展机器学习服务

在过去的三章中，你学习了如何使用内置算法来训练和部署模型，而无需编写一行机器学习代码。然而，这些算法并没有涵盖所有的机器学习问题。在许多情况下，你需要编写自己的代码。幸运的是，几个开源框架让这个过程相对容易。

在本章中，你将学习如何使用最流行的开源机器学习和深度学习框架来训练和部署模型。我们将涵盖以下主题：

+   发现 Amazon SageMaker 中的内置框架

+   在 Amazon SageMaker 上运行你的框架代码

+   使用内置框架

让我们开始吧！

# 技术要求

你需要一个 AWS 账户来运行本章中包含的示例。如果你还没有账户，请访问 [https://aws.amazon.com/getting-started/](https://aws.amazon.com/getting-started/) 创建一个。你还应该了解 AWS 免费套餐（[https://aws.amazon.com/free/](https://aws.amazon.com/free/)），它允许你在一定的使用限制内免费使用许多 AWS 服务。

你需要为你的账户安装并配置 AWS 命令行界面（[https://aws.amazon.com/cli/](https://aws.amazon.com/cli/)）。

你需要一个可用的 Python 3.x 环境。安装 Anaconda 发行版（[https://www.anaconda.com/](https://www.anaconda.com/)）不是必须的，但强烈建议安装，因为它包含了我们需要的许多项目（Jupyter、`pandas`、`numpy` 等）。

你需要一个可用的 Docker 安装。你可以在 [https://docs.docker.com](https://docs.docker.com) 找到安装说明和相关文档。

本书中包含的代码示例可以在 GitHub 上找到，网址是 [https://github.com/PacktPublishing/Learn-Amazon-SageMaker-second-edition](https://github.com/PacktPublishing/Learn-Amazon-SageMaker-second-edition)。你需要安装一个 Git 客户端才能访问它们（[https://git-scm.com/](https://git-scm.com/)）。

# 发现 Amazon SageMaker 中的内置框架

SageMaker 让你使用以下机器学习和深度学习框架来训练和部署模型：

+   **Scikit-learn**，无疑是最广泛使用的开源机器学习库。如果你是这个领域的新手，可以从这里开始： [https://scikit-learn.org](https://scikit-learn.org)。

+   **XGBoost**，一种非常流行且多功能的开源算法，适用于回归、分类和排序问题（[https://xgboost.ai](https://xgboost.ai)）。它也作为内置算法提供，如在[*第 4 章*](B17705_04_Final_JM_ePub.xhtml#_idTextAnchor069)《训练机器学习模型》中所展示的那样。以框架模式使用它将为我们提供更多的灵活性。

+   **TensorFlow**，一个极受欢迎的深度学习开源库（[https://www.tensorflow.org](https://www.tensorflow.org)）。SageMaker 还支持受人喜爱的 **Keras** API（[https://keras.io](https://keras.io)）。

+   **PyTorch**，另一个备受欢迎的深度学习开源库（[https://pytorch.org](https://pytorch.org)）。特别是研究人员喜欢它的灵活性。

+   **Apache MXNet**，一个有趣的深度学习挑战者。它是用 C++ 原生实现的，通常比其竞争对手更快且更具可扩展性。其 **Gluon** API 提供了丰富的计算机视觉工具包（[https://gluon-cv.mxnet.io](https://gluon-cv.mxnet.io)）、**自然语言处理**（**NLP**）（[https://gluon-nlp.mxnet.io](https://gluon-nlp.mxnet.io)）和时间序列数据（[https://gluon-ts.mxnet.io](https://gluon-ts.mxnet.io)）。

+   **Chainer**，另一个值得关注的深度学习挑战者（[https://chainer.org](https://chainer.org)）。

+   **Hugging Face**，最流行的、用于自然语言处理的最前沿工具和模型集合（[https://huggingface.co](https://huggingface.co)）。

+   **强化学习**框架，如 **Intel Coach**、**Ray RLlib** 和 **Vowpal Wabbit**。由于这可能需要一本书的篇幅，我在这里不会讨论这个话题！

+   **Spark**，借助一个专用的 SDK，允许你直接从 Spark 应用程序中使用 **PySpark** 或 **Scala** 训练和部署模型（[https://github.com/aws/sagemaker-spark](https://github.com/aws/sagemaker-spark)）。

你可以在 [https://github.com/awslabs/amazon-sagemaker-examples/tree/master/sagemaker-python-sdk](https://github.com/awslabs/amazon-sagemaker-examples/tree/master/sagemaker-python-sdk) 找到这些的许多示例。

在本章中，我们将重点关注最流行的框架：XGBoost、scikit-learn、TensorFlow、PyTorch 和 Spark。

最好的入门方式是先运行一个简单的示例。如你所见，工作流与内置算法是相同的。我们将在过程中强调一些差异，稍后将在本章深入讨论。

## 使用 XGBoost 运行第一个示例

在这个例子中，我们将使用 XGBoost 内置框架构建一个二分类模型。写这篇文章时，SageMaker 支持的最新版本是 1.3-1。

我们将使用基于 `xgboost.XGBClassifier` 对象和直接营销数据集的自定义训练脚本，这个数据集我们在 [*第 3 章*](B17705_03_Final_JM_ePub.xhtml#_idTextAnchor049)，*使用 Amazon SageMaker Autopilot 进行自动化机器学习* 中用过：

1.  首先，我们下载并解压数据集：

    [PRE0]

1.  我们导入 SageMaker SDK，并为任务定义一个 S3 前缀：

    [PRE1]

1.  我们加载数据集并进行非常基础的处理（因为这不是我们的重点）。简单地对分类特征进行独热编码，将标签移动到第一列（这是 XGBoost 的要求），打乱数据集，分割为训练集和验证集，并将结果保存在两个单独的 CSV 文件中：

    [PRE2]

1.  我们将这两个文件上传到 S3：

    [PRE3]

1.  我们定义两个输入，数据格式为 CSV：

    [PRE4]

1.  为训练任务定义一个估算器。当然，我们也可以使用通用的 `Estimator` 对象，并传递 XGBoost 容器在 `XGBoost` 估算器中的名称，这样就会自动选择正确的容器：

    [PRE5]

    这里有几个熟悉的参数：角色、基础设施要求和输出路径。其他参数呢？`entry_point` 是我们训练脚本的路径（可以在本书的GitHub仓库中找到）。`hyperparameters` 会传递给训练脚本。我们还需要选择一个 `framework_version` 值；这是我们想要使用的XGBoost版本。

1.  我们照常进行训练：

    [PRE6]

1.  我们也照常进行部署，创建一个唯一的端点名称：

    [PRE7]

    然后，我们从验证集加载一些样本，并将它们以CSV格式发送进行预测。响应包含每个样本的0到1之间的分数：

    [PRE8]

    这将输出以下概率：

    [PRE9]

1.  完成后，我们删除端点：

    [PRE10]

我们在这里使用了XGBoost，但这个工作流程对于其他框架也是完全相同的。这种标准的训练和部署方式使得从内置算法切换到框架，或从一个框架切换到下一个框架变得非常简单。

我们需要重点关注的要点如下：

+   **框架容器**：它们是什么？我们能看到它们是如何构建的吗？我们可以自定义它们吗？我们能用它们在本地机器上进行训练吗？

+   **训练**：SageMaker 训练脚本与普通框架代码有何不同？它如何接收超参数？它应该如何读取输入数据？模型应该保存在哪里？

+   **部署**：模型是如何部署的？脚本需要提供一些相关代码吗？预测的输入格式是什么？

+   `entry_point` 脚本？我们可以为训练和部署添加库吗？

所有这些问题现在都将得到解答！

## 使用框架容器

SageMaker 为每个内置框架提供训练和推理容器，并定期更新到最新版本。不同的容器也可供CPU和GPU实例使用。所有这些容器统称为**深度学习容器** ([https://aws.amazon.com/machine-learning/containers](https://aws.amazon.com/machine-learning/containers))。

正如我们在前面的例子中看到的，它们允许你使用自己的代码，而无需维护定制的容器。在大多数情况下，你不需要再进一步了解容器的细节，你可以高兴地忘记它们的存在。如果这个话题目前对你来说太高级，可以暂时跳过，继续阅读*本地训练与部署*部分。

如果你感到好奇或有定制需求，你会很高兴得知这些容器的代码是开源的：

+   **Scikit-learn**: [https://github.com/aws/sagemaker-scikit-learn-container](https://github.com/aws/sagemaker-scikit-learn-container)

+   **XGBoost**: [https://github.com/aws/sagemaker-xgboost-container](https://github.com/aws/sagemaker-xgboost-container)

+   **TensorFlow，PyTorch，Apache MXNet和Hugging Face**：[https://github.com/aws/deep-learning-containers](https://github.com/aws/deep-learning-containers)

+   **Chainer**：[https://github.com/aws/sagemaker-chainer-container](https://github.com/aws/sagemaker-chainer-container)

首先，这可以帮助你理解这些容器是如何构建的，以及SageMaker是如何使用它们进行训练和预测的。你还可以执行以下操作：

+   在本地机器上构建并运行它们进行本地实验。

+   在你最喜欢的托管Docker服务上构建并运行它们，例如**Amazon ECS**、**Amazon EKS**或**Amazon Fargate**（[https://aws.amazon.com/containers](https://aws.amazon.com/containers)）。

+   自定义它们，推送到Amazon ECR，并使用SageMaker SDK中的估算器。我们将在[*第8章*](B17705_08_Final_JM_ePub.xhtml#_idTextAnchor147)中演示这一点，*使用你的算法和代码*。

这些容器有一个很好的特性。你可以与SageMaker SDK一起使用它们，在本地机器上训练和部署模型。让我们看看这个是如何工作的。

## 本地训练和部署

**本地模式**是通过SageMaker SDK训练和部署模型，而无需启动AWS中的按需托管基础设施。你将使用本地机器代替。在此上下文中，“本地”指的是运行笔记本的机器：它可以是你的笔记本电脑、本地服务器，或者一个小型**笔记本实例**。

注意

在写本文时，本地模式在SageMaker Studio中不可用。

这是一个快速实验和迭代小型数据集的极好方式。你无需等待实例启动，也无需为此支付费用！

让我们重新审视之前的XGBoost示例，重点讲解使用本地模式时所需的更改：

1.  显式设置IAM角色的名称。`get_execution_role()`在本地机器上不起作用（在笔记本实例上有效）：

    [PRE11]

1.  从本地文件加载训练和验证数据集。将模型存储在本地`/tmp`目录中：

    [PRE12]

1.  在`XGBoost`估算器中，将`instance_type`设置为`local`。对于本地GPU训练，我们将使用`local_gpu`。

1.  在`xgb_estimator.deploy()`中，将`instance_type`设置为`local`。

这就是使用与AWS大规模环境中相同的容器在本地机器上进行训练所需的一切。此容器会被拉取到本地机器，之后你将一直使用它。当你准备好大规模训练时，只需将`local`或`local_gpu`实例类型替换为适当的AWS实例类型，就可以开始训练了。

故障排除

如果遇到奇怪的部署错误，可以尝试重启Docker（`sudo service docker restart`）。我发现它在部署过程中经常被中断，尤其是在Jupyter Notebooks中工作时！

现在，让我们看看在这些容器中运行自己代码所需的条件。这个功能叫做**脚本模式**。

## 使用脚本模式进行训练

由于您的训练代码运行在 SageMaker 容器内，它需要能够执行以下操作：

+   接收传递给估算器的超参数。

+   读取输入通道中可用的数据（训练数据、验证数据等）。

+   将训练好的模型保存到正确的位置。

脚本模式是 SageMaker 使这一切成为可能的方式。该名称来自于您的代码在容器中被调用的方式。查看我们 XGBoost 作业的训练日志，我们可以看到：

[PRE13]

我们的代码像普通的 Python 脚本一样被调用（因此称为脚本模式）。我们可以看到，超参数作为命令行参数传递，这也回答了我们应该在脚本中使用什么来读取它们：`argparse`。

这是我们脚本中相应的代码片段：

[PRE14]

那么输入数据和保存模型的位置呢？如果我们稍微仔细查看日志，就会看到：

[PRE15]

这三个环境变量定义了**容器内的本地路径**，指向训练数据、验证数据和保存模型的相应位置。这是否意味着我们必须手动将数据集和模型从 S3 复制到容器中并返回？不！SageMaker 会为我们自动处理这一切。这是容器中支持代码的一部分。

我们的脚本只需要读取这些变量。我建议再次使用 `argparse`，这样我们可以在 SageMaker 之外训练时将路径传递给脚本（稍后会详细介绍）。

这是我们脚本中相应的代码片段：

[PRE16]

通道名称

`SM_CHANNEL_xxx` 变量是根据传递给 `fit()` 的通道命名的。例如，如果您的算法需要一个名为 `foobar` 的通道，您需要在 `fit()` 中将其命名为 `foobar`，并在脚本中使用 `SM_CHANNEL_FOOBAR`。在您的容器中，该通道的数据会自动保存在 `/opt/ml/input/data/foobar` 目录下。

总结一下，为了在 SageMaker 上训练框架代码，我们只需要做以下几件事：

1.  使用 `argparse` 读取作为命令行参数传递的超参数。您可能已经在代码中这样做了！

1.  读取 `SM_CHANNEL_xxx` 环境变量并从中加载数据。

1.  读取 `SM_MODEL_DIR` 环境变量并将训练好的模型保存到该位置。

现在，让我们讨论在脚本模式下部署训练好的模型。

## 理解模型部署

通常，您的脚本需要包括以下内容：

+   一个加载模型的函数。

+   一个在将输入数据传递给模型之前处理数据的函数。

+   一个在返回预测结果给调用方之前处理预测结果的函数。

所需的实际工作量取决于您使用的框架和输入格式。让我们看看这对 TensorFlow、PyTorch、MXNet、XGBoost 和 scikit-learn 意味着什么。

### 使用 TensorFlow 部署

TensorFlow 推理容器依赖于**TensorFlow Serving**模型服务器进行模型部署（[https://www.tensorflow.org/tfx/guide/serving](https://www.tensorflow.org/tfx/guide/serving)）。因此，你的训练代码必须以这种格式保存模型。模型加载和预测功能会自动提供。

JSON是预测的默认输入格式，并且由于自动序列化，它也适用于`numpy`数组。JSON Lines和CSV也被支持。对于其他格式，你可以实现自己的预处理和后处理函数，`input_handler()`和`output_handler()`。你可以在[https://sagemaker.readthedocs.io/en/stable/using_tf.html#deploying-from-an-estimator](https://sagemaker.readthedocs.io/en/stable/using_tf.html#deploying-from-an-estimator)找到更多信息。

你也可以深入了解TensorFlow推理容器，访问[https://github.com/aws/deep-learning-containers/tree/master/tensorflow/inference](https://github.com/aws/deep-learning-containers/tree/master/tensorflow/inference)。

### 使用PyTorch进行部署

PyTorch推理容器依赖于`__call__()`方法。如果没有，你应该在推理脚本中提供`predict_fn()`函数。

对于预测，`numpy`是默认的输入格式。JSON Lines和CSV也被支持。对于其他格式，你可以实现自己的预处理和后处理函数。你可以在[https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/using_pytorch.html#serve-a-pytorch-model](https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/using_pytorch.html#serve-a-pytorch-model)找到更多信息。

你可以深入了解PyTorch推理容器，访问[https://github.com/aws/deep-learning-containers/tree/master/pytorch/inference](https://github.com/aws/deep-learning-containers/tree/master/pytorch/inference)。

### 使用Apache MXNet进行部署

Apache MXNet 推理容器依赖于**多模型服务器**（**MMS**）进行模型部署（[https://github.com/awslabs/multi-model-server](https://github.com/awslabs/multi-model-server)）。它使用默认的MXNet模型格式。

基于`Module` API的模型不需要模型加载函数。对于预测，它们支持JSON、CSV或`numpy`格式的数据。

Gluon模型确实需要一个模型加载函数，因为参数需要显式初始化。数据可以通过JSON或`numpy`格式发送。

对于其他数据格式，你可以实现自己的预处理、预测和后处理函数。你可以在[https://sagemaker.readthedocs.io/en/stable/using_mxnet.html](https://sagemaker.readthedocs.io/en/stable/using_mxnet.html)找到更多信息。

你可以深入了解MXNet推理容器，访问[https://github.com/aws/deep-learning-containers/tree/master/mxnet/inference/docker](https://github.com/aws/deep-learning-containers/tree/master/mxnet/inference/docker)。

### 使用XGBoost和scikit-learn进行部署

同样，XGBoost和scikit-learn分别依赖于[https://github.com/aws/sagemaker-xgboost-container](https://github.com/aws/sagemaker-xgboost-container)和[https://github.com/aws/sagemaker-scikit-learn-container](https://github.com/aws/sagemaker-scikit-learn-container)。

你的脚本需要提供以下内容：

+   一个`model_fn()`函数用于加载模型。与训练类似，加载模型的位置通过`SM_MODEL_DIR`环境变量传递。

+   两个可选函数，用于反序列化和序列化预测数据，分别命名为`input_fn()`和`output_fn()`。只有当你需要其他格式的输入数据（例如非JSON、CSV或`numpy`）时，才需要这些函数。

+   一个可选的`predict_fn()`函数将反序列化的数据传递给模型并返回预测结果。仅当你需要在预测之前对数据进行预处理，或对预测结果进行后处理时才需要这个函数。

对于XGBoost和scikit-learn，`model_fn()`函数非常简单且通用。以下是一些大多数情况下都能正常工作的示例：

[PRE17]

SageMaker还允许你导入和导出模型。你可以将现有模型上传到S3并直接在SageMaker上部署。同样，你也可以将训练好的模型从S3复制到其他地方进行部署。我们将在[*第11章*](B17705_11_Final_JM_ePub.xhtml#_idTextAnchor237)，《部署机器学习模型》中详细介绍这一点。

现在，让我们来讨论训练和部署所需的依赖关系。

## 管理依赖关系

在许多情况下，你需要向框架的容器中添加额外的源文件和库。让我们看看如何轻松做到这一点。

### 添加训练所需的源文件

默认情况下，所有估算器都会从当前目录加载入口脚本。如果你需要额外的源文件来进行训练，估算器允许你传递一个`source_dir`参数，指向存储额外文件的目录。请注意，入口脚本必须位于同一目录中。

在以下示例中，`myscript.py`和所有额外的源文件必须放在`src`目录中。SageMaker将自动打包该目录并将其复制到训练容器中：

[PRE18]

### 添加训练所需的库

你可以使用不同的技术来添加训练所需的库。

对于可以通过`pip`安装的库，最简单的方法是将`requirements.txt`文件与入口脚本放在同一文件夹中。SageMaker会自动在容器内安装这些库。

另外，你可以通过在训练脚本中执行`pip install`命令，使用`pip`直接安装库。我们在[*第6章*](B17705_06_Final_JM_ePub.xhtml#_idTextAnchor108)，《训练自然语言处理模型》中使用了这个方法，处理了LDA和NTM。这个方法在你不想或者不能修改启动训练作业的SageMaker代码时非常有用：

[PRE19]

对于不能通过 `pip` 安装的库，你应该使用 `dependencies` 参数。这个参数在所有估算器中都可用，它允许你列出要添加到训练作业中的库。这些库需要在本地、虚拟环境或特定目录中存在。SageMaker 会将它们打包并复制到训练容器中。

在以下示例中，`myscript.py` 需要 `mylib` 库。我们将在 `lib` 本地目录中安装它：

[PRE20]

然后，我们将其位置传递给估算器：

[PRE21]

最后的技术是将库安装到 Dockerfile 中的容器里，重建镜像并将其推送到 Amazon ECR。如果在预测时也需要这些库（例如，用于预处理），这是最好的选择。

### 为部署添加库

如果你需要在预测时提供特定的库，可以使用一个 `requirements.txt` 文件，列出那些可以通过 `pip` 安装的库。

对于其他库，唯一的选择是自定义框架容器。你可以通过`image_uri`参数将其名称传递给估算器：

[PRE22]

我们在本节中涵盖了许多技术主题。现在，让我们来看一下大局。

## 将所有内容整合在一起

使用框架时的典型工作流程如下所示：

1.  在你的代码中实现脚本模式；也就是说，读取必要的超参数、输入数据和输出位置。

1.  如有需要，添加一个 `model_fn()` 函数来加载模型。

1.  在本地测试你的训练代码，避免使用任何 SageMaker 容器。

1.  配置适当的估算器（如`XGBoost`、`TensorFlow`等）。

1.  使用估算器在本地模式下训练，使用内置容器或你自定义的容器。

1.  在本地模式下部署并测试你的模型。

1.  切换到托管实例类型（例如，`ml.m5.large`）进行训练和部署。

这个逻辑进展每一步所需的工作很少。它最大程度地减少了摩擦、错误的风险和挫败感。它还优化了实例时间和成本——如果你的代码因为一个小错误立即崩溃，就不必等待并支付托管实例的费用。

现在，让我们开始运用这些知识。在接下来的部分中，我们将运行一个简单的 scikit-learn 示例。目的是确保我们理解刚刚讨论的工作流程。

# 在 Amazon SageMaker 上运行你的框架代码

我们将从一个简单的 scikit-learn 程序开始，该程序在波士顿住房数据集上训练并保存一个线性回归模型，数据集在[*第4章*](B17705_04_Final_JM_ePub.xhtml#_idTextAnchor069)中使用过，*训练机器学习模型*：

[PRE23]

让我们更新它，使其可以在 SageMaker 上运行。

### 实现脚本模式

现在，我们将使用框架实现脚本模式，如下所示：

1.  首先，读取命令行参数中的超参数：

    [PRE24]

1.  将输入和输出路径作为命令行参数读取。我们可以决定去除拆分代码，改为传递两个输入通道。我们还是坚持使用一个通道，也就是`training`：

    [PRE25]

1.  由于我们使用的是scikit-learn，我们需要添加`model_fn()`以便在部署时加载模型：

    [PRE26]

到此为止，我们完成了。是时候测试了！

### 本地测试

首先，我们在本地机器上的Python 3环境中测试我们的脚本，不依赖任何SageMaker容器。我们只需要确保安装了`pandas`和scikit-learn。

我们将环境变量设置为空值，因为我们将在命令行上传递路径：

[PRE27]

很好。我们的代码在命令行参数下运行得很顺利。我们可以使用它进行本地开发和调试，直到我们准备好将其迁移到SageMaker本地模式。

### 使用本地模式

我们将按照以下步骤开始：

1.  仍然在我们的本地机器上，我们配置一个`SKLearn`估算器以本地模式运行，根据我们使用的设置来设定角色。只使用本地路径：

    [PRE28]

1.  如预期的那样，我们可以在训练日志中看到如何调用我们的代码。当然，我们得到的是相同的结果：

    [PRE29]

1.  我们在本地部署并发送一些CSV样本进行预测：

    [PRE30]

    通过打印响应，我们将看到预测值：

    [PRE31]

    使用本地模式，我们可以快速迭代模型。我们仅受限于本地机器的计算和存储能力。当达到限制时，我们可以轻松迁移到托管基础设施。

### 使用托管基础设施

当需要进行大规模训练并在生产环境中部署时，我们只需确保输入数据存储在S3中，并将“本地”实例类型替换为实际的实例类型：

[PRE32]

由于我们使用的是相同的容器，我们可以放心训练和部署会按预期工作。再次强调，我强烈建议您遵循以下逻辑流程：首先进行本地工作，然后是SageMaker本地模式，最后是SageMaker托管基础设施。这将帮助你集中精力处理需要做的事以及何时做。

在本章的其余部分，我们将运行更多示例。

# 使用内建框架

我们已经覆盖了XGBoost和scikit-learn。现在，是时候看看如何使用深度学习框架了。让我们从TensorFlow和Keras开始。

## 使用TensorFlow和Keras

在这个示例中，我们将使用TensorFlow 2.4.1来训练一个简单的卷积神经网络，数据集使用Fashion-MNIST（[https://github.com/zalandoresearch/fashion-mnist](https://github.com/zalandoresearch/fashion-mnist)）。

我们的代码分成了两个源文件：一个是入口点脚本（`fmnist.py`），另一个是模型（`model.py`，基于Keras层）。为了简洁起见，我只讨论SageMaker的步骤。你可以在本书的GitHub仓库中找到完整代码：

1.  `fmnist.py`首先从命令行读取超参数：

    [PRE33]

1.  接下来，我们读取环境变量，即训练集和验证集的输入路径、模型的输出路径以及实例上可用的 GPU 数量。这是我们第一次使用后者。它对于调整多 GPU 训练的批量大小非常有用，因为通常做法是将初始批量大小乘以 GPU 数量：

    [PRE34]

1.  将参数存储在本地变量中。然后，加载数据集。每个通道为我们提供一个压缩的`numpy`数组，用于存储图像和标签：

    [PRE35]

1.  然后，通过重新调整图像张量的形状、标准化像素值、进行独热编码图像标签，并创建将数据传输给模型的`tf.data.Dataset`对象，为训练准备数据。

1.  创建模型、编译并拟合它。

1.  训练完成后，将模型以 TensorFlow Serving 格式保存到适当的输出位置。此步骤非常重要，因为这是 SageMaker 用于 TensorFlow 模型的模型服务器：

    [PRE36]

我们使用常规工作流程训练和部署模型：

1.  在一个由 TensorFlow 2 内核支持的笔记本中，我们下载数据集并将其上传到 S3：

    [PRE37]

1.  我们配置`TensorFlow`估算器。我们还设置`source_dir`参数，以便将模型文件也部署到容器中：

    [PRE38]

1.  像往常一样训练和部署。我们将直接使用托管基础设施，但相同的代码也可以在本地模式下在你的本地机器上正常运行：

    [PRE39]

1.  验证准确率应为 91-92%。通过加载并显示一些验证数据集中的样本图像，我们可以预测它们的标签。`numpy`负载会自动序列化为 JSON，这是预测数据的默认格式：

    [PRE40]

    输出应如下所示：

    ![图 7.1 – 查看预测类别

    ](img/B17705_07_1.jpg)

    图 7.1 – 查看预测类别

1.  完成后，我们删除端点：

    [PRE41]

如你所见，脚本模式与内置容器的结合使得在 SageMaker 上运行 TensorFlow 变得非常简单。一旦进入常规流程，你会惊讶于将模型从笔记本电脑迁移到 AWS 的速度有多快。

现在，让我们来看一下 PyTorch。

## 使用 PyTorch

PyTorch 在计算机视觉、自然语言处理等领域非常流行。

在此示例中，我们将训练一个**图神经网络**（**GNN**）。这种类型的网络在图结构数据上表现特别好，如社交网络、生命科学等。事实上，我们的 PyTorch 代码将使用**深度图书馆**（**DGL**），这是一个开源库，可以更轻松地使用 TensorFlow、PyTorch 和 Apache MXNet 构建和训练 GNN（[https://www.dgl.ai/](https://www.dgl.ai/)）。DGL 已经安装在这些容器中，所以我们可以直接开始工作。

我们将使用 Zachary 空手道俱乐部数据集（[http://konect.cc/networks/ucidata-zachary/](http://konect.cc/networks/ucidata-zachary/))。以下是该图的内容：

![图 7.2 – Zachary 空手道俱乐部数据集

](img/B17705_07_2.jpg)

图 7.2 – Zachary空手道俱乐部数据集

节点0和33是教师，而其他节点是学生。边表示这些人之间的关系。故事是这样的，两位老师发生了争执，俱乐部需要被分成两部分。

训练任务的目的是找到“最佳”分割。这可以定义为一个半监督分类任务。第一位老师（节点0）被分配为类别0，而第二位老师（节点33）被分配为类别1。所有其他节点是未标记的，它们的类别将由**图卷积网络**计算。在最后一个周期结束时，我们将提取节点类别，并根据这些类别来分割俱乐部。

数据集被存储为一个包含边的pickle格式的Python列表。以下是前几条边：

[PRE42]

SageMaker的代码简洁明了。我们将数据集上传到S3，创建一个`PyTorch`估算器，然后进行训练：

[PRE43]

这一点几乎无需任何解释，对吧？

让我们来看一下简化的训练脚本，在这里我们再次使用了脚本模式。完整版本可在本书的GitHub仓库中找到：

[PRE44]

预测了以下类别。节点0和1是类别0，节点2是类别1，依此类推：

[PRE45]

通过绘制它们，我们可以看到俱乐部已经被干净利落地分开了：

![图 7.3 – 查看预测类别

](img/B17705_07_3.jpg)

图 7.3 – 查看预测类别

再次强调，SageMaker的代码并不会妨碍你。工作流和API在各个框架之间保持一致，你可以专注于机器学习问题本身。现在，让我们做一个新的示例，使用Hugging Face，同时还会看到如何通过内建的PyTorch容器部署一个PyTorch模型。

## 与Hugging Face合作

**Hugging Face** ([https://huggingface.co](https://huggingface.co)) 已迅速成为最受欢迎的自然语言处理开源模型集合。在撰写本文时，他们托管了近10,000个最先进的模型（[https://huggingface.co/models](https://huggingface.co/models)），并在超过250种语言的预训练数据集上进行了训练（[https://huggingface.co/datasets](https://huggingface.co/datasets)）。

为了快速构建高质量的自然语言处理应用，Hugging Face积极开发了三个开源库：

+   **Transformers**：使用Hugging Face模型进行训练、微调和预测（[https://github.com/huggingface/transformers](https://github.com/huggingface/transformers)）。

+   **Datasets**：下载并处理Hugging Face数据集（[https://github.com/huggingface/datasets](https://github.com/huggingface/datasets)）。

+   **Tokenizers**：为Hugging Face模型的训练和预测进行文本标记化（[https://github.com/huggingface/tokenizers](https://github.com/huggingface/tokenizers)）。

    Hugging Face教程

    如果你完全是Hugging Face的新手，请先通过他们的教程：[https://huggingface.co/transformers/quicktour.html](https://huggingface.co/transformers/quicktour.html)。

SageMaker在2021年3月增加了对Hugging Face的支持，涵盖TensorFlow和PyTorch。正如你所料，你可以使用`HuggingFace`估算器和内置容器。让我们运行一个示例，构建一个英文客户评论的情感分析模型。为此，我们将微调一个**DistilBERT**模型（[https://arxiv.org/abs/1910.01108](https://arxiv.org/abs/1910.01108)），该模型使用PyTorch实现，并在两个大型英语语料库（Wikipedia和BookCorpus数据集）上进行了预训练。

### 准备数据集

在这个例子中，我们将使用一个名为`generated_reviews_enth`的Hugging Face数据集（[https://huggingface.co/datasets/generated_reviews_enth](https://huggingface.co/datasets/generated_reviews_enth)）。它包含英文评论、其泰语翻译、一个标记指示翻译是否正确，以及星级评分：

[PRE46]

这是DistilBERT分词器所期望的格式：一个`labels`变量（`0`代表负面情绪，`1`代表正面情绪）和一个`text`变量，包含英文评论：

[PRE47]

让我们开始吧！我将向你展示每个步骤，你也可以在这本书的GitHub仓库中找到一个**SageMaker处理**版本：

1.  我们首先安装`transformers`和`datasets`库：

    [PRE48]

1.  我们下载了数据集，该数据集已经分为训练集（141,369个实例）和验证集（15,708个实例）。所有数据均为JSON格式：

    [PRE49]

1.  在每个评论中，我们创建一个名为`labels`的新变量。当`review_star`大于或等于4时，我们将其设置为`1`，否则设置为`0`：

    [PRE50]

1.  这些评论是嵌套的JSON文档，这使得移除我们不需要的变量变得困难。让我们将两个数据集扁平化：

    [PRE51]

1.  现在我们可以轻松丢弃不需要的变量。我们还将`translation.en`变量重命名为`text`：

    [PRE52]

现在，训练和验证实例已经具有DistilBERT分词器所期望的格式。我们已经在[*第6章*](B17705_06_Final_JM_ePub.xhtml#_idTextAnchor108)，*训练自然语言处理模型*中讲解了分词。一个显著的区别是，我们使用的是一个在与模型相同的英语语料库上预训练的分词器：

1.  我们下载我们预训练模型的分词器：

    [PRE53]

1.  我们对两个数据集进行分词。单词和标点符号会被相应的标记替换。如果需要，每个序列会被填充或截断，以适应模型的输入层（512个标记）：

    [PRE54]

1.  我们丢弃`text`变量，因为它不再需要：

    [PRE55]

1.  打印出一个实例，我们可以看到注意力掩码（全为1，意味着输入序列中的没有标记被掩盖）、输入ID（标记序列）和标签：

    [PRE56]

数据准备工作完成。接下来我们进入模型训练阶段。

### 微调Hugging Face模型

我们不打算从头开始训练：那样会花费太长时间，而且我们可能数据量也不足。相反，我们将对模型进行微调。从一个在大规模文本语料库上训练的模型开始，我们只会在我们自己的数据上再训练一个epoch，以便模型能够学习到数据中的特定模式：

1.  我们首先将两个数据集上传到S3。`datasets`库提供了一个方便的API来实现这一点：

    [PRE57]

1.  我们定义超参数并配置一个`HuggingFace`估算器。请注意，我们将仅对模型进行一个epoch的微调：

    [PRE58]

    为了简洁起见，我不会讨论训练脚本（`train.py`），该脚本可以在本书的GitHub仓库中找到。它没有特别之处：我们使用`Trainer` Hugging Face API，并通过脚本模式与SageMaker进行接口。由于我们只训练一个epoch，因此禁用了检查点保存（`save_strategy='no'`）。这有助于缩短训练时间（不保存检查点）和部署时间（模型工件较小）。

1.  还值得注意的是，你可以在Hugging Face网站上为你的估算器生成模板代码。如以下截图所示，你可以点击**Amazon SageMaker**，选择任务类型，然后复制并粘贴生成的代码：![图7.4 – 在Hugging Face网站上查看我们的模型](img/B17705_07_4.jpg)

    ](img/B17705_07_4.jpg)

    图7.4 – 在Hugging Face网站上查看我们的模型

1.  我们像往常一样启动训练作业，持续了大约42分钟：

    [PRE59]

就像其他框架一样，我们可以调用`deploy()` API来将模型部署到SageMaker端点。你可以在[https://aws.amazon.com/blogs/machine-learning/announcing-managed-inference-for-hugging-face-models-in-amazon-sagemaker/](https://aws.amazon.com/blogs/machine-learning/announcing-managed-inference-for-hugging-face-models-in-amazon-sagemaker/)找到一个示例。

相反，让我们看看如何使用内置的PyTorch容器和**TorchServe**部署我们的模型。实际上，这个部署示例可以推广到任何你希望通过TorchServe提供的PyTorch模型。

我发现我同事Todd Escalona写的这篇精彩博客文章在理解如何通过TorchServe提供PyTorch模型方面非常有帮助：[https://aws.amazon.com/blogs/machine-learning/serving-pytorch-models-in-production-with-the-amazon-sagemaker-native-torchserve-integration/](https://aws.amazon.com/blogs/machine-learning/serving-pytorch-models-in-production-with-the-amazon-sagemaker-native-torchserve-integration/)。

### 部署一个Hugging Face模型

与之前的示例相比，唯一的区别是我们必须使用S3中的模型工件来创建一个`PyTorchModel`对象，并构建一个`Predictor`模型，我们可以在其上使用`deploy()`和`predict()`。

1.  从模型工件开始，我们定义一个`Predictor`对象，然后用它创建一个`PyTorchModel`：

    [PRE60]

1.  聚焦于推理脚本（`torchserve-predictor.py`），我们编写了一个模型加载函数，解决 Hugging Face 特有的 PyTorch 容器无法默认处理的情况：

    [PRE61]

1.  我们还添加了一个返回文本标签的预测函数：

    [PRE62]

1.  推理脚本还包括基本的 `input_fn()` 和 `output_fn()` 函数，用于检查数据是否为 JSON 格式。你可以在本书的 GitHub 仓库中找到相关代码。

1.  回到我们的笔记本，我们像往常一样部署模型：

    [PRE63]

1.  一旦端点启动，我们预测一个文本样本并打印结果：

    [PRE64]

1.  最后，我们删除端点：

    [PRE65]

如你所见，使用 Hugging Face 模型非常简单。这也是一种具有成本效益的构建高质量 NLP 模型的方式，因为我们通常只需对模型进行非常少的训练周期（epoch）微调。

为了结束本章，让我们看看 SageMaker 和 Apache Spark 如何协同工作。

## 使用 Apache Spark

除了我们一直在使用的 Python SageMaker SDK，SageMaker 还包括 Spark 的 SDK（[https://github.com/aws/sagemaker-spark](https://github.com/aws/sagemaker-spark)）。这使你能够直接从运行在 Spark 集群上的 PySpark 或 Scala 应用程序中运行 SageMaker 作业。

### 结合 Spark 和 SageMaker

首先，你可以将**提取-转换-加载**（**ETL**）步骤与机器学习步骤解耦。每个步骤通常有不同的基础设施需求（实例类型、实例数量、存储），这些需求需要在技术上和财务上都能满足。为 ETL 设置合适的 Spark 集群，并在 SageMaker 上使用按需基础设施进行训练和预测是一个强大的组合。

其次，尽管 Spark 的 MLlib 是一个令人惊叹的库，但你可能还需要其他工具，如不同语言的自定义算法或深度学习。

最后，将模型部署到 Spark 集群进行预测可能不是最佳选择。应考虑使用 SageMaker 端点，尤其是因为它们支持**MLeap**格式（[https://combust.github.io/mleap-docs/](https://combust.github.io/mleap-docs/)）。

在以下示例中，我们将结合 SageMaker 和 Spark 构建一个垃圾邮件检测模型。数据将托管在 S3 中，垃圾邮件和非垃圾邮件（“ham”）各有一个文本文件。我们将使用在 Amazon EMR 集群上运行的 Spark 进行数据预处理。然后，我们将使用 SageMaker 中可用的 XGBoost 算法训练并部署模型。最后，我们将在 Spark 集群上进行预测。为了语言的多样性，这次我们使用 Scala 编写代码。

首先，我们需要构建一个 Spark 集群。

### 创建 Spark 集群

我们将如下方式创建集群：

1.  从 `sagemaker-cluster` 开始，再次点击**下一步**，然后点击**创建集群**。你可以在[https://docs.aws.amazon.com/emr/](https://docs.aws.amazon.com/emr/)找到更多详细信息。

1.  在集群创建过程中，我们在左侧垂直菜单的**Notebooks**条目中定义我们的Git仓库。然后，我们点击**Add repository**：![图7.6 – 添加Git仓库

    ](img/B17705_07_6.jpg)

    图7.6 – 添加Git仓库

1.  然后，我们创建一个连接到集群的Jupyter笔记本。从左侧垂直菜单中的**Notebooks**条目开始，如下图所示，我们为其命名，并选择我们刚刚创建的EMR集群和仓库。然后，我们点击**Create notebook**：![图7.7 – 创建Jupyter笔记本

    ](img/B17705_07_7.jpg)

    图7.7 – 创建Jupyter笔记本

1.  一旦集群和笔记本准备好，我们可以点击**Open in Jupyter**，这将带我们进入熟悉的Jupyter界面。

一切准备就绪。让我们编写一个垃圾邮件分类器！

### 使用Spark和SageMaker构建垃圾邮件分类模型

在这个示例中，我们将利用Spark和SageMaker的结合优势，通过几行Scala代码来训练、部署和预测垃圾邮件分类模型：

1.  首先，我们需要确保数据集已在S3中可用。在本地机器上，将这两个文件上传到默认的SageMaker桶（也可以使用其他桶）：

    [PRE66]

1.  返回到Jupyter笔记本，确保它正在运行Spark内核。然后，从Spark MLlib和SageMaker SDK中导入必要的对象。

1.  从S3加载数据。将所有句子转换为小写字母。然后，移除所有标点符号和数字，并修剪掉任何空白字符：

    [PRE67]

1.  然后，将消息拆分成单词，并将这些单词哈希到200个桶中。这个技术比我们在[*第6章*](B17705_06_Final_JM_ePub.xhtml#_idTextAnchor108)，“*训练自然语言处理模型*”中使用的单词向量简单得多，但应该能奏效：

    [PRE68]

    例如，以下消息中，桶15中的单词出现了一次，桶83中的单词出现了一次，桶96中的单词出现了两次，桶188中的单词也出现了两次：

    [PRE69]

1.  我们为垃圾邮件消息分配`1`标签，为正常邮件消息分配`0`标签：

    [PRE70]

1.  合并消息并将其编码为**LIBSVM**格式，这是**XGBoost**支持的格式之一：

    [PRE71]

    现在样本看起来类似于这样：

    [PRE72]

1.  将数据分为训练集和验证集：

    [PRE73]

1.  配置SageMaker SDK中可用的XGBoost估算器。在这里，我们将一次性训练并部署：

    [PRE74]

1.  启动一个训练任务和一个部署任务，在托管基础设施上，就像我们在[*第4章*](B17705_04_Final_JM_ePub.xhtml#_idTextAnchor069)，“*训练机器学习模型*”中使用内置算法时那样。SageMaker SDK会自动将Spark DataFrame传递给训练任务，因此我们无需做任何工作：

    [PRE75]

    正如你所期待的，这些活动将在SageMaker Studio的**Experiments**部分中可见。

1.  部署完成后，转换测试集并对模型进行评分。这会自动调用 SageMaker 端点。再次提醒，我们无需担心数据迁移：

    [PRE76]

    准确率应该在 97% 左右，表现得还不错！

1.  完成后，删除作业创建的所有 SageMaker 资源。这将删除模型、端点以及端点配置（一个我们还没有讨论过的对象）：

    [PRE77]

1.  别忘了终止笔记本和 EMR 集群。你可以在 EMR 控制台轻松完成这一步。

这个示例演示了如何轻松地结合 Spark 和 SageMaker 各自的优势。另一种方法是构建包含 Spark 和 SageMaker 阶段的 MLlib 流水线。你可以在 [https://github.com/awslabs/amazon-sagemaker-examples/tree/master/sagemaker-spark](https://github.com/awslabs/amazon-sagemaker-examples/tree/master/sagemaker-spark) 找到相关示例。

# 摘要

开源框架，如 scikit-learn 和 TensorFlow，简化了机器学习和深度学习代码的编写。它们在开发者社区中非常受欢迎，原因显而易见。然而，管理训练和部署基础设施仍然需要大量努力和技能，而这些通常是数据科学家和机器学习工程师所不具备的。SageMaker 简化了整个过程。你可以迅速从实验阶段过渡到生产环境，无需担心基础设施问题。

在本章中，你了解了 SageMaker 中用于机器学习和深度学习的不同框架，以及如何自定义它们的容器。你还学习了如何使用脚本模式和本地模式进行快速迭代，直到你准备好在生产环境中部署。最后，你运行了几个示例，其中包括一个结合了 Apache Spark 和 SageMaker 的示例。

在下一章中，你将学习如何在 SageMaker 上使用你自己的自定义代码，而无需依赖内置容器。
