# 12

# Implementing a Multilayer Artificial Neural Network from Scratch

As you may know, deep learning is getting a lot of attention from the press and is without any doubt the hottest topic in the machine learning field. Deep learning can be understood as a subfield of machine learning that is concerned with training artificial neural networks (NNs) with many layers efficiently. In this chapter, you will learn the basic concepts of artificial NNs so that you are well equipped for the following chapters, which will introduce advanced Python-based deep learning libraries and **deep neural network** (**DNN**) architectures that are particularly well suited for image and text analyses.

The topics that we will cover in this chapter are as follows:

*   Gaining a conceptual understanding of multilayer NNs
*   Implementing the fundamental backpropagation algorithm for NN training from scratch
*   Training a basic multilayer NN for image classification

# Modeling complex functions with artificial neural networks

At the beginning of this book, we started our journey through machine learning algorithms with artificial neurons in *Chapter 2*, *Training Simple Machine Learning Algorithms for Classification*. Artificial neurons represent the building blocks of the multilayer artificial NNs that we will discuss in this chapter.

The basic concept behind artificial NNs was built upon hypotheses and models of how the human brain works to solve complex problem tasks. Although artificial NNs have gained a lot of popularity in recent years, early studies of NNs go back to the 1940s when Warren McCulloch and Walter Pitts first described how neurons could work. (*A logical calculus of the ideas immanent in nervous activity*, *W. S. McCulloch* and *W. Pitts*. *The Bulletin of Mathematical Biophysics*, 5(4):115–133, 1943.)

However, in the decades that followed the first implementation of the **McCulloch-Pitts neuron** model—Rosenblatt's perceptron in the 1950s — many researchers and machine learning practitioners slowly began to lose interest in NNs since no one had a good solution for training an NN with multiple layers. Eventually, interest in NNs was rekindled in 1986 when D.E. Rumelhart, G.E. Hinton, and R.J. Williams were involved in the (re)discovery and popularization of the backpropagation algorithm to train NNs more efficiently, which we will discuss in more detail later in this chapter (*Learning representations by back-propagating errors*, *D. E. Rumelhart*, *G. E. Hinton*, *R. J. Williams*, *Nature*, 323 (6088): 533–536, *1986*). Readers who are interested in the history of **artificial intelligence** (**AI**), machine learning, and NNs are also encouraged to read the Wikipedia article on the so-called *AI winters*, which are the periods of time where a large portion of the research community lost interest in the study of NNs ([https://en.wikipedia.org/wiki/AI_winter](https://en.wikipedia.org/wiki/AI_winter)).

However, NNs are more popular today than ever thanks to the many major breakthroughs that have been made in the previous decade, which resulted in what we now call deep learning algorithms and architectures—NNs that are composed of many layers. NNs are a hot topic not only in academic research but also in big technology companies, such as Facebook, Microsoft, Amazon, Uber, and Google, that invest heavily in artificial NNs and deep learning research.

As of today, complex NNs powered by deep learning algorithms are considered the state-of-the-art solutions for complex problem solving such as image and voice recognition. Popular examples of the products in our everyday life that are powered by deep learning are Google's image search and Google Translate—an application for smartphones that can automatically recognize text in images for real-time translation into more than 20 languages.

Many exciting applications of DNNs have been developed at major tech companies and the pharmaceutical industry as listed in the following, non-comprehensive list of examples:

*   Facebook's DeepFace for tagging images (*DeepFace: Closing the Gap to Human-Level Performance in Face Verification*, *Y. Taigman*, *M. Yang*, *M. Ranzato*, and *L. Wolf*, *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 1701–1708, *2014*)
*   Baidu's DeepSpeech, which is able to handle voice queries in Mandarin (*DeepSpeech: Scaling up end-to-end speech recognition*, *A. Hannun*, *C. Case*, *J. Casper*, *B. Catanzaro*, *G. Diamos*, *E. Elsen*, *R. Prenger*, *S. Satheesh*, *S. Sengupta*, *A. Coates*, and *Andrew Y. Ng*, arXiv preprint arXiv:1412.5567, *2014*)
*   Google's new language translation service (*Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation*, arXiv preprint arXiv:1412.5567, *2016*)
*   Novel techniques for drug discovery and toxicity prediction (*Toxicity prediction using Deep Learning*, *T. Unterthiner*, *A. Mayr*, *G. Klambauer*, and *S. Hochreiter*, arXiv preprint arXiv:1503.01445, *2015*)
*   A mobile application that can detect skin cancer with an accuracy similar to professionally trained dermatologists (*Dermatologist-level classification of skin cancer with deep neural networks*, *A. Esteva*, *B.Kuprel*, *R. A. Novoa*, *J. Ko*, *S. M. Swetter*, *H. M. Blau*, and *S.Thrun*, in *Nature* 542, no. 7639, *2017*, pages 115-118)
*   Protein 3D structure prediction from gene sequences (*De novo structure prediction with deep-learning based scoring*, *R. Evans, J. Jumper, J. Kirkpatrick, L. Sifre, T.F.G. Green, C. Qin, A. Zidek, A. Nelson, A. Bridgland, H. Penedones, S. Petersen, K. Simonyan, S. Crossan, D.T. Jones, D. Silver, K. Kavukcuoglu, D. Hassabis, and A.W. Senior*, in *Thirteenth Critical Assessment of Techniques for Protein Structure Prediction*, 1-4 December, *2018*)
*   Learning how to drive in dense traffic from purely observational data such as camera video streams (*Model-predictive policy learning with uncertainty regularization for driving in dense traffic, M. Henaff, A. Canziani, Y. LeCun, 2019*, in *Conference Proceedings of the International Conference on Learning Representations*, *ICLR*, 2019)

## Single-layer neural network recap

This chapter is all about multilayer NNs, how they work, and how to train them to solve complex problems. However, before we dig deeper into a particular multilayer NN architecture, let's briefly reiterate some of the concepts of single-layer NNs that we introduced in *Chapter 2*, *Training Simple Machine Learning Algorithms for Classification*, namely, the **ADAptive LInear NEuron** (**Adaline**) algorithm, which is shown in the following figure:

![](img/B13208_12_01.png)

In *Chapter 2*, *Training Simple Machine Learning Algorithms for Classification*, we implemented the Adaline algorithm to perform binary classification, and we used the gradient descent optimization algorithm to learn the weight coefficients of the model. In every epoch (pass over the training dataset), we updated the weight vector *w* using the following update rule:

![](img/B13208_12_001.png)

In other words, we computed the gradient based on the whole training dataset and updated the weights of the model by taking a step into the opposite direction of the gradient ![](img/B13208_12_002.png). In order to find the optimal weights of the model, we optimized an objective function that we defined as the **sum of squared errors** (**SSE**) cost function ![](img/B13208_12_003.png). Furthermore, we multiplied the gradient by a factor, the **learning rate** ![](img/B13208_12_004.png), which we had to choose carefully to balance the speed of learning against the risk of overshooting the global minimum of the cost function.

In gradient descent optimization, we updated all weights simultaneously after each epoch, and we defined the partial derivative for each weight ![](img/B13208_12_005.png) in the weight vector *w* as follows:

![](img/B13208_12_006.png)

Here, ![](img/B13208_12_007.png) is the target class label of a particular sample ![](img/B13208_12_008.png), and ![](img/B13208_12_009.png) is the activation of the neuron, which is a linear function in the special case of Adaline.

Furthermore, we defined the activation function ![](img/B13208_12_010.png) as follows:

![](img/B13208_12_011.png)

Here, the net input, *z*, is a linear combination of the weights that are connecting the input layer to the output layer:

![](img/B13208_12_012.png)

While we used the activation ![](img/B13208_12_013.png) to compute the gradient update, we implemented a threshold function to squash the continuous valued output into binary class labels for prediction:

![](img/B13208_12_014.png)

**Single-layer naming convention**

Note that although Adaline consists of two layers, one input layer and one output layer, it is called a single-layer network because of its single link between the input and output layers.

Also, we learned about a certain *trick* to accelerate the model learning, the so-called **stochastic gradient descent** (**SGD**) optimization. SGD approximates the cost from a single training sample (online learning) or a small subset of training examples (mini-batch learning). We will make use of this concept later in this chapter when we implement and train a multilayer perceptron (MLP). Apart from faster learning—due to the more frequent weight updates compared to gradient descent—its noisy nature is also regarded as beneficial when training multilayer NNs with nonlinear activation functions, which do not have a convex cost function. Here, the added noise can help to escape local cost minima, but we will discuss this topic in more detail later in this chapter.

## Introducing the multilayer neural network architecture

In this section, you will learn how to connect multiple single neurons to a multilayer feedforward NN; this special type of *fully connected* network is also called **MLP**.

The following figure illustrates the concept of an MLP consisting of three layers:

![](img/B13208_12_02.png)

The MLP depicted in the preceding figure has one input layer, one hidden layer, and one output layer. The units in the hidden layer are fully connected to the input layer, and the output layer is fully connected to the hidden layer. If such a network has more than one hidden layer, we also call it a **deep artificial NN**.

**Adding additional hidden layers**

We can add any number of hidden layers to the MLP to create deeper network architectures. Practically, we can think of the number of layers and units in an NN as additional hyperparameters that we want to optimize for a given problem task using cross-validation technique, which we discussed in *Chapter 6*, *Learning Best Practices for Model Evaluation and Hyperparameter Tuning*.

However, the error gradients, which we will calculate later via backpropagation, will become increasingly small as more layers are added to a network. This vanishing gradient problem makes the model learning more challenging. Therefore, special algorithms have been developed to help train such DNN structures; this is known as **deep learning**.

As shown in the preceding figure, we denote the *i*th activation unit in the *l*th layer as ![](img/B13208_12_015.png). To make the math and code implementations a bit more intuitive, we will not use numerical indices to refer to layers, but we will use the *in* superscript for the input layer, the *h* superscript for the hidden layer, and the *out* superscript for the output layer. For instance, ![](img/B13208_12_016.png) refers to the *i*th value in the input layer, ![](img/B13208_12_017.png) refers to the *i*th unit in the hidden layer, and ![](img/B13208_12_018.png) refers to the *i*th unit in the output layer. Here, the activation units ![](img/B13208_12_019.png) and ![](img/B13208_12_020.png) are the bias units, which we set equal to 1\. The activation of the units in the input layer is just its input plus the bias unit:

![](img/B13208_12_021.png)

**Notational convention for the bias units**

Later in this chapter, we will implement an MLP using separate vectors for the bias units, which makes code implementation more efficient and easier to read. This concept is also used by TensorFlow, a deep learning library that we will cover in *Chapter 13*, *Parallelizing Neural Network Training with TensorFlow*. However, the mathematical equations that will follow would appear more complex or convoluted if we had to work with additional variables for the bias. Note that the computation via appending 1s to the input vector (as shown previously) and using a weight variable as bias is exactly the same as operating with separate bias vectors; it is merely a different convention.

Each unit in layer *l* is connected to all units in layer *l* + 1 via a weight coefficient. For example, the connection between the *k*th unit in layer *l* to the *j*th unit in layer *l* + 1 will be written as ![](img/B13208_12_022.png). Referring back to the previous figure, we denote the weight matrix that connects the input to the hidden layer as ![](img/B13208_12_023.png), and we write the matrix that connects the hidden layer to the output layer as ![](img/B13208_12_024.png).

While one unit in the output layer would suffice for a binary classification task, we saw a more general form of NN in the preceding figure, which allows us to perform multiclass classification via a generalization of the **one-versus-all** (**OvA**) technique. To better understand how this works, remember the **one-hot** representation of categorical variables that we introduced in *Chapter 4*, *Building Good Training Datasets – Data Preprocessing*.

For example, we can encode the three class labels in the familiar Iris dataset (*0=Setosa, 1=Versicolor, 2=Virginica*) as follows:

![](img/B13208_12_025.png)

This one-hot vector representation allows us to tackle classification tasks with an arbitrary number of unique class labels present in the training dataset.

If you are new to NN representations, the indexing notation (subscripts and superscripts) may look a little bit confusing at first. What may seem overly complicated at first will make much more sense in later sections when we vectorize the NN representation. As introduced earlier, we summarize the weights that connect the input and hidden layers by a matrix ![](img/B13208_12_026.png), where *d* is the number of hidden units and *m* is the number of input units including the bias unit. Since it is important to internalize this notation to follow the concepts later in this chapter, let's summarize what we have just learned in a descriptive illustration of a simplified 3-4-3 MLP:

![](img/B13208_12_03.png)

## Activating a neural network via forward propagation

In this section, we will describe the process of **forward propagation** to calculate the output of an MLP model. To understand how it fits into the context of learning an MLP model, let's summarize the MLP learning procedure in three simple steps:

1.  Starting at the input layer, we forward propagate the patterns of the training data through the network to generate an output.
2.  Based on the network's output, we calculate the error that we want to minimize using a cost function that we will describe later.
3.  We backpropagate the error, find its derivative with respect to each weight in the network, and update the model.

Finally, after we repeat these three steps for multiple epochs and learn the weights of the MLP, we use forward propagation to calculate the network output and apply a threshold function to obtain the predicted class labels in the one-hot representation, which we described in the previous section.

Now, let's walk through the individual steps of forward propagation to generate an output from the patterns in the training data. Since each unit in the hidden layer is connected to all units in the input layers, we first calculate the activation unit of the hidden layer ![](img/B13208_12_027.png) as follows:

![](img/B13208_12_028.png)

Here, ![](img/B13208_12_029.png) is the net input and ![](img/B13208_12_030.png) is the activation function, which has to be differentiable to learn the weights that connect the neurons using a gradient-based approach. To be able to solve complex problems such as image classification, we need nonlinear activation functions in our MLP model, for example, the sigmoid (logistic) activation function that we remember from the section about logistic regression in *Chapter 3*, *A Tour of Machine Learning Classifiers Using scikit-learn*:

![](img/B13208_12_031.png)

As you may recall, the sigmoid function is an *S*-shaped curve that maps the net input *z* onto a logistic distribution in the range 0 to 1, which cuts the *y*-axis at *z* = 0, as shown in the following graph:

![](img/B13208_12_04.png)

MLP is a typical example of a feedforward artificial NN. The term **feedforward** refers to the fact that each layer serves as the input to the next layer without loops, in contrast to recurrent NNs—an architecture that we will discuss later in this chapter and discuss in more detail in *Chapter 16*, *Modeling Sequential Data Using Recurrent Neural Networks*. The term *multilayer perceptron* may sound a little bit confusing since the artificial neurons in this network architecture are typically sigmoid units, not perceptrons. We can think of the neurons in the MLP as logistic regression units that return values in the continuous range between 0 and 1.

For purposes of code efficiency and readability, we will now write the activation in a more compact form using the concepts of basic linear algebra, which will allow us to vectorize our code implementation via NumPy rather than writing multiple nested and computationally expensive Python `for` loops:

![](img/B13208_12_032.png)

Here, ![](img/B13208_12_033.png) is our ![](img/B13208_12_034.png) dimensional feature vector of a sample ![](img/B13208_12_035.png) plus a bias unit.

![](img/B13208_12_036.png) is an ![](img/B13208_12_037.png) dimensional weight matrix where *d* is the number of units in the hidden layer. After matrix-vector multiplication, we obtain the ![](img/B13208_12_038.png) dimensional net input vector ![](img/B13208_12_039.png) to calculate the activation ![](img/B13208_12_040.png) (where ![](img/B13208_12_041.png)).

Furthermore, we can generalize this computation to all *n* examples in the training dataset:

![](img/B13208_12_042.png)

Here, ![](img/B13208_12_043.png) is now an ![](img/B13208_12_044.png) matrix, and the matrix-matrix multiplication will result in an ![](img/B13208_12_045.png) dimensional net input matrix ![](img/B13208_12_046.png). Finally, we apply the activation function ![](img/B13208_12_047.png) to each value in the net input matrix to get the ![](img/B13208_12_048.png) activation matrix the next layer (here, the output layer):

![](img/B13208_12_049.png)

Similarly, we can write the activation of the output layer in vectorized form for multiple examples:

![](img/B13208_12_050.png)

Here, we multiply the ![](img/B13208_12_051.png) matrix ![](img/B13208_12_052.png) (*t* is the number of output units) by the ![](img/B13208_12_053.png) dimensional matrix ![](img/B13208_12_054.png) to obtain the ![](img/B13208_12_055.png) dimensional matrix ![](img/B13208_12_056.png) (the columns in this matrix represent the outputs for each sample).

Lastly, we apply the sigmoid activation function to obtain the continuous valued output of our network:

![](img/B13208_12_057.png)

# Classifying handwritten digits

In the previous section, we covered a lot of the theory around NNs, which can be a little bit overwhelming if you are new to this topic. Before we continue with the discussion of the algorithm for learning the weights of the MLP model, backpropagation, let's take a short break from the theory and see an NN in action.

**Additional resources on backpropagation**

The NN theory can be quite complex; thus, it is recommended that you refer to two additional resources, which cover some of the concepts that we discuss in this chapter in more detail:

*   *Chapter 6*, *Deep Feedforward Networks*, *Deep Learning*, *I. Goodfellow*, *Y. Bengio*, and *A. Courville*, MIT Press, *2016* (Manuscripts freely accessible at [http://www.deeplearningbook.org](http://www.deeplearningbook.org)).
*   *Pattern Recognition and Machine Learning*, *C. M. Bishop* and others, Volume 1\. *Springer New York*, *2006*.
*   Lecture slides from the deep learning course at the University of Wisconsin–Madison:
    1.  [https://sebastianraschka.com/pdf/lecture-notes/stat479ss19/L08_logistic_slides.pdf](https://sebastianraschka.com/pdf/lecture-notes/stat479ss19/L08_logistic_slides.pdf)
    2.  [https://sebastianraschka.com/pdf/lecture-notes/stat479ss19/L09_mlp_slides.pdf](https://sebastianraschka.com/pdf/lecture-notes/stat479ss19/L09_mlp_slides.pdf%20)

In this section, we will implement and train our first multilayer NN to classify handwritten digits from the popular **Mixed National Institute of Standards and Technology** (**MNIST**) dataset that has been constructed by Yann LeCun and others and serves as a popular benchmark dataset for machine learning algorithms (*Gradient-Based Learning Applied to Document Recognition*, *Y. LeCun*, *L. Bottou*, *Y. Bengio*, and *P. Haffner*, *Proceedings of the IEEE*, 86(11): 2278-2324, *November 1998*).

## Obtaining and preparing the MNIST dataset

The MNIST dataset is publicly available at [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/) and consists of the following four parts:

*   **Training dataset images**: `train-images-idx3-ubyte.gz` (9.9 MB, 47 MB unzipped, and 60,000 examples)
*   **Training dataset labels**: `train-labels-idx1-ubyte.gz` (29 KB, 60 KB unzipped, and 60,000 labels)
*   **Test dataset images**: `t10k-images-idx3-ubyte.gz` (1.6 MB, 7.8 MB unzipped, and 10,000 examples)
*   **Test dataset labels**: `t10k-labels-idx1-ubyte.gz` (5 KB, 10 KB unzipped, and 10,000 labels)

The MNIST dataset was constructed from two datasets of the US **National Institute of Standards and Technology** (**NIST**). The training dataset consists of handwritten digits from 250 different people, 50 percent high school students and 50 percent employees from the Census Bureau. Note that the test dataset contains handwritten digits from different people following the same split.

After you download the files, it is recommended that you unzip them using the Unix/Linux `gzip` tool from the terminal for efficiency, using the following command in your local MNIST download directory:

[PRE0]

Alternatively, you can use your favorite unzipping tool if you are working with a machine running Microsoft Windows.

The images are stored in byte format, and we will read them into NumPy arrays that we will use to train and test our MLP implementation. In order to do that, we will define the following helper function:

[PRE1]

The `load_mnist` function returns two arrays, the first being an ![](img/B13208_12_058.png) dimensional NumPy array (`images`), where *n* is the number of examples and *m* is the number of features (here, pixels). The training dataset consists of 60,000 training digits and the test dataset contains 10,000 examples, respectively.

The images in the MNIST dataset consist of ![](img/B13208_12_059.png) pixels, and each pixel is represented by a grayscale intensity value. Here, we unroll the ![](img/B13208_12_060.png) pixels into one-dimensional row vectors, which represent the rows in our `images` array (784 per row or image). The second array (`labels`) returned by the `load_mnist` function contains the corresponding target variable, the class labels (integers 0-9) of the handwritten digits.

The way we read in the image might seem a little bit strange at first:

[PRE2]

To understand how those two lines of code work, let's take a look at the dataset description from the MNIST website:

[PRE3]

Using the two preceding lines of code, we first read in the magic number, which is a description of the file protocol, as well as the number of items (`n`) from the file buffer, before we load the following bytes into a NumPy array using the `fromfile` method. The `fmt` parameter value, `'>II'`, that we passed as an argument to `struct.unpack` can be composed into the two following parts:

*   `>`: This is big-endian—it defines the order in which a sequence of bytes is stored; if you are unfamiliar with the terms big-endian and little-endian, you can find an excellent article about *Endianness* on Wikipedia: [https://en.wikipedia.org/wiki/Endianness](https://en.wikipedia.org/wiki/Endianness)
*   `I`: This is an unsigned integer

Finally, we also normalized the pixels values in MNIST to the range –1 to 1 (originally 0 to 255) via the following code line:

[PRE4]

The reason behind this is that gradient-based optimization is much more stable under these conditions as discussed in *Chapter 2*, *Training Simple Machine Learning Algorithms for Classification*. Note that we scaled the images on a pixel-by-pixel basis, which is different from the feature scaling approach that we took in previous chapters.

Previously, we derived scaling parameters from the training dataset and used these to scale each column in the training dataset and test dataset. However, when working with image pixels, centering them at zero and rescaling them to a [–1, 1] range is also common and usually works well in practice.

**Batch normalization**

A commonly used trick for improving convergence in gradient-based optimization through input scaling is *batch normalization*, which is an advanced topic that we will cover in *Chapter 17*, *Generative Adversarial Networks for Synthesizing New Data*. Also, you can read more about batch normalization in the excellent research article *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift* by *Sergey Ioffe* and *Christian Szegedy* (2015, [https://arxiv.org/abs/1502.03167](https://arxiv.org/abs/1502.03167)).

[PRE5]

[PRE6]

To get an idea of how those images in MNIST look, let's visualize examples of the digits 0-9 after reshaping the 784-pixel vectors from our feature matrix into the original ![](img/B13208_12_061.png) image that we can plot via Matplotlib's `imshow` function:

[PRE7]

We should now see a plot of the ![](img/B13208_12_062.png) subfigures showing a representative image of each unique digit:

![](img/B13208_12_05.png)

In addition, let's also plot multiple examples of the same digit to see how different the handwriting for each really is:

[PRE8]

After executing the code, we should now see the first 25 variants of the digit 7:

![](img/B13208_12_06.png)

After we've gone through all the previous steps, it is a good idea to save the scaled images in a format that we can load more quickly into a new Python session to avoid the overhead of reading in and processing the data again. When we are working with NumPy arrays, an efficient and convenient method to save multidimensional arrays to disk is NumPy's `savez` function. (The official documentation can be found here: [https://docs.scipy.org/doc/numpy/reference/generated/numpy.savez.html](https://docs.scipy.org/doc/numpy/reference/generated/numpy.savez.html).)

[PRE9]

[PRE10]

After we create the `.npz` files, we can load the preprocessed MNIST image arrays using NumPy's `load` function as follows:

[PRE11]

The `mnist` variable now refers to an object that can access the four data arrays that we provided as keyword arguments to the `savez_compressed` function. These input arrays are now listed under the `files` attribute list of the `mnist` object:

[PRE12]

For instance, to load the training data into our current Python session, we will access the `X_train` array as follows (similar to a Python dictionary):

[PRE13]

Using a list comprehension, we can retrieve all four data arrays as follows:

[PRE14]

Note that while the preceding `np.savez_compressed` and `np.load` examples are not essential for executing the code in this chapter, they serve as a demonstration of how to save and load NumPy arrays conveniently and efficiently.

**Loading MNIST using scikit-learn**

Using scikit-learn's new `fetch_openml` function, it is now also possible to load the MNIST dataset more conveniently. For example, you can use the following code to create a 50,000-example training dataset and and a 10,000-example test dataset by fetching the dataset from [https://www.openml.org/d/554](https://www.openml.org/d/554):

[PRE15]

Please note that the distribution of MNIST records into training and test datasets will be different from the manual approach outlined in this section. Thus, you will observe slightly different results in the following sections if you load the dataset using the `fetch_openml` and `train_test_split` functions.

## Implementing a multilayer perceptron

In this subsection, we will now implement an MLP from scratch to classify the images in the MNIST dataset. To keep things simple, we will implement an MLP with only one hidden layer. Since the approach may seem a little bit complicated at first, you are encouraged to download the sample code for this chapter from the Packt Publishing website or from GitHub ([https://github.com/rasbt/python-machine-learning-book-3rd-edition](https://github.com/rasbt/python-machine-learning-book-3rd-edition)) so that you can view this MLP implementation annotated with comments and syntax highlighting for better readability.

If you are not running the code from the accompanying Jupyter Notebook file or don't have access to the Internet, copy the `NeuralNetMLP` code from this chapter into a Python script file in your current working directory (for example, `neuralnet.py``)`, which you can then import into your current Python session via the following command:

[PRE16]

The code will contain parts that we have not talked about yet, such as the backpropagation algorithm, but most of the code should look familiar to you based on the Adaline implementation in *Chapter 2*, *Training Simple Machine Learning Algorithms for Classification*, and the discussion of forward propagation in earlier sections.

Do not worry if not all of the code makes immediate sense to you; we will follow up on certain parts later in this chapter. However, going over the code at this stage can make it easier to follow the theory later.

The following is the implementation of an MLP:

[PRE17]

After executing this code, we next initialize a new 784-100-10 MLP—an NN with 784 input units (`n_features`), 100 hidden units (`n_hidden`), and 10 output units (`n_output`):

[PRE18]

If you read through the `NeuralNetMLP` code, you've probably already guessed what these parameters are for. Here, you find a short summary of them:

*   `l2`: This is the ![](img/B13208_12_063.png) parameter for L2 regularization to decrease the degree of overfitting.
*   `epochs`: This is the number of passes over the training dataset.
*   `eta`: This is the learning rate ![](img/B13208_12_064.png).
*   `shuffle`: This is for shuffling the training set prior to every epoch to prevent the algorithm getting stuck in circles.
*   `seed`: This is a random seed for shuffling and weight initialization.
*   `minibatch_size`: This is the number of training examples in each mini-batch when splitting the training data in each epoch for SGD. The gradient is computed for each mini-batch separately instead of the entire training data for faster learning.

Next, we train the MLP using 55,000 examples from the already shuffled MNIST training dataset and use the remaining 5,000 examples for validation during training. Note that training the NN may take up to five minutes on standard desktop computer hardware.

As you may have noticed from the preceding code, we implemented the `fit` method so that it takes four input arguments: training images, training labels, validation images, and validation labels. In NN training, it is really useful to compare training and validation accuracy, which helps us judge whether the network model performs well, given the architecture and hyperparameters. For example, if we observe a low training and validation accuracy, there is likely an issue with the training dataset, or the hyperparameters settings are not ideal. A relatively large gap between the training and the validation accuracy indicated that the model is likely overfitting the training dataset so that we want to reduce the number of parameters in the model or increase the regularization strength. If both the training and validation accuracies are high, the model is likely to generalize well to new data, for example, the test dataset, which we use for the final model evaluation.

In general, training (deep) NNs is relatively expensive compared with the other models we've discussed so far. Thus, we want to stop it early in certain circumstances and start over with different hyperparameter settings. On the other hand, if we find that it increasingly tends to overfit the training data (noticeable by an increasing gap between training and validation dataset performance), we may want to stop the training early as well.

Now, to start the training, we execute the following code:

[PRE19]

In our `NeuralNetMLP` implementation, we also defined an `eval_` attribute that collects the cost, training, and validation accuracy for each epoch so that we can visualize the results using Matplotlib:

[PRE20]

The preceding code plots the cost over the 200 epochs, as shown in the following graph:

![](img/B13208_12_07.png)

As we can see, the cost decreased substantially during the first 100 epochs and seems to slowly converge in the last 100 epochs. However, the small slope between epoch `175` and epoch `200` indicates that the cost would further decrease with a training over additional epochs.

Next, let's take a look at the training and validation accuracy:

[PRE21]

The preceding code examples plot those accuracy values over the 200 training epochs, as shown in the following figure:

![](img/B13208_12_08.png)

The plot reveals that the gap between training and validation accuracy increases as we train for more epochs. At approximately the 50th epoch, the training and validation accuracy values are equal, and then, the network starts overfitting the training data.

Note that this example was chosen deliberately to illustrate the effect of overfitting and demonstrate why it is useful to compare the validation and training accuracy values during training. One way to decrease the effect of overfitting is to increase the regularization strength—for example, by setting `l2=0.1`. Another useful technique to tackle overfitting in NNs is *dropout*, which will be covered in *Chapter 15*, *Classifying Images with Deep Convolutional Neural Networks*.

Finally, let's evaluate the generalization performance of the model by calculating the prediction accuracy on the test dataset:

[PRE22]

Despite the slight overfitting on the training data, our relatively simple one-hidden-layer NN achieved a relatively good performance on the test dataset, similar to the validation dataset accuracy (97.98 percent).

To further fine-tune the model, we could change the number of hidden units, values of the regularization parameters, and the learning rate, or use various other tricks that have been developed over the years but are beyond the scope of this book. In *Chapter 15*, *Classifying Images with Deep Convolutional Neural Networks*, you will learn about a different NN architecture that is known for its good performance on image datasets. Also, the chapter will introduce additional performance-enhancing tricks such as adaptive learning rates, more sophisticated SGD-based optimization algorithms, batch normalization, and dropout.

Other common tricks that are beyond the scope of the following chapters include:

*   Adding skip-connections, which are the main contribution of residual NNs (*Deep residual learning for image recognition*. K. He, X. Zhang, S. Ren, J. Sun (2016). In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, pp. 770-778)
*   Using learning rate schedulers that change the learning rate during training (*Cyclical learning rates for training neural networks*. L.N. Smith (2017). In *2017 IEEE Winter Conference on Applications of Computer Vision (WACV)*, pp. 464-472)
*   Attaching loss functions to earlier layers in the networks as it's being done in the popular Inception v3 architecture (*Rethinking the Inception architecture for computer vision*. C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, Z. Wojna (2016). In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, pp. 2818-2826)

Lastly, let's take a look at some of the images that our MLP struggles with:

[PRE23]

We should now see a ![](img/B13208_12_065.png) subplot matrix where the first number in the subtitles indicates the plot index, the second number represents the true class label (`t`), and the third number stands for the predicted class label (`p`):

![](img/B13208_12_09.png)

As we can see in the preceding figure, some of those images are even challenging for us humans to classify correctly. For example, the 6 in subplot `8` really looks like a carelessly drawn 0, and the 8 in subplot `23` could be a 9 due to the narrow lower part combined with the bold line.

# Training an artificial neural network

Now that we have seen an NN in action and have gained a basic understanding of how it works by looking over the code, let's dig a little bit deeper into some of the concepts, such as the logistic cost function and the backpropagation algorithm that we implemented to learn the weights.

## Computing the logistic cost function

The logistic cost function that we implemented as the `_compute_cost` method is actually pretty simple to follow since it is the same cost function that we described in the logistic regression section in *Chapter 3*, *A Tour of Machine Learning Classifiers Using scikit-learn*:

![](img/B13208_12_066.png)

Here, ![](img/B13208_12_067.png) is the sigmoid activation of the *i*th sample in the dataset, which we compute in the forward propagation step:

![](img/B13208_12_068.png)

Again, note that in this context, the superscript [*i*] is an index for training examples, not layers.

Now, let's add a regularization term, which allows us to reduce the degree of overfitting. As you recall from earlier chapters, the L2 regularization term is defined as follows (remember that we don't regularize the bias units):

![](img/B13208_12_069.png)

By adding the L2 regularization term to our logistic cost function, we obtain the following equation:

![](img/B13208_12_070.png)

Previously, we implemented an MLP for multiclass classification that returns an output vector of *t* elements that we need to compare to the ![](img/B13208_12_071.png) dimensional target vector in the one-hot encoding representation. If we predict the class label of an input image with class label 2, using this MLP, the activation of the third layer and the target may look like this:

![](img/B13208_12_072.png)

Thus, we need to generalize the logistic cost function to all *t* activation units in our network.

The cost function (without the regularization term) becomes the following:

![](img/B13208_12_073.png)

Here, again, the superscript [*i*] is the index of a particular sample in our training dataset.

The following generalized regularization term may look a little bit complicated at first, but here we are just calculating the sum of all weights of an *l* layer (without the bias term) that we added to the first column:

![](img/B13208_12_074.png)

Here, ![](img/B13208_12_075.png) refers to the number of units in a given layer *l*, and the following expression represents the penalty term:

![](img/B13208_12_076.png)

Remember that our goal is to minimize the cost function *J*(*W*); thus, we need to calculate the partial derivative of the parameters *W* with respect to each weight for every layer in the network:

![](img/B13208_12_077.png)

In the next section, we will talk about the backpropagation algorithm, which allows us to calculate those partial derivatives to minimize the cost function.

Note that *W* consists of multiple matrices. In an MLP with one hidden layer, we have the weight matrix, ![](img/B13208_12_078.png), which connects the input to the hidden layer, and ![](img/B13208_12_079.png), which connects the hidden layer to the output layer. A visualization of the three-dimensional tensor *W* is provided in the following figure:

![](img/B13208_12_10.png)

In this simplified figure, it may seem that both ![](img/B13208_12_080.png) and ![](img/B13208_12_081.png) have the same number of rows and columns, which is typically not the case unless we initialize an MLP with the same number of hidden units, output units, and input features.

If this sounds confusing, stay tuned for the next section, where we will discuss the dimensionality of ![](img/B13208_12_082.png) and ![](img/B13208_12_083.png) in more detail in the context of the backpropagation algorithm. Also, you are encouraged to read through the code of the `NeuralNetMLP` again, which is annotated with helpful comments about the dimensionality of the different matrices and vector transformations. You can obtain the annotated code either from Packt or the book's GitHub repository at [https://github.com/rasbt/python-machine-learning-book-3rd-edition](https://github.com/rasbt/python-machine-learning-book-3rd-edition).

## Developing your understanding of backpropagation

Although backpropagation was rediscovered and popularized more than 30 years ago (*Learning representations by back-propagating errors*, *D. E. Rumelhart*, *G. E. Hinton*, and *R. J. Williams*, *Nature*, 323: 6088, pages 533–536, *1986*), it still remains one of the most widely used algorithms to train artificial NNs very efficiently. If you are interested in additional references regarding the history of backpropagation, Juergen Schmidhuber wrote a nice survey article, *Who Invented Backpropagation?*, which you can find online at [http://people.idsia.ch/~juergen/who-invented-backpropagation.html](http://people.idsia.ch/~juergen/who-invented-backpropagation.html).

This section will provide both a short, clear summary and the bigger picture of how this fascinating algorithm works before we dive into more mathematical details. In essence, we can think of backpropagation as a very computationally efficient approach to compute the partial derivatives of a complex cost function in multilayer NNs. Here, our goal is to use those derivatives to learn the weight coefficients for parameterizing such a multilayer artificial NN. The challenge in the parameterization of NNs. is that we are typically dealing with a very large number of weight coefficients in a high-dimensional feature space. In contrast to cost functions of single-layer NNs such as Adaline or logistic regression, which we have seen in previous chapters, the error surface of an NN cost function is not convex or smooth with respect to the parameters. There are many bumps in this high-dimensional cost surface (local minima) that we have to overcome in order to find the global minimum of the cost function.

You may recall the concept of the chain rule from your introductory calculus classes. The chain rule is an approach to compute the derivative of a complex, nested function, such as *f*( *g*(*x*)), as follows:

![](img/B13208_12_084.png)

Similarly, we can use the chain rule for an arbitrarily long function composition. For example, let's assume that we have five different functions, *f*(*x*), *g*(*x*), *h*(*x*), *u*(*x*), and *v*(*x*), and let *F* be the function composition: *F*(*x*) = *f*(*g*(*h*(*u*(*v*(*x*))))). Applying the chain rule, we can compute the derivative of this function as follows:

![](img/B13208_12_085.png)

In the context of computer algebra, a set of techniques has been developed to solve such problems very efficiently, which is also known as **automatic differentiation**. If you are interested in learning more about automatic differentiation in machine learning applications, read A. G. Baydin and B. A. Pearlmutter's article *Automatic Differentiation of Algorithms for Machine Learning*, arXiv preprint arXiv:1404.7456, *2014*, which is freely available on arXiv at [http://arxiv.org/pdf/1404.7456.pdf](http://arxiv.org/pdf/1404.7456.pdf).

Automatic differentiation comes with two modes, the forward and reverse modes; backpropagation is simply a special case of reverse-mode automatic differentiation. The key point is that applying the chain rule in the forward mode could be quite expensive since we would have to multiply large matrices for each layer (Jacobians) that we would eventually multiply by a vector to obtain the output.

The trick of reverse mode is that we start from right to left: we multiply a matrix by a vector, which yields another vector that is multiplied by the next matrix and so on. Matrix-vector multiplication is computationally much cheaper than matrix-matrix multiplication, which is why backpropagation is one of the most popular algorithms used in NN training.

**A basic calculus refresher**

To fully understand backpropagation, we need to borrow certain concepts from differential calculus, which is outside the scope of this book. However, you can refer to a review chapter of the most fundamental concepts, which you might find useful in this context. It discusses function derivatives, partial derivatives, gradients, and the Jacobian. This text is freely accessible at [https://sebastianraschka.com/pdf/books/dlb/appendix_d_calculus.pdf](https://sebastianraschka.com/pdf/books/dlb/appendix_d_calculus.pdf). If you are unfamiliar with calculus or need a brief refresher, consider reading this text as an additional supporting resource before reading the next section.

## Training neural networks via backpropagation

In this section, we will go through the math of backpropagation to understand how you can learn the weights in an NN very efficiently. Depending on how comfortable you are with mathematical representations, the following equations may seem relatively complicated at first.

In a previous section, we saw how to calculate the cost as the difference between the activation of the last layer and the target class label. Now, we will see how the backpropagation algorithm works to update the weights in our MLP model from a mathematical perspective, which we implemented after the `# Backpropagation` code comment inside the `fit` method. As we recall from the beginning of this chapter, we first need to apply forward propagation in order to obtain the activation of the output layer, which we formulated as follows:

![](img/B13208_12_086.png)

Concisely, we just forward-propagate the input features through the connection in the network, as shown in the following illustration:

![](img/B13208_12_11.png)

In backpropagation, we propagate the error from right to left. We start by calculating the error vector of the output layer:

![](img/B13208_12_087.png)

Here, *y* is the vector of the true class labels (the corresponding variable in the `NeuralNetMLP` code is `delta_out`).

Next, we calculate the error term of the hidden layer:

![](img/B13208_12_088.png)

Here, ![](img/B13208_12_089.png) is simply the derivative of the sigmoid activation function, which we computed as `sigmoid_derivative_h = a_h * (1\. - a_h)` in the `fit` method of the `NeuralNetMLP`:

![](img/B13208_12_090.png)

Note that the ![](img/B13208_12_091.png) symbol means element-wise multiplication in this context.

**Activation function derivative**

Although it is not important to follow the next equations, you may be curious how the derivative of the activation function was obtained; it is summarized step by step here:

![](img/B13208_12_092.png)

Next, we compute the ![](img/B13208_12_093.png) layer error matrix (`delta_h`) as follows:

![](img/B13208_12_094.png)

To better understand how we computed this ![](img/B13208_12_095.png) term, let's walk through it in more detail. In the preceding equation, we used the transpose ![](img/B13208_12_096.png) of the ![](img/B13208_12_097.png)-dimensional matrix ![](img/B13208_12_098.png). Here, *t* is the number of output class labels and *h* is the number of hidden units. The matrix multiplication between the ![](img/B13208_12_099.png)-dimensional ![](img/B13208_12_100.png) matrix and the ![](img/B13208_12_101.png)-dimensional matrix ![](img/B13208_12_102.png) results in an ![](img/B13208_12_103.png)-dimensional matrix that we multiplied element-wise by the sigmoid derivative of the same dimension to obtain the ![](img/B13208_12_104.png)-dimensional matrix ![](img/B13208_12_105.png).

Eventually, after obtaining the ![](img/B13208_12_106.png) terms, we can now write the derivation of the cost function as follows:

![](img/B13208_12_107.png)

Next, we need to accumulate the partial derivative of every node in each layer and the error of the node in the next layer. However, remember that we need to compute ![](img/B13208_12_108.png) for every sample in the training dataset. Thus, it is easier to implement it as a vectorized version like in our `NeuralNetMLP` code implementation:

![](img/B13208_12_109.png)

And after we have accumulated the partial derivatives, we can add the following regularization term:

![](img/B13208_12_110.png)

(Please note that the bias units are usually not regularized.)

The two previous mathematical equations correspond to the code variables `delta_w_h`, `delta_b_h`, `delta_w_out`, and `delta_b_out` in `NeuralNetMLP`.

Lastly, after we have computed the gradients, we can update the weights by taking an opposite step toward the gradient for each layer *l*:

![](img/B13208_12_111.png)

This is implemented as follows:

[PRE24]

To bring everything together, let's summarize backpropagation in the following figure:

![](img/B13208_12_12.png)

# About the convergence in neural networks

You might be wondering why we did not use regular gradient descent but instead used mini-batch learning to train our NN for the handwritten digit classification. You may recall our discussion on SGD that we used to implement online learning. In online learning, we compute the gradient based on a single training example (*k* = 1) at a time to perform the weight update. Although this is a stochastic approach, it often leads to very accurate solutions with a much faster convergence than regular gradient descent. Mini-batch learning is a special form of SGD where we compute the gradient based on a subset *k* of the *n* training examples with 1 < *k* < *n*. Mini-batch learning has the advantage over online learning that we can make use of our vectorized implementations to improve computational efficiency. However, we can update the weights much faster than in regular gradient descent. Intuitively, you can think of mini-batch learning as predicting the voter turnout of a presidential election from a poll by asking only a representative subset of the population rather than asking the entire population (which would be equal to running the actual election).

Multilayer NNs are much harder to train than simpler algorithms such as Adaline, logistic regression, or support vector machines. In multilayer NNs, we typically have hundreds, thousands, or even billions of weights that we need to optimize. Unfortunately, the output function has a rough surface and the optimization algorithm can easily become trapped in local minima, as shown in the following figure:

![](img/B13208_12_13.png)

Note that this representation is extremely simplified since our NN has many dimensions; it makes it impossible to visualize the actual cost surface for the human eye. Here, we only show the cost surface for a single weight on the *x*-axis. However, the main message is that we do not want our algorithm to get trapped in local minima. By increasing the learning rate, we can more readily escape such local minima. On the other hand, we also increase the chance of overshooting the global optimum if the learning rate is too large. Since we initialize the weights randomly, we start with a solution to the optimization problem that is typically hopelessly wrong.

# A few last words about the neural network implementation

You may be wondering why we went through all of this theory just to implement a simple multilayer artificial network that can classify handwritten digits instead of using an open source Python machine learning library. In fact, we will introduce more complex NN models in the next chapters, which we will train using the open source TensorFlow library ([https://www.tensorflow.org](https://www.tensorflow.org)).

Although the from-scratch implementation in this chapter seems a bit tedious at first, it was a good exercise for understanding the basics behind backpropagation and NN training, and a basic understanding of algorithms is crucial for applying machine learning techniques appropriately and successfully.

Now that you have learned how feedforward NNs work, we are ready to explore more sophisticated DNNs by using TensorFlow, which allows us to construct NNs more efficiently, as we will see in *Chapter 13*, *Parallelizing Neural Network Training with TensorFlow*.

Over the past two years, since its release in November 2015, TensorFlow has gained a lot of popularity among machine learning researchers, who use it to construct DNNs because of its ability to optimize mathematical expressions for computations on multidimensional arrays utilizing **graphics processing units** (**GPUs**). While TensorFlow can be considered a low-level deep learning library, simplifying APIs such as Keras have been developed that make the construction of common deep learning models even more convenient, which we will see in *Chapter 13*, *Parallelizing Neural Network Training with TensorFlow*.

# Summary

In this chapter, you have learned the basic concepts behind multilayer artificial NNs, which are currently the hottest topic in machine learning research. In *Chapter 2*, *Training Simple Machine Learning Algorithms for Classification*, we started our journey with simple single-layer NN structures and now we have connected multiple neurons to a powerful NN architecture to solve complex problems such as handwritten digit recognition. We demystified the popular backpropagation algorithm, which is one of the building blocks of many NN models that are used in deep learning. After learning about the backpropagation algorithm in this chapter, we are well equipped for exploring more complex DNN architectures. In the remaining chapters, we will cover TensorFlow, an open source library geared toward deep learning, which allows us to implement and train multilayer NNs more efficiently.