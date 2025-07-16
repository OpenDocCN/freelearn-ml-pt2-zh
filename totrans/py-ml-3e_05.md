# 5

# Compressing Data via Dimensionality Reduction

In *Chapter 4*, *Building Good Training Datasets – Data Preprocessing*, you learned about the different approaches for reducing the dimensionality of a dataset using different feature selection techniques. An alternative approach to feature selection for dimensionality reduction is **feature extraction**. In this chapter, you will learn about three fundamental techniques that will help you to summarize the information content of a dataset by transforming it onto a new feature subspace of lower dimensionality than the original one. Data compression is an important topic in machine learning, and it helps us to store and analyze the increasing amounts of data that are produced and collected in the modern age of technology.

In this chapter, we will cover the following topics:

*   **Principal component analysis** (**PCA**) for unsupervised data compression
*   **Linear discriminant analysis** (**LDA**) as a supervised dimensionality reduction technique for maximizing class separability
*   Nonlinear dimensionality reduction via **kernel principal component analysis** (**KPCA**)

# Unsupervised dimensionality reduction via principal component analysis

Similar to feature selection, we can use different feature extraction techniques to reduce the number of features in a dataset. The difference between feature selection and feature extraction is that while we maintain the original features when we use feature selection algorithms, such as **sequential backward selection**, we use feature extraction to transform or project the data onto a new feature space.

In the context of dimensionality reduction, feature extraction can be understood as an approach to data compression with the goal of maintaining most of the relevant information. In practice, feature extraction is not only used to improve storage space or the computational efficiency of the learning algorithm, but can also improve the predictive performance by reducing the **curse of dimensionality**—especially if we are working with non-regularized models.

## The main steps behind principal component analysis

In this section, we will discuss PCA, an unsupervised linear transformation technique that is widely used across different fields, most prominently for feature extraction and dimensionality reduction. Other popular applications of PCA include exploratory data analyses and the denoising of signals in stock market trading, and the analysis of genome data and gene expression levels in the field of bioinformatics.

PCA helps us to identify patterns in data based on the correlation between features. In a nutshell, PCA aims to find the directions of maximum variance in high-dimensional data and projects the data onto a new subspace with equal or fewer dimensions than the original one. The orthogonal axes (principal components) of the new subspace can be interpreted as the directions of maximum variance given the constraint that the new feature axes are orthogonal to each other, as illustrated in the following figure:

![](img/B13208_05_01.png)

In the preceding figure, ![](img/B13208_05_001.png) and ![](img/B13208_05_002.png) are the original feature axes, and **PC1** and **PC2** are the principal components.

If we use PCA for dimensionality reduction, we construct a ![](img/B13208_05_003.png)-dimensional transformation matrix, *W*, that allows us to map a vector, *x*, the features of a training example, onto a new *k*-dimensional feature subspace that has fewer dimensions than the original *d*-dimensional feature space. For instance, the process is as follows. Suppose we have a feature vector, *x*:

![](img/B13208_05_005.png)

which is then transformed by a transformation matrix, ![](img/B13208_05_006.png):

![](img/B13208_05_007.png)

resulting in the output vector:

![](img/B13208_05_008.png)

As a result of transforming the original *d*-dimensional data onto this new *k*-dimensional subspace (typically *k* << *d*), the first principal component will have the largest possible variance. All consequent principal components will have the largest variance given the constraint that these components are uncorrelated (orthogonal) to the other principal components—even if the input features are correlated, the resulting principal components will be mutually orthogonal (uncorrelated). Note that the PCA directions are highly sensitive to data scaling, and we need to standardize the features *prior* to PCA if the features were measured on different scales and we want to assign equal importance to all features.

Before looking at the PCA algorithm for dimensionality reduction in more detail, let's summarize the approach in a few simple steps:

1.  Standardize the *d*-dimensional dataset.
2.  Construct the covariance matrix.
3.  Decompose the covariance matrix into its eigenvectors and eigenvalues.
4.  Sort the eigenvalues by decreasing order to rank the corresponding eigenvectors.
5.  Select *k* eigenvectors, which correspond to the *k* largest eigenvalues, where *k* is the dimensionality of the new feature subspace (![](img/B13208_05_009.png)).
6.  Construct a projection matrix, *W*, from the "top" *k* eigenvectors.
7.  Transform the *d*-dimensional input dataset, *X*, using the projection matrix, *W*, to obtain the new *k*-dimensional feature subspace.

In the following sections, we will perform a PCA step by step, using Python as a learning exercise. Then, we will see how to perform a PCA more conveniently using scikit-learn.

## Extracting the principal components step by step

In this subsection, we will tackle the first four steps of a PCA:

1.  Standardizing the data.
2.  Constructing the covariance matrix.
3.  Obtaining the eigenvalues and eigenvectors of the covariance matrix.
4.  Sorting the eigenvalues by decreasing order to rank the eigenvectors.

First, we will start by loading the Wine dataset that we were working with in *Chapter 4*, *Building Good Training Datasets – Data Preprocessing*:

[PRE0]

**Obtaining the Wine dataset**

You can find a copy of the Wine dataset (and all other datasets used in this book) in the code bundle of this book, which you can use if you are working offline or the UCI server at [https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data](https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data) is temporarily unavailable. For instance, to load the Wine dataset from a local directory, you can replace the following line:

[PRE1]

with the following one:

[PRE2]

Next, we will process the Wine data into separate training and test datasets—using 70 percent and 30 percent of the data, respectively—and standardize it to unit variance:

[PRE3]

After completing the mandatory preprocessing by executing the preceding code, let's advance to the second step: constructing the covariance matrix. The symmetric ![](img/B13208_05_010.png)-dimensional covariance matrix, where *d* is the number of dimensions in the dataset, stores the pairwise covariances between the different features. For example, the covariance between two features, ![](img/B13208_05_011.png) and ![](img/B13208_05_012.png), on the population level can be calculated via the following equation:

![](img/B13208_05_013.png)

Here, ![](img/B13208_05_014.png) and ![](img/B13208_05_015.png) are the sample means of features *j* and *k*, respectively. Note that the sample means are zero if we standardized the dataset. A positive covariance between two features indicates that the features increase or decrease together, whereas a negative covariance indicates that the features vary in opposite directions. For example, the covariance matrix of three features can then be written as follows (note that ![](img/B13208_05_016.png) stands for the Greek uppercase letter sigma, which is not to be confused with the summation symbol):

![](img/B13208_05_017.png)

The eigenvectors of the covariance matrix represent the principal components (the directions of maximum variance), whereas the corresponding eigenvalues will define their magnitude. In the case of the Wine dataset, we would obtain 13 eigenvectors and eigenvalues from the ![](img/B13208_05_018.png)-dimensional covariance matrix.

Now, for our third step, let's obtain the eigenpairs of the covariance matrix. As you will remember from our introductory linear algebra classes, an eigenvector, *v*, satisfies the following condition:

![](img/B13208_05_019.png)

Here, ![](img/B13208_04_020.png) is a scalar: the eigenvalue. Since the manual computation of eigenvectors and eigenvalues is a somewhat tedious and elaborate task, we will use the `linalg.eig` function from NumPy to obtain the eigenpairs of the Wine covariance matrix:

[PRE4]

Using the `numpy.cov` function, we computed the covariance matrix of the standardized training dataset. Using the `linalg.eig` function, we performed the eigendecomposition, which yielded a vector (`eigen_vals`) consisting of 13 eigenvalues and the corresponding eigenvectors stored as columns in a ![](img/B13208_05_021.png)-dimensional matrix (`eigen_vecs`).

**Eigendecomposition in NumPy**

The `numpy.linalg.eig` function was designed to operate on both symmetric and non-symmetric square matrices. However, you may find that it returns complex eigenvalues in certain cases.

A related function, `numpy.linalg.eigh`, has been implemented to decompose Hermetian matrices, which is a numerically more stable approach to working with symmetric matrices such as the covariance matrix; `numpy.linalg.eigh` always returns real eigenvalues.

## Total and explained variance

Since we want to reduce the dimensionality of our dataset by compressing it onto a new feature subspace, we only select the subset of the eigenvectors (principal components) that contains most of the information (variance). The eigenvalues define the magnitude of the eigenvectors, so we have to sort the eigenvalues by decreasing magnitude; we are interested in the top *k* eigenvectors based on the values of their corresponding eigenvalues. But before we collect those *k* most informative eigenvectors, let's plot the **variance explained ratios** of the eigenvalues. The variance explained ratio of an eigenvalue, ![](img/B13208_05_023.png), is simply the fraction of an eigenvalue, ![](img/B13208_05_023.png), and the total sum of the eigenvalues:

![](img/B13208_05_024.png)

Using the NumPy `cumsum` function, we can then calculate the cumulative sum of explained variances, which we will then plot via Matplotlib's `step` function:

[PRE5]

The resulting plot indicates that the first principal component alone accounts for approximately 40 percent of the variance.

Also, we can see that the first two principal components combined explain almost 60 percent of the variance in the dataset:

![](img/B13208_05_02.png)

Although the explained variance plot reminds us of the feature importance values that we computed in *Chapter 4*, *Building Good Training Datasets – Data Preprocessing*, via random forests, we should remind ourselves that PCA is an unsupervised method, which means that information about the class labels is ignored. Whereas a random forest uses the class membership information to compute the node impurities, variance measures the spread of values along a feature axis.

## Feature transformation

Now that we have successfully decomposed the covariance matrix into eigenpairs, let's proceed with the last three steps to transform the Wine dataset onto the new principal component axes. The remaining steps we are going to tackle in this section are the following:

1.  Select *k* eigenvectors, which correspond to the *k* largest eigenvalues, where *k* is the dimensionality of the new feature subspace (![](img/B13208_05_025.png)).
2.  Construct a projection matrix, *W*, from the "top" *k* eigenvectors.

1.  Transform the *d*-dimensional input dataset, *X*, using the projection matrix, *W*, to obtain the new *k*-dimensional feature subspace.

Or, in less technical terms, we will sort the eigenpairs by descending order of the eigenvalues, construct a projection matrix from the selected eigenvectors, and use the projection matrix to transform the data onto the lower-dimensional subspace.

We start by sorting the eigenpairs by decreasing order of the eigenvalues:

[PRE6]

Next, we collect the two eigenvectors that correspond to the two largest eigenvalues, to capture about 60 percent of the variance in this dataset. Note that two eigenvectors have been chosen for the purpose of illustration, since we are going to plot the data via a two-dimensional scatter plot later in this subsection. In practice, the number of principal components has to be determined by a tradeoff between computational efficiency and the performance of the classifier:

[PRE7]

By executing the preceding code, we have created a ![](img/B13208_05_026.png)-dimensional projection matrix, *W*, from the top two eigenvectors.

**Mirrored projections**

Depending on which versions of NumPy and LAPACK you are using, you may obtain the matrix, *W*, with its signs flipped. Please note that this is not an issue; if *v* is an eigenvector of a matrix, ![](img/B13208_05_027.png), we have:

![](img/B13208_05_028.png)

Here, *v* is the eigenvector, and *–v* is also an eigenvector, which we can show as follows. Using basic algebra, we can multiply both sides of the equation by a scalar, ![](img/B13208_05_029.png):

![](img/B13208_05_030.png)

Since matrix multiplication is associative for scalar multiplication, we can then rearrange this to the following:

![](img/B13208_05_031.png)

Now, we can see that ![](img/B13208_05_032.png) is an eigenvector with the same eigenvalue, ![](img/B13208_05_033.png), for both ![](img/B13208_05_034.png) and ![](img/B13208_05_035.png). Hence, both *v* and *–v* are eigenvectors.

Using the projection matrix, we can now transform an example, *x* (represented as a 13-dimensional row vector), onto the PCA subspace (the principal components one and two) obtaining ![](img/B13208_05_037.png), now a two-dimensional example vector consisting of two new features:

![](img/B13208_05_038.png)

[PRE8]

Similarly, we can transform the entire ![](img/B13208_05_039.png)-dimensional training dataset onto the two principal components by calculating the matrix dot product:

![](img/B13208_05_040.png)

[PRE9]

Lastly, let's visualize the transformed Wine training dataset, now stored as an ![](img/B13208_05_041.png)-dimensional matrix, in a two-dimensional scatterplot:

[PRE10]

As we can see in the resulting plot, the data is more spread along the *x*-axis—the first principal component—than the second principal component (*y*-axis), which is consistent with the explained variance ratio plot that we created in the previous subsection. However, we can tell that a linear classifier will likely be able to separate the classes well:

![](img/B13208_05_03.png)

Although we encoded the class label information for the purpose of illustration in the preceding scatter plot, we have to keep in mind that PCA is an unsupervised technique that doesn't use any class label information.

## Principal component analysis in scikit-learn

Although the verbose approach in the previous subsection helped us to follow the inner workings of PCA, we will now discuss how to use the `PCA` class implemented in scikit-learn.

The `PCA` class is another one of scikit-learn's transformer classes, where we first fit the model using the training data before we transform both the training data and the test dataset using the same model parameters. Now, let's use the `PCA` class from scikit-learn on the Wine training dataset, classify the transformed examples via logistic regression, and visualize the decision regions via the `plot_decision_regions` function that we defined in *Chapter 2*, *Training Simple Machine Learning Algorithms for Classification*:

[PRE11]

For your convenience, you can place the `plot_decision_regions` code shown above into a separate code file in your current working directory, for example, `plot_decision_regions_script.py`, and import it into your current Python session.

[PRE12]

By executing the preceding code, we should now see the decision regions for the training data reduced to two principal component axes:

![](img/B13208_05_04.png)

When we compare PCA projections via scikit-learn with our own PCA implementation, it can happen that the resulting plots are mirror images of each other. Note that this is not due to an error in either of those two implementations; the reason for this difference is that, depending on the eigensolver, eigenvectors can have either negative or positive signs.

Not that it matters, but we could simply revert the mirror image by multiplying the data by –1 if we wanted to; note that eigenvectors are typically scaled to unit length 1\. For the sake of completeness, let's plot the decision regions of the logistic regression on the transformed test dataset to see if it can separate the classes well:

[PRE13]

After we plot the decision regions for the test dataset by executing the preceding code, we can see that logistic regression performs quite well on this small two-dimensional feature subspace and only misclassifies a few examples in the test dataset:

![](img/B13208_05_05.png)

If we are interested in the explained variance ratios of the different principal components, we can simply initialize the `PCA` class with the `n_components` parameter set to `None`, so all principal components are kept and the explained variance ratio can then be accessed via the `explained_variance_ratio_` attribute:

[PRE14]

Note that we set `n_components=None` when we initialized the `PCA` class so that it will return all principal components in a sorted order, instead of performing a dimensionality reduction.

# Supervised data compression via linear discriminant analysis

LDA can be used as a technique for feature extraction to increase the computational efficiency and reduce the degree of overfitting due to the curse of dimensionality in non-regularized models. The general concept behind LDA is very similar to PCA, but whereas PCA attempts to find the orthogonal component axes of maximum variance in a dataset, the goal in LDA is to find the feature subspace that optimizes class separability. In the following sections, we will discuss the similarities between LDA and PCA in more detail and walk through the LDA approach step by step.

## Principal component analysis versus linear discriminant analysis

Both PCA and LDA are linear transformation techniques that can be used to reduce the number of dimensions in a dataset; the former is an unsupervised algorithm, whereas the latter is supervised. Thus, we might think that LDA is a superior feature extraction technique for classification tasks compared to PCA. However, A.M. Martinez reported that preprocessing via PCA tends to result in better classification results in an image recognition task in certain cases, for instance, if each class consists of only a small number of examples (*PCA Versus LDA*, *A. M. Martinez* and *A. C. Kak*, *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 23(2): 228-233, *2001*).

**Fisher LDA**

LDA is sometimes also called Fisher's LDA. Ronald A. Fisher initially formulated *Fisher's Linear Discriminant* for two-class classification problems in 1936 (*The Use of Multiple Measurements in Taxonomic Problems*, *R. A. Fisher*, *Annals of Eugenics*, 7(2): 179-188, *1936*). Fisher's linear discriminant was later generalized for multi-class problems by C. Radhakrishna Rao under the assumption of equal class covariances and normally distributed classes in 1948, which we now call LDA (*The Utilization of Multiple Measurements in Problems of Biological Classification*, *C. R. Rao*, *Journal of the Royal Statistical Society*. Series B (Methodological), 10(2): 159-203, *1948*).

The following figure summarizes the concept of LDA for a two-class problem. Examples from class 1 are shown as circles, and examples from class 2 are shown as crosses:

![](img/B13208_05_06.png)

A linear discriminant, as shown on the *x*-axis (LD 1), would separate the two normal distributed classes well. Although the exemplary linear discriminant shown on the *y*-axis (LD 2) captures a lot of the variance in the dataset, it would fail as a good linear discriminant since it does not capture any of the class-discriminatory information.

One assumption in LDA is that the data is normally distributed. Also, we assume that the classes have identical covariance matrices and that the training examples are statistically independent of each other. However, even if one, or more, of those assumptions is (slightly) violated, LDA for dimensionality reduction can still work reasonably well (*Pattern Classification 2nd Edition*, *R. O. Duda*, *P. E. Hart*, and *D. G. Stork*, *New York*, *2001*).

## The inner workings of linear discriminant analysis

Before we dive into the code implementation, let's briefly summarize the main steps that are required to perform LDA:

1.  Standardize the *d*-dimensional dataset (*d* is the number of features).
2.  For each class, compute the *d*-dimensional mean vector.
3.  Construct the between-class scatter matrix, ![](img/B13208_05_042.png), and the within-class scatter matrix, ![](img/B13208_05_043.png).
4.  Compute the eigenvectors and corresponding eigenvalues of the matrix, ![](img/B13208_05_044.png).
5.  Sort the eigenvalues by decreasing order to rank the corresponding eigenvectors.
6.  Choose the *k* eigenvectors that correspond to the *k* largest eigenvalues to construct a ![](img/B13208_05_045.png)-dimensional transformation matrix, *W*; the eigenvectors are the columns of this matrix.
7.  Project the examples onto the new feature subspace using the transformation matrix, *W*.

As we can see, LDA is quite similar to PCA in the sense that we are decomposing matrices into eigenvalues and eigenvectors, which will form the new lower-dimensional feature space. However, as mentioned before, LDA takes class label information into account, which is represented in the form of the mean vectors computed in step 2\. In the following sections, we will discuss these seven steps in more detail, accompanied by illustrative code implementations.

## Computing the scatter matrices

Since we already standardized the features of the Wine dataset in the PCA section at the beginning of this chapter, we can skip the first step and proceed with the calculation of the mean vectors, which we will use to construct the within-class scatter matrix and between-class scatter matrix, respectively. Each mean vector, ![](img/B13208_05_046.png), stores the mean feature value, ![](img/B13208_05_047.png), with respect to the examples of class *i*:

![](img/B13208_05_048.png)

This results in three mean vectors:

![](img/B13208_05_049.png)

[PRE15]

Using the mean vectors, we can now compute the within-class scatter matrix, ![](img/B13208_05_050.png):

![](img/B13208_05_051.png)

This is calculated by summing up the individual scatter matrices, ![](img/B13208_05_052.png), of each individual class *i*:

![](img/B13208_05_053.png)

[PRE16]

The assumption that we are making when we are computing the scatter matrices is that the class labels in the training dataset are uniformly distributed. However, if we print the number of class labels, we see that this assumption is violated:

[PRE17]

Thus, we want to scale the individual scatter matrices, ![](img/B13208_05_054.png), before we sum them up as scatter matrix ![](img/B13208_05_055.png). When we divide the scatter matrices by the number of class-examples, ![](img/B13208_05_056.png), we can see that computing the scatter matrix is in fact the same as computing the covariance matrix, ![](img/B13208_05_057.png)—the covariance matrix is a normalized version of the scatter matrix:

![](img/B13208_05_058.png)

The code for computing the scaled within-class scatter matrix is as follows:

[PRE18]

After we compute the scaled within-class scatter matrix (or covariance matrix), we can move on to the next step and compute the between-class scatter matrix ![](img/B13208_05_059.png):

![](img/B13208_05_060.png)

Here, *m* is the overall mean that is computed, including examples from all *c* classes:

[PRE19]

## Selecting linear discriminants for the new feature subspace

The remaining steps of the LDA are similar to the steps of the PCA. However, instead of performing the eigendecomposition on the covariance matrix, we solve the generalized eigenvalue problem of the matrix, ![](img/B13208_05_061.png):

[PRE20]

After we compute the eigenpairs, we can sort the eigenvalues in descending order:

[PRE21]

In LDA, the number of linear discriminants is at most *c*−1, where *c* is the number of class labels, since the in-between scatter matrix, ![](img/B13208_05_062.png), is the sum of *c* matrices with rank one or less. We can indeed see that we only have two nonzero eigenvalues (the eigenvalues 3-13 are not exactly zero, but this is due to the floating-point arithmetic in NumPy).

**Collinearity**

Note that in the rare case of perfect collinearity (all aligned example points fall on a straight line), the covariance matrix would have rank one, which would result in only one eigenvector with a nonzero eigenvalue.

To measure how much of the class-discriminatory information is captured by the linear discriminants (eigenvectors), let's plot the linear discriminants by decreasing eigenvalues, similar to the explained variance plot that we created in the PCA section. For simplicity, we will call the content of class-discriminatory information **discriminability**:

[PRE22]

As we can see in the resulting figure, the first two linear discriminants alone capture 100 percent of the useful information in the Wine training dataset:

![](img/B13208_05_07.png)

Let's now stack the two most discriminative eigenvector columns to create the transformation matrix, *W*:

[PRE23]

## Projecting examples onto the new feature space

Using the transformation matrix, *W*, that we created in the previous subsection, we can now transform the training dataset by multiplying the matrices:

![](img/B13208_05_063.png)

[PRE24]

As we can see in the resulting plot, the three Wine classes are now perfectly linearly separable in the new feature subspace:

![](img/B13208_05_08.png)

## LDA via scikit-learn

That step-by-step implementation was a good exercise to understand the inner workings of an LDA and understand the differences between LDA and PCA. Now, let's look at the `LDA` class implemented in scikit-learn:

[PRE25]

Next, let's see how the logistic regression classifier handles the lower-dimensional training dataset after the LDA transformation:

[PRE26]

Looking at the resulting plot, we can see that the logistic regression model misclassifies one of the examples from class 2:

![](img/B13208_05_09.png)

By lowering the regularization strength, we could probably shift the decision boundaries so that the logistic regression model classifies all examples in the training dataset correctly. However, and more importantly, let's take a look at the results on the test dataset:

[PRE27]

As we can see in the following plot, the logistic regression classifier is able to get a perfect accuracy score for classifying the examples in the test dataset by only using a two-dimensional feature subspace, instead of the original 13 Wine features:

![](img/B13208_05_10.png)

# Using kernel principal component analysis for nonlinear mappings

Many machine learning algorithms make assumptions about the linear separability of the input data. You have learned that the perceptron even requires perfectly linearly separable training data to converge. Other algorithms that we have covered so far assume that the lack of perfect linear separability is due to noise: Adaline, logistic regression, and the (standard) SVM to just name a few.

However, if we are dealing with nonlinear problems, which we may encounter rather frequently in real-world applications, linear transformation techniques for dimensionality reduction, such as PCA and LDA, may not be the best choice.

In this section, we will take a look at a kernelized version of PCA, or KPCA, which relates to the concepts of kernel SVM that you will remember from *Chapter 3*, *A Tour of Machine Learning Classifiers Using scikit-learn*. Using KPCA, we will learn how to transform data that is not linearly separable onto a new, lower-dimensional subspace that is suitable for linear classifiers.

![](img/B13208_05_11.png)

## Kernel functions and the kernel trick

As you will remember from our discussion about kernel SVMs in *Chapter 3*, *A Tour of Machine Learning Classifiers Using scikit-learn*, we can tackle nonlinear problems by projecting them onto a new feature space of higher dimensionality where the classes become linearly separable. To transform the examples ![](img/B13208_05_064.png) onto this higher *k*-dimensional subspace, we defined a nonlinear mapping function, ![](img/B13208_05_065.png):

![](img/B13208_05_066.png)

We can think of ![](img/B13208_05_067.png) as a function that creates nonlinear combinations of the original features to map the original *d*-dimensional dataset onto a larger, *k*-dimensional feature space.

For example, if we had a feature vector ![](img/B13208_05_068.png) (*x* is a column vector consisting of *d* features) with two dimensions (*d* = 2), a potential mapping onto a 3D-space could be:

![](img/B13208_05_069.png)

In other words, we perform a nonlinear mapping via KPCA that transforms the data onto a higher-dimensional space. We then use standard PCA in this higher-dimensional space to project the data back onto a lower-dimensional space where the examples can be separated by a linear classifier (under the condition that the examples can be separated by density in the input space). However, one downside of this approach is that it is computationally very expensive, and this is where we use the **kernel trick**. Using the kernel trick, we can compute the similarity between two high-dimension feature vectors in the original feature space.

Before we proceed with more details about the kernel trick to tackle this computationally expensive problem, let's think back to the standard PCA approach that we implemented at the beginning of this chapter. We computed the covariance between two features, *k* and *j*, as follows:

![](img/B13208_05_070.png)

Since the standardizing of features centers them at mean zero, for instance, ![](img/B13208_05_071.png) and ![](img/B13208_05_072.png), we can simplify this equation as follows:

![](img/B13208_05_073.png)

Note that the preceding equation refers to the covariance between two features; now, let's write the general equation to calculate the covariance matrix, ![](img/B13208_05_074.png):

![](img/B13208_05_075.png)

Bernhard Scholkopf generalized this approach (*Kernel principal component analysis*, *B. Scholkopf*, *A. Smola*, and *K.R. Muller*, pages 583-588, *1997*) so that we can replace the dot products between examples in the original feature space with the nonlinear feature combinations via ![](img/B13208_05_076.png):

![](img/B13208_05_077.png)

To obtain the eigenvectors—the principal components—from this covariance matrix, we have to solve the following equation:

![](img/B13208_05_078.png)

Here, ![](img/B13208_05_079.png) and *v* are the eigenvalues and eigenvectors of the covariance matrix, ![](img/B13208_05_080.png), and *a* can be obtained by extracting the eigenvectors of the kernel (similarity) matrix, *K*, as you will see in the following paragraphs.

**Deriving the kernel matrix**

The derivation of the kernel matrix can be shown as follows. First, let's write the covariance matrix as in matrix notation, where ![](img/B13208_05_081.png) is an ![](img/B13208_05_082.png)-dimensional matrix:

![](img/B13208_05_083.png)

Now, we can write the eigenvector equation as follows:

![](img/B13208_05_084.png)

Since ![](img/B13208_05_085.png), we get:

![](img/B13208_05_086.png)

Multiplying it by ![](img/B13208_05_087.png) on both sides yields the following result:

![](img/B13208_05_088.png)

Here, *K* is the similarity (kernel) matrix:

![](img/B13208_05_089.png)

As you might recall from the *Solving nonlinear problems using a kernel SVM* section in *Chapter 3*, *A Tour of Machine Learning Classifiers Using scikit-learn*, we use the kernel trick to avoid calculating the pairwise dot products of the examples, *x*, under ![](img/B13208_05_090.png) explicitly by using a kernel function, ![](img/B13208_05_091.png), so that we don't need to calculate the eigenvectors explicitly:

![](img/B13208_05_092.png)

In other words, what we obtain after KPCA are the examples already projected onto the respective components, rather than constructing a transformation matrix as in the standard PCA approach. Basically, the kernel function (or simply kernel) can be understood as a function that calculates a dot product between two vectors—a measure of similarity.

The most commonly used kernels are as follows:

*   The polynomial kernel:![](img/B13208_05_093.png)

    Here, ![](img/B13208_05_094.png) is the threshold and *p* is the power that has to be specified by the user.

*   The hyperbolic tangent (sigmoid) kernel:![](img/B13208_05_095.png)
*   The **radial basis function** (**RBF**) or Gaussian kernel, which we will use in the following examples in the next subsection:![](img/B13208_05_096.png)

    It is often written in the following form, introducing the variable ![](img/B13208_05_097.png).

    ![](img/B13208_05_098.png)

To summarize what we have learned so far, we can define the following three steps to implement an RBF KPCA:

1.  We compute the kernel (similarity) matrix, *K*, where we need to calculate the following:![](img/B13208_05_099.png)

    We do this for each pair of examples:

    ![](img/B13208_05_100.png)

    For example, if our dataset contains 100 training examples, the symmetric kernel matrix of the pairwise similarities would be ![](img/B13208_05_101.png)-dimensional.

2.  We center the kernel matrix, *K*, using the following equation:![](img/B13208_05_102.png)

    Here, ![](img/B13208_05_103.png) is an ![](img/B13208_05_104.png)-dimensional matrix (the same dimensions as the kernel matrix) where all values are equal to ![](img/B13208_05_105.png).

3.  We collect the top *k* eigenvectors of the centered kernel matrix based on their corresponding eigenvalues, which are ranked by decreasing magnitude. In contrast to standard PCA, the eigenvectors are not the principal component axes, but the examples already projected onto these axes.

At this point, you may be wondering why we need to center the kernel matrix in the second step. We previously assumed that we are working with standardized data, where all features have mean zero when we formulate the covariance matrix and replace the dot-products with the nonlinear feature combinations via ![](img/B13208_05_106.png). Thus, the centering of the kernel matrix in the second step becomes necessary, since we do not compute the new feature space explicitly so that we cannot guarantee that the new feature space is also centered at zero.

In the next section, we will put those three steps into action by implementing a KPCA in Python.

## Implementing a kernel principal component analysis in Python

In the previous subsection, we discussed the core concepts behind KPCA. Now, we are going to implement an RBF KPCA in Python following the three steps that summarized the KPCA approach. Using some SciPy and NumPy helper functions, we will see that implementing a KPCA is actually really simple:

[PRE28]

One downside of using an RBF KPCA for dimensionality reduction is that we have to specify the ![](img/B13208_05_107.png) parameter a priori. Finding an appropriate value for ![](img/B13208_05_108.png) requires experimentation and is best done using algorithms for parameter tuning, for example, performing a grid search, which we will discuss in more detail in *Chapter 6*, *Learning Best Practices for Model Evaluation and Hyperparameter Tuning*.

### Example 1 – separating half-moon shapes

Now, let us apply our `rbf_kernel_pca` on some nonlinear example datasets. We will start by creating a two-dimensional dataset of 100 example points representing two half-moon shapes:

[PRE29]

For the purposes of illustration, the half-moon of triangle symbols will represent one class, and the half-moon depicted by the circle symbols will represent the examples from another class:

![](img/B13208_05_12.png)

Clearly, these two half-moon shapes are not linearly separable, and our goal is to *unfold* the half-moons via KPCA so that the dataset can serve as a suitable input for a linear classifier. But first, let's see how the dataset looks if we project it onto the principal components via standard PCA:

[PRE30]

Clearly, we can see in the resulting figure that a linear classifier would be unable to perform well on the dataset transformed via standard PCA:

![](img/B13208_05_13.png)

Note that when we plotted the first principal component only (right subplot), we shifted the triangular examples slightly upward and the circular examples slightly downward to better visualize the class overlap. As the left subplot shows, the original half-moon shapes are only slightly sheared and flipped across the vertical center—this transformation would not help a linear classifier in discriminating between circles and triangles. Similarly, the circles and triangles corresponding to the two half-moon shapes are not linearly separable if we project the dataset onto a one-dimensional feature axis, as shown in the right subplot.

**PCA versus LDA**

Please remember that PCA is an unsupervised method and does not use class label information in order to maximize the variance in contrast to LDA. Here, the triangle and circle symbols were just added for visualization purposes to indicate the degree of separation.

Now, let's try out our kernel PCA function, `rbf_kernel_pca`, which we implemented in the previous subsection:

[PRE31]

We can now see that the two classes (circles and triangles) are linearly well separated so that we have a suitable training dataset for linear classifiers:

![](img/B13208_05_14.png)

Unfortunately, there is no universal value for the tuning parameter, ![](img/B13208_05_109.png), that works well for different datasets. Finding a ![](img/B13208_05_110.png) value that is appropriate for a given problem requires experimentation. In *Chapter 6*, *Learning Best Practices for Model Evaluation and Hyperparameter Tuning*, we will discuss techniques that can help us to automate the task of optimizing such tuning parameters. Here, we will use values for ![](img/B13208_05_111.png) that have been found to produce good results.

### Example 2 – separating concentric circles

In the previous subsection, we saw how to separate half-moon shapes via KPCA. Since we put so much effort into understanding the concepts of KPCA, let's take a look at another interesting example of a nonlinear problem, concentric circles:

[PRE32]

Again, we assume a two-class problem where the triangle shapes represent one class, and the circle shapes represent another class:

![](img/B13208_05_15.png)

Let's start with the standard PCA approach to compare it to the results of the RBF kernel PCA:

[PRE33]

Again, we can see that standard PCA is not able to produce results suitable for training a linear classifier:

![](img/B13208_05_16.png)

Given an appropriate value for ![](img/B13208_05_112.png), let's see if we are luckier using the RBF KPCA implementation:

[PRE34]

Again, the RBF KPCA projected the data onto a new subspace where the two classes become linearly separable:

![](img/B13208_05_17.png)

## Projecting new data points

In the two previous example applications of KPCA, the half-moon shapes and the concentric circles, we projected a single dataset onto a new feature. In real applications, however, we may have more than one dataset that we want to transform, for example, training and test data, and typically also new examples we will collect after the model building and evaluation. In this section, you will learn how to project data points that were not part of the training dataset.

As you will remember from the standard PCA approach at the beginning of this chapter, we project data by calculating the dot product between a transformation matrix and the input examples; the columns of the projection matrix are the top *k* eigenvectors (*v*) that we obtained from the covariance matrix.

Now, the question is how we can transfer this concept to KPCA. If we think back to the idea behind KPCA, we will remember that we obtained an eigenvector (*a*) of the centered kernel matrix (not the covariance matrix), which means that those are the examples that are already projected onto the principal component axis, *v*. Thus, if we want to project a new example, ![](img/B13208_05_113.png), onto this principal component axis, we will need to compute the following:

![](img/B13208_05_114.png)

Fortunately, we can use the kernel trick so that we don't have to calculate the projection, ![](img/B13208_05_115.png), explicitly. However, it is worth noting that KPCA, in contrast to standard PCA, is a memory-based method, which means that we have to *reuse the original training dataset each time to project new examples*.

We have to calculate the pairwise RBF kernel (similarity) between each *i*th example in the training dataset and the new example, ![](img/B13208_05_116.png):

![](img/B13208_05_117.png)

Here, the eigenvectors, *a*, and eigenvalues, ![](img/B13208_05_118.png), of the kernel matrix, *K*, satisfy the following condition in the equation:

![](img/B13208_05_119.png)

After calculating the similarity between the new examples and the examples in the training dataset, we have to normalize the eigenvector, *a*, by its eigenvalue. Thus, let's modify the `rbf_kernel_pca` function that we implemented earlier so that it also returns the eigenvalues of the kernel matrix:

[PRE35]

Now, let's create a new half-moon dataset and project it onto a one-dimensional subspace using the updated RBF KPCA implementation:

[PRE36]

To make sure that we implemented the code for projecting new examples, let's assume that the 26th point from the half-moon dataset is a new data point, ![](img/B13208_05_120.png), and our task is to project it onto this new subspace:

[PRE37]

By executing the following code, we are able to reproduce the original projection. Using the `project_x` function, we will be able to project any new data example as well. The code is as follows:

[PRE38]

Lastly, let's visualize the projection on the first principal component:

[PRE39]

As we can now also see in the following scatterplot, we mapped the example, ![](img/B13208_05_121.png), onto the first principal component correctly:

![](img/B13208_05_18.png)

## Kernel principal component analysis in scikit-learn

For our convenience, scikit-learn implements a KPCA class in the `sklearn.decomposition` submodule. The usage is similar to the standard PCA class, and we can specify the kernel via the `kernel` parameter:

[PRE40]

To check that we get results that are consistent with our own KPCA implementation, let's plot the transformed half-moon shape data onto the first two principal components:

[PRE41]

As we can see, the results of scikit-learn's `KernelPCA` are consistent with our own implementation:

![](img/B13208_05_19.png)

**Manifold learning**

The scikit-learn library also implements advanced techniques for nonlinear dimensionality reduction that are beyond the scope of this book. The interested reader can find a nice overview of the current implementations in scikit-learn, complemented by illustrative examples, at [http://scikit-learn.org/stable/modules/manifold.html](http://scikit-learn.org/stable/modules/manifold.html).

# Summary

In this chapter, you learned about three different, fundamental dimensionality reduction techniques for feature extraction: standard PCA, LDA, and KPCA. Using PCA, we projected data onto a lower-dimensional subspace to maximize the variance along the orthogonal feature axes, while ignoring the class labels. LDA, in contrast to PCA, is a technique for supervised dimensionality reduction, which means that it considers class information in the training dataset to attempt to maximize the class-separability in a linear feature space.

Lastly, you learned about a nonlinear feature extractor, KPCA. Using the kernel trick and a temporary projection into a higher-dimensional feature space, you were ultimately able to compress datasets consisting of nonlinear features onto a lower-dimensional subspace where the classes became linearly separable.

Equipped with these essential preprocessing techniques, you are now well prepared to learn about the best practices for efficiently incorporating different preprocessing techniques and evaluating the performance of different models in the next chapter.