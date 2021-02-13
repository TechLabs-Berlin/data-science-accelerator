# Support Vector Machines
Support vector machines (SVMs) are another subset of supervised learning methods that can be used for either classification or regression tasks. To find out what makes Support Vector Machines unique, and get a more expansive overview of SVMs, we would encourage students to follow the tutorial linked below:  
[Support Vector Machines Tutorial – Learn to implement SVM in Python](https://data-flair.training/blogs/svm-support-vector-machine-tutorial/)
> This tutorial will guide you through the following understanding:
> * What are SVMs?
> * A visual introduction to how they work
> * Implementation in Python
> * Pros and Cons of this algorithm
> * Model Parameter Tuning
> * Applications of SVM

## What makes SVMs such an important algorithm?
1. SVMs are highly effective in high-dimensional spaces, making them one of the most frequently used solutions for dimensional data.
2. SVMs tend to generally find an optimal solution due to underlying convex optimization.
3. SVMs can deal with linearity or non-linearity in the data, by choice of appropriate kernels
4. SVMs make use of the the **Kernel Trick** to handle non-linearly separable data
### Kernel Trick
> A kernel is simply a transformation of a two-dimensional plane to a higher dimensional space.

Sometimes separability cannot be achieved in a linear space. By application of a kernel trick, non-linear data is projected in to a higher dimensional space, where separation is much easier.

![kernel-trick](./svm_kernel_trick.jpg)

#### Some general intuitions behind the Kernel trick:
- Kernels generalize the notion of "inner product similarity"
- The goal with SVMs is to maximize the margin of separation
- 2 vectors (examples used in training) that are similar and predict different classes maximize the width of margin
- On the other hand, two vectors that are similar and of the same class, add redundant information to the model
- The inner dot product offers a measure of similarity between vectors. By disregarding features that have a dot product of zero, the kernel ensures the SVM focussing on critical examples.

## Implementing SVMs with `scikit-learn`
Fortunately there are a host of SVM implementations in most modern languages. One of the most popular (dueto ease of use) pythonic flavors is collection of SVMs in `scikit-learn`. Follow along for a simple example with the [Iris dataset](https://archive.ics.uci.edu/ml/datasets/iris):
```python
from sklearn.datasets import load_iris
iris = load_iris()
```
Let's process the dataset to create predictors and targets.
```python
X = iris.data
y = iris.target
```
It's common in machine learning to evaluate performance using train-test splits. We use a simple 80/20 split in this example.
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
```
Let's train a Support Vector Classifier1
```python
from sklearn.svm import SVC
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)
```
Making predictions is as simple as:
```python
y_pred = clf.predict(X_test)
```
Let's evaluate this model using standard PRF metrics (precision, recall, F1-score).
```python
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))
```
##### OUTPUT
```
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        10
           1       1.00      1.00      1.00         8
           2       1.00      1.00      1.00        12

    accuracy                           1.00        30
   macro avg       1.00      1.00      1.00        30
weighted avg       1.00      1.00      1.00        30
```

### Some Useful Kernels
1. The Polynomial Kernel
   ```python
   poly_clf = SVC(kernel='poly', degree=8)
   ```
   > For higher-order relationships. Requires parameter `degree` to be specified for degree  of polynomial.
2. Gaussian Kernel
   ```python
   gauss_clf = SVC(kernel='rbf')
   ```
3. Sigmoid Kernel
   ```python
   sig_clf = SVC(kernel='sigmoid')
   ```

#### Suggested Exercise
Create some classification reports to compare performance with different kernels.

## Optional Reading
Here are some additional resources to explore SVMs further if you are so inclined:
- [Idiot's Guide to SVMs](https://web.mit.edu/6.034/wwwbob/svm.pdf)
- [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/05.07-support-vector-machines.html)