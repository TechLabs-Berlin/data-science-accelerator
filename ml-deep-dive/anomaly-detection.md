# Anomaly Detection
Let's start with an overview of the anomaly detection task, it's definition and a short intuitive introduction to the topic on Coursera:
- [Introduction to Anomaly Detection](https://www.coursera.org/lecture/ai/introduction-to-anomaly-detection-ASPv0)

You now have some basic knowledge of the topic. You should know that:
1. An outlier is an observation within a group that can have an unusual value that does not belong to the general set of observations (pertaining to that group).
2. Neural nets are quite commonly used in application, to differentiate outliers in a dataset today. However, they were'nt always the state-of-the-art method for this task.
3. Some other key algorithms that can be used to detect anomalies are tabulated below.

### Summary Table of Key Algorithms

 Abbreviation | Algorithm                              | Overview
--------------|----------------------------------------|------------------------------------------------------
KNN           | K Nearest Neighbors                    | Proposes a new definition of distance-based outlier<br/> Considers for each point the sum of the distances from its k nearest neighbors, called weight<br/>Outliers are those points having the largest values of weight
LOF           | Local Outlier Factor                   | Uses density-based outlier detection to identify local outliers, i.e, w.r.t local neighborhood (higher the LOF, more likely to be an anomaly)
IForest       | Isolated Forest                        | Underlying principle, is that its easier to separate isolated points over points that have a lot of neighbors. [Additional reading](https://quantdare.com/isolation-forest-algorithm/)
OCSVM         | One-Class Support Vector Machine (SVM) | [Reading](http://rvlasveld.github.io/blog/2013/07/12/introduction-to-one-class-support-vector-machines/)
\-            | Auto-encoder Ensembles                 | [Linked Paper](http://saketsathe.net/downloads/autoencode.pdf)
COPOD         | Copula-Based Outlier Detection         | [Linked Paper](http://www.andrew.cmu.edu/user/yuezhao2/papers/20-icdm-copod.pdf)

## Additional Resources
For a deeper dive in to the topic, our recommended extra reading:
- [Anomaly Detection: A Tutorial](http://webdocs.cs.ualberta.ca/~icdm2011/downloads/ICDM2011_anomaly_detection_tutorial.pdf)
- [Tutorial: Anomaly Detection in Networks](https://veena-mendiratta.blog/tutorial-anomaly-detection-in-networks/)