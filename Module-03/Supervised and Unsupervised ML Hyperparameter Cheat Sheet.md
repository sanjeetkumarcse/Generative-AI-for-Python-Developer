# Supervised and Unsupervised ML Hyperparameter Cheat Sheet

## Supervised Learning

| Algorithm                            | Hyperparameter     | Purpose                       | Typical Range / Values       | Example Use Case                 |
| ------------------------------------ | ------------------ | ----------------------------- | ---------------------------- | -------------------------------- |
| Linear Regression                    | fit_intercept      | Include intercept term        | True / False                 | House price prediction           |
|                                      | normalize          | Scale features                | True / False                 | Sales forecasting                |
|                                      | alpha              | Regularization strength       | 0.01–10                      | Ridge/Lasso regression           |
| Logistic Regression                  | C                  | Inverse regularization        | 0.01–100                     | Spam detection, churn prediction |
|                                      | penalty            | Regularization type           | l1, l2, elasticnet, none     | Classification                   |
|                                      | solver             | Optimization algorithm        | liblinear, saga, lbfgs       | Logistic regression training     |
|                                      | max_iter           | Max iterations                | 100–1000                     | Convergence control              |
| Decision Tree                        | max_depth          | Max tree depth                | 1–None                       | Fraud detection                  |
|                                      | min_samples_split  | Min samples to split          | 2–100                        | Customer segmentation            |
|                                      | min_samples_leaf   | Min samples at leaf           | 1–50                         | Classification tasks             |
|                                      | max_features       | Features considered per split | auto, sqrt, log2, int        | Decision splits                  |
|                                      | criterion          | Split quality metric          | gini, entropy                | Tree construction                |
| Random Forest                        | n_estimators       | Number of trees               | 100–1000+                    | Feature importance               |
|                                      | max_depth          | Max tree depth                | 1–None                       | Classification/regression        |
|                                      | min_samples_split  | Min samples to split          | 2–100                        | Tree growth control              |
|                                      | max_features       | Features per split            | auto, sqrt, log2, int        | Feature sampling                 |
|                                      | bootstrap          | Sample with replacement       | True / False                 | Bagging control                  |
| Gradient Boosting (XGBoost/LightGBM) | learning_rate      | Step size shrinkage           | 0.01–0.3                     | Predictive modeling              |
|                                      | n_estimators       | Number of boosting rounds     | 100–1000                     | Model complexity                 |
|                                      | max_depth          | Max tree depth                | 3–10                         | Overfitting control              |
|                                      | subsample          | Sample fraction per tree      | 0.5–1                        | Regularization                   |
|                                      | colsample_bytree   | Features per tree fraction    | 0.5–1                        | Boosting diversity               |
|                                      | gamma              | Min loss reduction for split  | 0–5                          | Pruning threshold                |
| SVM                                  | C                  | Regularization parameter      | 0.01–100                     | Image classification             |
|                                      | kernel             | Kernel type                   | linear, poly, rbf, sigmoid   | Classification tasks             |
|                                      | gamma              | Kernel coefficient            | scale, auto, 0.001–1         | Influence radius                 |
|                                      | degree             | Polynomial degree             | 2–5                          | Poly kernel only                 |
| KNN                                  | n_neighbors        | Number of neighbors           | 1–50                         | Recommendation systems           |
|                                      | weights            | Uniform/distance weighting    | uniform, distance            | Prediction control               |
|                                      | p                  | Distance metric               | 1 = Manhattan, 2 = Euclidean | Clustering / classification      |
| Neural Network (MLP)                 | hidden_layer_sizes | Neurons per layer             | (16–1024+)                   | Handwriting recognition          |
|                                      | activation         | Non-linearity                 | relu, tanh, logistic         | Learning complex patterns        |
|                                      | solver             | Optimizer                     | adam, sgd, lbfgs             | Training control                 |
|                                      | alpha              | L2 regularization             | 0.0001–0.01                  | Overfitting prevention           |
|                                      | learning_rate      | Step size control             | constant, adaptive           | Gradient updates                 |
|                                      | batch_size         | Samples per update            | 32–512                       | Memory and convergence           |
|                                      | max_iter           | Max iterations                | 200–1000+                    | Training control                 |

## Unsupervised Learning

| Algorithm               | Hyperparameter     | Purpose                      | Typical Range / Values         | Example Use Case         |
| ----------------------- | ------------------ | ---------------------------- | ------------------------------ | ------------------------ |
| K-Means                 | n_clusters         | Number of clusters           | 2–10+                          | Customer segmentation    |
|                         | init               | Centroid initialization      | k-means++, random              | Convergence speed        |
|                         | max_iter           | Max iterations               | 100–500                        | Algorithm stopping       |
|                         | n_init             | Number of initializations    | 10–50                          | Avoid poor local minima  |
|                         | tol                | Convergence tolerance        | 1e-4–1e-2                      | Stopping criterion       |
| Hierarchical Clustering | n_clusters         | Number of clusters           | 2–10+                          | Document clustering      |
|                         | linkage            | Distance between clusters    | ward, complete, average        | Agglomeration method     |
|                         | affinity           | Distance metric              | euclidean, manhattan, cosine   | Cluster computation      |
| DBSCAN                  | eps                | Max distance for neighbor    | 0.1–10                         | Anomaly detection        |
|                         | min_samples        | Min points to form cluster   | 3–10                           | Density threshold        |
|                         | metric             | Distance metric              | euclidean, manhattan, cosine   | Neighbor calculation     |
| Gaussian Mixture (GMM)  | n_components       | Number of mixture components | 2–10+                          | Overlapping clusters     |
|                         | covariance_type    | Covariance shape             | full, tied, diag, spherical    | Cluster shape modeling   |
|                         | tol                | Convergence tolerance        | 1e-3–1e-5                      | EM stopping criterion    |
|                         | max_iter           | Max EM iterations            | 100–500                        | Training control         |
| PCA                     | n_components       | Number of components         | 1–n_features                   | Dimensionality reduction |
|                         | svd_solver         | Decomposition method         | auto, full, arpack, randomized | Algorithm selection      |
|                         | whiten             | Scale components             | True / False                   | Standardization          |
| t-SNE                   | n_components       | Output dimensions            | 2 or 3                         | Visualization            |
|                         | perplexity         | Balances local/global        | 5–50                           | Neighbor weighting       |
|                         | learning_rate      | Step size                    | 10–1000 (typical 200)          | Optimization control     |
|                         | n_iter             | Iterations                   | 250–1000+                      | Training control         |
| Autoencoders            | hidden_layer_sizes | Encoder/decoder neurons      | 64–512                         | Anomaly detection        |
|                         | activation         | Non-linearity                | relu, sigmoid, tanh            | Feature learning         |
|                         | optimizer          | Training optimizer           | adam, sgd, rmsprop             | Weight updates           |
|                         | learning_rate      | Step size                    | 0.001–0.01                     | Gradient control         |
|                         | batch_size         | Samples per update           | 32–256                         | Memory / convergence     |
|                         | epochs             | Number of passes             | 50–500                         | Training time            |
| Isolation Forest        | n_estimators       | Number of trees              | 100–500                        | Outlier detection        |
|                         | max_samples        | Samples per tree             | auto / int                     | Randomization            |
|                         | contamination      | Fraction of outliers         | 0–0.5                          | Threshold control        |
|                         | max_features       | Features per tree            | 1–n_features                   | Subspace sampling        |
