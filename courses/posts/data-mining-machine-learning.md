---
title: Data Mining & Machine Learning
category: AI/ML
semester: 2024 F
---

This course presents core machine learning algorithms implemented $\textbf{from scratch}$ to expose their underlying objective functions, optimization procedures, and inductive biases, independent of library abstractions. The focus is on understanding how and why these algorithms work, rather than on performance tuning or software engineering details.

---
# 1. Linear Regression
Given a dataset with n observations:
$$\{(x^{(i)}, y^{(i)})\}_{i=1}^n$$
where:
- $x^{(i)} \in \mathbb{R}^d$: The $d$-dimensional feature vector for the $i$-th observation
- $y^{(i)} \in \mathbb{R}$: The target value for the $i$-th observation
  
We aim to model the relationship using a linear function:
$$\hat{y} = \mathbf{w}^\top \mathbf{x} + b$$
where:
- $\mathbf{w} \in \mathbb{R}^d$: The weight vector (coefficients).
- $b \in \mathbb{R}$: The bias term (intercept).

## 1.1. Loss Function: Mean Squared Error (MSE)
To learn the parameters $\mathbf{w}$ and $b$, we minimize the **Mean Squared Error** loss:
$$\mathcal{L}(\mathbf{w}, b) = \frac{1}{n} \sum_{i=1}^n \left( y^{(i)} - \hat{y}^{(i)} \right)^2$$
Substituting $\hat{y}^{(i)} = \mathbf{w}^\top \mathbf{x}^{(i)} + b$, the loss becomes:
$$\mathcal{L}(\mathbf{w}, b) = \frac{1}{n} \sum_{i=1}^n \left( y^{(i)} - (\mathbf{w}^\top \mathbf{x}^{(i)} + b) \right)^2$$

## 1.2. Optimization: Normal Equation
The optimal weights $\mathbf{w}$ and bias $b$ can be derived by solving the following closed-form equation (Normal Equation):

$$\mathbf{w} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}$$

## 1.3. Gradient Descent 
Alternatively, we can iteratively update the parameters using **Gradient Descent**:
- Compute the gradient of the loss:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{w}} = -\frac{2}{n} \mathbf{X}^\top (\mathbf{y} - \mathbf{X}\mathbf{w})$$
$$\frac{\partial \mathcal{L}}{\partial b} = -\frac{2}{n} \sum_{i=1}^n (y^{(i)} - \hat{y}^{(i)})$$
- Update the parameters:
$$\mathbf{w} \leftarrow \mathbf{w} - \eta \frac{\partial \mathcal{L}}{\partial \mathbf{w}}$$
$$b \leftarrow b - \eta \frac{\partial \mathcal{L}}{\partial b}$$
Where $\eta$ is the learning rate.
---

# 2. Naive Bayes algorithm

## 2.1. Preprocessing
```python
from sklearn.preprocessing import LabelEncoder

def q1(df, output_file):
    # (i) Strip quotes 
    columns_to_strip = ['race', 'race_o', 'field']
    removed_cnt = 0
    
    # count how many cells are changed after this pre-processing step
    for col in columns_to_strip:
        before_strip = df[col].copy()
        df[col] = df[col].str.strip("'")
        removed_cnt += (before_strip != df[col]).sum()

    # (ii) Convert 'field' to lowercase
    field_before = df['field'].copy()
    df['field'] = df['field'].str.lower()
    standardized_cnt = (field_before != df['field']).sum()
    
    # (iii) Label Encoding for 'gender', 'race', 'race_o', and 'field'
    label_encoders = {}
    columns_to_encode = ['gender', 'race', 'race_o', 'field']

    for col in columns_to_encode:
        # mapping each categorical value of an attribute to an integer number
        label_encoders[col] = LabelEncoder()
        df[col] = label_encoders[col].fit_transform(df[col].astype(str))

    # (iv) Normalize preference scores
    # Normalize preference scores for participant
    df['participant_total'] = df[PREFERENCE_SCORES_PARTICIPANT].sum(axis=1)
    for col in PREFERENCE_SCORES_PARTICIPANT:
        df[col] = df[col] / df['participant_total']

    # Normalize preference scores for partner
    df['partner_total'] = df[PREFERENCE_SCORES_PARTNER].sum(axis=1)
    for col in PREFERENCE_SCORES_PARTNER:
        df[col] = df[col] / df['partner_total']

    # Save the new CSV
    df.drop(columns=['participant_total', 'partner_total'], inplace=True)
    df.rename(columns={"intelligence_parter": "intelligence_partner"}, inplace=True)
    df.to_csv(output_file, index=False)
```

## 2.2. Naive Bayes Classifier
### Prior Probability
The prior probability, denoted as P(Class), represents the likelihood of each class occurring in the dataset. 
```python
self.class_prior = {
    'yes': len(class_yes) / len(df_sampled),
    'no': len(class_no) / len(df_sampled)
}
```
### Likelihood Estimation with Discretization and DP
After determining the bin edges, each feature value is assigned to a discrete bin index, which is used to compute the likelihood of each feature given a class. The likelihood estimation is defined as:
```python
self.likelihood_dp = {'yes': {}, 'no': {}}
self.likelihood_dp['yes'][feature] = self._calculate_likelihood(class_yes, feature)
self.likelihood_dp['no'][feature] = self._calculate_likelihood(class_no, feature)

def _calculate_likelihood(self, subset, feature):
    likelihoods = {}
    for bin_value in range(self.bin_count):
        bin_count = subset[feature].value_counts().get(bin_value, 0) + 1
        likelihoods[bin_value] = bin_count / (len(subset) + self.bin_count)
    return likelihoods
```

### Logarithmic Probability for Numerical Stability
To prevent numerical underflow caused by multiplying small probability values, the model uses the logarithmic transformation.
```python
log_prob = np.log(prior)
log_prob += np.log(self.likelihood_dp[class_label][feature].get(feature_value, 1e-6))
Evaluation with Accuracy Metric
The performance of the model is evaluated using accuracy, defined as:
correct_predictions = 0
total_samples = len(df)
for _, row in df.iterrows():
    true_class = "yes" if row["decision"] == 1 else "no"
    predicted_class = self.predict(row)
    if predicted_class == true_class:
        correct_predictions += 1
return correct_predictions / total_samples
```

## 2.3. Naive Bayes Class
```python
class MyNaiveBayes:
    def __init__(self, continuous_columns, bin_count=5):
        self.continuous_columns = continuous_columns
        self.bin_count = bin_count
        self.class_prior = {} 
        self.likelihood_dp = {}  
        
    def discretize_columns(self, df):
        for col in self.continuous_columns:
            bin_edges = np.linspace(df[col].min(), df[col].max(), self.bin_count + 1)
            df[col] = pd.cut(df[col], bins=bin_edges, labels=False, include_lowest=True)
        return df

    def train(self, df, t_frac=1.0):
        df_sampled = df.sample(frac=t_frac, random_state=47)
        
        class_yes = df_sampled[df_sampled["decision"] == 1]
        class_no = df_sampled[df_sampled["decision"] == 0]

        # Prior Probability
        self.class_prior = {
            'yes': len(class_yes) / len(df_sampled),
            'no': len(class_no) / len(df_sampled)
        }
        
        self.likelihood_dp = {'yes': {}, 'no': {}}

        for feature in self.continuous_columns:
            self.likelihood_dp['yes'][feature] = self._calculate_likelihood(class_yes, feature)
            self.likelihood_dp['no'][feature] = self._calculate_likelihood(class_no, feature)

    def _calculate_likelihood(self, subset, feature):
        likelihoods = {}
        for bin_value in range(self.bin_count):
            # Conditional Probability
            bin_count = subset[feature].value_counts().get(bin_value, 0) + 1
            likelihoods[bin_value] = bin_count / (len(subset) + self.bin_count)
        return likelihoods

    def predict(self, instance):
        best_class, highest_prob = None, float('-inf')

        for class_label, prior in self.class_prior.items():
            log_prob = np.log(prior) 
            
            for feature in self.continuous_columns:
                feature_value = instance.get(feature)
                if feature_value is not None:
                    num = self.likelihood_dp[class_label][feature].get(feature_value, 1e-6)
                    log_prob += np.log(num)

            if log_prob > highest_prob:
                highest_prob = log_prob
                best_class = class_label

        return best_class

    def evaluate(self, df):
        correct_predictions = 0
        total_samples = len(df)

        for _, row in df.iterrows():
            true_class = "yes" if row["decision"] == 1 else "no"
            predicted_class = self.predict(row)
            if predicted_class == true_class:
                correct_predictions += 1

        # Accuracy
        return correct_predictions / total_samples
```
---

# 3. Logistic Regression
## 3.1. Sigmoid Function 
Map any real-valued number into a value between 0 and 1: $\sigma(z) = \frac{1}{1 + e^{-z}}$
```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```

## 3.2. Cost Function
$$J(w) = \frac{1}{m} \sum_{i=1}^{m} \left[ -y^{(i)} \log(h_\theta(x^{(i)})) - (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)})) \right] + \frac{\lambda}{2m} \|w\|^2$$

- m is the number of data points
- $y^{(i)} $ is actual label
- $ h_\theta(x^{(i)}) $ is the predicted value (using the sigmoid function)
- w is the weight vector
- λ is the regularization parameter
```python
def compute_cost(X, y, w, lambda_):
    m = len(y)
    h = sigmoid(np.dot(X, w))
    # L2 regularization term
    reg_term = (lambda_ / (2 * m)) * np.sum(np.square(w))
    cost = (-1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h)) + reg_term
    return cost
```

## 3,3. Gradient Descent 
$$w := w - \alpha \nabla J(w)$$
- α is the learning rate,
- ∇ 𝐽(𝑤) is the gradient of the cost function with respect to the weights.
```python
def lr_gradient_descent(X, y, w, learning_rate, lambda_, tol, max_iter):
    m = len(y)
    for i in range(max_iter):
        h = sigmoid(np.dot(X, w))
        gradient = (1/m) * np.dot(X.T, (h - y)) + (lambda_/m) * w
        w_new = w - learning_rate * gradient
        if np.linalg.norm(w_new - w) < tol: break
        w = w_new    
    return w
```

## 3.4. Accuracy Calculation
$$\text{Accuracy} = \frac{\text{Number of correct predictions}}{\text{Total number of predictions}} \times 100$$

---
# 4. Linear SVM 
## 4.1. Hinge Loss
$$L(w, x^{(i)}, y^{(i)}) = \max(0, 1 - y^{(i)} (w^T x^{(i)}))$$
```python
def compute_hinge_loss(X, y, w, lambda_):
    m = len(y)
    margins = y * np.dot(X, w)
    loss = np.maximum(0, 1 - margins)
    cost = np.mean(loss) + (lambda_ / 2) * np.sum(np.square(w))
    return cost
```

## 4.2. Subgradient Descent
$$w := w - \alpha \cdot \nabla_w J(w)$$
```python
def compute_subgradient(X, y, w, lambda_):
    margins = y * np.dot(X, w)
    gradient = np.where(margins[:, np.newaxis] < 1, -y[:, np.newaxis] * X, 0)
    gradient = np.mean(gradient, axis=0) + lambda_ * w
    return gradient

def svm_subgradient_descent(X, y, w, learning_rate, lambda_, tol, max_iter):
    for i in range(max_iter):
        gradient = compute_subgradient(X, y, w, lambda_)
        w_new = w - learning_rate * gradient
        if np.linalg.norm(w_new - w) < tol:
            break
        w = w_new
    return w
```
---
# 5. Decision Trees
## 5.1. Splitting Criterion (Gini Index)
Gini Index measures the impurity of a node $S$: $Gini(S) = 1 - \sum_{k=1}^K p_k^2 $
- $K$ is the number of classes,
- $p_k$ is the proportion of class $$k$$ in $S$.
```python
def gini(self, y):
    """Calculate Gini index for a split."""
    unique, counts = np.unique(y, return_counts=True)
    prob = counts / len(y)
    return 1 - np.sum(prob ** 2)
```
## 5.2. Gini-gain
Gini-gain measures the reduction in impurity when a split is applied based on attribute $A$:
$$GiniGain(S, A) = Gini(S) - \sum_{v \in A} \frac{|S_v|}{|S|} Gini(S_v) $$
- $|S|$ is the total number of samples,
- $S_v$ is the subset of $S$ where attribute $$A$$ takes value $v$.
```python
def gini_gain(self, y, splits):
    """Calculate Gini gain after a split."""
    total = len(y)
    weighted_gini = 0
    for split in splits:
        weighted_gini += (len(split) / total) * self.gini(split)
    return self.gini(y) - weighted_gini
```

## 5.3. Information Gain (Entropy)
An alternative splitting criterion is Information Gain, which is based on entropy:
$$Entropy(S) = - \sum_{k=1}^K p_k \log_2 p_k $$
$$IG(S, A) = Entropy(S) - \sum_{v \in A} \frac{|S_v|}{|S|} Entropy(S_v) $$

Stopping Criteria
- Depth Limit: $Depth \leq D_{max} $
- Minimum Sample Limit: $|S| \geq N_{min} $
  
Prediction
- At a leaf node, the predicted class is the one with the highest count:
$$\hat{y} = \text{argmax}_{k} \, \text{count}(y_k \in S) $$
```python
def predict_one(self, x, node):
    if isinstance(node, dict):
        if x[node["feature"]] == 0:
            return self.predict_one(x, node["left"])
        else:
            return self.predict_one(x, node["right"])
    else:
        return node

def predict(self, X):
    """Predict the class for all samples in X."""
    return np.array([self.predict_one(x, self.tree) for x in X])
```
---
# 6. Bagging
## 6.1. Bootstrap Sampling
Generate $T$ bootstrap samples from the dataset $D = \{(x_1, y_1), \dots, (x_n, y_n)\}$:

$$D_i = \{(x_1', y_1'), \dots, (x_n', y_n')\} \quad \text{where } (x_j', y_j') \sim D $$
```python
def bootstrap_sample(self, X, y):
    """Create a bootstrap sample of the dataset."""
    n_samples = X.shape[0]
    indices = np.random.choice(n_samples, n_samples, replace=True)
    return X[indices], y[indices]
```

## 6.2. Majority Voting
Combine predictions from $T$ models using majority voting:
$$\hat{y} = \text{argmax}_k \sum_{t=1}^T I(\hat{y}_t = k)$$
where:
- $\hat{y}_t$ is the prediction from the $t$-th model,
- $I(\cdot)$ is the indicator function.
```python
def predict(self, X):
    predictions = np.array([tree.predict(X) for tree in self.trees])
    # Majority voting
    return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)
```

### Bias-Variance Decomposition
- The expected error of the ensemble model can be expressed as:
$$\text{Error}(x) = \text{Bias}^2(x) + \text{Variance}(x) + \sigma^2 $$

---
# 7. Random Forest
## 7.1. Random Feature Subsampling
- For each node, select a random subset of features:
$$F \sim \text{Uniform}(1, \sqrt{p}) $$
where:
- $$p$$ is the total number of features,
- $$F$$ is the subset of features.
 
## 7.2. Gini Index and Gini-gain (on Feature Subset)
- Use the selected features $$F$$ to calculate Gini-gain:
$$GiniGain(S, A) = Gini(S) - \sum_{v \in A} \frac{|S_v|}{|S|} Gini(S_v) $$

## 7.3. Majority Voting
- Combine predictions from $$T$$ models using majority voting:
$$\hat{y} = \text{argmax}_k \sum_{t=1}^T I(\hat{y}_t = k) $$
```python
def predict(self, X):
    predictions = []
    for tree, features in self.trees:
        preds = tree.predict(X[:, features])
        predictions.append(preds)

    predictions = np.array(predictions)
    # Majority voting
    return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)
```
## 7.4. Out-of-Bag (OOB) Error
Evaluate model performance on samples not included in the bootstrap:
$$\text{OOB Error} = \frac{1}{n} \sum_{i=1}^n I(\hat{y}_\text{OOB}^i \neq y_i) $$

---
# 8. Neural Network

## 8.1. Forward Propagation
$\textbf{Hidden Layer Calculation}$
Weighted Sum: $z^{(1)} = X W^{(1)} + b^{(1)}$
- $X$: Input matrix $(n_{\text{samples}} \times n_{\text{features}})$  
- $W^{(1)}$: Weights from input to hidden layer $(n_{\text{features}} \times n_{\text{hidden}})$ 
- $b^{(1)}$: Biases for hidden layer $(1 \times n_{\text{hidden}})$ 
Activation of the hidden layer:
$$a^{(1)} = \text{ReLU}(z^{(1)})$$

## 8.2. Output Layer Calculation  
$$z^{(2)} = a^{(1)} W^{(2)} + b^{(2)}$$
- $a^{(1)}$: Activation of the hidden layer $(n_{\text{samples}} \times n_{\text{hidden}})$ 
- $W^{(2)}$: Weights from hidden to output layer $(n_{\text{hidden}} \times n_{\text{output}})$ 
- $b^{(2)}$: Biases for output layer $(1 \times n_{\text{output}})$ 
- Activation of the output layer (Softmax for multi-class classification):
$$a^{(2)} = \text{Softmax}(z^{(2)})$$
$$\text{Softmax}(z)_i = \frac{e^{z_i}}{\sum_{j} e^{z_j}}$$
```python
def forward(self, X):
    # Forward pass
    self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
    self.hidden_output = self.relu(self.hidden_input)  # Using ReLU for hidden layers
    bias = self.bias_output
    z_output = np.dot(self.hidden_output, self.weights_hidden_output)
    self.output_input = z_output + bias
    self.output_output = self.softmax(self.output_input)

    return self.output_output
```

## 8.3. Loss Function
For classification problems, Cross-Entropy Loss is used:
$$\mathcal{L} = -\frac{1}{n} \sum_{i=1}^n \sum_{k=1}^K y_{i,k} \log(\hat{y}_{i,k})$$
- $n$: Number of samples  
- $K$: Number of classes  
- $y_{i,k}$: One-hot encoded true label for sample $i$, class $$k$$  
- $\hat{y}_{i,k}$: Predicted probability for sample $i$, class $$k$$ 

## 8.4. Backward Propagation
### Output Layer Gradient
Gradient of the loss with respect to the output layer pre-activation:
$$\frac{\partial \mathcal{L}}{\partial z^{(2)}} = a^{(2)} - y$$

### Gradients for Weights and Biases (Output Layer):
$$\frac{\partial \mathcal{L}}{\partial W^{(2)}} = a^{(1)^T} \cdot \frac{\partial \mathcal{L}}{\partial z^{(2)}}$$
$$\frac{\partial \mathcal{L}}{\partial b^{(2)}} = \sum \frac{\partial \mathcal{L}}{\partial z^{(2)}}$$

### Hidden Layer Gradient
Gradient of the hidden layer activation:
$$\frac{\partial \mathcal{L}}{\partial a^{(1)}} = \frac{\partial \mathcal{L}}{\partial z^{(2)}} \cdot W^{(2)^T}$$
$$\frac{\partial \mathcal{L}}{\partial z^{(1)}} = \frac{\partial \mathcal{L}}{\partial a^{(1)}} \cdot \text{ReLU}'(z^{(1)})$$

### Gradients for Weights and Biases (Hidden Layer):
$$\frac{\partial \mathcal{L}}{\partial W^{(1)}} = X^T \cdot \frac{\partial \mathcal{L}}{\partial z^{(1)}}$$
$$\frac{\partial \mathcal{L}}{\partial b^{(1)}} = \sum \frac{\partial \mathcal{L}}{\partial z^{(1)}}$$
```python
def backward(self, X, y_true, y_pred):
    n_samples = X.shape[0]

    # Output layer gradients
    d_output_input = y_pred - y_true
    d_weights_hidden_output = np.dot(self.hidden_output.T, d_output_input) / n_samples
    d_bias_output = np.sum(d_output_input, axis=0) / n_samples

    # Hidden layer gradients
    d_hidden_output = np.dot(d_output_input, self.weights_hidden_output.T)
    d_hidden_input = d_hidden_output * self.relu_derivative(self.hidden_input)
    d_weights_input_hidden = np.dot(X.T, d_hidden_input) / n_samples
    d_bias_hidden = np.sum(d_hidden_input, axis=0) / n_samples

    # Update weights and biases
    self.weights_hidden_output -= self.learning_rate * d_weights_hidden_output
    self.bias_output -= self.learning_rate * d_bias_output
    self.weights_input_hidden -= self.learning_rate * d_weights_input_hidden
    self.bias_hidden -= self.learning_rate * d_bias_hidden
```

## 8.5. Weight Update
Using Gradient Descent, weights and biases are updated as follows:
$$W^{(l)} \leftarrow W^{(l)} - \eta \cdot \frac{\partial \mathcal{L}}{\partial W^{(l)}}$$
$$b^{(l)} \leftarrow b^{(l)} - \eta \cdot \frac{\partial \mathcal{L}}{\partial b^{(l)}}$$

---
# 9. K means
```python
def kmeans(X, K, max_iter=50, random_seed=0):
    np.random.seed(random_seed)
    N, D = X.shape
    
    initial_indices = np.random.choice(N, K, replace=False)
    centroids = X[initial_indices]

    for _ in range(max_iter):
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        cluster_labels = np.argmin(distances, axis=1)

        new_centroids = np.array([X[cluster_labels == k].mean(axis=0) for k in range(K)])
        
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    wc_ssd = sum(
        np.sum(np.linalg.norm(X[cluster_labels == k] - centroids[k], axis=1)**2)
        for k in range(K)
    )
    
    return cluster_labels, centroids, wc_ssdx
```
---
# 10. Gaussian Mixture Models (GMM) and Expectation–Maximization (EM)
## 10.1 Model Definition
A Gaussian Mixture Model represents the data distribution as a weighted sum of Gaussian components:
$$p(x) = \sum_{k=1}^{K} \pi_k \, \mathcal{N}(x \mid \mu_k, \Sigma_k)$$
where:
- $\pi_k \ge 0$ and $\sum_{k=1}^K \pi_k = 1$ are mixing coefficients,
- $\mu_k$ is the mean of component $k$,
- $\Sigma_k$ is the covariance of component $k$.
## 10.2. Expectation–Maximization (EM)
EM maximizes the log-likelihood by alternating between soft assignment and parameter updates.
## 10.3. E-step (Responsibilities)
$$\gamma_{ik} = P(z_i = k \mid x_i) = \frac{\pi_k \mathcal{N}(x_i \mid \mu_k, \Sigma_k)}{\sum_{j=1}^{K} \pi_j \mathcal{N}(x_i \mid \mu_j, \Sigma_j)}$$
M-step (Parameter Updates)
Let
$$N_k = \sum_{i=1}^n \gamma_{ik}$$
Then the updates are:
$$\mu_k = \frac{1}{N_k} \sum_{i=1}^n \gamma_{ik} x_i$$
$$\Sigma_k = \frac{1}{N_k} \sum_{i=1}^n \gamma_{ik}
(x_i - \mu_k)(x_i - \mu_k)^\top$$
$$\pi_k = \frac{N_k}{n}$$

---
# 11. Stacking (Mathematical and Conceptual View)
Stacking combines multiple base models by learning a meta-model over their predictions.
## 11.1. Model Formulation
Given base learners $f_1, f_2, \dots, f_M$, the stacked predictor is:
$$\hat{y} = g\big(f_1(x), f_2(x), \dots, f_M(x)\big)$$
where:
- $f_m(x)$ are base model predictions,
- $g(\cdot)$ is a meta-learner (typically linear or logistic).