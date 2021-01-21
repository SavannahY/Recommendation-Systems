## Experiment Design and Recommendation Engines

### introduction to Recommendation Engines

Techniques and measures of effectiveness

### Matrix Factorization for Recommendations

One of the most populur techniques for recommendation engines known as FunkSVD. 

### Concept of Experiment Design

Correlation does not imply causation

#### Key feature

1) Comparison between groups

2) Control other variables

#### Types

Between-subject design A/B Test: Experiment comparing the performance of two groups: a control group "A" and an experiment group "B".

#### Sample

simple random sample

tratified random sampling



## Recommendation engines

- Knowledge Based Recommendations
- Collaborative Filtering Based Recommendations
  -  Model Based Collaborative Filtering
  - Neighborhood Based Collaborative Filtering
- Content Based Recommendations

Similarity Metrics
In order to implement Neighborhood Based Collaborative Filtering, you will learn about some common ways to measure the similarity between two users (or two items) including:

- Pearson's correlation coefficient
- Spearman's correlation coefficient
- Kendall's 
- Euclidean Distance
- Manhattan Distance



## Matrix Factorization for Recommendation

### Metrics

Classificaiton: Precision, Recall, Accuracy

Regression: Mean-Squared Error, R-Squared

### FunkSVD

Funk SIngular Value Decomposition

#### latent factor

DIfferent from 'observed' factor

Values that are not directly obserable in our data, but may be recognized when looking at relationships and trends that exist between obeserved data values.

Let k be the number of latent features used, n be the number of users, and m be the number of items. With this in mind, match each matrix to its corresponding dimensions. For the below, consider rows-columns as the structure.

U n-k

Sigma k-k

V m-k

V-transpose k-m



## Matrix Factorization with Distributed SGD

## Introduction

In this assignment, we will work on a recommendation problem based on the real world dataset from both Yelp and Amazon Review Datasets. The ultimate goal of this project is to build the popular matrix factorization based recommendation model and optionally on Spark.

During this assignment, you will need to implement the algorithm from scratch to achive this goal by yourself. But don't worry, this notebook will guide you through those process step by step. To be specific, the tasks in this assignment include:

- Data Pre-processing and Exploratory Data Analysis (EDA)
- Matrix Factorization Algorithm Implementation
- Distributed Stochastic Gradient Descent

Hopefully, after this assignment you can

- Get the idea of how the matrix factorization algorithm works
- Understand how to train the matrix factorization model through SGD
- Get familiar with the basic idea of Spark and how it can help facilitate matrix factorization model learning
- Have hands-on experiences for implementing recommendation algorithm and optionally on Spark

## Dataset

For this project, we will use 2 big datasets, one from Yelp Reviews and another from Amazon Product Reviews. Each of these two datasets consists of a few sub-datasets, for example, Yelp datasets have reviews from 6 different states and Amazon datasets have reviews from 18 different product categories. These sub-datasets are
organized into different folders, and the data is stored in a `.csv` file called `matrix.csv`. To help you focus more on the model/algorithm side, we have pre-processed the raw data for you.

#### Data Format
All the `matrix.csv` files have exactly the same format, there will be 3 columns in the `.csv` file, first column is user_id, second column is item_id and 3rd one is user rating. For the user rating column, both datasets have rating among 1, 2, 3, 4 and 5 (maps to 1-5 stars).


#### Dataset Statistics
Below you can find the dataset statistics, you may want to try your implementation on some smaller datasets first. Given that the rating behaviors can vary over different geo regions or product categories, we use these many sub-datasets to make sure any performance change on your algorithm can be statistical significantly presented.

| Sub-dataset Name (Yelp Reviews) | Number of Reviews |
| ------------------------------- | ----------------- |
| Illinois (IL)                   | 12K               |
| Wisconsin (WI)                  | 43K               |
| Pennsylvania (PA)               | 66K               |
| North Carolina (NC)             | 95K               |
| Arizona (AZ)                    | 586K              |
| Nevada (NV)                     | 674K              |

| Sub-dataset Name (Amazon Reviews) | Number of Reviews |
| --------------------------------- | ----------------- |
| Musical Instruments               | 500K              |
| Instant Video                     | 583K              |
| Digital Music                     | 836K              |
| Baby                              | 915K              |
| Patio, Lawn and Garden            | 993K              |
| Pet Supplies                      | 1.23M             |
| Office Products                   | 1.24M             |
| Grocery & Groumet Food            | 1.29M             |
| Video Games                       | 1.32M             |
| Automotives                       | 1.37M             |
| Tools & Home Improvements         | 1.92M             |
| Beauty                            | 2.02M             |
| Toys & Games                      | 2.25M             |
| Apps for Android                  | 2.63M             |
| Health & Personal Care            | 2.98M             |
| Kindle Stores                     | 3.20M             |
| Sports & Outdoors                 | 3.26M             |
| Cell Phones & Accessories         | 3.44M             |
| CDs & Vinyl                       | 3.74M             |
| Home & Kitchen                    | 4.25M             |
| Movies & TV                       | 4.60M             |
| Clothing, Shoes & Jewelry         | 5.74M             |
| Electronics                       | 7.82M             |


Note, if possible you can apply your algorithm to all the sub-datasets, but this is not required.

## Data Pre-processing and Exploratory Data Analysis (EDA)

Before we start dive into the recommendation algorithm, it's worth taking a look at the rating distribution for each dataset. Different rating distribution may indicate different difficulty level and different expected results.

Also, we have already re-indexed the user_id and item_id for you, so they all now start from 0 and end at MAX_ID, the total number of unique IDs should be MAX_ID+1.

```python
def preprocess(data_file: str) -> Tuple:
    reviews = []    
    
    # For verification purpose
    user_id_set = set()
    item_id_set = set()
    max_user_id = 0
    max_item_id = 0
    
    with open(data_file) as f:
        for line in f:
            if not line:
                continue
            user_id, item_id, rating = line.strip().split(',')
            user_id, item_id, rating = int(user_id), int(item_id), float(rating)
            
            max_user_id = max(max_user_id, user_id)
            max_item_id = max(max_item_id, item_id)
            user_id_set.add(user_id)
            item_id_set.add(item_id)
            reviews.append((user_id, item_id, rating))
    
    # Verify the indexing, DO NOT CHANGE
    assert len(user_id_set) == max_user_id+1, \
        f"User IDs are not properly re-indexed, {len(user_id_set)} vs {max_user_id+1}"
    assert len(item_id_set) == max_item_id+1, \
        f"Item IDs are not properly re-indexed, {len(item_id_set)} vs {max_item_id+1}"
    
    
    return reviews, max_user_id+1, max_item_id+1
```

Now let's load one dataset

```python
# (Optional) Feel free to change the dataset here
dataset_file = './yelp_datasets/IL/matrix.csv'
reviews, user_dim, item_dim = preprocess(dataset_file)
print(f'Number of reviews: {len(reviews)}')
print(f'User dimension: {user_dim}')
print(f'Item dimension: {item_dim}')
# Remember the `reviews` variable has the format [(user_id, item_id, rating), ...]
star_list = [int(x[2]) for x in reviews]
print(star_list[:10])

def get_label_counts(star_list: List[int]) -> List[int]:
    stars_count = [0] * 5 # 1-5 stars
    for star in star_list:
        stars_count[star-1] += 1
    print('star counts:', stars_count)
    print('distribution:', np.array(stars_count, dtype=float) / sum(stars_count))
    return stars_count

def plot_bar_chart(labels, values, title, ylabel):
    fig, ax = plt.subplots()
    ax.bar(labels, values)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.show()
   
stars_count = get_label_counts(star_list)
plot_bar_chart(
    labels=['1 Star', '2 Stars', '3 Stars', '4 Stars', '5 Stars'],
    values=np.array(stars_count, dtype=float) / sum(stars_count),
    title='Rating (stars) distribution',
    ylabel='Percentage'
)

def visualize_rating_distribution(datasets):
    for dataset in datasets:
        print(dataset)
        # preprocess
        reviews, user_dim, item_dim = preprocess(dataset)
        # star list
        star_list = [int(x[2]) for x in reviews]
        stars_count = get_label_counts(star_list)
        
        plot_bar_chart(
            labels=['1 Star', '2 Stars', '3 Stars', '4 Stars', '5 Stars'],
            values=np.array(stars_count, dtype=float) / sum(stars_count),
            title='Rating (stars) distribution',
            ylabel='Percentage'
        )
visualize_rating_distribution(yelp_datasets + amazon_datasets)
```

## Matrix Factorization

Our goal here is to implement the Matrix Factorization model for recommendation using Stochastic Gradient Descent (SGD). We will start by implementing a standard local version, and then we will extend it to a distributed version introducted from this [paper](https://dl.acm.org/doi/abs/10.1145/2020408.2020426) in Spark.

To recap, in Matrix Factorization, the $m\times n$ user-rating matrix (i.e., each role of the matrix is an user, each column of the metrix is an item, and the cell value is the rating given to the item of that column from the user of that role) is decomposed into two smaller matrix, one is the $m\times k$ user matrix where each row represents a user's hidden vector, another is $k\times n$ item matrix where each column represents an item's hidden vector.

### Problem Definition

In Matrix Factorization, the optimization goal is 

$$
\min_{{W}\in \Bbb R^{m\times k},\ {H}\in \Bbb R^{k\times n} } \sum_{(i,j)\in Z} \big(v_{i,j}-{w}_i^T {h}_j \big)^2 + \lambda(\|{W}\|^2 +\|{H}\|^2)
$$

If we include bias terms, the final loss function for our problem is defined as 

$$
L= \sum_{(i,j)\in Z} \big(v_{i,j}-{w}_i^T {h}_j- b^{(w)}_i - b^{(h)}_j -\mu \big)^2 + \lambda(\sum_i \|{w_i}\|^2 +\sum_j \|{h_j}\|^2 + \|b^{(w)}\|^2 + \|b^{(h)}\|^2)
$$

where $Z$ is the user rating set, $W$ is the user matrix and $H$ is the item matrix, $b^{(w)}$ is the user bias, $b^{(h)}$ is the item bias, $\mu$ is the constant global bias (Note the notation here is different with course material as we want to keep consistent with that paper in case you read it).

### Stochastic Gradient Descent for Matrix Factorization (SGD-MF)

In SGD-MF, we select a single datapoint and update the corresponding row of W and column
of H in the direction of negative gradient.

The gradients for the loss are given as follows:

$$
\frac{\partial L}{\partial w_i} = -2\big(v_{i,j}-{w}_i^T {h}_j- b^{(w)}_i - b^{(h)}_j -\mu \big){h}_j+2\lambda w_i
$$

$$
\frac{\partial L}{\partial h_j} = -2\big(v_{i,j}-{w}_i^T {h}_j- b^{(w)}_i - b^{(h)}_j -\mu \big){w}_i+2\lambda h_j
$$

$$
\frac{\partial L}{\partial b_i^{(w)}} = -2\big(v_{i,j}-{w}_i^T {h}_j- b^{(w)}_i - b^{(h)}_j -\mu \big)+2\lambda b_i^{(w)}
$$

$$
\frac{\partial L}{\partial b_j^{(h)}} = -2\big(v_{i,j}-{w}_i^T {h}_j- b^{(w)}_i - b^{(h)}_j -\mu \big)+2\lambda b_j^{(h)}
$$

The SGD will update the matrix values iteratively using following rule:

$$
w_i = w_i - \alpha\frac{\partial L}{\partial w_i}
$$

$$h_j = h_j - \alpha\frac{\partial L}{\partial h_j}$$

$$b_i^{(w)} = b_i^{(w)} - \alpha\frac{\partial L}{\partial b_i^{(w)}}$$

$$b_j^{(h)} = b_j^{(h)} - \alpha\frac{\partial L}{\partial b_j^{(h)}}$$

where $\alpha$ is the learning rate (how fast it converges) and $\lambda$ is the regularization parameter (biases against extreme models).

Make sure you can derive them by yourself, then you can start to implement.

```python
# TODO: Let's first initialize our model parameter randomly.
#       You need to fill the correct dimensions for each parameters
def get_initial_model_parameters(user_dim: int, item_dim: int, k: int) -> Tuple:
    
    W = np.random.rand(user_dim, k)
    H = np.random.rand(k, item_dim)
    user_bias = np.random.rand(user_dim)
    item_bias = np.random.rand(item_dim)
    
    return W, H, user_bias, item_bias
# TODO: implement this function to calculate global bias as mean rating from dataset
def get_global_bias(reviews: List[Tuple]) -> float:
    return np.mean([float(x[2]) for x in reviews])
```

If you test your function with Yelp IL dataset, you should get something around 3.7

```
print(get_global_bias(reviews))
```

```python
# TODO: implement this function to make prediction on one single instance,
#       the output should be a real number
def predict(
    W: np.ndarray,
    H: np.ndarray,
    user_bias: np.ndarray,
    item_bias: np.ndarray,
    global_bias: np.ndarray,
    user_idx: int,
    item_idx: int
) -> np.ndarray:
    predicted_rating = np.dot(W[user_idx], H.T[item_idx])+user_bias[user_idx] + item_bias[item_idx]+ global_bias
    
    return predicted_rating
```

Let's make sure all your functions can work

```python
W, H, user_bias, item_bias = get_initial_model_parameters(user_dim=user_dim, item_dim=item_dim, k=5)
mu = get_global_bias(reviews)
predict(W, H, user_bias, item_bias, mu, 0, 0)
```

### Evaluation

This an iterative algorithm, to help make sure your implementation is correct, we need to implement the evaluation function to verify the training process is moving towards the direction we want.

Since we are predicting the user rating, we can use MSE as the evaluation metrics:

$$\text{MSE}=\frac{1}{n}\sum_{i=1}^n(\hat{r}_i-r_i)^2$$

where $\hat{r}_i$ is your predicted rating for i-th instance, $r_i$ is the corresponding true rating and $n$ is the total number of rating instances in the dataset used for evaluation. 

Now let's implement the following function to caculate the MSE for the given dataset. 

```python
# TODO: implement the evaluate function to return the MSE
# user_id, item_id, rating
def evaluate(reviews, W, H ,user_bias, item_bias, mu) -> float:
    se = 0 
    for review in reviews:
        user_idx = review[0]
        item_idx = review[1]
        predicted_rating = np.dot(W[user_idx], H.T[item_idx])+user_bias[user_idx] + item_bias[item_idx]+ global_bias
        se += (review[2]- predicted_rating)^2
        
    mse = se/len(reviews)
    
    
    return mse
```

### Dataset Split

Since you only have a whole dataset with user ratings, to better evaluate the algorithm results, you should split your dataset into training and validation. The simplest way to do this is setting a split ratio, say 0.8 (e.g., 80% training and 20% validation), then use a random number generator to generate random numbers between zero and one for each sample, if the random number is smaller than your split ratio (i.e., 0.8), you put this sample in training data, otherwise put it in validation data.

PS: You can also try to implement K-Fold cross validation and use it for evaluation during the training, which should get you more accurate results.

```python
# TODO: implement this function using the logic mentioned above
def prob_dataset_spliter(dataset, split_ratio=0.8):
    train = []
    val = []
    for data in dataset:
        random_num = np.random.rand()
        if random_num < split_ratio:
            train.append(data)
        else:
            val.append(data)

    return train, val
```

Let's test it, while if the number does not exactly match, do you have any idea?

```python
# Let's test it
np.random.seed(999)
test_data = list(range(1000))
split_ratio = 0.6
train, val = prob_dataset_spliter(test_data, split_ratio)
real_ratio = len(train) / len(test_data)
assert abs(real_ratio - split_ratio) < 0.01, "Please double check your function"
```

### Matrix Factorization Algorithm


Great!! Now let's put everything together and implement the training function for Matrix Factorization. During the training process, you will implement the stochastic gradient descent, use the helper functions you have implemented in the previous section to help complete the training function.


To help start, the pseudocode of the algorithm can be described as:

Require: Training indices $Z$, Training data $V$, randomly initialized $W$, $H$, user bias, item bias and pre-compute global bias

```
while not converged do:
    Select a training point (i, j) ∈ Z uniformly at random
    Compute gradients of each variable
    Update the gradient of corresponding row in W, column in H and biases
end while
```

Some additional thoughts:
- To help make sure the trainig in on the right direction, you should check the MSE value on both training and validation datasets regularly during the training (e.g., after certain number of iterations)
- Think about what other stopping criteria you should use for the training (the default one is fixed 100 iterations)
- You may want to shuffle the order of training samples each time you used the entire dataset
- After you figure out the training process, you can tune the hyper-parameters based on your understanding about how the training works (we have already used a dynamic learning rate here)

