# Matrix Factorization Using DSGD on Spark


import sys
import numpy as np
from pyspark import SparkContext


self_eval = True   # Evaluate the MES using spark during training
verbose = True    # Whether to print the debug info

def prob_dataset_spliter(infile, split_ratio=0.8):

    train = 'train.data'
    test = 'test.data'
    
    f_train = open(train, 'w')
    f_test = open(test, 'w')

    with open(infile, 'r') as f:
        for line in f:
            if np.random.random() < split_ratio:
                f_train.write(line)
            else:
                f_test.write(line)
    f_train.close()
    f_test.close()

    return train, test

def train_mf_with_dsgd_spark(train, test):
    # Parameters
    num_factors = 5
    num_workers = 4
    num_iter = 100
    beta_v = 0.5
    lambda_v = 1
    outputW = 'w_out.csv'
    outputH = 'h_out.csv'

    # Set up spark
    sc = SparkContext(appName='DSGD_MF')

    # Load file, data format:
    # <user_id>,<item_id>,<rating>
    data_train = sc.textFile(train)
    data_test = sc.textFile(test)
    V = data_train.map(lambda line: parse_line(line)).cache()
    V_test = data_test.map(lambda line: parse_line(line)).cache()
    V_whole = V.union(V_test)
    
    # Find the max value for user_id and item_id
    max_user_id, max_item_id = V_whole.reduce(lambda x, y: (max(x[0], y[0]), max(x[1], y[1])))
    if verbose:
        print(f'max_item_id={max_item_id} max_user_id={max_user_id}')

    # Get the N_i* and N_*j
    row_counts = V.map(lambda x: (x[0], 1)).countByKey()
    col_counts = V.map(lambda x: (x[1], 1)).countByKey()

    # Partition using max ids
    # Compute boundary
    item_bound, usr_bound = compute_boundary(max_item_id, max_user_id, num_workers)
    if verbose:
        print(f'item_bound={item_bound}\nuser_bound={usr_bound}')
    broadcastVar = sc.broadcast([num_workers, item_bound, usr_bound])

    # Index the V matrix
    Vidx = V.map(lambda record: data_indexing(record, broadcastVar, row_counts, col_counts)).cache()
    if not self_eval:
        V.unpersist()

    # Create the strata
    strata_list = []
    strata_count = []
    for i in range(num_workers):
        strata = Vidx.filter(lambda x: x[0]==i).map(lambda x: (x[1], (x[2], x[3], x[4]))) \
                    .partitionBy(num_workers).cache()
        count = strata.count()
        strata_list.append(strata)
        strata_count.append(count)
    Vidx.unpersist()

    # Validate partitions
    # l = strata_list[0].glom().collect()
    # for part in l:
    #    print('partition=', part[1:10])
    # print(strata_count)

    # Build local parameters
    W = np.random.rand(max_user_id + 1, num_factors)
    H = np.random.rand(max_item_id + 1, num_factors)
    params = sc.broadcast([beta_v, lambda_v])

    if self_eval:
        sumSL, n = V.map(lambda x: evaluate(x, W, H)).reduce(lambda x,y: (x[0]+y[0], x[1]+y[1]))
        sumSL_test, n_test = V_test.map(lambda x: evaluate(x, W, H)).reduce(lambda x,y: (x[0]+y[0], x[1]+y[1]))
        print(f'Initial MSE -> train: {sumSL / n}, test: {sumSL_test / n_test}')

    # Perform stochastic gradient descent
    m = 0
    for t in range(num_iter):
        for strata in strata_list:
            results = strata.mapPartitions(lambda x: update_weights(x, W, H, m, params)).collect()
            # Update weights
            for x in results:
                if x[0] == 'W':
                    W[x[1]] = x[2]
                elif x[0] == 'H':
                    H[x[1]] = x[2]

        # Update the # of iteration
        m += np.sum(strata_count)
        
        # Evaluation
        if self_eval:
            sumSL, n = V.map(lambda x: evaluate(x, W, H)).reduce(lambda x,y: (x[0]+y[0], x[1]+y[1]))
            sumSL_test, n_test = V_test.map(lambda x: evaluate(x, W, H)).reduce(lambda x,y: (x[0]+y[0], x[1]+y[1]))
            print(f'MSE iter:{t+1} -> train: {sumSL / n}, test: {sumSL_test / n_test}')

    # Save the matrix W and H for future prediction
    # not needed for our project here
    # save_results(W, outputW, H, outputH)


def parse_line(record):
    """
    Parse the raw input of the data
    """
    user_id, item_id, rating = record.split(',')
    return int(user_id), int(item_id), int(rating)
    

def data_indexing(record, broadcastVar, row_counts, col_counts):
    """
    Index the data matrix V to create strata and partition
    """
    (user_id, item_id, rating) = record
    num_workers, item_bound, usr_bound = broadcastVar.value
    i = -1
    j = -1
    for idx in range(num_workers):
        (lower, upper) = item_bound[idx]
        if item_id >= lower and item_id <= upper:
            j = idx
            break
    for idx in range(num_workers):
        (lower, upper) = usr_bound[idx]
        if user_id >= lower and user_id <= upper:
            i = idx
            break
    if j - i < 0:
        i = j + num_workers - i
    else:
        i = j - i
    return (i, j, user_id, item_id, rating)


def compute_boundary(max_item_id, max_user_id, num_workers):
    """
    Compute the indexing boundary
    """
    # Assume index is from 0
    item_step = int((max_item_id+1) / num_workers)
    user_step = int((max_user_id+1) / num_workers)

    mov_interval = []
    usr_interval = []

    for idx in range(num_workers):
        if idx != num_workers-1:
            mov_interval.append((idx*item_step, (idx+1)*item_step-1))
            usr_interval.append((idx*user_step, (idx+1)*user_step-1))
        else:
            mov_interval.append((idx*item_step, max_item_id))
            usr_interval.append((idx*user_step, max_user_id))
    return mov_interval, usr_interval


def evaluate(record, W, H):
    """
    Evaluate the MSE
    """
    (user_id, item_id, rating) = record
    err = (rating - np.dot(W[user_id], H[item_id]))**2
    return (err, 1)


def update_weights(iterator, W, H, m, params):
    """
    Update the weights of W and H using SGD
    """
    beta_v, lambda_v = params.value
    W_set = set()
    H_set = set()
    t = 0
    for record in iterator:
        alpha = (100 + t + m)**(-beta_v)
        t += 1
        # user_id, item_id, rating
        (i, j, v) = record[1]
        tmp = -2 * (v - np.dot(W[i], H[j]))
        grad_w = tmp * H[j] + 2 * lambda_v * W[i]
        grad_h = tmp * W[i] + 2 * lambda_v * H[j]

        W[i] -= alpha * grad_w
        H[j] -= alpha * grad_h
        W_set.add(i)
        H_set.add(j)
    results = []

    for i in W_set:
        results.append(('W', i, W[i]))
    for j in H_set:
        results.append(('H', j, H[j]))

    return results


def save_results(W, outputW, H, outputH):
    """
    Save the matrix W and H to files
    """
    np.savetxt(outputW, W, delimiter=",")
    np.savetxt(outputH, H.T, delimiter=",")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: spark-submit mf_dsgd_spark.py <input_user_rating_file_full_path>')
        print('To keep only standard logs: add 2>/dev/null at the end')
        sys.exit(-1)
    np.random.seed(0)
    train, test = prob_dataset_spliter(sys.argv[1])
    train_mf_with_dsgd_spark(train, test)
    