from partition_data_dirichlet import *
import fedml
import logging
from fedml.ml.engine import ml_engine_adapter
import params

def batch_data(args, data, batch_size):

    """
    data is a dict := {'x': [numpy array], 'y': [numpy array]} (on one client)
    returns x, y, which are both numpy array of length: batch_size
    """
    data_x = data["x"]
    data_y = data["y"]

    # randomly shuffle data
    np.random.seed(100)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)

    # loop through mini-batches
    batch_data = list()
    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i : i + batch_size]
        batched_y = data_y[i : i + batch_size]
        # time.sleep(1)
        batched_x, batched_y = ml_engine_adapter.convert_numpy_to_ml_engine_data_format(args, batched_x, batched_y)
        batch_data.append((batched_x, batched_y))
        #print(batched_x[0])
    #print(batch_data)
        #time.sleep(10)
    return batch_data

def read_data(args):
    # 训练集读取
    dr = ds_r(params.MNIST_TRAIN_IMAGES_PATH)
    lr = lb_r(params.MNIST_TRAIN_LABELS_PATH)
    dr.open()
    lr.open()
    index, data = dr.read(params.DATASET_SIZE_USED_TO_TRAIN)  # data:ndarray:(60000,28,28),index:range(1,60001)
    index, label = lr.read(params.DATASET_SIZE_USED_TO_TRAIN)  # label:ndarray(60000,),index:range(1,60001)
    data = data[..., np.newaxis] / 255.0  # 添加一个维度，data:(60000,28,28,1),还归一化了一下
    dr.close()
    lr.close()

    # 测试集读取
    edr = ds_r(params.MNIST_EVAL_IMAGES_PATH)
    elr = lb_r(params.MNIST_EVAL_LABELS_PATH)
    edr.open()
    elr.open()
    index, edata = edr.read(params.DATASET_SIZE_USED_TO_EVAL)  # edata:ndarray:(10000,28,28)
    index, elabel = elr.read(params.DATASET_SIZE_USED_TO_EVAL)  # elabel:ndarray:(10000,）,index:range(1,100001)
    edata = edata[..., np.newaxis] / 255.0  # edata:(10000,28,28,1)，还归一化了一下
    edr.close()
    elr.close()

    # client_idcs: {list：3}每个客户端被分配到的数据的索引，是按类别0, 1, ..., 9顺序排的
    # df_client: {ndarray: 60000, }表示数据集中该行数据分给哪个客户端
    # 训练的
    client_idcs, df_client = split_index(label, args.client_num_in_total, args.split_policy, args.partition_alpha)
    # label_name = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # show_split_img(label, args.client_num_in_total, client_idcs, label.max() + 1, label_name)
    # 测试的
    eclient_idcs, edf_client = split_index(elabel, args.client_num_in_total, args.split_policy, args.partition_alpha)

    clients = []
    groups = []
    train_data = {}
    test_data = {}

    for i in range(args.client_num_in_total):
        clients.append('f_0000{}'.format(i))
        groups.append(None)

        # 训练划分
        index_client_i = np.argwhere(df_client == i).reshape(-1)
        label_i = label[index_client_i]
        data_i = data[index_client_i, :, :, :]
        train_data['f_0000{}'.format(i)] = {'y': label_i.astype('float').tolist(),
                                            'x': data_i.reshape(data_i.shape[0], -1).astype('float').tolist()}

        # 测试划分
        eindex_client_i = np.argwhere(edf_client == i).reshape(-1)
        elabel_i = elabel[eindex_client_i]
        edata_i = edata[eindex_client_i, :, :, :]
        test_data['f_0000{}'.format(i)] = {'y': elabel_i.astype('float').tolist(),
                                           'x': edata_i.reshape(edata_i.shape[0], -1).astype('float').tolist()}

    return clients, groups, train_data, test_data

def load_partition_data_mnist(args, batch_size):
    users, groups, train_data, test_data = read_data(args)
    if len(groups) == 0:
        groups = [None for _ in users]
    train_data_num = 0
    test_data_num = 0
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    train_data_local_num_dict = dict()
    train_data_global = list()
    test_data_global = list()
    client_idx = 0
    logging.info("loading data...")
    # print(train_data,11111111111111111111111)
    for u, g in zip(users, groups):
        user_train_data_num = len(train_data[u]["x"])
        user_test_data_num = len(test_data[u]["x"])
        train_data_num += user_train_data_num
        test_data_num += user_test_data_num
        train_data_local_num_dict[client_idx] = user_train_data_num

        # transform to batches
        train_batch = batch_data(args, train_data[u], batch_size)
        test_batch = batch_data(args, test_data[u], batch_size)

        # index using client index
        train_data_local_dict[client_idx] = train_batch
        test_data_local_dict[client_idx] = test_batch
        train_data_global += train_batch
        test_data_global += test_batch
        client_idx += 1
    logging.info("finished the loading data")
    client_num = client_idx
    class_num = 10
    # print(train_data_local_dict[9])

    return (
        client_num,
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    )



def load_data(args):

    """
    Please read through the data loader at to see how to customize the dataset for FedML framework.
    """
    (
        client_num,
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    ) = load_partition_data_mnist(
        args,
        args.batch_size,
    )
    """
    For shallow NN or linear models, 
    we uniformly sample a fraction of clients each round (as the original FedAvg paper)
    """
    args.client_num_in_total = client_num
    dataset = [
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    ]
    return dataset, class_num


if __name__ == '__main__':


    pass

    # dataset, output = load_data()