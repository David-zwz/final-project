import numpy as np
import csv

class Readdata:

    data = []
    labeltemp = []
    testtext = []
    testlabel = []

# 数据样本
    input_word=input("please input a dataset(we have rude sentence dataset and iris dataset):")
    if input_word.lower()=="rude sentence":
        with open('./Datasets/sentence.csv', 'r') as f0:
            df0 = csv.reader(f0)

            for col in df0:
                labeltemp.append(col[-1])

            a = labeltemp[1:]
            label = list(map(int, a))

        with open('./Datasets/sentence.csv', 'r') as f1:
            df1 = csv.reader(f1)

            for row in df1:
                data.append(row)
            dataset = data[1:]

            for i in range(len(dataset)):# this part is used to dealete blank and label in dataset
                datarow = dataset[i]
                datarow = datarow[:-1]
                while '' in datarow:
                    datarow.remove('')
                dataset[i] = datarow

    elif input_word.lower()=="iris":
        with open('./Datasets/iris.csv', 'r') as f0:
            df0 = csv.reader(f0)

            for col in df0:
                labeltemp.append(col[-1])

            a = labeltemp[1:]
            label = list(map(int, a))

        with open('./Datasets/iris.csv', 'r') as f1:
            df1 = csv.reader(f1)

            for row in df1:
                data.append(row)

            dataset = data[1:]

            for i in range(len(dataset)):
                datarow = dataset[i]
                datarow = datarow[:-1]
                while '' in datarow:
                    datarow.remove('')
                dataset[i] = datarow

    else:
        print("must input legal datasets")


    if input_word != "rude sentence":#对于文字的数据集不需要进行处理

        dataset=list(zip(*dataset))#转置来进行归一化处理，之后还得转置回去
        dataset=np.array(dataset).astype(float)


        input_word1 = input("please preprocess the dataset(we have normalization and standardization):")
        if input_word1.lower() == "normalization":
            for i in range(len(dataset)):

                max_value = np.max(dataset[i])  # 获取矩阵中的最大值
                min_value = np.min(dataset[i])  # 获取矩阵中的最小值
                dataset[i] = (dataset[i] - min_value) / (max_value - min_value)  # 按照归一化公式进行计算
        elif input_word1.lower() == "standardization":

            for i in range(len(dataset)):
            # 计算均值和标准差
                mean = np.mean(dataset[i])
                std = np.std(dataset[i])

            # 使用公式进行标准化
                dataset[i] = (dataset[i] - mean) / std
        else:
            print("it seems you do not need it, that is ok")

        dataset = dataset.tolist()
        dataset = list(zip(*dataset))


    train_size = 0.7
    train_dataindex = int(len(dataset) * train_size)
    train_labelindex = int(len(label) * train_size)
    train_data = dataset[:train_dataindex]  # 切片得到训练集
    test_data = dataset[train_dataindex:]  # 切片得到测试集
    train_label = label[:train_labelindex]  # 切片得到训练集
    test_label = label[train_labelindex:]  # 切片得到测试集
    #These four data will be used in the next part, They are all lists, we need to transfer them into np.array using a=np.array(b)


