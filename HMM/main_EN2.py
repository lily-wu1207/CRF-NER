#coding=utf-8


statuDict2 = {"O":0,
              "B-PER":1, "I-PER":2, "B-ORG":3, "I-ORG":4,
              "B-LOC":5, "I-LOC":6, "B-MISC":7, "I-MISC":8}
label2= ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC" , "I-MISC"]


type = 9
statuDict = statuDict2
label = label2
filename = 'NER/test/English_my_result.txt'
train_set = './NER/English/train.txt'
val_set = './NER/test/english_test.txt'
d = []


import numpy as np
import time

def trainParameter(fileName):
    '''
    依据训练文本统计PI、A、B
    :param fileName: 训练文本
    :return: 三个参数
    '''

    #初始化PI的一维数组，因为对应四种状态，大小为type
    PI = np.zeros(type)
    #初始化状态转移矩阵A
    A = np.zeros((type, type))
    #初始化观测概率矩阵，分别为四种状态到每个字的发射概率
    #因为是中文分词，使用ord(汉字)即可找到其对应编码，这里用一个65536的空间来保证对于所有的汉字都能
    #找到对应的位置来存储
    B = np.zeros((type, len(d)))

    #去读训练文本
    fr = open(fileName, encoding='utf-8')
    #文本中的每一行认为是一个训练样本
    #在统计上，三个参数依据“10.3.2” Baum-Welch算法内描述的统计
    #PI依据式10.35
    #A依据10.37
    #B依据10.38
    #注：并没有使用Baum-Welch算法，只是借助了其内部的三个参数生成公式，其实
    #公式并不是Baum-Welch特有的，只是在那一节正好有描述
    i = -1
    for line in fr.readlines():
        # print(line)

        #对单行按空格进行切割
        curLine = line.strip().split()
        if(curLine == []):
            i = -1
            continue
        i = i + 1
        state = curLine[1] # 状态
        watch = curLine[0] # 观测

        #对词性的标记放在该列表中
        wordLabel = []




        #如果是单行开头第一个字，PI中对应位置加1,
        if i == 0:
            PI[statuDict[state]] += 1
        else:
            A[statuDict[prestate]][statuDict[state]] += 1
        # 对于每一个字，在生成的状态链中统计B
        B[statuDict[state]][d.index(watch)] += 1


        prestate = state



    #上面代码在统计上全部是统计的次数，实际运算需要使用概率，
    #下方代码是将三个参数的次数转换为概率
    #----------------------------------------
    #对PI求和，概率生成中的分母
    sum = np.sum(PI)
    #遍历PI中每一个元素，元素出现的次数/总次数即为概率
    for i in range(len(PI)):

        if PI[i] == 0:  PI[i] = -9e+1000
        #如果不为0，则计算概率，再对概率求log
        else: PI[i] = np.log(PI[i] / sum)

    #与上方PI思路一样，求得A的概率对数
    for i in range(len(A)):
        sum = np.sum(A[i])
        for j in range(len(A[i])):
            if A[i][j] == 0: A[i][j] = -9e+1000
            else: A[i][j] = np.log(A[i][j] / sum)

    #与上方PI思路一样，求得B的概率对数
    for i in range(len(B)):
        sum = np.sum(B[i])
        for j in range(len(B[i])):
            if B[i][j] == 0: B[i][j] = -9e+1000
            else:B[i][j] = np.log(B[i][j] / sum)

    #返回统计得到的三个参数
    return PI, A, B

def loadArticle(fileName):
    '''
    加载文章
    :param fileName:文件路径
    :return: 按句子切割过的文章内容
    '''
    #初始化文章列表
    artical = []
    #打开文件
    fr = open(fileName, encoding='utf-8')
    #按行读取文件
    for line in fr.readlines():
        #读到的每行最后都有一个\n，使用strip将最后的回车符去掉
        line = line.strip()
        #将该行放入文章列表中
        artical.append(line)
    #将文章返回
    j = 0
    a = []
    for i in range(len(artical)):
        if(artical[i] == ''):
            a.append(artical[j:i])
            j = i + 1

    return a

def participle(artical, PI, A, B, file):
    '''
    分词
    算法依据“10.4.2 维特比算法”
    :param artical:要分词的文章
    :param PI: 初始状态概率向量PI
    :param A: 状态转移矩阵
    :param B: 观测概率矩阵
    :return: 分词后的文章
    '''
    #初始化分词后的文章列表
    retArtical = []

    #对文章按行读取
    for i in range(len(artical)):

        slice = artical[i]

        l = len(slice)
        # 初始化δ，δ存放四种状态的概率值，因为状态链中每个状态都有四种概率值，因此长度时该行的长度
        delta = [[0 for _ in range(type)] for _ in range(l)]
        # 依据算法10.5 第一步：初始化
        for j in range(type):
            # 初始化δ状态链中第一个状态的四种状态概率
            firstword = slice[0].split(" ")[0]
            if (firstword not in d):
                # print(firstword)
                firstword = '_'
            delta[0][j] = PI[j] + B[j][d.index(firstword)]

        # 初始化ψ，初始时为0
        psi = [[0 for _ in range(type)] for _ in range(l)]

        #算法10.5中的第二步：递推
        #for循环的符号与书中公式一致，可以对比着看来理解
        #依次处理整条链
        for t in range(1, l):  # time
            #对于链中的每个状态，求状态概率
            for q in range(type):
                #初始化一个临时列表，用于存放概率
                tmpDelta = [0] * type
                for j in range(type):
                    tmpDelta[j] = delta[t - 1][j] + A[j][q]

                #找到最大的那个δ * a，
                maxDelta = max(tmpDelta)
                #记录最大值对应的状态
                maxDeltaIndex = tmpDelta.index(maxDelta)

                #将找到的最大值乘以b放入，
                #注意：这里同样因为log变成了加法
                watch = slice[t].split(" ")[0]
                if (watch not in d):
                    # print('watch',watch)
                    watch = '_'
                delta[t][q] = maxDelta + B[q][d.index(watch)]
                #在ψ中记录对应的最大状态索引
                psi[t][q] = maxDeltaIndex

        #建立一个状态链列表，开始生成状态链
        sequence = []
        #算法10.5 第三步：终止
        #获取最后一个状态的最大状态概率对应的索引
        i_opt = delta[l - 1].index(max(delta[l - 1]))
        #在状态链中添加索引
        sequence.append(label[i_opt])

        #算法10.5 第四步：最优路径回溯
        #从后往前遍历整条链
        for t in range(l - 1, 0, -1):
            #不断地从当前时刻t的ψ列表中读取到t-1的最优状态
            i_opt = psi[t][i_opt]
            #将状态放入列表中
            sequence.append(label[i_opt])
        #因为是从后往前将状态放入的列表，所以这里需要翻转一下，变成了从前往后
        sequence.reverse()
        # print(sequence)

        for m in range(len(sequence)):
            text = slice[m].split(" ")[0] + ' ' + sequence[m] + '\n'
            file.write(text)
        file.write('\n')
        # print("write!")

if __name__ == '__main__':
    # 开始时间
    start = time.time()
    fr = open(train_set, encoding='utf-8')


    def file_lines_to_list(filename):
        with open(filename) as f:
            return f.read().splitlines()


    # fsave = open('en_dict.txt', 'w', encoding='utf-8')
    # fr = open(train_set, encoding='utf-8')
    # for line in fr.readlines():
    #     curLine = line.strip().split()
    #     if (curLine == []):
    #         continue
    #     watch = curLine[0]  # 观测
    #     if (watch not in d):
    #         d.append(watch)
    #         fsave.write(watch + '\n')
    # fsave.write('_')  # 表示未出现的字符
    d = file_lines_to_list('en_dict.txt')

    PI, A, B = trainParameter(train_set)
    #依据现有训练集统计PI、A、B
    # PI, A, B = trainParameter(train_set)
    # np.save("PI_EN.npy", PI)
    # np.save("A_EN.npy", A)
    # np.save("B_EN.npy", B)

    # PI = np.load('PI_CH.npy')
    # A = np.load('A_CH.npy')
    # B = np.load('B_CH.npy')
    # print(PI, A, B)
    #读取测试文章
    artical = loadArticle(val_set)
    # print(len(artical))

    # #打印原文
    # print('-------------------原文----------------------')
    # for line in artical:
    #     print(line)
    #
    #进行分词
    file = open(filename, 'w', encoding='utf-8')
    partiArtical = participle(artical, PI, A, B, file)
    #
    # #打印分词结果
    # print('-------------------分词后----------------------')
    # for line in partiArtical:
    #     print(line)
    #
    # #结束时间
    # print('time span:', time.time() - start)