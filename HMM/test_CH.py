#coding=utf-8

statuDict1 = {'O':0,
              'B-NAME':1, 'M-NAME':2, 'E-NAME':3, 'S-NAME':4,
              'B-CONT':5, 'M-CONT':6, 'E-CONT':7, 'S-CONT':8,
              'B-EDU':9, 'M-EDU':10, 'E-EDU':11, 'S-EDU':12,
              'B-TITLE':13, 'M-TITLE':14, 'E-TITLE':15, 'S-TITLE':16,
              'B-ORG':17, 'M-ORG':18, 'E-ORG':19, 'S-ORG':20,
              'B-RACE':21, 'M-RACE':22, 'E-RACE':23, 'S-RACE':24,
              'B-PRO':25, 'M-PRO':26, 'E-PRO':27, 'S-PRO':28,
              'B-LOC':29, 'M-LOC':30, 'E-LOC':31, 'S-LOC':32,}
label1 = ['O',
        'B-NAME', 'M-NAME', 'E-NAME', 'S-NAME'
        , 'B-CONT', 'M-CONT', 'E-CONT', 'S-CONT'
        , 'B-EDU', 'M-EDU', 'E-EDU', 'S-EDU'
        , 'B-TITLE', 'M-TITLE', 'E-TITLE', 'S-TITLE'
        , 'B-ORG', 'M-ORG', 'E-ORG', 'S-ORG'
        , 'B-RACE', 'M-RACE', 'E-RACE', 'S-RACE'
        , 'B-PRO', 'M-PRO', 'E-PRO', 'S-PRO'
        , 'B-LOC', 'M-LOC', 'E-LOC', 'S-LOC']

type = 33
statuDict = statuDict1
label = label1
val_set = './NER/test/chinese_test.txt' # 测试文件地址
filename = './NER/test/Chinese_my_result.txt' # 输出的文件地址


import numpy as np
import time

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
            delta[0][j] = PI[j] + B[j][ord(firstword)]
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
                delta[t][q] = maxDelta + B[q][ord(watch)]
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
if __name__ == '__main__':

    PI = np.load('PI_CH.npy')
    A = np.load('A_CH.npy')
    B = np.load('B_CH.npy')

    # print(PI, A, B)
    #读取测试文章
    artical = loadArticle(val_set)
    #进行分词
    file = open(filename, 'w', encoding='utf-8')
    partiArtical = participle(artical, PI, A, B, file)
