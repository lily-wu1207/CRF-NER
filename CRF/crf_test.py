#!/usr/bin/env python

import argparse
from crf import LinearChainCRF

if __name__ == '__main__':

    crf = LinearChainCRF()
    # # CH
    # crf.load('Chinese2.json')
    # crf.test('data/NER/Chinese/chinese_test.txt')
    # filename = 'data/NER/Chinese/Chinese_my_result.txt'
    # file = open(filename, 'w', encoding='utf-8')
    # crf.print_test_result('data/NER/Chinese/chinese_test.txt', file)
    # EN
    crf.load('English4.json')
    crf.test('data/NER/English/english_test.txt')
    filename = 'data/NER/English/English_my_result.txt'
    file = open(filename, 'w')
    crf.print_test_result('data/NER/English/english_test.txt', file)
