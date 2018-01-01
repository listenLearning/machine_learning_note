#!/usr/bin/env python
__coding__ = "utf-8"
__author__ = " Ng WaiMing "

from training.action.supervised.bayes import Bayes
import feedparser

if __name__ == "__main__":
    # bayes: p(c_i|w)=\frac{p(w|c_i)p(c_i)}{p(w)}
    # listOPosts, listClasses = loadDataSet()
    # myVocabList = createVocabList(listOPosts)
    # trainMat = []
    # for postinDoc in listOPosts:
    #     trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    # p0V, p1V, pAb = trainNB0(trainMat, listClasses)
    # print(p0V)
    # testingNB()
    # Bayes.spamTest()
    # np = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
    # sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
    # vocabList, p0V, p1V = Bayes.localWords(np, sf)
    # print(p0V, p1V)
    ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
    sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
    # vocab_list, p_sf, p_ny = Bayes.localWords(ny, sf)
    # print(vocab_list)
    Bayes.getTopWords(ny, sf)
