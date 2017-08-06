# -*- coding: utf-8 -*-
import jieba

#全模式
seg_list1 = jieba.cut("我来到杭州电子科技大学",cut_all=True)
print ("Full Mode: " + "/ ".join(seg_list1))

#精确模式
seg_list2 = jieba.cut("我来到杭州电子科技大学",cut_all=False)
print ("Default Mode: " + "/ ".join(seg_list2))

#默认是精确模式
seg_list3 = jieba.cut("我来到了网易杭州研究院大厦")
print (", ".join(seg_list3))

seg_list4 = jieba.cut_for_search("高鹏昺毕业于中国科学院计算所，后在日本京都大学深造")
print (", ".join(seg_list4))

import jieba.analyse
import jieba.posseg as pseg
text = u'中国共产党已经走过90年的光辉历程。90年来，一代又一代共产党人团结带领人民在艰难困苦中奋起、在艰辛探索中前进，完成了民族独立和人民解放的伟大历史任务，建立了社会主义制度，开辟了中国特色社会主义道路，创造了中华民族发展史上最辉煌的业绩。坚持党的建设为党领导的伟大事业服务，是我们党在领导革命、建设、改革的伟大事业中，不断加强和改进党的自身建设的一条宝贵经验。深刻认识和把握这条经验，对于加强新形势下党的建设具有重要意义。'
#基于TF-IDF
keywords = jieba.analyse.extract_tags(text, topK=20, withWeight=True, allowPOS=())
for item in keywords:
    print (item[0],item[1])

#基于TextRank
keywords = jieba.analyse.textrank(text, topK=20, withWeight=True, allowPOS=('ns','n','vn','v'))
for item in keywords:
    print (item[0],item[1])

#分词词性
words = pseg.cut(text)
for word, flag in words:
    print ('%s, %s' % (word, flag))