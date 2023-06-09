import os
en=[]
zh=[]
with open('/data/zclfe/transformer/corpora/training.en','r',encoding='utf-8')as fenin, open('/data/zclfe/transformer/corpora/training.zh','r',encoding='utf-8')as fzhin:
    for en_line,zh_line in zip(fenin,fzhin):
        en.append(en_line)
        zh.append(zh_line)
cnt=0
with open('/data/zclfe/transformer/corpora/test_news_filtered.en','w',encoding='utf-8')as fen, open('/data/zclfe/transformer/corpora/test_news_filtered.zh','w',encoding='utf-8')as fzh:
    with open('/data/zclfe/transformer/corpora/test_news.en','r',encoding='utf-8')as fenin, open('/data/zclfe/transformer/corpora/test_news.zh','r',encoding='utf-8')as fzhin:
        for en_line,zh_line in zip(fenin,fzhin):
            if en_line in en or zh_line in zh:
                continue
            fen.write(en_line)
            fzh.write(zh_line)