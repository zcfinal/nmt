
file_root='/data/zclfe/transformer/corpora/'
with open(file_root+'news-commentary-v15.en-zh.tsv','r',encoding='utf-8')as f,open(file_root+'test_news.en','w')as fouten,open(file_root+'test_news.zh','w',encoding='utf-8')as foutzh:
    for line in f:
        try:
            en,zh = line.strip().split('\t')
        except:
            continue
        fouten.write(en+'\n')
        foutzh.write(zh+'\n')