
file_root='/data/zclfe/transformer/corpora/'

with open(file_root+'ch_en_all.txt','r',encoding='utf-8')as f,open(file_root+'test.en','w')as fouten,open(file_root+'test.zh','w',encoding='utf-8')as foutzh:
    for line in f:
        en,zh = line.strip().split('\t')
        zh=zh.replace(' ','')
        fouten.write(en+'\n')
        foutzh.write(zh+'\n')