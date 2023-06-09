
# with open(file_root+'news-commentary-v15.en-zh.tsv','r',encoding='utf-8')as f,open(file_root+'test_news.en','w')as fouten,open(file_root+'test_news.zh','w',encoding='utf-8')as foutzh:
#     for line in f:
#         try:
#             en,zh = line.strip().split('\t')
#         except:
#             continue
#         fouten.write(en+'\n')
#         foutzh.write(zh+'\n')

def write(fin_name,fout):
    if 'Book' in fin_name:
        file_root='/data/zclfe/transformer/corpora/datum2017/'
    else:
        file_root=''
    if 'cn' in fin_name:
        with open(file_root+fin_name,'r',encoding='utf-8')as f:
            for line in f:
                fout.write(line.replace(' ',''))
    else:
        with open(file_root+fin_name,'r',encoding='utf-8')as f:
            for line in f:
                fout.write(line)



# with open('/data/zclfe/transformer/corpora/datum_en_train.txt','w',encoding='utf-8')as fen,open('/data/zclfe/transformer/corpora/datum_zh_train.txt','w',encoding='utf-8')as fzh:
#     for i in range(1,20,1):
#         write(f'Book{i}_cn.txt',fzh)
#         write(f'Book{i}_en.txt',fen)
#     write(f'/data/zclfe/transformer/corpora/training.zh',fzh)
#     write(f'/data/zclfe/transformer/corpora/training.en',fen)
#     write(f'/data/zclfe/transformer/corpora/test.zh',fzh)
#     write(f'/data/zclfe/transformer/corpora/test.en',fen)
#     write(f'/data/zclfe/transformer/corpora/test_news_filtered.zh',fzh)
#     write(f'/data/zclfe/transformer/corpora/test_news_filtered.en',fen)

# with open('/data/zclfe/transformer/corpora/datum_en_test.txt','w',encoding='utf-8')as fen,open('/data/zclfe/transformer/corpora/datum_zh_test.txt','w',encoding='utf-8')as fzh:
#     write(f'Book{20}_cn.txt',fzh)
#     write(f'Book{20}_en.txt',fen)

# with open('/data/zclfe/transformer/corpora/en_corpora.txt','w',encoding='utf-8')as fen,open('/data/zclfe/transformer/corpora/zh_corpora.txt','w',encoding='utf-8')as fzh:
#     for i in range(1,20,1):
#         write(f'Book{i}_cn.txt',fzh)
#         write(f'Book{i}_en.txt',fen)
#     write(f'/data/zclfe/transformer/corpora/training.zh',fzh)
#     write(f'/data/zclfe/transformer/corpora/training.en',fen)
#     write(f'/data/zclfe/transformer/corpora/test.zh',fzh)
#     write(f'/data/zclfe/transformer/corpora/test.en',fen)
#     write(f'/data/zclfe/transformer/corpora/test_news_filtered.zh',fzh)
#     write(f'/data/zclfe/transformer/corpora/test_news_filtered.en',fen)
#     write(f'Book{20}_cn.txt',fzh)
#     write(f'Book{20}_en.txt',fen)
import random
with open('/data/zclfe/transformer/corpora/en_corpora.txt','r',encoding='utf-8')as fen,open('/data/zclfe/transformer/corpora/zh_corpora.txt','r',encoding='utf-8')as fzh:
    with open('/data/zclfe/transformer/corpora/datum_en_train.txt','w',encoding='utf-8')as fen_train,open('/data/zclfe/transformer/corpora/datum_zh_train.txt','w',encoding='utf-8')as fzh_train:
        with open('/data/zclfe/transformer/corpora/datum_en_test.txt','w',encoding='utf-8')as fen_test,open('/data/zclfe/transformer/corpora/datum_zh_test.txt','w',encoding='utf-8')as fzh_test:
            for en,zh in zip(fen,fzh):
                if random.uniform(0,1)<0.95:
                    fen_train.write(en)
                    fzh_train.write(zh)
                else:
                    fen_test.write(en)
                    fzh_test.write(zh)