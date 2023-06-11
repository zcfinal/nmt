from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
from statistics import mean
import sys
import sentencepiece as spm
import sacrebleu


global sm_func
sm_func = {}
sm_func["None"] = None
sm_func["sm1"] = SmoothingFunction().method1
sm_func["sm2"] = SmoothingFunction().method2
sm_func["sm3"] = SmoothingFunction().method3
sm_func["sm4"] = SmoothingFunction().method4
sm_func["sm5"] = SmoothingFunction().method5
sm_func["sm6"] = SmoothingFunction().method6
sm_func["sm7"] = SmoothingFunction().method7


class BLEU(object):
    

    def compute_bleu(self, refs, systems, sm):
        bleu_1 = []
        bleu_2 = []
        bleu_3 = []
        bleu_4 = []
        bleu_all = []
        cnt = 0
        for i in range(len(systems)):
            refs[i] = refs[i].split()
            systems[i] = systems[i].split()
            B1 = sentence_bleu(refs[i], systems[i], weights = (1, 0, 0, 0), smoothing_function=sm)
            bleu_1.append(float(B1))
            B2 = sentence_bleu(refs[i], systems[i], weights = (0, 1, 0, 0), smoothing_function=sm)
            bleu_2.append(float(B2))
            B3 = sentence_bleu(refs[i], systems[i], weights = (0, 0, 1, 0), smoothing_function=sm)
            bleu_3.append(float(B3))
            B4 = sentence_bleu(refs[i], systems[i], weights = (0, 0, 0, 1), smoothing_function=sm)
            bleu_4.append(float(B4))    
            BA = sentence_bleu(refs[i], systems[i], smoothing_function=sm)
            bleu_all.append(float(BA))   
            cnt+=1
        print(cnt,len(systems))
        return mean(bleu_1), mean(bleu_2), mean(bleu_3), mean(bleu_4), mean(bleu_all)

    def print_score(self, ref_corpus, gen_corpus, sm):
        
        b1= corpus_bleu(ref_corpus, gen_corpus, weights = (1, 0, 0, 0), smoothing_function=sm)
        b2= corpus_bleu(ref_corpus, gen_corpus, weights = (0, 1, 0, 0), smoothing_function=sm)
        b3 = corpus_bleu(ref_corpus, gen_corpus, weights = (0, 0, 1, 0), smoothing_function=sm)
        b4 = corpus_bleu(ref_corpus, gen_corpus, weights = (0, 0, 0, 1), smoothing_function=sm)
        ba = corpus_bleu(ref_corpus, gen_corpus, weights = (0.25, 0.25, 0.25, 0.25), smoothing_function=sm)
        print("------------------------------------------")
        print(" BLEU-ALL: %02.3f, BLEU-1: %02.3f, BLEU-2: %02.3f, BLEU-3: %02.3f, BLEU-4: %02.3f" \
            %(ba * 100, b1 * 100, b2 * 100, b3 * 100, b4 * 100))


if __name__ == "__main__":
    bleu = BLEU()
    generated_file = sys.argv[1]
    sm = 'sm2'

    gen = open(generated_file, 'r', encoding='utf-8')

    gen_corpus = []
    ref_corpus = []

    gen_f=True
    for g in gen:
        if 'machine:\n' == g:
            gen_f=True
            continue
        elif 'origin:\n' == g:
            gen_f=False
            continue
        if gen_f:
            gen_corpus.append(g.strip())
        else:
            ref_corpus.append(g.strip())   

    print(sacrebleu.corpus_bleu(ref_corpus, gen_corpus).score)

    b1, b2, b3, b4, ba = bleu.compute_bleu(ref_corpus, gen_corpus, sm_func[sm])
    print("------------------------------------------")
    print("S_FUNC: %s, BLEU-ALL: %02.3f, BLEU-1: %02.3f, BLEU-2: %02.3f, BLEU-3: %02.3f, BLEU-4: %02.3f" \
        %(sm, ba * 100, b1 * 100, b2 * 100, b3 * 100, b4 * 100))
    print('corpus')
    bleu.print_score(ref_corpus,gen_corpus,sm_func['sm7'])
    gen.close()