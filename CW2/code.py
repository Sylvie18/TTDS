import re
import pandas as pd
import numpy as np
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
from collections import Counter
from scipy.sparse import dok_matrix
from sklearn.svm import LinearSVC

sys_res = {}
qrels = {}
trntst = {}
mires = []
randnum = 18

ST = []
with open('englishST.txt', 'r', encoding='UTF-8-sig') as f:
    for line in f:
        ST.append(line.replace('\n', ''))
f.close()


def Readcsv():
    file1 = pd.read_csv('system_results.csv')
    for i in range(len(file1)):
        item1 = []
        system_number = file1['system_number'][i]
        query_number = file1['query_number'][i]
        item1.append(file1['doc_number'][i])
        item1.append(file1['rank_of_doc'][i])
        item1.append(file1['score'][i])

        if system_number not in sys_res:
            sys_res.setdefault(system_number, {})

        if query_number in sys_res[system_number]:
            sys_res[system_number][query_number].append(item1)
        else:
            sys_res[system_number].setdefault(query_number, []).append(item1)

    file2 = pd.read_csv('qrels.csv')
    for i in range(len(file2)):
        item2 = []
        query_id = file2['query_id'][i]
        item2.append(file2['doc_id'][i])
        item2.append(file2['relevance'][i])

        if query_id in qrels:
            qrels[query_id].append(item2)
        else:
            qrels.setdefault(query_id, []).append(item2)

def CalPR(docs, query_number, num):
    rank = 0
    for doc in docs[:num]:
        for doc_id in qrels[query_number]:
            if doc[0] in doc_id:
                rank += 1
                break
    return rank

def CalGrade(k, docs, query_number):
    grade = [0]*k
    for i in range(k):
        for doc_id in qrels[query_number]:
            if docs[i][0] in doc_id:
                grade[i] = doc_id[1]
                break
    return grade

def CalDCG(grade, k):
    DCG = grade[0]
    for i in range(2, k+1):
        DCG += grade[i-1] / np.log2(i)
    return DCG

def CaliGrade(k, query_number):
    igrade = []
    for rel in qrels[query_number]:
        igrade.append(rel[1])

    off = k - len(igrade)
    if off > 0:
        for i in range(off):
            igrade.append(0)
    return igrade[:k]

def EVAL():
    f = open("ir_eval.csv", "w")
    hdr = "system_number,query_number,P@10,R@50,r-precision,AP,nDCG@10,nDCG@20\n"
    f.write(hdr)

    Readcsv()
    P = 10
    R = 50
    for sys_number in range(1, 7):
        rankP = []
        rankR = []
        r_precision = []
        AP = []
        DCGlist = {10: [], 20: []}

        for query_number in range(1, 11):
            docs = sys_res[sys_number][query_number]
            rankP.append(CalPR(docs, query_number, P) / P)
            relR = len(qrels[query_number])
            rankR.append(CalPR(docs, query_number, R) / relR)
            r_precision.append(CalPR(docs, query_number, relR) / relR)

            res = 0
            for i in range(len(docs)):
                relk = 0
                pk = CalPR(docs, query_number, i+1) / (i+1)
                if pk != 0:
                    for doc_id in qrels[query_number]:
                        if docs[i][0] in doc_id:
                            relk = 1
                            break
                res += pk * relk
            AP.append(res/relR)

            for cutoff in [10,20]:
                grade = CalGrade(cutoff, docs, query_number)
                DCG = CalDCG(grade, cutoff)
                igrade = CaliGrade(cutoff, query_number)
                iDCG = CalDCG(igrade, cutoff)

                if iDCG:
                    DCGlist[cutoff].append(DCG/iDCG)
                else:
                    DCGlist[cutoff].append(0)

            line = "{},{},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}\n".format(sys_number, query_number, rankP[-1], rankR[-1], r_precision[-1], AP[-1], DCGlist[10][-1], DCGlist[20][-1])
            f.write(line)

            if query_number == 10:
                line = "{},mean,{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}\n".format(sys_number, np.mean(rankP), np.mean(rankR), np.mean(r_precision), np.mean(AP), np.mean(DCGlist[10]), np.mean(DCGlist[20]))
                f.write(line)
    f.close()

def ProcessText(text):
    token = re.compile(r'\b[a-zA-Z0-9]+\b', re.I).findall(text)
    lower_token = [word.lower() for word in token]
    result = [word for word in lower_token if word not in ST]
    return result

def Readtsv():
    file = pd.read_csv('train_and_dev.tsv', sep='\t', header=None)
    text = ''
    for i in range(len(file)):
        corpora = file[0][i]
        verses = file[1][i]

        if corpora in trntst:
            trntst[corpora].append(verses)
        else:
            trntst.setdefault(corpora, []).append(verses)

        text += verses + '\n'

    for corpora in trntst:
        for i, verses in enumerate(trntst[corpora]):
            trntst[corpora][i] = ProcessText(verses)

    allpreterms = list(set(ProcessText(text)))
    return allpreterms

def WordLevel():
    allpreterms = Readtsv()
    orders = [['Quran', 'OT', 'NT'], ['OT', 'Quran', 'NT'], ['NT', 'Quran', 'OT']]
    chires = []

    for order in orders:
        target = order[0]
        targetlen = len(trntst[target])
        otherlen = len(trntst[order[1]]) + len(trntst[order[2]])
        N = targetlen + otherlen
        onemires = []
        onechires = []

        for term in allpreterms:
            N11 = 0
            for item in trntst[target]:
                if term in item:
                    N11 += 1
            N01 = targetlen - N11

            N10 = 0
            for corpora in order[1:]:
                for item in trntst[corpora]:
                    if term in item:
                        N10 += 1
            N00 = otherlen - N10

            N1x = N11 + N10
            Nx1 = N11 + N01
            N0x = N00 + N01
            Nx0 = N00 + N10

            sub1 = np.log2(N*N11 / (N1x*Nx1)) if N*N11 != 0 and N1x*Nx1 != 0 else 0
            sub2 = np.log2(N*N01 / (N0x*Nx1)) if N*N01 != 0 and N0x*Nx1 != 0 else 0
            sub3 = np.log2(N*N10 / (N1x*Nx0)) if N*N10 != 0 and N1x*Nx0 != 0 else 0
            sub4 = np.log2(N*N00 / (N0x*Nx0)) if N*N00 != 0 and N0x*Nx0 != 0 else 0
            mi = (N11/N)*sub1 + (N01/N)*sub2 + (N10/N)*sub3 + (N00/N)*sub4

            below = Nx1 * N1x * Nx0 * N0x
            chi = N * np.square(N11*N00-N10*N01) / below if below != 0 else 0

            onemires.append([term, mi])
            onechires.append([term, chi])

        mires.append(sorted(onemires, key=lambda x: x[-1], reverse=-True))
        chires.append(sorted(onechires, key=lambda x: x[-1], reverse=-True))

    print("MI")
    for each in mires:
        print(each[:10])

    print("CHI")
    for each in chires:
        print(each[:10])

def CalTopicScore(scoreall, start, end):
    score = [0] * 20
    for doc in scoreall[start:end]:
        for item in doc:
            score[item[0]] += item[1]
    return np.array(score)/(end-start)

def TopicLevel():
    texts = trntst['Quran'] + trntst['OT'] + trntst['NT']
    len1 = len(trntst['Quran'])
    len2 = len(trntst['OT'])
    lenall = len(texts)
    scoreall = []

    dictionary = Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    lda = LdaModel(corpus, id2word=dictionary, num_topics=20)

    for i in range(lenall):
        scoreall.append(lda.get_document_topics(corpus[i]))

    print('LDA')
    order = [[0, len1], [len1, len1 + len2], [len1 + len2, lenall]]
    for i in range(3):
        avgscore = CalTopicScore(scoreall, order[i][0], order[i][1])
        print(avgscore)

    for topic in lda.print_topics(num_topics=20, num_words=10):
        print(topic)

def SimpleProcessText(text):
    token = re.compile(r'\b[a-zA-Z0-9]+\b', re.I).findall(text)
    result = [word.lower() for word in token]
    return result

def GetUse(file, type):
    label = ['Quran', 'OT', 'NT']
    text = ''
    X = []
    y = []

    for i in range(len(file)):
        corpora = file[0][i]
        verses = file[1][i]

        X.append(SimpleProcessText(verses))
        y.append(label.index(corpora))
        text += verses + '\n'

    if type == 'trn':
        return X, y, text
    return X, y

def SparseMatrix(use, classterms):
    S = dok_matrix((len(use), len(classterms)))
    for i in range(len(use)):
        num = Counter(use[i])
        for term, value in num.items():
            if term in classterms:
                j = classterms.index(term)
                S[i, j] = value
    return S

def CreateMatrix(highestmi):
    filetrn = pd.read_csv('train_and_dev.tsv', sep='\t', header=None)
    filetst = pd.read_csv('test.tsv', sep='\t', header=None)

    Xbase, ybase, text = GetUse(filetrn, 'trn')
    Xtst, ytst = GetUse(filetst, 'tst')

    np.random.seed(randnum)
    np.random.shuffle(Xbase)
    np.random.seed(randnum)
    np.random.shuffle(ybase)

    trnlen = int(0.9*len(Xbase))
    Xtrn = Xbase[:trnlen]
    ytrn = ybase[:trnlen]
    Xdev = Xbase[trnlen:]
    ydev = ybase[trnlen:]

    if highestmi:
        classterms = highestmi
    else:
        classterms = list(set(SimpleProcessText(text)))

    Strn = SparseMatrix(Xtrn, classterms)
    Sdev = SparseMatrix(Xdev, classterms)
    Stst = SparseMatrix(Xtst, classterms)
    return Strn, Sdev, Stst, Xdev, ytrn, ydev, ytst

def CalAcc(y_pred, y_true, system, split):
    f = open("classification.csv", "a")
    if system == 'baseline' and split == 'train':
        hdr = "system,split,p-quran,r-quran,f-quran,p-ot,r-ot,f-ot,p-nt,r-nt,f-nt,p-macro,r-macro,f-macro\n"
        f.write(hdr)

    dfpred = pd.DataFrame(y_pred)
    dftrue = pd.DataFrame(y_true)
    labels = [0, 1, 2]
    precision = []
    recall = []
    F1 = []

    for label in labels:
        pred = dfpred[dfpred[0] == label]
        index_pred = pred.index.tolist()
        true = dftrue[dftrue[0] == label]
        index_true = dftrue.reindex(index=index_pred)

        precision.append(sum(np.array(pred) == np.array(index_true)) / len(pred))
        recall.append(sum(np.array(pred) == np.array(index_true)) / len(true))
        F1.append(2*precision[label]*recall[label] / (precision[label]+recall[label]))

    macro_P = np.mean(precision)
    macro_R = np.mean(recall)
    macro_F1 = 2*macro_P*macro_R / (macro_P+macro_R)

    line = system+','+split+','+"{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}\n".format(precision[0][0], recall[0][0], F1[0][0], precision[1][0], recall[1][0], F1[1][0], precision[2][0], recall[2][0], F1[2][0], macro_P, macro_R, macro_F1)
    f.write(line)

def ThreeIns(ydev_pred, ydev, Xdev):
    instance = 0
    print('ThreeIns')
    for i in range(len(ydev_pred)):
        if ydev_pred[i] != ydev[i] and instance < 3:
            print(ydev_pred[i], ydev[i], Xdev[i])
            instance += 1

def BaseLine():
    Strn, Sdev, Stst, Xdev, ytrn, ydev, ytst = CreateMatrix([])
    model = LinearSVC(C=1e3, random_state=0, max_iter=1e5)
    model.fit(Strn, ytrn)

    ytrn_pred = model.predict(Strn)
    ydev_pred = model.predict(Sdev)
    ytst_pred = model.predict(Stst)

    CalAcc(ytrn_pred, ytrn, 'baseline', 'train')
    CalAcc(ydev_pred, ydev, 'baseline', 'dev')
    CalAcc(ytst_pred, ytst, 'baseline', 'test')

    ThreeIns(ydev_pred, ydev, Xdev)

# def HighestMI():
#     use = mires[0] + mires[1] + mires[2]
#     rank = sorted(use, key=lambda x: x[-1], reverse=-True)
#     all = []
#
#     for each in rank:
#         all.append(each[0])
#
#     highestmi = list(set(all))
#     highestmi.sort(key=all.index)
#     return highestmi[:int(len(highestmi)/5)]

def Improve():
    # highestmi = HighestMI()
    # Strn, Sdev, Stst, Xdev, ytrn, ydev, ytst = CreateMatrix(highestmi)

    # for i in range[100, 300, 500, 10, 30, 50]:
    Strn, Sdev, Stst, Xdev, ytrn, ydev, ytst = CreateMatrix([])
    model = LinearSVC(C=10, random_state=0, max_iter=1e5)
    model.fit(Strn, ytrn)

    ytrn_pred = model.predict(Strn)
    ydev_pred = model.predict(Sdev)
    ytst_pred = model.predict(Stst)

    CalAcc(ytrn_pred, ytrn, 'improved', 'train')
    CalAcc(ydev_pred, ydev, 'improved', 'dev')
    CalAcc(ytst_pred, ytst, 'improved', 'test')

if __name__ == '__main__':
    EVAL()
    WordLevel()
    TopicLevel()
    BaseLine()
    Improve()
