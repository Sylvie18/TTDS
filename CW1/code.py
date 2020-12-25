import re
import numpy as np
from xml.dom.minidom import parse
from nltk.stem.porter import PorterStemmer

# Pre-processes text
def process_text(text):
    # Tokenisation
    token = re.compile(r'\b[a-zA-Z0-9]+\b', re.I).findall(text)
    # Case folding
    lower_token = [word.lower() for word in token]
    # Stopping
    stop = [word for word in lower_token if word not in ST]
    # Porter stemming
    porter_stemmer = PorterStemmer()
    result = [porter_stemmer.stem(word) for word in stop]

    return result

# Pre-processes query word. The parameter is a token or a list of tokens.
def process_word(word):
    if word not in ST:
        porter_stemmer = PorterStemmer()
        word = porter_stemmer.stem(word)
        return word
    return ''

# Find the position of the word in the document
def find_position(words, word, docno):
    document = {}
    document = document.fromkeys([docno])

    position = np.where(np.array(words) == word)
    for each in position:
        each = [i + 1 for i in each]
        document[docno] = list(each)

    return document

# Creates a positional inverted index
def creat_index():
    global docnumber  # the last DOCNO
    global docamount  # the document amount

    # Parse XML file
    domTree = parse("CW1collection/trec.5000.xml")
    docs = domTree.documentElement.getElementsByTagName("DOC")

    docamount = len(docs)
    index = {}

    for doc in docs:
        docno = doc.getElementsByTagName("DOCNO")[0].childNodes[0].data
        text = doc.getElementsByTagName("TEXT")[0].childNodes[0].data
        if doc.getElementsByTagName("HEADLINE"):
            headline = doc.getElementsByTagName("HEADLINE")[0].childNodes[0].data
            text = headline + text

        words = process_text(text)
        for word in words:
            position = find_position(words, word, docno)
            if word in index:
                index[word].update(position)
            else:
                index[word] = position

        docnumber = docno

    return index

# Write index to the file in a formatted version
def write_index_file(index):
    with open('CW1result/trec.5000.index.txt', 'w', encoding='UTF-8-sig') as f:
        for word, docs in index.items():
            indexing = word + ':' + str(docs.__len__()) + '\n'
            for doc, pos in docs.items():
                indexing += '\t' + doc + ':' + ' '
                for each in pos:
                    if each == pos[-1]:
                        indexing += str(each)
                    else:
                        indexing += str(each) + ','
                indexing += '\n'
            f.write(indexing+'\n')
    f.close()

# Load index into memory
def extract_index(file_name):
    index = {}
    with open(file_name, 'r', encoding='UTF-8-sig') as f:
        use = f.read().split('\n\n')
        for each in use:
            temp = {}
            each = each.split('\n\t')
            word = each[0].split(':')[0]
            if word:
                for item in each[1:]:
                    doc = item.split(':')[0]
                    pos = item.split(':')[1].strip().split(',')
                    convert = []
                    for one in pos:
                        convert.append(int(one))
                    temp[doc]=convert
                index[word]=temp
    f.close()
    return index

# Return the list of docID
def getresult(word):
    result = []
    for key in index[word].keys():
        result.append(int(key))
    return result

# Search a single word
def single(word):
    result = []
    word = process_word(word)
    if word in index.keys():
        return getresult(word)
    return result

# Uniform result format
def answerformat(set):
    result = []
    for each in set:
        result.append(int(each))
    return result

# Set operations
def calculate(word1, word2, operation):
    result = []
    if operation == 'OR':
        if word1 not in index.keys() and word2 in index.keys():
            return getresult(word2)
        if word1 in index.keys() and word2 not in index.keys():
            return getresult(word1)
        if word1 not in index.keys() and word2 not in index.keys():
            return result

    if word1 in index.keys() and word2 in index.keys():
        compare1 = index.get(word1)
        compare2 = index.get(word2)
        if operation == 'NOT':
            return answerformat(compare1.keys() - compare2.keys())
        elif operation == 'AND':
            return answerformat(compare1.keys() & compare2.keys())
        else:
            return answerformat(compare1.keys() | compare2.keys())

    return result

# Boolean search
def bool_search(querywords):
    if len(querywords) == 1:
        return single(querywords[0])

    # Process the search with phrases
    word1 = querywords[0]
    word2 = querywords[-1]
    result1 = []
    result2 = []
    haveNot = False

    if '"' in word1 and '"' not in word2:
        result1 = phrase_search(word1)
        if word2.find('NOT') == 0:
            haveNot = True
            word2 = process_word(word2.split(' ')[-1])
        result2 = single(word2)
    if '"' not in word1 and '"' in word2:
        result1 = single(process_word(word1))
        result2 = phrase_search(word2)
    if '"' in word1 and '"' in word2:
        result1 = phrase_search(word1)
        result2 = phrase_search(word2)

    if len(result1) > 0:
        if haveNot:
            return list(set(result1) - set(result2))
        if 'AND' in querywords:
            return list(set(result1) & set(result2))
        if 'OR' in querywords:
            return list(set(result1) | set(result2))

    # Process the search only has words
    else:
        word1 = process_word(word1)

        if word2.find('NOT') == 0:
            word2 = process_word(word2.split(' ')[-1])
            return calculate(word1, word2, 'NOT')

        word2 = process_word(word2)

        if 'AND' in querywords:
            return calculate(word1, word2, 'AND')
        if 'OR' in querywords:
            return calculate(word1, word2, 'OR')

# Phrase search
def phrase_search(phrase):
    result = []
    phrase = phrase.strip('"').split(' ')

    word1 = process_word(phrase[0])
    word2 = process_word(phrase[-1])

    if word1 in index.keys() and word2 in index.keys():
        compare1 = index.get(word1)
        compare2 = index.get(word2)

        for key1 in compare1.keys():
            if key1 in compare2.keys():
                value1 = compare1[key1]
                value2 = compare2[key1]
                for pos in value1:
                    if pos+1 in value2:
                        result.append(int(key1))
                        break

    return result

# Proximity search
def proximity_search(proximity):
    result = []

    num = proximity.split('#')[1].split('(')[0].strip(' ')
    word1 = proximity.split('(')[1].split(',')[0].strip(' ')
    word2 = proximity.split(',')[1].split(')')[0].strip(' ')

    word1 = process_word(word1)
    word2 = process_word(word2)

    if word1 in index.keys() and word2 in index.keys():
        compare1 = index.get(word1)
        compare2 = index.get(word2)

        for key1 in compare1.keys():
            if key1 in compare2.keys():
                value1 = compare1[key1]
                value2 = compare2[key1]
                for pos in value1:
                    for i in range(1, int(num)+1):
                        if pos + i in value2 or pos - i in value2:
                            result.append(int(key1))
                            break
                    else:
                        continue
                    break

    return result

# Ranked IR based on TFIDF
def tfidf(querywords):
    result = []
    idf = []
    subindex = []

    for term in querywords[1:]:
        term = process_word(term.lower().strip())
        if term in index.keys():
            docs = index.get(term)
            df = docs.__len__()
            idf.append(np.log10(docamount/df))
            subindex.append(docs)
        else:
            idf.append(0)
            subindex.append({})

    for i in range(1, int(docnumber)+1):
        w = 0
        for num in range(len(querywords)-1):
            if str(i) in subindex[num].keys():
                tf = subindex[num].get(str(i)).__len__()
                value = 1 + (np.log10(tf))
            else:
                value = 0
            w = w + (value * idf[num])

        result.append([int(querywords[0]), i, "{:.4f}".format(w)])

    return sorted(result, key=lambda x: x[-1], reverse=-True)

def result_boolean():
    fread = open('CW1collection/queries.boolean.txt', 'r', encoding='UTF-8-sig')
    fwrite = open('CW1result/results.boolean.txt', 'w', encoding='UTF-8-sig')

    for query in fread:
        querywords = []
        query = re.split(r' +(AND|OR) +', query)
        num = query[0].split(' ', 1)[0].strip()
        word1 = query[0].split(' ', 1)[1].strip()

        querywords.append(word1)
        for item in query[1:]:
            querywords.append(item.strip())

        if word1.startswith('"') and len(querywords) == 1:
            result = phrase_search(word1)
        elif word1.startswith('#'):
            result = proximity_search(word1)
        else:
            result = bool_search(querywords)

        result = sorted(result)
        for each in result:
            line = num + ',' + str(each) + '\n'
            fwrite.write(line)

    fread.close()
    fwrite.close()

def result_ranked():
    fread = open('CW1collection/queries.ranked.txt', 'r', encoding='UTF-8-sig')
    fwrite = open('CW1result/results.ranked.txt', 'w', encoding='UTF-8-sig')

    for query in fread:
        querywords = re.compile(r'\b[a-zA-Z0-9]+\b', re.I).findall(query)

        result = tfidf(querywords)

        for each in range(150):
            line = ''
            for item in result[each]:
                if item == result[each][-1]:
                    line += str(item)
                else:
                    line += str(item) + ','
            line += '\n'
            fwrite.write(line)

    fread.close()
    fwrite.close()

if __name__ == '__main__':
    global ST
    global index

    ST = []
    with open('englishST.txt', 'r', encoding='UTF-8-sig') as f:
        for line in f:
            ST.append(line.replace('\n', ''))
    f.close()

    creat_index = creat_index()
    write_index_file(creat_index)

    index = extract_index('CW1result/trec.5000.index.txt')
    result_boolean()
    result_ranked()
