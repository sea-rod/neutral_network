from nltk import word_tokenize
import nltk
nltk.download("punkt_tab")
def preprocess(d):
    d=d.lower()
    d="eos "+ d
    d=d.replace("."," eos")
    return d
def generate_tokens(d):
    tokens = word_tokenize(d)
    return tokens
def generate_tokens_freq(tokens):
    dct={}
    for i in tokens:
        dct[i]=0
    for i in tokens:
        dct[i]+=1
    return dct
def generate_ngrams(tokens,k):
    l=[]
    i=0
    while(i<len(tokens)):
        l.append(tokens[i:i+k])
        i=i+1
    l=l[:-1]
    return l
def generate_ngram_freq(bigram):
    dct1={}
    for i in bigram:
        st=" ".join(i)
        dct1[st]=0
    for i in bigram:
        st=" ".join(i)
        dct1[st]+=1
    return dct1
def find1(s,dct1):
    try:
        return dct1[s]
    except:
        return 0
def print_probability_table(distinct_tokens,dct,dct1):
    n=len(distinct_tokens)
    l=[[]*n for i in range(n)]
    for i in range(n):
        denominator = dct[distinct_tokens[i]]
        for j in range(n):
            numerator = find1(distinct_tokens[i]+" "+distinct_tokens[j],dct1)
            l[i].append(float("{:.3f}".format(numerator/denominator)))
    return l
d=input("Enter corpus = ")
print("\n"+'\033[1m'+"Given Corpus"+'\033[0m')
print(d)

d=preprocess(d)
print("\n"+'\033[1m'+"Preprocessing"+'\033[0m')
print(d)

tokens=generate_tokens(d)
print("\n"+'\033[1m'+"Generate Tokens"+'\033[0m')
print(tokens)

distinct_tokens = list(set(sorted(tokens)))
dct=generate_tokens_freq(tokens)
print("\n"+'\033[1m'+"Generate Frequency of Tokens"+'\033[0m')
print(dct)

bigram = generate_ngrams(tokens,2)
print("\n"+'\033[1m'+"Generate bigrams"+'\033[0m')
for i in bigram:
    print("'{}'".format(' '.join(i)), end=", ")

dct1=generate_ngram_freq(bigram)
print("\n\n"+'\033[1m'+"Generate Frequency of bigrams"+'\033[0m')
print(dct1)

probability_table=print_probability_table(distinct_tokens,dct,dct1)
print("\n"+'\033[1m'+"Probability table"+'\033[0m'+"\n")

n=len(distinct_tokens)
print("\t"+'\033[1m', end="")
for i in range(n):
    print(distinct_tokens[i],end="\t")
print('\033[0m'+"\n")

for i in range(n):
    print('\033[1m',distinct_tokens[i],'\033[0m',end="\t")
    for j in range(n):
        print(probability_table[i][j],end="\t")
    print("")

print("\n","-"*100)
text = input("\nEnter text to check its probability = ")
print("\n"+'\033[1m'+"Given Text"+'\033[0m')
print(text)

p = preprocess(text)
print("\n"+'\033[1m'+"Preprocessing"+'\033[0m')
print(p)

t=generate_tokens(p)
print("\n"+'\033[1m'+"Generate Tokens"+'\033[0m')
print(t)

n = generate_ngrams(t,2)
print("\n"+'\033[1m'+"Generate bigrams"+'\033[0m')
for i in n:
    print("'{}'".format(' '.join(i)), end=", ")
print("\n\n"+'\033[1m'+"Calculate bigram probability"+'\033[0m')
s=1
dct2={}
for i in n:
    dct2[" ".join(i)]=0

for i in n:
    k=distinct_tokens.index(i[0])
    m=distinct_tokens.index(i[1])
    dct2[" ".join(i)]=probability_table[k][m]
    print("P('{}')\t=  ".format(' '.join(i)),probability_table[k][m])
    s*=probability_table[k][m]

print("\n"+'\033[1m'+ "Calculate Probability of the sentence"+'\033[0m')
print(f"P('{text}') \n= ",end="")
x=dct2.popitem()
for i in dct2:
    print(f"P('{i}')", end=" * ")
print(f"P('{x[0]}')\n= ", end='')

for i in dct2:
    print(dct2[i], end=" * ")
print(x[1],"\n=",s)

print("\n"+'\033[1m'+f"Probability('{text}') = "+"{:.5f}".format(s))
