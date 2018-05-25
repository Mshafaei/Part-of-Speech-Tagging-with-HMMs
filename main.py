import nltk
from chardet import detect
import collections, numpy
import random
trainSet = open("train.conll", 'r')
sentences=[]
temp=[]
probabilitiesDic={}
probabilitiesDic2={}
words_tags=[]
tags=[]
wordDic={}
numberOfWordForTag={}
tagForWrod={}
numberOfline=0
# read the dataset and build array of sentences.
for line in trainSet:
    numberOfline=numberOfline+1
    if len(line)>1:
        temp.append(line.strip())
    else:
        sentences.append(temp)
        temp=[]
#build array of tuples of words and their corresponding tag(POS tag+ language tag)
for sentence in sentences:
    words_tags.append(("","*start*"))
    words_tags.append(("","*start*"))
    for part in sentence:
        p = part.split("\t")
        words_tags.append((p[0],p[1]+p[2]))

words_tags.append(("","*start*"))
words_tags.append(("","*start*"))

# extract olnly tags
for t in words_tags:
    tags.append(t[1])

#calculating bigram and trigram and find frequency of them for tags
fd_tags = nltk.FreqDist(tags)
bigrams = nltk.bigrams(tags)
trigram=nltk.trigrams(tags)
fdist = nltk.FreqDist(trigram)
fdist2 = nltk.FreqDist(bigrams)


#calculating trigram probability
def get_trigram_probability(first,second,third):
    
    trigram_frequency = fdist[(first, second , third)]
    bigram_frequency = fdist2[(first,second)]
    if first=="*start*" and second =="*start*":
        trigram_probability = float(trigram_frequency)/float(fd_tags[first])
    else:
        trigram_probability = float(trigram_frequency)/float(bigram_frequency)
    
    return(trigram_probability)

#calculating bigram probability
def get_bigram_probability(first,second):
    
    bigram_frequency = fdist2[(first,second)]
    unigram_frequency = fd_tags[first]
    bigram_probability = float(bigram_frequency)/float(unigram_frequency)
    return(bigram_probability)

pre="*start*"
pre2="*start*"
#we are iterating over the tags and finding all possible trigram combination, and calculate trigram probability for them, and make a dictionary of all two consecutive tags which returen the probability of three consecutive tag
for tag in tags:
    nextP=get_trigram_probability(pre,pre2,tag)
    if tuple([pre,pre2]) in probabilitiesDic:
        dic=probabilitiesDic[tuple([pre,pre2])]
    else:
        dic={}
    dic[tag]=nextP
    probabilitiesDic[tuple([pre,pre2])]=dic
    pre=pre2
    pre2=tag

nextP=get_trigram_probability(pre,"*start*","*start*")
if pre in probabilitiesDic:
    dic=probabilitiesDic[pre]
else:
    dic={}
dic[tag]=nextP
probabilitiesDic[pre]=dic
#================
#we are iterating over the tags and finding all possible bigram combination, and calculate bigram probability for them, and make a dictionary of all  tags which return probabilty of two consecutive tags
pre="*start*"
for tag in tags:
    nextP=get_bigram_probability(pre,tag)
    if pre in probabilitiesDic2:
        dic=probabilitiesDic2[pre]
    else:
        dic={}
    dic[tag]=nextP
    probabilitiesDic2[pre]=dic
    pre=tag

nextP=get_bigram_probability(pre,"*start*")

if pre in probabilitiesDic2:
    dic=probabilitiesDic2[pre]
else:
    dic={}
dic[tag]= nextP
probabilitiesDic2[pre]=dic

# make a dictionary of tags to words and save word count in this dictionary
for wt in words_tags:
    word=wt[0].lower()
    tag=wt[1]
    if tag=="*start*":
        continue
    if tag in numberOfWordForTag:
        wordDic=numberOfWordForTag[tag]
    else:
        numberOfWordForTag[tag]={}
        wordDic = numberOfWordForTag[tag]
    if word in wordDic:
        wordDic[word]=wordDic[word]+1
    else:
        wordDic[word]=1
# iterate over above dictionary and change count to probability
countOfWord=0
for tag in numberOfWordForTag:
    listOfWords= numberOfWordForTag[tag]
    for word in listOfWords:
        countOfWord=listOfWords[word]+countOfWord
    for word in listOfWords:
        probOfWord=float(listOfWords[word])/float(countOfWord)
        listOfWords[word]=probOfWord
        numberOfWordForTag[tag]=listOfWords
    countOfWord=0

# make a list of tags for each word
for wt in words_tags:
    word=wt[0].lower()
    tag=wt[1]
    listOfTags=[]
    if tag=="*start*":
        continue
    if word in tagForWrod:
        listOfTags=tagForWrod[word]
        if tag not in listOfTags:
            listOfTags.append(tag)
    else:
        tagForWrod[word]=[]
        listOfTags= tagForWrod[word]
        listOfTags.append(tag)
    tagForWrod[word]=listOfTags




# read the test file and save words and correspondig language tags in an array
dev = open("test.conll", "r")
output = open("result.txt", "w")
testdata=[]
lan=[]
for line in dev:
    line=line.strip()
    data=line.split("\t")
    testdata.append(data[0])
    if (len(data)>1):
        lan.append(data[1])
    else:
        lan.append("alaki")


preTag="*start*"
preTag2="*start*"
preTagProb={tuple([preTag,preTag2]):1}
viterbi={}
result={}
#empty line means start of a new sentence so we reset the pre-tags to "start" tag
for i in range(len(testdata)):
    word=testdata[i]
    wLan=lan[i]
    if len(word)==0:
        output.write('\n')
        preTag="*start*"
        preTag2="*start*"
        preTagProb={tuple([preTag,preTag2]):1}
        continue
    #I found out some errors are repeating for some special words, so I am handling them here
    if ( word=='her' and wLan=='eng') or( word=='to' and wLan=='eng'):
        if word=='to':
            assigned_tag='PART'
        elif word=='her':
            assigned_tag='PRON'


        output.write('{}\t{}\t{} \n'.format(word, wLan, assigned_tag))
        temp={}
        for w in preTagProb:
            temp [tuple([w[1],wLan+assigned_tag])]=1
        preTagProb=temp
        continue
    # if a word is starting with capital letter with high probability it is 'Propn'
    if  word!='I' and (word.istitle()):
        assigned_tag='PROPN'
        output.write('{}\t{}\t{} \n'.format(word, wLan, assigned_tag))
        temp={}
        for w in preTagProb:
            temp [tuple([w[1],wLan+assigned_tag])]=1
        preTagProb=temp
        continue
    #a in spanish language mostly is ADP
    if (word == 'a') and wLan=="spa":
        if word.lower()=='a':
            assigned_tag='ADP'
        output.write('{}\t{}\t{} \n'.format(word, wLan, assigned_tag))
        temp={}
        for w in preTagProb:
            temp [tuple([w[1],wLan+assigned_tag])]=1
        preTagProb=temp
        continue
    if word.lower() == '<unintelligible>':
        assigned_tag='UNK'
        output.write('{}\t{}\t{} \n'.format(word, wLan, assigned_tag))
        temp={}
        for w in preTagProb:
            temp [tuple([w[1],wLan+'UNK'])]=1
        preTagProb=temp
        continue
    #handling new word
    if word.lower() not in tagForWrod:
        if word.endswith("ed") or word.endswith("ing") :
            assigned_tag='VERB'
        else:
            assigned_tag='NOUN'
        output.write('{}\t{}\t{} \n'.format(word, wLan, assigned_tag))
        temp={}
        for w in preTagProb:
            temp [tuple([w[1],wLan+'NOUN'])]=1
        preTagProb=temp
        continue
    # if word only has one tag
    elif len(tagForWrod[word.lower()])==1:
        output.write('{}\t{}\t{}\n'.format(word,wLan ,tagForWrod[word.lower()][0].replace('&spa','').replace('spa','').replace('eng','').replace('&','')))
        temp={}
        for w in preTagProb:
            temp [tuple([w[1],tagForWrod[word.lower()][0]])]=1
        preTagProb=temp
#word exist in train data set and has more than one tag
    else:
        #extract all tags for word and iterate on them
        for tag in tagForWrod[word.lower()]:
            #extract probabilty of having the word in each of those tags
            wordsProb=numberOfWordForTag[tag]
            wordProb=wordsProb[word.lower()]
            #iterate over possible tags in the previouse state
            for each_tag in preTagProb:

                probOfTags={}
                #extract trigram probablity ond bigram probabilty according to previous tag and current tag
                if each_tag in probabilitiesDic:
                    probOfTags=probabilitiesDic[each_tag]
                probOfTagbBi=probabilitiesDic2[each_tag[1]]
                
                probOfTag=0
                #if we have trigram sequence
                if tag in probOfTags:
                    probOfTag=probOfTags[tag]
                #if we have bigram sequence
                if tag in probOfTagbBi:
                    probOfTag=probOfTagbBi[tag]
                #else set value with somthing rather than 0
                else:
                    probOfTag=0.0001
                probabilty=float(probOfTag)*float(wordProb)*float(preTagProb[each_tag])
                viterbi[each_tag]=probabilty
            #get the tag which has maximum amount of probabilty for one possible tag
            maxProb=max(viterbi.values())
            for w in preTagProb:
                result[tuple([w[1],tag])]=maxProb
            viterbi={}
        #update the dictionary of possible previous tags
        preTagProb=result
        result={}
        #get the tag among all possible tags which has maximum probability as tag for current word
        mostProTag=max(preTagProb,key=preTagProb.get)[1]

        output.write('{}\t{}\t{}\n'.format(word, wLan, mostProTag.replace('&spa','').replace('spa','').replace('eng','').replace('&','')))























