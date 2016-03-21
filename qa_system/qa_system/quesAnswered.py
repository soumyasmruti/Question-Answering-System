import sys
reload(sys)  
sys.setdefaultencoding('utf8')
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from corenlp import *
import glob, os, re, ast, operator, json, collections, unicodedata, string
import phrase_process as pp

corenlp = StanfordCoreNLP()
WNL = WordNetLemmatizer()

scorer = {"clue": 3, "good_clue":4, "confident":5, "slam_dunk":6}

with open("utils/prep_list.txt") as pl:
    locationprep = set(pl.read().split())

with open("utils/stop_words.txt") as sl:
    stop = list(set(sl.read().split()) - locationprep)

def scoreSent(ques, sentList, qTags):
    return 0

units = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
        "sixteen", "seventeen", "eighteen", "nineteen",  "twenty", "thirty", 
        "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
        "hundred", "thousand", "million", "billion", "trillion"]

def removeStopWords(query):
    querywords = query.split()
    resultwords  = [WNL.lemmatize(word.lower()) for word in querywords if word.lower() not in stop]
    result = ' '.join(resultwords)
    return result

def scoreSent(ques, sentList, qTags, ptree, date):
    quesW = set(ques.replace("?", "").split()) #set of words in a question
    quesW = [WNL.lemmatize(x) for x in quesW] #question words lemmatized
    quesWma = [WNL.lemmatize(x.lower()) for x in quesW]
    sentScore = {} #Final list of sentence with scores
    qverbL = [] #
    qner = set() #NER tags for questions only tags
    qppL = re.findall('name[\s|)|(]*PP', ptree) #preposition list for name+PP for whatRules
    p = pp.parse(ques) #Finding Head Noun for question
    np = pp.find(p, 'PP') #Prepositional Phrase for Finding Head Noun for question
    npl = [x for x in np] #Finding Head Noun for question
    head_noun = 0 
    if len(npl) > 0:
        ppl = pp.unpack(npl)
        head_noun = ppl[ppl.index("NN")+1]#Get Head Noun word
    hqTag = []
    for t in qTags:
        qner.add(t[1]['NamedEntityTag'])
        if 'VB' in t[1]['PartOfSpeech']: #Verb Matching
            qverbL.append(t[0])
            qverbL.append(WNL.lemmatize(t[0].lower(), 'v'))
        if 'RB' or 'JJ' in t[1]['PartOfSpeech']: #Verb Matching
            hqTag.append(t[0])
    for sent, values in sentList.iteritems():
        sent = sent.replace("\n", " ").replace('"', "").replace(',', "")
        sentScore[sent] = 0 
        sentScore[date] = 0 
        sverbL = [] 
        sentL= removeStopWords(sent).split()#Lemmatized Sentence List
        tempL = set(quesWma)&(set(sentL)-set(qverbL)) #Word Matched Set
        #print "Matched Words: "+str(tempL)
        #print "Question Words: "+str(quesW)
        #print "Sentence Words: "+str(sentL)
        sentScore[sent] += len(tempL)*3
        st = values['tags']
        sner, stags = set(), set()
        head_noun_sent = 0
        for s in st:
            sner.add(s[1]['NamedEntityTag'])
            stags.add(s[1]['PartOfSpeech'])
            if 'VB' in s[1]['PartOfSpeech']:
                sverbL.append(WNL.lemmatize(s[0].lower(), 'v'))
            if 'NN' in s[1]['PartOfSpeech']:
                head_noun_sent = s[0]
        verbL = set(qverbL)&set(sverbL)
        #print qverbL, sverbL, verbL
        sentScore[sent] += len(verbL)*6
        if "Who" in quesW:
            sentScore[sent] += whoRules(qner,sner, quesW, sentL)
            
        if "When" in quesW:
            sentScore[sent] = whenRules(qner,sner, quesW, sentL, sentScore[sent])
            if len(date) >0:
                sentScore[date] += dateline(quesW)
            
        if "Where" in quesW:
            sentScore[sent] += whereRules(sentL, sner)
            if len(date) >0:
                sentScore[date] += dateline(quesW)
            
        if "What" in quesW:
            sentScore[sent] += whatRules(qner, quesW, sentL, stags, qppL, head_noun, head_noun_sent)
            #print sentScore[sent]
            
        if "how" in quesW:
            sentScore[sent] += howRules(quesW, sentL, hqTag, sent)
            
    if "Why" in quesW:
        sentScore = whyRules(sentScore, sentList)
        
    return sentScore

def whyRules(sentSc, sentL):
    whySentDic = {}
    #maxim = max(sentSc.values())
    #bestList = [k for k,v in sentSc.items() if v == maxim]
    #best = bestList[len(bestList)-1]
    sentL = [str(x).replace("\n", " ") for x in sentL]
    best = max(sentSc, key=sentSc.get) #Find the best sentence
    
    for idx, sent in enumerate(sentL):
        sent = sent.replace("\n", " ")
        score = 0
        if sent == best:
            score += scorer['clue']
        try: 
            if idx == sentL.index(best)-1:
                score += scorer['clue']
        except IndexError:
            pass
        except ValueError:
            pass
        try:
            if idx == sentL.index(best)+1:
                score += scorer['good_clue']
        except IndexError:
            pass
        except ValueError:
            pass
        sentW = [WNL.lemmatize(x) for x in set(sent.split())]
        if "want" in sentW:
            score += scorer['good_clue']
        if "so" in sentW or "because" in sentW:
            score += scorer['good_clue']
        whySentDic[sent] = score
    return whySentDic

def whoRules(qNers, sNers, qw, sw):
    score = 0
    count = 0
    for x in sNers:
        if x == 'PERSON':
            count += 1
    if "PERSON" in qNers and count > 1:
            score += scorer['good_clue']
    if "PERSON" not in qNers and "PERSON" in sNers:
        score += scorer['good_clue']
        #print score
    if "PERSON" not in qNers and "name" in sw:
        score += scorer['good_clue']
    if "PERSON" in sNers:
        score += scorer['good_clue']
    return score
    
def dateTime(qw):
    mscore = 0
    if "happen" in qw:
        mscore += scorer['good_clue']
    if "take" in qw and "place" in qw:
        mscore += scorer['good_clue']
    if "this" in qw:
        mscore += scorer['slam_dunk']
    if "story" in qw:
        mscore += scorer['slam_dunk']
    return mscore
    
def whenRules(qNers, sNers, qw, sw, mscore):
    score = mscore
    if "DATE" in sNers or "p.m." in sw or "a.m." in sw:
        score += scorer['good_clue'] 
    if len(set(['start', 'begin'])&set(qw)) > 0 and len(set(['start', 'begin', 'since', 'year'])&set(sw)) > 0:
        score += scorer['slam_dunk']
    if ['the', 'last'] in qw and len(set(['first', 'last', 'since', 'ago'])&set(sw)) > 0:
        score += scorer['slam_dunk']
    
    return score
    
def whereRules(sw, sTags):
    score = 0
    if len(set(sw)&locationprep) > 0:
        score += scorer['good_clue']
    #print sTags
    if "LOCATION" in sTags:
        score += scorer['confident']
    return score
    
def whatRules(qNers, qw, sw, sTags, pl, head_noun, head_noun_sent):
    score = 0
    if "DATE" in qNers and (len(set(['today', 'yesterday', 'tomorrow'])&set(sw)) > 0 or ['last', 'night'] in sw):
        score += scorer['clue']
    if "kind" in qw and len(set(['call', 'from'])&set(sw)) > 0:
        score += scorer['good_clue']
    if "name" in qw and len(set(['call', 'known', 'name'])&set(sw)) > 0:
        score += scorer['slam_dunk']
    if len(pl) > 0 and "NNP" in sTags:
        if head_noun_sent == head_noun and head_noun_sent != 0 and head_noun != 0:
            score += scorer['slam_dunk']
    return score

def howRules(qw, sw, qTag, sent):
    score = 0
    checkDig = [x.isdigit() for x in sw]
    if "many" or "much" or "long" in qw:
        if "True" in checkDig:
            score += scorer['confident']
        if any(word in sent for word in units):
            score += scorer['confident']
    if len(qw) > 1:
        if qw[1] in qTag:
            if "True" in checkDig:
                score += scorer['confident']
            if any(word in sent for word in units):
                score += scorer['confident']
    return score

def findAnswers(sentDic, ques, sent_dic, date):
    #ques = ques.replace("?", "")
    #print sentDic
    
    if len(sentDic) != 0:
        maxim = max(sentDic.values())
        bestList = [k for k,v in sentDic.items() if v == maxim]
        best = bestList[0]
        st = sent_dic[best]['tags']
        quesW = ques.split()
        bestW = best.split()
        resultWords = [word for word in bestW if word not in quesW]
        newBest = " ".join(resultWords)
        #print best
        if "Who" in ques:
            whol = []
            for who in st:
                if who[1]['NamedEntityTag'] == 'PERSON': #Human Matching
                    whol.append(who[0])
            if len(whol) == 0:
                bestL = newBest.split()
                if 'by' in bestL:
                    whol.append(bestL[bestL.index("by")+1])
            #print sentDic[newBest]
            return ' '.join(str(elem) for elem in whol), newBest
            
        if "When" in ques:
            ws = set(["first", "last", "since", "ago", "start", "begin", "year"])&set(newBest.split())
            if len(ws) != 0:
                return ' '.join(str(elem) for elem in ws), newBest
            else:
                for when in st:
                    if when[1]['NamedEntityTag'] == 'DATE': #Date Matching
                        return when[0], newBest
            if ("happen" or "take place" or "this" or "story" in ques) and len(date) >0:
                return date, date
            
        if "Where" in ques:
            whr = set(newBest.split())&locationprep
            whr = list(whr)
            for where in st:
                if where[1]['NamedEntityTag'] == 'LOCATION': #Location Matching
                    whr.append(where[0])     
            if len(whr) != 0:
                return ' '.join(str(elem) for elem in whr), newBest
            if ("happen" or "take place" or "this" or "story" in ques) and len(date) >0:
                return date, date
                
        if "What" in ques:
            wht = set(["name", "call", "known", "from", "today", "yesterday", "tomorrow", "last", "night"])&set(newBest.split())
            for what in st:
                if 'NNP' in what[1]['PartOfSpeech']: #Proper Noun Matching
                    return what[0], newBest
            if len(wht) != 0:
                return ' '.join(str(elem) for elem in wht), newBest
            if "What are" in ques:
                lastWor = ques.split()[-1]
                tempbb = ""
                try:
                    if "(" in newBest:
                        tempbb = newBest.split(lastWor, 1)[1].split("(")[1].split(")")[0]
                    elif "--" in newBest:
                        tempbb = newBest.split(lastWor, 1)[1].split("--")[1]
                    elif "means" in newBest:
                        tempbb = newBest.split(lastWor, 1)[1].split("means")[1]
                    elif "mean" in newBest:
                        tempbb = newBest.split(lastWor, 1)[1].split("mean")[1]
                    elif "are" in newBest:
                        tempbb = newBest.split(lastWor, 1)[1].split("are")[1]
                    elif "is" in newBest:
                        tempbb = newBest.split(lastWor, 1)[1].split("is")[1]
                    return tempbb, tempbb
                except IndexError:
                    pass
            
        if "how" in ques:
            checkDig = [x.isdigit() for x in quesW]
            idx = 0
            if "TRUE" in checkDig:
                idx = checkDig.index("TRUE")
                return quesW[idx], quesW[idx]
            cUnits = set(units)&set(newBest.split())
            if len(cUnits) > 1:
                return ' '.join(str(elem) for elem in cUnits), ' '.join(str(elem) for elem in cUnits)
                    
            
        if "Why" in ques:
            best = bestList[len(bestList)-1]
            bestW = best.split()
            rWords = [word for word in bestW if word not in quesW]
            nBest = " ".join(rWords)
            why = set(["so", "because", "want"])&set(nBest.split())
            if len(why) != 0:
                return ' '.join(str(elem) for elem in why), nBest
            return nBest, nBest
        #print sentDic[newBest]
        return newBest, newBest
        
    return "",""


perList = ["person", "people", "man", "woman", "women", "men", "male", "female", "human", "social class", "working class"]
quesIdList = []
quesDic, stories = {}, {}
def quesExtractor(dirpath, storyID):
    sent_dic = {}
    with open(dirpath+storyID+".story")as s:
        doc = s.read()
        try:
            head = re.findall('^DATE: ?(.*)', doc)[0]
        except IndexError:
            head = ""
        text = re.findall(r'\nTEXT:(((\n|^)*.*)*)', doc)[0][0]
        sentences = sent_tokenize(text.strip())
        for sent in sentences:
            sent = sent.replace("\n"," ").replace('"', "").replace(',', "")
            s = corenlp.parse(sent)
            sd = ast.literal_eval(s)
            altsd = sd["sentences"][0]['words']
            for lis in altsd:
                try:
                    defn = wn.synset(list[0]+'.n.01').definition().lower()
                    if any(word in defn for word in perList):
                        lis[1]['NamedEntityTag'] = 'PERSON'
                except:
                    pass
            sent_dic[sent] = {"parsetree": sd["sentences"][0]['parsetree'], "tags": altsd}
        stories[storyID] = {"head": head, "sentences" :sent_dic}
        
    with open(dirpath+storyID+".questions") as f:
        while True:
            quesId = re.findall('^QuestionID: ?(.*)', f.readline())
            ques = re.findall('^Question: ?(.*)', f.readline())
            diff = f.readline()
            line4 = f.readline()
            if len(ques) != 0 and len(quesId) !=0:
                ques = ques[0].replace("?", "")
                #ques = removeStopWords(ques[0].replace("?", ""))
                quesId = quesId[0]
                quesIdList.append(quesId)
                p = corenlp.parse(ques)
                pd = ast.literal_eval(p)
                quesDic[quesId] = {"storyID" : storyID, "Question" : ques, "tags": pd['sentences'][0]['words'], 'parsetree': pd['sentences'][0]['parsetree']}
            if not line4: break
        
    return quesDic, stories  
    
inputIdList = []
if __name__ == '__main__':
    with open(sys.argv[1]) as f:
        dirPath = f.readline().rstrip()
        for line in f:
            inputIdList.append(line.rstrip().split(".")[0])

            
    #os.chdir(dirPath.rstrip())
    
    allQues, allStories, answerDic, ansSenDic, ansBestwa = {}, {}, {}, {}, {}
    #allQues, allStories = quesExtractor(dirPath+"/", "1999-W09-5".rstrip())
    for sid in inputIdList:
        allQues, allStories = quesExtractor(dirPath+"/", sid.rstrip())
    count = 0
    for key, value in allQues.iteritems():
        count+=1
        scoredSentDic = scoreSent(value["Question"], allStories[value["storyID"]]["sentences"], value["tags"], value["parsetree"], allStories[value["storyID"]]["head"])
        answer, senten = findAnswers(scoredSentDic, value["Question"], allStories[value["storyID"]]["sentences"], allStories[value["storyID"]]["head"])
        answerDic[key] = answer
        ansSenDic[key] = senten
            
    with open("response_file.txt", 'w') as rf:
        for key in quesIdList:
            rf.write("QuestionID: " + key)
            rf.write("\n")
            rf.write("Answer: " + ansSenDic[key])
            rf.write("\n")
            rf.write("\n") 
