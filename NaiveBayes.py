#NaiveBayes.py
from sys import argv
from random import randint
from math import *

#1. Read the titanic.dat file
#input: the filename (full path)
#output: people: list[diary{“attribute”:list[3],”label”:int}]
#        pclass: list of the attribute, class
#        age:    list of the attribute, age
#        sex:    list of the attribute, sex
def readDat(filename):
    f = open(filename,'r')
    lines = f.readlines()
    pclass,age,sex = [],[],[]
    people = []
    for i in lines:
        #ignore the character @
        if i[0] != '@':
            values = i.split(',')
            #store the attributes separatedly
            pclass.append(float(values[0]))
            age.append(float(values[1]))
            sex.append(float(values[2]))
            attributes = list(map(float,values[:-1]))
            label = float(values[-1][:-1]) #remove the \n
            person = {"attribute":attributes,"label":label}
            people.append(person)
    return people,pclass,age,sex
    #print(people)
    #print(len(people))

#Entropy calculator
#input: list[num,num,...]  (num is the number of the label, the size of the list is the label count)
#output: float Entropy
def CalEntropy(lst):
    num = len(lst)
    Entropy = 0
    for i in lst:
        Entropy+=i/num*log(i/num,2)
    return Entropy

#Using Entropy to calculate the optimal threshold of dividing the values of a specific attribute into
#two spaces, left and right, which are below or above the threshold.
#input: undeplicate list of the values of one attribute,list udup_lst; dataset, list people; the index
#of the attribute in diary{“attribute”:list[3],”label”:int}.
# #output: the threshold, float Threshold.  
def CalThreByEntropy(udup_lst,people,index):
    Threshold = 0
    LeftItem, RightItem = [],[]
    MaxEntropy = 10000
    #首先扫描people，根据udup_lst分裂people
    for i in range(len(udup_lst)-1):
        Thres = 1/2*(udup_lst[i]+udup_lst[i+1])
        TotalNum = len(people)
        for j in range(TotalNum):
            if people[j]["attribute"][index] <= Thres:
                LeftItem.append(people[j])
            else:
                RightItem.append(people[j])
        LeftPosit = 0
        RightPosit = 0
        for j in range(len(LeftItem)):
            if LeftItem[j]["label"] == 1.0:
                LeftPosit+=1
        for j in range(len(RightItem)):
            if RightItem[j]["label"] == 1.0:
                RightPosit+=1
        Numlist1 = [LeftPosit,len(LeftItem)-LeftPosit]
        Numlist2 = [RightPosit,len(RightItem)-RightPosit]
        TotalEntropy = len(LeftItem)/TotalNum*CalEntropy(Numlist1)+len(RightItem)/TotalNum*CalEntropy(Numlist2)
        if TotalEntropy < MaxEntropy:
            MaxEntropy = TotalEntropy
            Threshold = Thres
    return Threshold
    

#2. Use information gain to discretize the data, converting values of each attribute into boolean values.
#Prerequisite: CalThreByEntropy
#input: dataset, list people; Attributes, list[list pclass, list age, list sex]
#output: Discreted people; thresholds of the attributes.
def DiscretData(people,Attributes):
    pclass,age,sex = Attributes
    #unduplicate the attribute lists
    udup_pclass = sorted(list(set(pclass)))
    udup_age = sorted(list(set(age)))
    udup_sex = sorted(list(set(sex)))
    #calculate the threshold to devide dataset using entropy.
    Threpclass,Threage,Thresex = 0,0,0
    Threpclass = CalThreByEntropy(udup_pclass,people,0)
    Threage = CalThreByEntropy(udup_age,people,1)
    Thresex = CalThreByEntropy(udup_sex,people,2)
    #print(Threpclass,Threage,Thresex)
    #Discret the attributes according to entropy.
    for i in people:
        if i["attribute"][0] <= Threpclass:
            i["attribute"][0] = 0
        else:
            i["attribute"][0] = 1
        if i["attribute"][1] <= Threage:
            i["attribute"][1] = 0
        else:
            i["attribute"][1] = 1
        if i["attribute"][2] <= Thresex:
            i["attribute"][2] = 0
        else:
            i["attribute"][2] = 1
    return people,[Threpclass,Threage,Thresex]

#3. Divide the data into two sets, with the ratio of 7:3
#people: the standardlised data structure
#ratio: ratio of training data and test data, from 0 to 1
#output:traindata; testdata
def DataDevide(people,ratio):
    randomintlist = []
    traindata = []
    testdata = []
    for i in range(int(floor(len(people)*ratio))):
        index = randint(0,len(people)-1)
        while index in randomintlist:
            index = randint(0,len(people)-1)
        randomintlist.append(index)
        traindata.append(people[index])
    for i in range(len(people)-1):
        if i not in randomintlist:
            testdata.append(people[i])
    return traindata,testdata
    #print(len(traindata))
    #print(len(testdata))

#4. Naive Bayes algorithm
#output: the predicted label $ possibility
#input: traindata, case (case should have been normailised)
def NaiveBayes(people,case):
    AttrNum = len(people[0]["attribute"])
    TotalNum = len(people)
    LabelValueLst = []
    attributes = case["attribute"]
    for label in (1.0,-1.0):
        AttrValue = 1
        Labellst = []
        Labelnum = 0
        for i in range(TotalNum):
            if people[i]["label"] == label:
                Labellst.append(people[i])
                Labelnum+=1
        #print(Labelnum)
        for i in range(AttrNum):
            MatchNum = 0
            for j in range(len(Labellst)):
                if Labellst[j]["attribute"][i] == attributes[i]:
                    MatchNum+=1
            AttrValue*=MatchNum/len(Labellst)
        LabelValue = AttrValue*Labelnum/TotalNum
        #print(LabelValue)
        LabelValueLst.append(LabelValue)
    if LabelValueLst[0] > LabelValueLst[1]:
        return 1,LabelValueLst
    else:
        return -1,LabelValueLst

#5. Test the ratio of the successfully predicted cases in testdata
def ClassTest(traindata,testdata):
    n = len(testdata)
    result = []
    for case in testdata:
        test = NaiveBayes(traindata,case)
        label = case["label"]
        temp = {"test":test,"label":label}
        result.append(temp)
    #print(result)
    num = 0
    for i in result:
        if i["test"][0] == i["label"]:
            num+=1
    ratio = num/n
    print("%.4f" %ratio)       

def main(argv):
    if len(argv) == 1:
        print("NaiveBayes.py: A Naive Bayes algorithm program to predict the survival of a person in the titanic incident.\n")
        print("USage:\npython NaiveBayes.py filepath [DataDivideRatio=0.7]\n")
        print("filepath\n\tthe full path of the .dat or .csv file, relative or absolute all both accepted.")
        print("k\n\tthe number of nearest neighbor, which has to be set by the user.")
    elif len(argv) > 1:
        filename = argv[1]
        if len(argv) == 2:
            ratio = 0.7
        else:
            ratio = float(argv[2])

        people,pclass,age,sex = readDat(filename)
        Attributes = [pclass,age,sex]
        DiscretPeople,thresholds = DiscretData(people,Attributes)
        traindata,testdata = DataDevide(people,0.7)
        #print(len(traindata),len(testdata),len(people))
        ClassTest(traindata,testdata)

main(argv)