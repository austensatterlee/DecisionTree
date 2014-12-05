import numpy as np
import heapq
import collections

def build_forest(training_features,training_labels,N=10,bagsize=1,verbose=False,**kwargs):
    n,d=training_features.shape
    forest=[]
    forestgains=[]
    N=int(N)
    for i in xrange(N):
        print "Growing tree ",i,"!"
        bag = np.random.choice(np.arange(n),size=int(n*bagsize),replace=True)
        treegain,tree = build_tree(training_features[bag],training_labels[bag],**kwargs)
        if treegain>0:
            forest.append(tree)
            forestgains.append(treegain)
        print "Tree ",i," has grown!"
        if len(forest):
            print forestgains
    if verbose:
        return forest,forestgains
    else:
        return forest

def build_tree(training_features,training_labels,depth=1,MAXDEPTH=np.inf,MINGOODNESS=0.0,K=8):
    if depth>=MAXDEPTH or training_labels.size<=1:
        return 0,most_frequent(training_labels)

    n,d = np.shape(training_features)

    k=max(0,min(K,d))#2*int(np.ceil(np.sqrt(d))) #####:(

    candidates = np.random.choice(np.arange(d),size=k)
    bestCandidate,bestPartition,bestThreshold,bestGoodness = 0,0,0,0
    for candidate in candidates:
        goodness,threshold,partition = getBestThreshold(training_features,training_labels,candidate)
        if goodness>=bestGoodness:
            bestGoodness = goodness
            bestCandidate = candidate
            bestThreshold = threshold
            bestPartition = partition
    if bestGoodness<MINGOODNESS or bestThreshold==0:
        return 0,most_frequent(training_labels)
    leftgain,leftTree = build_tree(training_features[bestPartition[0]],training_labels[bestPartition[0]],depth=depth+1,)
    rightgain,rightTree = build_tree(training_features[bestPartition[1]],training_labels[bestPartition[1]],depth=depth+1,)
    print ("{:<25}"*5).format("Depth: "+str(depth), "Attr: "+str(bestCandidate), "Best Goodness: {:.6f}".format(bestGoodness),"Tree Goodness: {:.6f}".format(bestGoodness+leftgain+rightgain),"Threshold: {}".format(bestThreshold))
    return bestGoodness+leftgain+rightgain,(bestCandidate,bestThreshold,leftTree,rightTree)


def getBestThreshold(features,labels,feature_ind):
    n,d = np.shape(features)
    total_entropy = get_entropy(labels)
    sortedValues=[]
    for j in xrange(n):
        fval = features[j,feature_ind]
        flabel = labels[j]
        heapq.heappush(sortedValues,(fval,flabel,j))

    sortedFeatures=collections.deque([])
    sortedLabels=collections.deque([])
    sortedIndices=[]
    while len(sortedValues)>0:
        feature,label,index=heapq.heappop(sortedValues)
        sortedFeatures.append(feature)
        sortedLabels.append(label)
        sortedIndices.append(index)

    leftgroup = []
    bestGoodness,bestThreshold,bestIndex = 0,None,None
    i=0
    lastval=0
    while(len(sortedFeatures)>0):
        currval,currlabel = sortedFeatures.popleft(),sortedLabels.popleft()
        if (i==n-1 or currval!=sortedFeatures[0]):
            rightgroup = sortedLabels
            t = (lastval+currval)/2.0
            leftentropy = get_entropy(leftgroup)
            rightentropy = get_entropy(rightgroup)
            goodness = total_entropy-len(leftgroup)/float(n)*leftentropy-(n-len(leftgroup))/float(n)*rightentropy
            if goodness>bestGoodness or bestThreshold==None:
                bestGoodness = goodness
                bestThreshold = t
                bestIndex = i
        leftgroup.append(currlabel)
        lastval=currval
        i+=1
    rightindices = sortedIndices[bestIndex:]
    leftindices = sortedIndices[:bestIndex]
    return bestGoodness,bestThreshold,(leftindices,rightindices)

def get_entropy(samples):
    N=float(len(samples))
    values = {}
    for s in samples:
        values[s]=values.get(s,0)+1

    entropy=0
    for value in values:
        prob = values[value]/N
        entropy=entropy-prob*np.log2(prob)
    return entropy

def most_frequent(samples):
    counter={}
    maxCount,maxElement = 0,0
    for s in samples:
        counter[s]=counter.get(s,0)+1
        if counter[s]>maxCount:
            maxCount=counter[s]
            maxElement=s
    return maxElement

def treeclassify(tree,features):
    curr_node = tree
    while True:
        if not isinstance(curr_node,collections.Container):
            return curr_node
        attribute = curr_node[0]
        threshold = curr_node[1]
        left,right = curr_node[2:]
        if features[attribute]<=threshold:
            curr_node = left
        else:
            curr_node = right
    return curr_node

def forestclassify(forest,features,verbose=False,forestgains=None):
    if forestgains==None:
        forestgains = np.ones(len(forest))
    votes={}
    maxVotes,maxLabel = 0,0
    winningGain = 0.0
    for i,tree in enumerate(forest):
        label=treeclassify(tree,features)
        votes[label]=votes.get(label,0)+forestgains[i]
        if votes[label]>maxVotes:
            maxVotes = votes[label]
            maxLabel = label
            winningGain = forestgains[i]
        elif votes[label]==maxVotes:
            if forestgains[i]*votes[label]>winningGain*maxVotes:
                maxVotes=votes[label]
                maxLabel=label
                winningGain=forestgains[i]
    if verbose:
        return maxLabel,votes
    else:
        return maxLabel

def validation_accuracy(forest,evf,evl,verbose=False,forestgains=None):
    """

    Arguments:
    forest - could be trees
    evf - validation features
    evl - validation labels

    """
    if not isinstance(forest[0],collections.Container):
        forest = [forest]
    numcorrect=0.0
    incorrect=[]
    allvotes=[]
    for i in xrange(len(evf)):
        feature,label = evf[i],evl[i]
        forestguess = forestclassify(forest,feature,forestgains=forestgains,verbose=verbose)
        if isinstance(forestguess,collections.Container):
            forestguess,votes = forestguess
            allvotes.append(votes)
        if label == forestguess:
            numcorrect+=1
        else:
            incorrect.append((feature,label))
    acc = numcorrect/len(evf)
    if verbose:
        return acc,incorrect,allvotes
    else:
        return acc

if __name__=="__main__":
    import sys
    import hw12loader as hwl
    etf,etl=hwl.load_email_training_data()
    evf,evl=hwl.load_email_validation_data()
    T=sys.argv[1:]
    for t in T:
        forest=build_forest(etf,etl,t)
        guessedlabels=[]
        for vfeature in evf:
            label=forestclassify(vfeature,etf)
            guessedlabels.append(label)
        fp=open('emailOutput'+str(t)+'.csv','w')
        fp.write('\n'.join(map(str,guessedlabels)))
        fp.close()
