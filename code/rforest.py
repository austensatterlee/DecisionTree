import numpy as np
import heapq
import collections

def build_forest(training_features,training_labels,N=10,bagsize=1,verbose=False,**kwargs):
    """
    Grow many trees.

    training_features - data to train from
    training_labels - labels for training_features
    N - number of trees to grow
    bagsize - number of random data points to train each tree on

    Other keyword arguments are passed to build_tree.

    """
    n,d=training_features.shape
    forest=[]
    N=int(N)
    for i in xrange(N):
        junk=True
        while junk:
            print "Growing tree",i,"!"
            bag = np.random.choice(np.arange(n),size=int(n*bagsize),replace=True)
            tree = build_tree(training_features[bag],training_labels[bag],verbose=verbose,**kwargs)
            if treegain>0:
                forest.append(tree)
                print "Tree",i,"has grown!"
                junk=False
            else:
                print "Tree",i,"is junk yo"
                print treegain,tree
    return forest

def build_tree(training_features,training_labels,**kwargs):
    """
    Grow a tree.

    The data structure for the decision tree is recursively defined as:
    (
     dimension,
     threshold,
     left subtree,
     right subtree
    )

    A simplification of the decision rule defined by a tree is: given a
    data point X, if X[dimension]>threshold, then the right subtree should
    be used. Otherwise, the left subtree should be used.

    A leaf in the tree is a label for the data point.

    training_features   Data to train on.
    training_labels     Labels for training_features.
    MAXDEPTH            The maximum depth (roughly the number of
                        decisions) the tree will be allowed to grow to.
    MINGOODNESS         The minimum tolerable "goodness" of each decision
                        in order for the tree to add it.
    K                   Maximum number of feature dimensions to consider when determining
                        which should be used for the next decision.

    """
    depth = kwargs.pop('depth',0)
    MAXDEPTH = kwargs['MAXDEPTH'] = kwargs.get('MAXDEPTH',2**training_features.shape[1])
    MINGOODNESS = kwargs['MINGOODNESS'] = kwargs.get('MINGOODNESS',-np.inf)
    K = kwargs['K'] = kwargs.get('K',8)
    verbose = kwargs['verbose'] = kwargs.get('verbose',False)

    if depth>=MAXDEPTH or training_labels.size<=5:
        leaf = most_frequent(training_labels)
        if depth>=MAXDEPTH:
            print "Leaf {:25} label: {}".format("(depth lim.)",leaf)
        else:
            print "Leaf {:25} label: {}".format("(too few data: {} pts)".format(training_labels.size),leaf)
        return leaf

    n,d = np.shape(training_features)

    #k=2*int(np.ceil(np.sqrt(d+1)))
    k=max(1,min(K,d))

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
        leaf = most_frequent(training_labels)
        print "Leaf (goodness lim.) - label: {}".format(leaf)
        return leaf
    if verbose:
        print ("{:<25}"*4).format("Depth: "+str(depth), "Dim: "+str(bestCandidate), "Best Goodness: {:.6f}".format(bestGoodness),"Threshold: {}".format(bestThreshold))
    leftTree = build_tree(training_features[bestPartition[0]],training_labels[bestPartition[0]],depth=depth+1,**kwargs)
    rightTree = build_tree(training_features[bestPartition[1]],training_labels[bestPartition[1]],depth=depth+1,**kwargs)
    return (bestCandidate,bestThreshold,leftTree,rightTree)


def getBestThreshold(features,labels,feature_ind):
    """
    Choose a plane lying on the specified dimension of the feature space
    such that the 'goodness' of the partition of the given data points is
    maximized.

    features - the points on the feature space to be partitioned
    labels - the labels corresponding to each feature
    feature_ind - the dimension on which to partition

    Given a plane which divides the set of features `F` into two
    partitions `F_l` and `F_r`, 'goodness' is defined as the difference
    between the entropy of `F` and the sum of the entropies of
    `F_l` and `F_r`.

    `G = H(F) - [Pr(F_l) H(F_l) - ( 1 - Pr(F_l) ) H(F_r)]`

    Here, `Pr(F_l)` represents the probability of
    randomly selecting an element from the left set.

    ------------------------------

    Returns a tuple (X,Y,Z), where
    X - goodness of the partition
    Y - location of the dividing plane
    Z - (indices of features in the left partition,
        indices of features in the right partition)

    """
    n,d = np.shape(features)
    total_entropy = get_entropy(labels)

    # Sort the data points by along the specified dimension
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

    # Choose the threshold (location of the dividing plane) that maximizes
    # goodness
    leftgroup = []
    bestGoodness,bestThreshold,bestIndex = 0,None,None
    i=0
    lastval=0
    while(len(sortedFeatures)>0):
        currval,currlabel = sortedFeatures.popleft(),sortedLabels.popleft()
        if (i==n-1 or currval!=sortedFeatures[0]):
            rightgroup = sortedLabels
            t = (lastval+currval)/2.0
            if t!=0:
                leftentropy = get_entropy(leftgroup)
                rightentropy = get_entropy(rightgroup)
                goodness = total_entropy-(len(leftgroup)/float(n)*leftentropy+(n-len(leftgroup))/float(n)*rightentropy)
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
    """
    compute the entropy (uncertainty) of a data set

    """
    # Compute the distribution of the samples
    N=float(len(samples))
    values = {}
    for s in samples:
        values[s]=values.get(s,0)+1

    # entropy = sum_{i} p_i log(1/p_i)
    entropy=0
    for value in values:
        prob = values[value]/N
        entropy=entropy-prob*np.log2(prob)
    return entropy

def most_frequent(samples):
    """
    compute the mode of a data set

    """
    counter={}
    maxCount,maxElement = 0,0
    for s in samples:
        counter[s]=counter.get(s,0)+1
        if counter[s]>maxCount:
            maxCount=counter[s]
            maxElement=s
    return maxElement

def treeclassify(tree,features):
    """
    Use a tree to classify a data point

    """
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

def forestclassify(forest,features,verbose=False,forestweights=None):
    """
    Use a forest to classify a data point

    forest - the forest to use
    features - the data point
    forestweights - optionally provide a list of weights to prioritize the
    votes cast by each participating tree.

    """
    if forestweights==None:
        forestweights = np.ones(len(forest))
    votes={}
    maxVotes,maxLabel = 0,None
    winningGain = 0.0
    for i,tree in enumerate(forest):
        label=treeclassify(tree,features)
        votes[label]=votes.get(label,0)+1
        if votes[label]>maxVotes:
            maxVotes = votes[label]
            maxLabel = label
            winningGain = forestweights[i]
    for label in votes:
        if votes[label]==maxVotes and label!=maxLabel:
            if forestweights[i]*votes[label]>winningGain*maxVotes:
                maxVotes=votes[label]
                maxLabel=label
                winningGain=forestweights[i]
    if verbose:
        return maxLabel,votes
    else:
        return maxLabel

def validation_accuracy(forest,evf,evl,verbose=False,forestweights=None):
    """

    Arguments:
    forest - a forest or a single tree
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
        forestguess = forestclassify(forest,feature,forestweights=forestweights,verbose=verbose)
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

def print_tree(tree):
    def _print_tree(tree,depth,prefixchar):
        padchar=' -'
        if isinstance(tree,collections.Container):
            dim,thresh,left,right = tree
            treestr = "({}{}){} Dim. {} > {}\n".format(prefixchar,depth,padchar*depth,dim,thresh)
            return treestr + _print_tree(right,depth+1,'>') + _print_tree(left,depth+1,'<')
        else:
            leafstr = "({}{}){} Label: {}\n".format(prefixchar,depth,padchar*depth,tree)
            return leafstr
    print _print_tree(tree,depth=1,prefixchar='|')

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
