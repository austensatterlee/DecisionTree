import numpy as np
import heapq
def knn_classify(samplefeatures,trainingfeatures,traininglabels,k=1):
    """

    Arguments:
    sampledatum - feature vector (size d)
    trainingdata - list of (feature vector,label) tuples

    """
    neighbors=[]
    for i in xrange(len(trainingfeatures)):
        candidatefeatures=trainingfeatures[i]
        candidatelabel=traininglabels[i]
        featurediff=samplefeatures-candidatefeatures
        distance=np.sqrt(np.dot(featurediff,featurediff))
        heapq.heappush(neighbors,(distance,candidatelabel))
    votesdict={}
    winningvote=None
    for i in xrange(k):
        vote=heapq.heappop(neighbors)[1]
        votesdict[vote]=votesdict.get(vote,0)+1
        if winningvote==None or votesdict[vote]>votesdict[winningvote]:
            winningvote=vote
    return winningvote

def validation_accuracy(valf,vall,trf,trl,k=1,verbose=False):
    """

    Arguments:
    valf - validation features
    vall - validation labels
    trf - training features
    trl - training labels

    """
    numcorrect=0.0
    incorrect=[]
    for i in xrange(len(valf)):
        features=valf[i]
        label=vall[i]
        knn_label = knn_classify(features,trf,trl,k)
        if knn_label==label:
            numcorrect+=1
        elif verbose:
            incorrect.append((i,label,knn_label))
    acc = numcorrect/len(valf)
    if verbose:
        return acc,incorrect
    else:
        return acc

if __name__=="__main__":
    import hw12loader as hwl
    import sys
    dtf,dtl=hwl.load_digit_training_data()
    dvf,dvl=hwl.load_digit_validation_data()
    K=sys.argv[1:]
    for k in K:
        guessedlabels=[]
        for vfeature in dvf:
            label=knn_classify(vfeature,dtf,dtl,int(k))
            guessedlabels.append(label)
        fp=open('digitsOutput'+str(k)+'.csv','w')
        fp.write('\n'.join(map(str,guessedlabels)))
        fp.close()
