import sys
import cPickle as pickle
import DSSM_blstm_neg
import numpy as np 


model = DSSM_blstm_neg.DSSM_BLSTM_Model()

print "model construction..."
model.model_constructor(wemb_matrix_path='../../data/pickles/index_wemb_matrix.pkl')
print "construction complete"

print "loading parameters..."
model.reload_model("val")

word2index_dict = pickle.load(open('../../data/pickles/ROC_train_vocab_dict.pkl','r'))

def convert2index(sent):
    '''
    parameters: 
    -----------
    sent ==> type: list of strings(words)

    return:
    -----------
    tokens ==> type: list of ints(index of word in dictionary)
    unknown_words_ls ==> type: list of strings(unknown words)
    '''
    tokens = []
    unknown_words_ls = []
    for word in sent:
        if word in word2index_dict:
            tokens.append(word2index_dict[word])
        else:
            tokens.append(word2index_dict['UUUNKNOWNNN'])
            unknown_words_ls.append(word)
    return tokens, unknown_words_ls



while 1:

    print "please enter the story you have"
    print "please use space as delimitor :) "
    try:
        story = sys.stdin.readline()
        story_words = [word.lower() for word in story.split()]
        story_tokens, unknowns = convert2index(story_words)
        print "unknown words in story: ", " ".join(unknowns)

    except KeyboardInterrupt:
        print "Ended by user"
        break

    print "please enter the first end you have"
    try:
        end1 = sys.stdin.readline()
        end1_words = [word.lower() for word in end1.split()]
        end1_tokens, unknowns = convert2index(end1_words)
        print "unknown words in end1: ", " ".join(unknowns)

    except KeyboardInterrupt:
        print "Ended by user"
        break

    print "please enter the second end you have"
    try:
        end2 = sys.stdin.readline()
        end2_words = [word.lower() for word in end2.split()]
        end2_tokens, unknowns = convert2index(end2_words)
        print "unknown words in end2: ", " ".join(unknowns)

    except KeyboardInterrupt:
        print "Ended by user"
        break

    story_input = np.asarray(story_tokens, dtype='int64').reshape((1,-1))
    story_mask = np.ones((1,len(story_tokens)))

    ending1 = np.asarray(end1_tokens, dtype='int64').reshape((1,-1))
    ending1_mask = np.ones((1,len(end1_tokens)))

    ending2 = np.asarray(end2_tokens, dtype='int64').reshape((1,-1))
    ending2_mask = np.ones((1, len(end2_tokens)))

    cos1 = model.compute_cost(story_input, story_mask, ending1, ending1_mask)
    cos2 = model.compute_cost(story_input, story_mask, ending2, ending2_mask)

    # Answer denotes the index of the anwer
    print "reasoning probability of these two ending with the story are:"
    print cos1
    print cos2

