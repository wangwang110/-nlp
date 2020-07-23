import ESCmain
import LangModel
import nltk
class LangModel:
    def __init__(self, order, alpha, sentences):
        self.index = 0
        self.order = order
        self.alpha = alpha
        if order > 1:
            self.backoff = LangModel(order - 1, alpha, sentences)
            self.lexicon = None
        else:
            self.backoff = None
            self.n = 0

        self.ngramFD = nltk.FreqDist()
        lexicon = set()
        # set NUll
        for sentence in sentences:
            self.index += 1
            words = sentence.split(" ")
            wordNGrams = nltk.ngrams(words, order)
            for wordNGram in wordNGrams:
                self.ngramFD[wordNGram] += 1
                if order == 1 and wordNGram[0] != "<s>" and wordNGram[0] != "</s>":
                    lexicon.add(wordNGram)
                    self.n += 1
        self.v = len(lexicon)

    def logprob(self, ngram):
        t = self.prob(ngram)
        print (t)
        #if (t<=0):
        #    t=0.00000000000000001
        return math.log(t)

    def prob(self, ngram):
        current = self
        print len(ngram)
        while current.order > len(ngram):
            current = current.backoff

        # current.order==len(ngram)

        if current.backoff != None:
            print "111"
            freq = current.ngramFD[ngram]
            backoffFreq = current.backoff.ngramFD[ngram[:-1]]
            if freq == 0:
                print "1111"
                if len(ngram) > 1:
                    backprob = current.backoff.prob(ngram[1:])
                    return current.alpha * backprob
                else:
                    backprob = current.backoff.prob(ngram)
                    return current.alpha * backprob

            else:
                print "1112"
                if backoffFreq > 0:
                    return freq / backoffFreq
                else:
                    return freq / current.n

        else:
            # laplace smoothing to handle unknown unigrams
            # print current.n, "-=-=-=-=-"
            print"222"
            k = float(float(current.ngramFD[ngram] + 1) / float(current.n + current.v))
            return k
print ESCmain.process_txt('''Love is everywhere. We have love from our parents and friends, which makes us become stronger. Without love, we can't survive long, or we will just like the walking dead. So many great people have owed their success to the love from families, but the love for nature always forgotten by the public.
Love from families and frieds are praised in so many works. Like the great poets, they wrote so many famous works to give applause to the people who suport them all the time. In the movies and TV shows, family love and friendship occupy most themes. They shows the great power of human being. With love, we can conquered all the difficulties.
But love for nature is not often mentioned by the public. We love the beautiful scenery and the lovely animals, so we have the desire to protect the enviroment, for the purpose of keeping clean scenery and make sure the animals won't disappear. Animals and human being are part of nature. We live and die together.
Love is not just around families and friends, but we also need to care for nature.''')