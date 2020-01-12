import numpy as np
import math

def sample(logits):
    noise = np.random.gumbel(size=len(logits))
    sample = np.log(logits) + noise
    return sample


if __name__=="__main__":
    n_cats = 7
    cats = np.arange(n_cats)

    probs = np.random.randint(low=1, high=20, size=n_cats)
    probs = probs / sum(probs)
    logits = [0.3,0.4,0.5,0.6,0.2,0.5,0.8]

    samples = sample(logits)

    log_1 = []
    sum = 0
    for i in samples:
        a = math.exp(i)
        log_1.append(a)
        sum += a


    out = [log_1[i] / sum for i in range(n_cats)]
    he = 0
    for i in range(n_cats):
        he += out[i]*i
    #np.array([n_cats,])
    print(out)
    print(he)
