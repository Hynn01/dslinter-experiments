#!/usr/bin/env python
# coding: utf-8

# This notebook is based on below discussion:
# https://www.kaggle.com/competitions/birdclef-2022/discussion/321883#1772162

# In[ ]:


get_ipython().system('pip install nb_black > /dev/null')


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("ggplot")

get_ipython().run_line_magic('load_ext', 'lab_black')


# # Evaluation metrics

# ## Balanced Accuracy
# 
# https://scikit-learn.org/stable/modules/model_evaluation.html#balanced-accuracy-score

# In[ ]:


def balanced_accuracy(pred, target, eps=1e-6):
    tp = (pred * target).sum(axis=-1)
    fn = ((1 - pred) * target).sum(axis=-1)
    fp = (pred * (1 - target)).sum(axis=-1)
    tn = ((1 - pred) * (1 - target)).sum(axis=-1)
    tpr = tp / (tp + fn + eps)
    tnr = tn / (tn + fp + eps)
    return 0.5 * (tpr + tnr)


# ## Mean F1 and inverted F1 (MFIF)
# 
# This metrics is suggested by [Enrique Gurdiel](https://www.kaggle.com/gurdiel) in [this post](https://www.kaggle.com/competitions/birdclef-2022/discussion/321883#1772071).
# 
# This score calculates the mean of F1 and F1 for inverted both prediction and target.
# We call this score MFIF for simplicity.
# 
# ```
# MFIF Score = 0.5 * (F1(pred, target) + F1(1 - pred, 1 - target))
# ```

# In[ ]:


def f1(pred, target, eps=1e-6):
    tp = (pred * target).sum(axis=-1)
    fn = ((1 - pred) * target).sum(axis=-1)
    fp = (pred * (1 - target)).sum(axis=-1)
    tn = ((1 - pred) * (1 - target)).sum(axis=-1)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    return 2 * precision * recall / (precision + recall + eps)


def mean_f1_and_inv_f1(pred, target):
    return 0.5 * (f1(pred, target) + f1(1 - pred, 1 - target))


# # Numerical Simulation
# 
# ## Condistion of simulation
# 
# * Here, there are 10,560 samples (which is identical to the number of samples of public test data) with negative and positive probability is $p_\text{pos}$ and $p_\text{neg}$ respectively.
# * The probability of correct prediction of the simulated model is constant: the probability of correct answer is $p_\text{correct}$ for both positive and negative targets.

# ## Prepare target and simulated model prediction

# ## Experiment

# In[ ]:


class TargetGenerator:
    def __init__(self, p_pos=0.1):
        self.p_pos = p_pos

    def generate(self, n_samples, N):
        return np.random.choice(2, (N, n_samples), p=[1 - self.p_pos, self.p_pos])


class PredictorGenerator:
    def __init__(self, p_correct):
        self.p_correct = p_correct

    def generate(self, n_samples, target, N):
        base = np.random.choice(
            2, (N, n_samples), p=[1 - self.p_correct, self.p_correct]
        )
        pred = target * base + (1 - target) * (1 - base)
        return pred


# In[ ]:


def simulation(
    target_generator,
    predictor_generator,
    n_samples=10_560,
    N=1_000,
    n_disp=50,
    n_bins=30,
    stat="percent",
):
    """
    n_samples = 5500 * 12 * 0.16 = 10560
    """
    print("=" * 14 + " Experiment " + "=" * 14)
    print(f"- {target_generator.p_pos} of positive samples")
    if hasattr(predictor_generator, "p_correct"):
        print(f"- {predictor_generator.p_correct} of correct prediction")
    print(f"- {n_samples} samples")
    print(f"- {N} trials")
    print("=" * 40)

    target = target_generator.generate(n_samples, N)
    pred = predictor_generator.generate(n_samples, target, N)

    print(f"target (first {n_disp} samples): {target[0, :n_disp].tolist()}")
    print("")
    print(f"prediction (first {n_disp} samples): {pred[0, :n_disp].tolist()}")
    print("")
    print("** Calculated Score **")
    ba = balanced_accuracy(pred, target)
    mfif = mean_f1_and_inv_f1(pred, target)
    f1_ = f1(pred, target)

    print(f"F1: {f1_.mean(axis=0):.4f} (std={f1_.std(axis=0):.4f})")
    print(f"MFIF: {mfif.mean(axis=0):.4f} (std={mfif.std(axis=0):.4f})")
    print(f"balanced accuracy: {ba.mean(axis=0):.4f} (std={ba.std(axis=0):.4f})")

    _, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(f1_, label="F1", color="blue", ax=ax, stat=stat, bins=n_bins)
    sns.histplot(mfif, label="MFIF", color="green", ax=ax, stat=stat, bins=n_bins)
    sns.histplot(
        ba, label="balanced accuracy", color="red", ax=ax, stat=stat, bins=n_bins
    )
    ax.set(title=f"Distribution of calculated score (N={N})", xlabel="Score")
    ax.legend()
    plt.show()


# In[ ]:


target_gen = TargetGenerator()
pred_gen = PredictorGenerator(0.7)
simulation(target_gen, pred_gen)


# In[ ]:


target_gen = TargetGenerator()
pred_gen = PredictorGenerator(0.8)
simulation(target_gen, pred_gen)


# In[ ]:


target_gen = TargetGenerator(p_pos=0.01)
pred_gen = PredictorGenerator(0.7)
simulation(target_gen, pred_gen)


# In[ ]:


target_gen = TargetGenerator(p_pos=0.001)
pred_gen = PredictorGenerator(0.7)
simulation(target_gen, pred_gen)


# ## Discussion: features of balanced accuracy
# 
# ### Tolerance for False Positives
# 
# If there is a class imbalance where there are fewer positive examples than negative examples, balanced accuracy will be more tolerant of false positives. For example, if there are 100 samples, 10 positive examples and 90 negative examples, 10 false positives and 1 false negative will have the same impact on the score.
# 
# ### Invariant with respect to the ratio of positive examples
# 
# If the model's probability of correct prediction is constant, the balanced accuracy is constant regardless of the proportion of positive examples. This makes it difficult to estimate the proportion of positive and negative cases in the private test data by LB probing.
# 
# On the other hand, the smaller the proportion of positive examples, the larger the variance of the balanced accuracy score. Since balanced accuracy treats the percentage of positive and negative examples equally, the smaller the number of positive examples, the greater the impact on the score when one positive example is answered correctly or incorrectly.

# # Conclusion
# 
# * balanced accuracy is more tolerant of false positives than mfif. This explains our observation well. 

# # Appendix: Simulations of naive prediction

# ## A1: All true prediction

# In[ ]:


class PositivePredictorGenerator:
    def generate(self, n_samples, target, N):
        return np.ones((N, n_samples))


target_gen = TargetGenerator()
pred_gen = PositivePredictorGenerator()
simulation(target_gen, pred_gen, N=1_000)


# ## A2. all zero prediction

# In[ ]:


class FalsePredictorGenerator:
    def generate(self, n_samples, target, N):
        return np.zeros((N, n_samples))


target_gen = TargetGenerator()
pred_gen = FalsePredictorGenerator()
simulation(target_gen, pred_gen, N=1_000)


# ## A3. Random prediction

# In[ ]:


class RandomPredictorGenerator:
    def generate(self, n_samples, target, N):
        return np.random.choice(2, (N, n_samples))


target_gen = TargetGenerator()
pred_gen = RandomPredictorGenerator()
simulation(target_gen, pred_gen)


# ## Conclusion
# 
# When the evaluation metric is balanced accuracy, these simulation results fit well with observations from naive submissions.
