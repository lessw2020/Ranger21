# Ranger21
Ranger deep learning optimizer rewrite to use newest components 

Ranger with Radam + Lookahead core is now 1.5 years old.  In the interim, a number of new developments have happened including the rise of Transformers for Vision.

Thus, Ranger21 (as in 2021) is a rewrite with the following components planned to be added.  The idea is that these internals will be paramaterized so that you can mix and match to optimize the optimizer on your respective dataset. 

Current feature set planned:
1 - Linear and Exponential warmup in place of RAdam.  This is based on the findings of https://arxiv.org/abs/1910.04209v3

2 - MadGrad core engine in place of Adam internals.  This is based on my own testing with Vision Transformers as well as the compelling MadGrad paper:  https://arxiv.org/abs/2101.11075v1

3 - Stable Weight Decay instead of AdamW style or Adam style:  needs more testing but the paper is very compelling:  https://arxiv.org/abs/2011.11152v3

4 - Gradient Centralization will be continued - as always, you can turn it on or off.  https://arxiv.org/abs/2004.01461v2

5 - Lookahead may be brought forward - unclear how much it may help with the new MadGrad core, which already leverages dual averaging, but will probably include as a testable param. 

6 - ??  Some discussion of having Adam core present as well so that one could quickly test with both Madgrad and Adam (or AdamP) with the flip of a param. 

If you have ideas/feedback, feel free to open an issue. 



