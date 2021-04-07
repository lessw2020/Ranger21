# Ranger21 - integrating the latest deep learning components into a single optimizer
Ranger deep learning optimizer rewrite to use newest components 

Ranger, with Radam + Lookahead core, is now 1.5 years old.  In the interim, a number of new developments have happened including the rise of Transformers for Vision.

Thus, Ranger21 (as in 2021) is a rewrite with multiple new additions reflective of some of the most impressive papers this past year.  The focus for Ranger21 is that these internals will be parameterized, and where possible, automated, so that you can easily test and leverage some of the newest concepts in AI training, to optimize the optimizer on your respective dataset. 

### Ranger21 Status:</br>
<b>April 6 - Ranger21 alpha ready</b> - automatic warmup added.  Seeing impressive results with only 3 features implemented.  </br>Stable weight decay + GC + automated linear warmup seem to sync very nicely. 
Thus if you are feeling adventorous, Ranger21 is basically alpha usable.  Recommend you use the default warmup (automatic by default), but test lr and weight decay. 
</br>
Ranger21 will output the settings at init to make it clear what you are running with:
![Ranger21_initialization](https://user-images.githubusercontent.com/46302957/113806993-2de62980-9718-11eb-8291-9764b71a544d.JPG)


April 5 - stable weight decay added.  Quick testing shows nice results with 1e-4 weight decay on subset of ImageNet. 

Current feature set planned:</br>

1 - <b>feature complete</b> - automated, Linear and Exponential warmup in place of RAdam.  This is based on the findings of https://arxiv.org/abs/1910.04209v3

2 - MadGrad core engine in place of Adam internals.  This is based on my own testing with Vision Transformers as well as the compelling MadGrad paper:  https://arxiv.org/abs/2101.11075v1

3 - <b>feature complete</b> - Stable Weight Decay instead of AdamW style or Adam style:  needs more testing but the paper is very compelling:  https://arxiv.org/abs/2011.11152v3

4 - <b>feature complete</b> - Gradient Centralization will be continued - as always, you can turn it on or off.  https://arxiv.org/abs/2004.01461v2

5 - Lookahead may be brought forward - unclear how much it may help with the new MadGrad core, which already leverages dual averaging, but will probably include as a testable param. 

6 - ??  Some discussion of having Adam core present as well so that one could quickly test with both Madgrad and Adam (or AdamP) with the flip of a param. 

If you have ideas/feedback, feel free to open an issue. 



