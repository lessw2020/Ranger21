# Ranger21 - integrating the latest deep learning components into a single optimizer
Ranger deep learning optimizer rewrite to use newest components 

Ranger, with Radam + Lookahead core, is now 1.5 years old.  In the interim, a number of new developments have happened including the rise of Transformers for Vision.

Thus, Ranger21 (as in 2021) is a rewrite with multiple new additions reflective of some of the most impressive papers this past year.  The focus for Ranger21 is that these internals will be parameterized, and where possible, automated, so that you can easily test and leverage some of the newest concepts in AI training, to optimize the optimizer on your respective dataset. 

### Ranger21 Status:</br>
<b> April 12 - positive negative momentum added, madgrad core checked in </b> Testing over the weekend showed that positive negative momentum works really well, and even better with GC.  
Code is a bit messy atm b/c also tested Adaiw, but did not do that well so removed and added pos negative momentum.
Pos Neg momentum is a new technique to add parameter based, anisotropic noise to the gradient which helps it settle into flatter minima and also escape saddle points. In other words, better results.
Link to their excellent paper:
https://arxiv.org/abs/2103.17182

You can toggle between madgrad or not with the use_madgrad = True/False flag:
![ranger21_use_madgrad_toggle](https://user-images.githubusercontent.com/46302957/114484623-6c1f9500-9bbf-11eb-84f0-830859556856.JPG)


<b> April 10 - madgrad core engine integrated </b> Madgrad has been added in a way that you will be able to select to use MadGrad or Adam as the core 'engine' for the optimizer.  
Thus, you'll be able to simply toggle which opt engine to use, as well as the various enhancements (warmup, stable weight decay, gradient_centralization) and thus quickly find the best optimization setup for your specific dataset. 

Still testing things and then will update code here...
Gradient centralization good for both - first findings are gradient centralization definitely improves MadGrad (just like it does with Adam core) so will have GC on as default for both engines.

![madgrad_added_ranger21](https://user-images.githubusercontent.com/46302957/114292041-aca4d480-9a40-11eb-92b3-4243fd6d4390.JPG)


### LR selection is very different between MadGrad and Adam core engine:
One item - the starting lr for madgrad is very different (typically higher) than with Adam....have done some testing with automated LR scheduling (HyperExplorer and ABEL), but that will be added later if it's successful.  But if you simply plug your usual Adam LR's into Madgrad you won't be impressed :) 

Note that AdamP projection was also tested as an option, but impact was minimal, so will not be adding it atm. 

<b>April 6 - Ranger21 alpha ready</b> - automatic warmup added.  Seeing impressive results with only 3 features implemented.  </br>Stable weight decay + GC + automated linear warmup seem to sync very nicely. 
Thus if you are feeling adventorous, Ranger21 is basically alpha usable.  Recommend you use the default warmup (automatic by default), but test lr and weight decay. 
</br>
Ranger21 will output the settings at init to make it clear what you are running with:
![Ranger21_initialization](https://user-images.githubusercontent.com/46302957/113806993-2de62980-9718-11eb-8291-9764b71a544d.JPG)


April 5 - stable weight decay added.  Quick testing shows nice results with 1e-4 weight decay on subset of ImageNet. 

Current feature set planned:</br>

1 - <b>feature complete</b> - automated, Linear and Exponential warmup in place of RAdam.  This is based on the findings of https://arxiv.org/abs/1910.04209v3

2 - <b> Feature in progress - MadGrad core engine </b>.  This is based on my own testing with Vision Transformers as well as the compelling MadGrad paper:  https://arxiv.org/abs/2101.11075v1

3 - <b>feature complete</b> - Stable Weight Decay instead of AdamW style or Adam style:  needs more testing but the paper is very compelling:  https://arxiv.org/abs/2011.11152v3

4 - <b>feature complete</b> - Gradient Centralization will be continued - as always, you can turn it on or off.  https://arxiv.org/abs/2004.01461v2

5 - Lookahead may be brought forward - unclear how much it may help with the new MadGrad core, which already leverages dual averaging, but will probably include as a testable param. 

6 - <b>Feature implementation in progress - dual optimization engines </b> - Will have Adam and Madgrad core present as well so that one could quickly test with both Madgrad and Adam (or AdamP) with the flip of a param. 

If you have ideas/feedback, feel free to open an issue. 



