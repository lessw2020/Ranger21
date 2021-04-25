# Ranger21 - integrating the latest deep learning components into a single optimizer
Ranger deep learning optimizer rewrite to use newest components 

Ranger, with Radam + Lookahead core, is now 1.5 years old.  In the interim, a number of new developments have happened including the rise of Transformers for Vision.

Thus, Ranger21 (as in 2021) is a rewrite with multiple new additions reflective of some of the most impressive papers this past year.  The focus for Ranger21 is that these internals will be parameterized, and where possible, automated, so that you can easily test and leverage some of the newest concepts in AI training, to optimize the optimizer on your respective dataset. 

### Ranger21 Status:</br>
<b> April 24 - New record on benchmark with NormLoss, PosNeg momo, Stable decay etc. all combined </b> NormLoss integrated into Ranger21 set a new high on our simple benchmark (ResNet 18, subset of ImageWoof).  </br>
Best Accuracy = 71.69   Best Validation Loss = 15.33</br>

![bestloss_r21](https://user-images.githubusercontent.com/46302957/115976500-96f4dc00-a523-11eb-847c-c06502a8dc44.JPG)

For comparison, using plain Adam on this benchmark:</br>
Adam Only Accuracy = 64.84   Best Adam Val Loss = 17.19

In otherwords, 6.85% higher accuracy atm. </br>

Basically it shows that the integration of all these various new techniques is paying off, as currently combining them delivers better than any of them + Adam.

New code checked in - adds Lookahead and of course Norm Loss.  Also the settings is now callable via .show_settings() as an easy way to check settings.  
![Ranger21_424_settings](https://user-images.githubusercontent.com/46302957/115976494-8fcdce00-a523-11eb-9bc8-6c4e2e1c189c.JPG)


Given that the extensive settings may become overwhelming, planning to create config file support to make it easy to save out settings for various architectures and ideally have a 'best settings' recipe for CNN, Transformer for Image/Video, GAN, etc. 

<b> April 23 - Norm Loss will be added, initial benchmarking in progress for several features </b> A new soft regularizer, norm loss, was recently published in this paper on Arxiv:  https://arxiv.org/abs/2103.06583v1  

It's in the spirit of weight decay, but approaches it in a unique manner by nudging the weights towards the oblique manifold..this means unlike weight decay, it can actually push smaller weights up towards the norm 1 property vs weight decay only pushes down.  Their paper also shows norm less is less sensitive to hyperparams such as batch size, etc. unlike regular weight decay.</br>

One of the lead authors was kind enough to share their TF implemention, and have reworked it into PyTorch form and integrated into Ranger21.  Initial testing set a new high for validation loss on my very basic benchmark.  Thus, norm loss will be available with the next code update. </br>

Also did some initial benchmarking to set vanilla Adam as a baseline, and ablation style testing with pos negative momentum.  Pos neg momo alone is a big improvement over vanilla Adam, and looking forward to mapping out the contributions and synergies between all of the new features being rolled into Ranger21 including norm loss, adapt gradient clipping, gc, etc.  

<b> April 18 PM - Adaptive gradient clipping added, thanks for suggestion and code from @kayuksel. </b> AGC is used in NFNets to replace BN.  For our use case here, it's to have a smarter gradient clipping algo vs the usual hard clipping, and ideally better stabilize training.

Here's how the Ranger21 settings output looks atm:
![ranger21_settings](https://user-images.githubusercontent.com/46302957/115160522-7a513380-a04d-11eb-80a9-871f99da798e.JPG)


<b> April 18 AM - chebyshev fractals added, cosine warmdown (cosine decay) added </b></br>
Chebyshev performed reasonably well, but still needs more work before recommending so it's defaulting to off atm. 
There are two papers providing support for using Chebyshev, one of which is:
https://arxiv.org/abs/2010.13335v1 </br>
Cosine warmdown has been added so that the default lr schedule for Ranger21 is linear warmup, flat run at provided lr, and then cosine decay of lr starting at the X% passed in.  (Default is .65).  

<b> April 17 - building benchmark dataset(s)</b> As a cost effective way of testing Ranger21 and it's various options, currently taking a subset of ImageNet categories and building out at the high level an "ImageSubNet50" and also a few sub category datasets.  These are similar in spirit to ImageNette and ImageWoof, but hope to make a few relative improvements including pre-sizing to 224x224 for speed of training/testing.
First sub-dataset in progress in ImageBirds, which includes:  </br>
n01614925 bald eagle </br>
n01616318 vulture</br>
n01622779 grey owl</br>  
n01806143 peacock</br>
n01833805 hummingbird</br>
</br>
This is a medium-fine classification problem and will use as first tests for this type of benchmarking.  Ideally, will make a seperate repo for the ImageBirds shortly to make it available for people to use though hosting the dataset poses a cost problem... 

<b> April 12 - positive negative momentum added, madgrad core checked in </b> Testing over the weekend showed that positive negative momentum works really well, and even better with GC.  
Code is a bit messy atm b/c also tested Adaiw, but did not do that well so removed and added pos negative momentum.
Pos Neg momentum is a new technique to add parameter based, anisotropic noise to the gradient which helps it settle into flatter minima and also escape saddle points. 
In other words, better results.
</br>
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



