# data comprehension
## target
target means toxic or not

we have these subtypes:
- severe_toxicity
- obscene
- threat
- insult
- identity_attack
- sexual_explicit

*todo* check if all data has subtypes

## identity

*todo* split it out and train for more information

- **male**
- **female**
- transgender
- other_gender
- heterosexual
- **homosexual_gay_or_lesbian**
- bisexual
- other_sexual_orientation
- **christian**
- **jewish**
- **muslim**
- hindu
- buddhist
- atheist
- other_religion
- **black**
- **white**
- asian
- latino
- other_race_or_ethnicity
- physical_disability
- intellectual_or_learning_disability
- **psychiatric_or_mental_illness**
- other_disability

for bold ones, they are the identities with more than 500
 examples in the test set, and will be included in the evaluation
 calculation. i.e., subgroups, so, 
 
 **Do you remember how target was created? This is a fraction of voters who considered the comment to be toxic. Then is is completely normal that 0%, 1/6, 1/5 of voters could think the same.**
 add other features:
 
 
# todo
 
*todo* for these subgroups, we can handle !!!!!
we check one identity as example! white -> 
so steps:
1. filter out white data,25082 (80/20), only the ones with large difference (it just like ensemble learning)(should work i think)
2. continue to train
3. retest , check for this identity( and others), what is the effect (to speed things 20, all 20% test set)
4. re-iterate

## data
1. balance the data
2. run identity networks (check if it is usable)(to use as the data to reduce the bias)
3. add other features, if we decrease the identity affects to the final prediction
4. use other hand picked features, like length / complexity of grammar (feature engineering)

## what other people do 
debiased word embedding
gender swap (GS) augment the training data by swapping (somewhat not ideal, as some might asymmetric, but should more positive benefit overall)

so the key is like to re-balance data...

## embedding
Add back two embedding. it is like transfer learning


 differently? not just see gay and then treat it as toxic
 
 so we can use network for different subgroup? and given them different weight?
 
 Added attention, simple attention, as our target is not sequential data, so only one
 attention unit
 
*add back aux* might contain useful information, so make the model learn this task better, understand the information better
 
*add dann*

## what we found
for different subgroup, when you change the threshold, the auc will change 
for some subgroup. 

For example: for male subgroup
threshold - subgroup_auc:
0.5 - 0.918
0.6 - 0.923
0.7 - 0.932

as we change the threshold for deciding actual pos/neg, the predicted value
is not changed, it means our model, not mapping text to toxity very will?
Ans: **no**, just threshold larger, the ones above threshold should really toxic, 
so it is easier for our model to predict, so the score is high

## for identity
```
[INFO]2019-06-09 12:18:21,015:main:restore from the model file /content/gdrivedata/My Drive/_male_0.hdf5 -> done




2019-06-09 12:18:21.106084: I tensorflow/stream_executor/dso_loader.cc:152] successfully opened CUDA library libcublas.so.10.0 locally
[INFO]2019-06-09 12:19:45,525:main:for male, predict_sensitivity is 0.9684783694674792
[INFO]2019-06-09 12:19:45,528:main:restore from the model file
2019-06-09 12:19:48.241816: W tensorflow/core/framework/allocator.cc:124] Allocation of 388579200 exceeds 10% of system memory.
[INFO]2019-06-09 12:19:49,482:main:restore from the model file /content/gdrivedata/My Drive/_female_0.hdf5 -> done




[INFO]2019-06-09 12:21:12,640:main:for female, predict_sensitivity is 0.9713341774155259
[INFO]2019-06-09 12:21:12,648:main:restore from the model file
2019-06-09 12:21:15.465765: W tensorflow/core/framework/allocator.cc:124] Allocation of 388579200 exceeds 10% of system memory.
[INFO]2019-06-09 12:21:16,843:main:restore from the model file /content/gdrivedata/My Drive/_homosexual_gay_or_lesbian_0.hdf5 -> done




[INFO]2019-06-09 12:22:40,449:main:for homosexual_gay_or_lesbian, predict_sensitivity is 0.9630606860158312
[INFO]2019-06-09 12:22:40,453:main:restore from the model file
2019-06-09 12:22:44.097960: W tensorflow/core/framework/allocator.cc:124] Allocation of 388579200 exceeds 10% of system memory.
[INFO]2019-06-09 12:22:45,604:main:restore from the model file /content/gdrivedata/My Drive/_christian_0.hdf5 -> done




[INFO]2019-06-09 12:24:09,075:main:for christian, predict_sensitivity is 0.9234235502858591
[INFO]2019-06-09 12:24:09,082:main:restore from the model file
2019-06-09 12:24:12.828029: W tensorflow/core/framework/allocator.cc:124] Allocation of 388579200 exceeds 10% of system memory.
[INFO]2019-06-09 12:24:14,514:main:restore from the model file /content/gdrivedata/My Drive/_jewish_0.hdf5 -> done




[INFO]2019-06-09 12:25:38,179:main:for jewish, predict_sensitivity is 0.9930929686420776
[INFO]2019-06-09 12:25:38,181:main:restore from the model file
[INFO]2019-06-09 12:25:43,066:main:restore from the model file /content/gdrivedata/My Drive/_muslim_0.hdf5 -> done




[INFO]2019-06-09 12:27:06,766:main:for muslim, predict_sensitivity is 0.9664395403234008
[INFO]2019-06-09 12:27:06,768:main:restore from the model file
[INFO]2019-06-09 12:27:12,055:main:restore from the model file /content/gdrivedata/My Drive/_black_0.hdf5 -> done




[INFO]2019-06-09 12:28:35,736:main:for black, predict_sensitivity is 0.9805321219987021
[INFO]2019-06-09 12:28:35,737:main:restore from the model file
[INFO]2019-06-09 12:28:41,476:main:restore from the model file /content/gdrivedata/My Drive/_white_0.hdf5 -> done




[INFO]2019-06-09 12:30:05,492:main:for white, predict_sensitivity is 0.9902733523394265
[INFO]2019-06-09 12:30:05,493:main:restore from the model file
[INFO]2019-06-09 12:30:11,336:main:restore from the model file /content/gdrivedata/My Drive/_psychiatric_or_mental_illness_0.hdf5 -> done




[INFO]2019-06-09 12:31:35,339:main:for psychiatric_or_mental_illness, predict_sensitivity is 0.9063036546480255
[INFO]2019-06-09 12:31:35,340:main:Start run with lr 0.005, decay 0.5, gamma 2.0, BS 1024, NO_ID_IN_TRAIN True, EPOCHS 4, Y_TRAIN_BIN False ALPHA0.666
```
### loss function
if target value is float, not binary(True/False), then the 
BCE - ( y*log(y_pred) + (1-y)*log(1-y_pred) ) would not be suitable,
for example, when y=0.2, y_pred=0.6, the loss is -(0.2\*log(0.6)+0.8\*(log0.4)) will will be smaller than binary one
( - log(0.6) < - log(0.4))
(for binary, it will be -log(0.4), large, and the derivative pass back: )
For derivative:
it is dL/L(logits) = logits - y. so for binary, it will easily pushed to
two ends, but for linear, 

### loss and Z, sigmoid(z)
we can debug, to check the Z distribution, and to design our loss function.

### continue train...  
loading existed model and run as LearningRateScheduler setting learning rate to 0.0003125. (0.005/16), then 
continue training. the result is better...., so best so far runs (4+2) epochs. got 0.929 -> train one more epochs, it is 0.93

#### analysing
for 'white' subgroup:
7                          white      0.872848  0.966420  0.861708
For all the subgroups, (which is not know as input)

## need to consider group information, and improve the bias thing
thoughts:
1. training the wrong ones, so the model will know about bias and try to correct then 
    1. for this, the paper use logic pair, so network will learn that it means the same, to
    reduce the bias by learn they are the same. But this needs many handpicked example, and
    need to pay attention to the asymmetric pair. Really hand picked.
    2. how about we select the ones not learned well, just train them...
2. just remove all the identity related information? 
 
## votes
votes by netizens, can be helpful too, (don't know to what extent) but how do we use this?
most types votes are scare. the "like" and "disagree" might be helpful
## annotator count
identity and toxicity

# EDA from precedencer

## NaN missing values
for identities, around 80% is NaN
`Identity columns will have missing values when a comment didn't mention any identity. We can replace them with 0.`
## unique value

## like ratios (might be helpful)(only count might be not good)

## topics
use identity as topics:
*todo* how about we recognize the topics? and use these topics to check toxic
*todo* how do we know toxic? decision tree might be helpful

topics: we have 

- race and ethnicity
- gender
- sexual orientation
- religion
- disability

*todo* analyze topics with subtypes, subgroups(identity) (which is interesting too), 
how they are related

```python
features = ['asian', 'black', 'jewish', 'latino', 'other_race_or_ethnicity', 'white']
plot_features_distribution(features, "Distribution of race and ethnicity features values in the train set")
```

## relations
*todo* all with identity in training?
Really? check in toxic, non-toxic, how identity is distributed

- target with subgroup
- sensitive topics - target
- feedback(mono, votes)(but not in test data, so...)
- identity - target(0,1) #https://www.kaggle.com/ekhtiar/newbie-tutorial-jigsaw-unintended-bias-basic-eda
- identity - weighted target (0~1)  # same above
- time series - target
- time series - relative weighted toxic target score
- time series - relative weighted toxic target score per identity (topic will be fine, as topic is identity group)
- time series - relative maximum(local max) toxic (trend with time) - across identity (you can see white and black confrontation)
- correlation of identities in comment texts (between: female-male, hindu-buddhist, black-white, and so on)
*todo* sns relation, for toxic comments, find correlation, is it different with non-toxic # https://www.kaggle.com/ekhtiar/newbie-tutorial-jigsaw-unintended-bias-basic-eda
- toxicity_annotator_count - target frequency (but not for reducing bias)
- sentiment (neg-pos, complexity of comment) with toxic , non-toxic, "This shows that the toxic comments generally tend to be more negative on average."
    " This shows that toxic comments are generally less grammatically complex than non-toxic comments. Maybe, this is because toxic comments are generally more blunt and short. They often attack/insult people at a personal level. On the other hand, non-toxic comments generally try to share a perspective or make a point, and thus, they tend to be more gramatically complex or compounded on average. "

## missing values


# Todo in algorithm
first toxic, then reducing unintended bias
topics (subgroups), then analyze based topics if high topic affinity?
learn from the attention model, use small network to learn loss function composition? and dense_3_loss is smaller then dense 2
*Q how it is caculated?*
Still?

How to take subgroup into account? into our loss function
or weights different data?

So our problem: for the ones with 
(AUC problem, not the threshold problem), but subgroup indeed will affect 
the final ones. 
The logic order is reversed in my previous thoughts. Subgroup is not existed 
when training the model.

```python
      bce = target * math_ops.log(output + epsilon())
      bce += (1 - target) * math_ops.log(1 - output + epsilon())
return -bce # binary cross entropy
```
like wide and deep? just different feature, different combination
but for this bias thing, should use the group information!


## problems
non-binarized ones, on focal loss, no weights for unbalance
### comparison binary or not

subgroup

    male  0.9194	(0.09576205909252167, 0.1543089598417282)	(0.5339319109916687, 0.29221320152282715)
    femal 0.9194	(0.08456523716449738, 0.14278335869312286)	(0.5135778188705444, 0.2955287992954254)
    homos 0.8358	(0.197971910238266, 0.17625105381011963)	(0.4901445806026459, 0.24800816178321838)
    chris 0.9251	(0.057429440319538116, 0.11492059379816055)	(0.429908812046051, 0.2790357172489166)
    jewis 0.8835	(0.11788886785507202, 0.14946062862873077)	(0.4652176797389984, 0.2746000289916992)
    musli 0.854	(0.16775627434253693, 0.1603298932313919)	(0.47988712787628174, 0.26370587944984436)
    black 0.8358	(0.2118678092956543, 0.18535982072353363)	(0.5275276899337769, 0.265418142080307)
    white 0.8442	(0.19593653082847595, 0.1829969435930252)	(0.5289711356163025, 0.27579113841056824)
    psych 0.9149	(0.11351379752159119, 0.15796461701393127)	(0.5642567276954651, 0.2966223359107971)
    
bpsn
    
    male  0.9131	(0.09576205909252167, 0.1543089598417282)	(0.5362406373023987, 0.30515122413635254)
    femal 0.9253	(0.08456523716449738, 0.14278335869312286)	(0.5401219129562378, 0.30457088351249695)
    homos 0.8196	(0.197971910238266, 0.17625105381011963)	(0.5392336249351501, 0.3066760003566742)
    chris 0.9511	(0.057429440319538116, 0.11492059379816055)	(0.5450959205627441, 0.3035833537578583)
    jewis 0.895	(0.11788886785507202, 0.14946062862873077)	(0.5378623008728027, 0.30382877588272095)
    musli 0.849	(0.16775627434253693, 0.1603298932313919)	(0.5423970818519592, 0.306907594203949)
    black 0.8044	(0.2118678092956543, 0.18535982072353363)	(0.5368527173995972, 0.30728623270988464)
    white 0.8185	(0.19593653082847595, 0.1829969435930252)	(0.537156879901886, 0.30799585580825806)
    psych 0.8979	(0.11351379752159119, 0.15796461701393127)	(0.5352566242218018, 0.30342885851860046)
    
bnsp
    
    male  0.95	(0.060392823070287704, 0.1260470747947693)	(0.5339319109916687, 0.29221320152282715)
    femal 0.9419	(0.061103250831365585, 0.1274721622467041)	(0.5135778188705444, 0.2955287992954254)
    homos 0.9544	(0.06110946834087372, 0.12693394720554352)	(0.4901445806026459, 0.24800816178321838)
    chris 0.9167	(0.06487572938203812, 0.13133881986141205)	(0.429908812046051, 0.2790357172489166)
    jewis 0.9379	(0.06313309073448181, 0.12918923795223236)	(0.4652176797389984, 0.2746000289916992)
    musli 0.9509	(0.059209901839494705, 0.12605734169483185)	(0.47988712787628174, 0.26370587944984436)
    black 0.9595	(0.059782497584819794, 0.1251622885465622)	(0.5275276899337769, 0.265418142080307)
    white 0.9596	(0.0571410246193409, 0.12244977802038193)	(0.5289711356163025, 0.27579113841056824)
    psych 0.9531	(0.06357372552156448, 0.12931282818317413)	(0.5642567276954651, 0.2966223359107971)


bnsp better than bpsn, it means:
backgroup True negtive, Positive ones mapped to larger ones, 
but subgroup True negtive, will be predicted to higher probility, make it hard to differ with backgroup True positive


## useful links
- https://www.cerebriai.com/testing-for-overfitting-in-binary-classifiers/
- [keras custom losses, aux output](https://www.pyimagesearch.com/2018/06/04/keras-multiple-outputs-and-multiple-losses/)
