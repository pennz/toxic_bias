# data comprehension
## more data !

Classification of political social media: Social media messages from politicians classified by content. (4 MB)

https://github.com/niderhoff/nlp-datasets
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
**length** **word count** might be useful feature (also length/words)

## 
my val set, contains to many ones with identity (40%, so need 3 fold more data to the val, so it will be like normal)?

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


# thoughts

how about dinamic loss? so it can learn new things?
as assimilation bias, it won't learn new or slow to learn new things?

[('male', [0.0868988279903449, 0.024006818735868903, 0.031242945481204577, 0.07416071331123188, 0.13932945004135933, 0.1860850052194549, 0.18298757267356236, 0.1888153851299257, 0.12141408227441405, 0.08994548555068564, -0.09294516783403772]
  ('fema', [0.0858460457333715, 0.03998626965140316, 0.035134839321660186, 0.08014239709265442, 0.13069130597670467, 0.20103575435650542, 0.19648444055167236, 0.17253754502896734, 0.15636591704341837, 0.06390800533740948, -0.0642629398352049]
  ('chri', [0.07383960193847536, 0.0001525138778442787, 0.008113421241832041, 0.06929939356217822, 0.12123725133126122, 0.176555796776425, 0.17232984331296086, 0.18051083675004498, 0.07992384555347144, 0.05648397203345214, -0.08698685953461757]
  ('jewi', [0.09814064930646847, 0.07052894419587422, 0.02992634595319145, 0.011870180283804162, 0.12616045954211955, 0.1927126063005163, 0.2789278252509592, 0.2270403916813823, 0.06602803934785656, 0.07523033355075154, 0.024996043954576774]
[array([0.,               0.0795185 ,         0.17971314,              0.25032738,         0.35026588,        0.45057484,             0.55087489,          0.64984766,         0.7442968 ,            0.82651849,          0.92473269,        1.        ]
  ('homo', [0.16058013599093368, 0.04661187378478136, 0.0046392306093415115, 0.04502535375235045, 0.09615846539514336, 0.14508515359987917, 0.11613045466954847, 0.11503373370399786, 0.15670964717046992, -0.04125251988983094, -0.148541165722741]
  ('musl', [0.175797473705881, 0.0908275125069619, 0.08717591820880657, 0.07986940142175555, 0.11521488129356153, 0.1698934726878419, 0.21820685182355817, 0.13151431639035271, 0.12393779400865594, 0.06189930412936861, -0.10637003556798827]
  ('blac', [0.1960007584824854, 0.10478315693235739, 0.0915559899422432, 0.05309145844492202, 0.11899847962881561, 0.1745522076463109, 0.18411732378484894, 0.1813219463630721, 0.14092580113874234, -0.004915524340389267, -0.14145868425041705]
  ('whit', [0.1729280053536235, 0.06667576900789515, 0.07576588332371627, 0.0694295272306843, 0.10727775771229858, 0.17656470025690546, 0.18215378513520436, 0.15837744743992646, 0.14462884582142338, 0.05891528832673197, -0.03966006912103419]
  ('psyc', [0.1534715938734, 0.1715572774787318, 0.07450049093637635, 0.04877846018660041, 0.22132909577006502, 0.25624672781122954, 0.3060427953903993, 0.19067985141097027, 0.14635980250295152, 0.12182798568293494, -0.03182154893875122])]

male 0.9233 4475.0 3783, 0.1889, 0.268 692, 0.7802, 0.2675 3783, 0.1017, 0.1362 692, 0.6409, 0.1299
 femal 0.925 5359.0 4631, 0.1862, 0.2637 728, 0.7751, 0.2576 4631, 0.1012, 0.134 728, 0.6285, 0.1305
 homos 0.8567 1102.0 784, 0.313, 0.2891 318, 0.7459, 0.2624 784, 0.1722, 0.1477 318, 0.6278, 0.126
 chris 0.9352 4125.0 3755, 0.1538, 0.253 370, 0.7738, 0.2644 3755, 0.08989, 0.128 370, 0.6094, 0.1165
 jewis 0.8983 729.0 604, 0.2414, 0.2735 125, 0.7618, 0.256 604, 0.132, 0.1409 125, 0.619, 0.1308
 musli 0.8846 2095.0 1632, 0.3036, 0.2887 463, 0.7811, 0.2476 1632, 0.1471, 0.1447 463, 0.6243, 0.1223
 black 0.8553 1456.0 990, 0.3059, 0.2919 466, 0.7389, 0.2654 990, 0.1738, 0.154 466, 0.6261, 0.122
 white 0.862 2513.0 1807, 0.297, 0.2878 706, 0.7411, 0.2675 1807, 0.1706, 0.1515 706, 0.6213, 0.1136
 psych 0.9248 477.0 378, 0.2436, 0.3023 99, 0.8229, 0.2085 378, 0.1155, 0.1389 99, 0.6495, 0.119 

after use unfocal ( and beta bias for activation)

I0619 01:30:14.568189 140651875575680 lstm.py:935] error info: male, [0.06976694879404818, 0.021678299143853552, 0.021132456790180696, 0.040207538505457004, 0.05811422408576056, 0.21110787733493067, 0.17121297685417536, 0.2029319625161144, 0.12999203477487042, 0.055987557590970204, -0.07548467533195541]
I0619 01:30:14.568980 140651875575680 lstm.py:935] error info: female, [0.07141765812437627, 0.010710574579914662, 0.021946577322815092, 0.05366731395503924, 0.06716256893273545, 0.1717966735029047, 0.17685459531441733, 0.17978181325043274, 0.13412174006591504, 0.0623992743061014, -0.06906417553324348]
I0619 01:30:14.569115 140651875575680 lstm.py:935] error info: homosexual_gay_or_lesbian, [0.1441713744153579, 0.09970750658599635, 0.05874564974129875, 0.036267526913607864, 0.04170616146220986, 0.13014369919233765, 0.16717655152608704, 0.1314515768623192, 0.10710013774061605, 0.08245479066766584, -0.13679108023643494]
I0619 01:30:14.569193 140651875575680 lstm.py:935] error info: christian, [0.06300703750578629, 0.008353306610655576, -0.013825597710113187, 0.05649540465650727, 0.08091315719767189, 0.15112991641332027, 0.19604897038720567, 0.18798701539933843, 0.12771739962499795, 0.09100204265458953, -0.15298340717951456]
I0619 01:30:14.569263 140651875575680 lstm.py:935] error info: jewish, [0.10150076118345554, 0.01051388381929874, -0.012878425348033146, 0.0747033833080523, 0.1345577999439482, 0.17294953328867754, 0.21092493259078907, 0.12121616577253573, 0.09282761406780557, 0.10414400148069268, -0.11076110601425171]
I0619 01:30:14.569333 140651875575680 lstm.py:935] error info: muslim, [0.1591843567147826, 0.09329922628980544, 0.06286777313430153, 0.07175165679004696, 0.09588642642737273, 0.14318168239673779, 0.22605714922926032, 0.17222997692923248, 0.08344125530456324, 0.10717235460176733, -0.17353269440720948]
I0619 01:30:14.569396 140651875575680 lstm.py:935] error info: black, [0.17948942573534118, 0.1148249663727655, 0.0763595003813984, 0.05529621203390342, 0.07927967053646014, 0.17937225014570796, 0.15341170057344988, 0.16813585112843119, 0.09864428332477675, 0.04485196713486935, -0.003090201025463658]
I0619 01:30:14.569459 140651875575680 lstm.py:935] error info: white, [0.1479704718530914, 0.04190380087436939, 0.0698155003012461, 0.06945685591097933, 0.055813983728022265, 0.18648702358478317, 0.15622752879152965, 0.18059055056534615, 0.10848814601948328, 0.0474959653065239, -0.020777168604808962]
I0619 01:30:14.569521 140651875575680 lstm.py:935] error info: psychiatric_or_mental_illness, [0.14171911318871108, 0.1050696023928574, 0.08523304832870701, 0.06391022215163503, 0.06653034614470311, 0.14976314517048808, 0.2569822332495565, 0.1435285794402023, 0.1688979688257606, 0.0960054962795656, -0.192216237783432]

### subgroup auc
male  0.9453 4442.0	3779, 0.1584, 0.2529	663, 0.8044, 0.2471	3779, 0.101, 0.1356	663, 0.6417, 0.1315
femal 0.9433 5427.0	4666, 0.1554, 0.2497	761, 0.7895, 0.2524	4666, 0.09847, 0.1326	761, 0.634, 0.1291
homos 0.8894 1129.0	803, 0.2586, 0.2823	326, 0.747, 0.2512	803, 0.1712, 0.1476	326, 0.6222, 0.1304
chris 0.9468 3965.0	3580, 0.1318, 0.2385	385, 0.7727, 0.2638	3580, 0.08891, 0.1254	385, 0.6136, 0.1173
jewis 0.929 754.0	626, 0.1923, 0.263	128, 0.779, 0.2307	626, 0.1225, 0.1359	128, 0.6235, 0.123
musli 0.8965 2067.0	1592, 0.2632, 0.2915	475, 0.7741, 0.2407	1592, 0.152, 0.1443	475, 0.6296, 0.1307
black 0.8894 1522.0	1041, 0.2871, 0.2923	481, 0.7793, 0.2411	1041, 0.1694, 0.153	481, 0.6263, 0.1267
white 0.897 2498.0	1795, 0.2696, 0.2841	703, 0.7762, 0.2467	1795, 0.1776, 0.1507	703, 0.6239, 0.1209
psych 0.916 514.0	405, 0.2347, 0.2817	109, 0.7952, 0.2464	405, 0.1175, 0.1385	109, 0.648, 0.1401

