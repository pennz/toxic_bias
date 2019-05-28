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
 
*todo* for these subgroups, we can handle
 differently? not just see gay and then treat it as toxic
 
 so we can use network for different subgroup? and given them different weight?
 
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

```python
      bce = target * math_ops.log(output + epsilon())
      bce += (1 - target) * math_ops.log(1 - output + epsilon())
return -bce # binary cross entropy
```
like wide and deep? just different feature, different combination
but for this bias thing, should use the group information!
