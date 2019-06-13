import numpy as np
import pandas as pd
import tensorflow as tf
import os
import gc
import pickle
from tensorflow.keras.preprocessing import text, sequence

MAX_LEN = 180  # used in lstm definition ....

import lstm
import sklearn

from IPython.core.debugger import set_trace

# from tqdm import tqdm
# tqdm.pandas()


MODEL_NAME = "lstm"
EMBEDDING_FILES = [
    '../input/glove840b300dtxt/glove.840B.300d.txt',
'../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec']
# only read first 220 words in the comment sentence
IDENTITY_COLUMNS = [
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness'
]  # features, meaning
AUX_COLUMNS = ['severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat', 'sexual_explicit']
TEXT_COLUMN = 'comment_text'
TARGET_COLUMN = 'target'
TOXICITY_COLUMN = TARGET_COLUMN
# CHARS_TO_REMOVE = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n“”’\'∞θ÷α•à−β∅³π‘₹´°£€\×™√²—'
DATA_FILE_FLAG = '.tf_record_saved'


def maybe_download(url):
    # By default the file at the url origin is downloaded to the cache_dir ~/.keras,
    # placed in the cache_subdir datasets, and given the filename fname
    path = tf.keras.utils.get_file(url.split('/')[-1], url)
    return path


def train_input_fn_general(features, labels, batch_size, cv, split_id=None, n_splits=None, handle_large=False,
                           shuffle_size=None):
    # for boost tree, need to prepare feature columns
    # 2048? columns, all float
    if cv:
        return _input_fn_general(features, labels, batch_size, split_id=split_id, n_splits=n_splits, cv_train=True,
                                 handle_large=handle_large)
    else:
        return _input_fn_general(features, labels, batch_size, cv=False)


def eval_input_fn_general(features, labels, batch_size, cv, split_id=None, n_splits=None):
    if cv:
        return _input_fn_general(features, labels, batch_size, with_y=True, repeat=False, shuffle=False,
                                 split_id=split_id, n_splits=n_splits, cv_train=False)
    else:
        return _input_fn_general(features, labels, batch_size, with_y=True, repeat=False, shuffle=False, cv=False)


def pred_input_fn_general(features, batch_size):
    return _input_fn_general(features, None, batch_size, with_y=False, repeat=False, shuffle=False, cv=False)


def _input_fn_general(features, labels, batch_size, with_y=True, repeat=True, shuffle=True, split_id=-1, n_splits=10,
                      cv=True, cv_train=True, to_dict=False, handle_large=False, shuffle_size=None):
    # for these, we will need to extract all the points before:
    """_input_fn_general is used for (boot tree)?

    :param features:
    :param labels:
    :param batch_size:
    :param with_y:
    :param repeat:
    :param shuffle:
    :param split_id:
    :param n_splits:
    :param cv:
    :param cv_train: cv train or evaluation
    :return:
    """

    def _to_dict(f):
        # first to pandas data frame
        df = pd.DataFrame(f, columns=[str(i) for i in range(features.shape[-1])])
        return dict(df)

    if to_dict:
        features = _to_dict(features)
    if handle_large:
        # features_placeholder = tf.placeholder(features.dtype, features.shape)
        # labels_placeholder = tf.placeholder(labels.dtype, labels.shape)
        features = tf.keras.backend.placeholder(dtype=features.dtype, shape=features.shape)
        labels = tf.keras.backend.placeholder(dtype=labels.dtype, shape=labels.shape)
    if with_y:
        ds = tf.data.Dataset.from_tensor_slices((features, labels))
    else:
        ds = tf.data.Dataset.from_tensor_slices(features)

    if cv:
        assert split_id >= 0 and n_splits > 1 and split_id < n_splits
        if cv_train:
            ds = [ds.shard(n_splits, i) for i in range(n_splits)]

            shards_cross = [ds[val_id] for val_id in range(n_splits) if val_id != split_id]

            ds = shards_cross[0]
            for t in shards_cross[1:]:
                ds = ds.concatenate(t)

            if shuffle:
                if handle_large:
                    ds = ds.shuffle(buffer_size=shuffle_size)  # just memory is not enough ...
                else:
                    ds = ds.shuffle(
                        buffer_size=int(len(labels) * (n_splits - 1) / n_splits))  # just memory is not enough ...
        else:  # cv evaluate, no need to shuffle
            ds = ds.shard(n_splits, split_id)
    else:
        if shuffle:
            if handle_large:
                ds = ds.shuffle(buffer_size=shuffle_size)
            else:
                ds = ds.shuffle(buffer_size=len(labels))
    # after shuffle, we do cross validtation split

    # taken from Dan, https://stackoverflow.com/questions/39748660/how-to-perform-k-fold-cross-validation-with-tensorflow
    # will need to append id, then remove the id?
    # -> no need, we just split to 5 shards, then rearrange these shards
    if repeat:
        ds = ds.repeat()
    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    # Return the dataset.
    return ds.batch(batch_size).prefetch(1)


def _load_embeddings_only_for_fasttext_crawl_avg(path):
    """

    :param path: large embedding file
    :return: np array of
    """
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if i == 0: continue  # for crawl the first line is "2000000 300"
            # print(line)
            hidden_dim = len(line.split(' ')) - 2 if line.strip().split(' ')[
                                                         -1] == '\n' else 1  # for crawl, the '\n' is there to
            # print(hidden_dim)
            break

    vec_sum_place = np.zeros(hidden_dim, dtype=np.float32)
    # print(vec_sum_place.shape)
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if i == 0: continue
            # try:
            nums = line.split(' ')[1:]
            if nums[-1] == '\n':
                nums = nums[:-1]
            vec_sum_place = vec_sum_place + np.array([float(num) for num in nums], dtype=np.float32)
            # except ValueError as e:
            #    print(line)
            #    print(len(line.strip().split(' ')))
            #    print(e)
            #    raise e
    avg_vec = vec_sum_place / i  # i = 0 is the header line
    return avg_vec


# Define bias metrics, then evaluate our new model for bias using the validation set predictions
SUBGROUP_AUC = 'subgroup_auc'
BPSN_AUC = 'bpsn_auc'  # stands for background positive, subgroup negative
BNSP_AUC = 'bnsp_auc'  # stands for background negative, subgroup positive
OVERALL_AUC = 'overall_auc'

from sklearn import metrics

class TargetDistAnalyzer:
    def __init__(self, target):
        self.df = target  # df with target, with identity information, then we can sort it out
        self.descretizer = sklearn.preprocessing.KBinsDiscretizer(10+1, encode='ordinal', strategy='uniform')
        self.descretizer.fit(target[TARGET_COLUMN].values.reshape(-1, 1))
        self.id_used_in_train = None

    def get_distribution(self, target_data):
        """
        target_data: pandas series, need to get index, so need series

        :return: (type, cnt number, frequency, index) pair list for this distribution
        """
        dst = []

        y_t = self.descretizer.transform(target_data.values.reshape(-1,1))
        uniq_elements, element_counts = np.unique(y_t, return_counts=True)
        all_counts = len(y_t)
        for i, e in enumerate(uniq_elements):
            dst.append((e, element_counts[i], element_counts[i]/all_counts, target_data.loc[(y_t == e).ravel()].index))

        return dst

    def get_distribution_overall(self):
        return self.get_distribution(self.df[TARGET_COLUMN])

    def get_distribution_subgroups(self):
        dstr = {}

        for g in IDENTITY_COLUMNS:
            dstr[g] = self.get_distribution(self.df[self.df[g+'_in_train'].fillna(0.)>=0.5][TARGET_COLUMN])  # could use continuous data, might be helpful so calculate belongs (fuzzy logic)
        return dstr



class BiasBenchmark:
    def __init__(self, train_df_id_na_dropped, threshold=0.5):
        # we can use this to divide subgroups(beside, we need to calculate only the 45000, with identities
        self.threshold = threshold
        self.validate_df = BiasBenchmark.copy_convert_dataframe_to_bool(train_df_id_na_dropped[IDENTITY_COLUMNS + [TOXICITY_COLUMN, TEXT_COLUMN]], threshold)

    @staticmethod
    def compute_auc(y_true, y_pred):
        try:
            return metrics.roc_auc_score(y_true, y_pred)
        except ValueError:
            return np.nan

    @staticmethod
    def compute_predict_estimator(y_true, y_pred):
        assert pd.core.dtypes.common.is_dtype_equal(y_true.dtype, np.dtype('bool'))

        pos_cnt = sum(y_true)
        neg_cnt = len(y_true) - pos_cnt

        pred_pos_cnt = sum(y_pred > 0.5)
        pred_neg_cnt = len(y_pred) - pred_pos_cnt

        at_predict = y_pred[y_true]  # predictions for True actually, might predict to 0.4(bad)
        af_predict = y_pred[~y_true]

        return ((neg_cnt, pos_cnt, pos_cnt/(neg_cnt+pos_cnt)),
        (pred_neg_cnt, pred_pos_cnt),
        (len(af_predict), np.mean(af_predict), np.std(af_predict)),
        (len(at_predict), np.mean(at_predict), np.std(at_predict)))



    @staticmethod
    def compute_subgroup_classify_detail(df, subgroup, label, model_name):
        """
        Compute AUC for spefic subgroup
        :param df: dataframe which contains predictions for all subgroups
        :param subgroup: compute AUC for this subgroup
        :param label: target column name
        :param model_name:
        :return: just mean/std, for pos, neg, then we just use this information to make shift
        """
        subgroup_examples = df[df[subgroup]]  # innter df[subgroup] will get out boolean list, which is used to select this
        # subgroup examples
        return BiasBenchmark.compute_predict_estimator(subgroup_examples[label], subgroup_examples[model_name])

    @staticmethod
    def compute_subgroup_auc(df, subgroup, label, model_name):
        """
        Compute AUC for spefic subgroup. AUC only cares about ordering, not threshold
        :param df: dataframe which contains predictions for all subgroups
        :param subgroup: compute AUC for this subgroup
        :param label: target column name
        :param model_name:
        :return: auc score
        """
        subgroup_examples = df[df[subgroup]]  # innter df[subgroup] will get out boolean list, which is used to select this
                                                # subgroup examples
        return BiasBenchmark.compute_auc(subgroup_examples[label], subgroup_examples[model_name])

    @staticmethod
    def compute_bpsn_classify_detail(df, subgroup, label, model_name):
        """Computes the AUC of the within-subgroup negative examples and the background positive examples."""
        subgroup_negative_examples = df[df[subgroup] & ~df[label]]
        non_subgroup_positive_examples = df[~df[subgroup] & df[label]]  # background positive
        examples = subgroup_negative_examples.append(non_subgroup_positive_examples)

        # this example is background True positive, with subgroup True negative, and see our model's prediction's performance
        return BiasBenchmark.compute_predict_estimator(examples[label], examples[model_name])

    @staticmethod
    def compute_bpsn_auc(df, subgroup, label, model_name):
        """Computes the AUC of the within-subgroup negative examples and the background positive examples."""
        subgroup_negative_examples = df[df[subgroup] & ~df[label]]
        non_subgroup_positive_examples = df[~df[subgroup] & df[label]]
        examples = subgroup_negative_examples.append(non_subgroup_positive_examples)
        return BiasBenchmark.compute_auc(examples[label], examples[model_name])

    @staticmethod
    def compute_bnsp_classify_detail(df, subgroup, label, model_name):
        """Computes the AUC of the within-subgroup positive examples and the background negative examples."""
        subgroup_positive_examples = df[df[subgroup] & df[label]]
        non_subgroup_negative_examples = df[~df[subgroup] & ~df[label]]
        examples = subgroup_positive_examples.append(non_subgroup_negative_examples)
        return BiasBenchmark.compute_predict_estimator(examples[label], examples[model_name])

    @staticmethod
    def compute_bnsp_auc(df, subgroup, label, model_name):
        """Computes the AUC of the within-subgroup positive examples and the background negative examples."""
        subgroup_positive_examples = df[df[subgroup] & df[label]]
        non_subgroup_negative_examples = df[~df[subgroup] & ~df[label]]
        examples = subgroup_positive_examples.append(non_subgroup_negative_examples)
        return BiasBenchmark.compute_auc(examples[label], examples[model_name])

    @staticmethod
    def compute_bias_metrics_for_model(dataset,
                                       subgroups,
                                       model,
                                       label_col,
                                       include_asegs=False):
        """
        Computes per-subgroup metrics for all subgroups and one model.
        # bias_metrics_df = BiasBenchmark.compute_bias_metrics_for_model(validate_df, IDENTITY_COLUMNS, MODEL_NAME, TOXICITY_COLUMN)
        :param dataset: prediction result
        :param subgroups: all group names
        :param model: my model name
        :param label_col: target column name
        :param include_asegs: ?
        :return:
        """
        records = []
        subgroup_distribution = []
        for subgroup in subgroups:
            record = {
                'subgroup': subgroup,
                'subgroup_size': len(dataset[dataset[subgroup]])
            }
            record_pn = {
                'subgroup': subgroup,
                'subgroup_size': len(dataset[dataset[subgroup]])
            }
            record[SUBGROUP_AUC] = BiasBenchmark.compute_subgroup_auc(dataset, subgroup, label_col, model)
            record[BPSN_AUC] = BiasBenchmark.compute_bpsn_auc(dataset, subgroup, label_col, model)
            record[BNSP_AUC] = BiasBenchmark.compute_bnsp_auc(dataset, subgroup, label_col, model)

            record_pn[SUBGROUP_AUC] = BiasBenchmark.compute_subgroup_classify_detail(dataset, subgroup, label_col, model)
            record_pn[BPSN_AUC] = BiasBenchmark.compute_bpsn_classify_detail(dataset, subgroup, label_col, model)
            record_pn[BNSP_AUC] = BiasBenchmark.compute_bnsp_classify_detail(dataset, subgroup, label_col, model)

            records.append(record)
            subgroup_distribution.append(record_pn)
        return pd.DataFrame(records).sort_values('subgroup_auc', ascending=True), subgroup_distribution

    # Calculate the final score
    @staticmethod
    def calculate_overall_auc(df, model_name):
        true_labels = df[TOXICITY_COLUMN]
        predicted_labels = df[model_name]
        return metrics.roc_auc_score(true_labels, predicted_labels)

    @staticmethod
    def calculate_overall_auc_distribution(df, model_name):
        info = dict()
        info[OVERALL_AUC] = BiasBenchmark.calculate_overall_auc(df, model_name)
        info['subgroup'] = 'overall'
        info['subgroup_size'] = len(df)
        info['distribution'] = BiasBenchmark.compute_predict_estimator(df[TOXICITY_COLUMN], df[model_name])
        return info

    @staticmethod
    def power_mean(series, p):
        total = sum(np.power(series, p))
        return np.power(total / len(series), 1 / p)

    @staticmethod
    def get_final_metric(bias_df, overall_auc, POWER=-5, OVERALL_MODEL_WEIGHT=0.25):
        bias_metrics_on_subgroups = [
            BiasBenchmark.power_mean(bias_df[SUBGROUP_AUC], POWER),
            BiasBenchmark.power_mean(bias_df[BPSN_AUC], POWER),
            BiasBenchmark.power_mean(bias_df[BNSP_AUC], POWER)
        ]
        bias_score = np.average(bias_metrics_on_subgroups)
        lstm.logger.debug(f'bias metrics details (AUC, BPSN, BNSP): {bias_metrics_on_subgroups}')
        return (OVERALL_MODEL_WEIGHT * overall_auc) + ((1 - OVERALL_MODEL_WEIGHT) * bias_score)

    @staticmethod
    def convert_to_bool(df, col_name, threshold=0.5, keep_original=False):
        if keep_original:
            df[col_name+'_orig'] = df[col_name]
        df[col_name] = np.where(df[col_name] >= threshold, True, False)

    @staticmethod
    def copy_convert_dataframe_to_bool(df, threshold):
        bool_df = df.copy()
        for col in IDENTITY_COLUMNS:
            BiasBenchmark.convert_to_bool(bool_df, col)
        BiasBenchmark.convert_to_bool(bool_df, TARGET_COLUMN, threshold, keep_original=True)
        return bool_df

    def calculate_benchmark(self, pred=None, validate_df=None, model_name=MODEL_NAME):
        """

        :param pred:
        :param model_name:
        :return: final metric score, bias auc for subgroups, subgroup classification distribution details, overall auc
        """
        if validate_df is None:
            print('In caculating benchmark, the validate_df passed in None, use default validate_df, with given pred')
            validate_df = self.validate_df
            assert validate_df.shape[0] == len(pred)
            validate_df[model_name] = pred  # prediction

        print('In caculating benchmark...')
        validate_df = BiasBenchmark.copy_convert_dataframe_to_bool(validate_df[IDENTITY_COLUMNS + [TOXICITY_COLUMN, TEXT_COLUMN, model_name]], self.threshold)
        bias_metrics_df, subgroup_distribution = BiasBenchmark.compute_bias_metrics_for_model(validate_df,
                                                                       IDENTITY_COLUMNS, model_name, TOXICITY_COLUMN)
        overall_auc_dist = BiasBenchmark.calculate_overall_auc_distribution(validate_df, model_name)
        final_score = BiasBenchmark.get_final_metric(bias_metrics_df, overall_auc_dist[OVERALL_AUC])

        return final_score, bias_metrics_df, subgroup_distribution, overall_auc_dist


# embedding vocab
class EmbeddingHandler:
    contraction_mapping = {
        "Trump's": 'trump is', "'cause": 'because', ',cause': 'because', ';cause': 'because', "ain't": 'am not',
        'ain,t': 'am not',
        'ain;t': 'am not', 'ain´t': 'am not', 'ain’t': 'am not', "aren't": 'are not',
        'aren,t': 'are not', 'aren;t': 'are not', 'aren´t': 'are not', 'aren’t': 'are not', "can't": 'cannot',
        "can't've": 'cannot have', 'can,t': 'cannot', 'can,t,ve': 'cannot have',
        'can;t': 'cannot', 'can;t;ve': 'cannot have',
        'can´t': 'cannot', 'can´t´ve': 'cannot have', 'can’t': 'cannot', 'can’t’ve': 'cannot have',
        "could've": 'could have', 'could,ve': 'could have', 'could;ve': 'could have', "couldn't": 'could not',
        "couldn't've": 'could not have', 'couldn,t': 'could not', 'couldn,t,ve': 'could not have',
        'couldn;t': 'could not',
        'couldn;t;ve': 'could not have', 'couldn´t': 'could not',
        'couldn´t´ve': 'could not have', 'couldn’t': 'could not', 'couldn’t’ve': 'could not have',
        'could´ve': 'could have',
        'could’ve': 'could have', "didn't": 'did not', 'didn,t': 'did not', 'didn;t': 'did not', 'didn´t': 'did not',
        'didn’t': 'did not', "doesn't": 'does not', 'doesn,t': 'does not', 'doesn;t': 'does not', 'doesn´t': 'does not',
        'doesn’t': 'does not', "don't": 'do not', 'don,t': 'do not', 'don;t': 'do not', 'don´t': 'do not',
        'don’t': 'do not',
        "hadn't": 'had not', "hadn't've": 'had not have', 'hadn,t': 'had not', 'hadn,t,ve': 'had not have',
        'hadn;t': 'had not',
        'hadn;t;ve': 'had not have', 'hadn´t': 'had not', 'hadn´t´ve': 'had not have', 'hadn’t': 'had not',
        'hadn’t’ve': 'had not have', "hasn't": 'has not', 'hasn,t': 'has not', 'hasn;t': 'has not', 'hasn´t': 'has not',
        'hasn’t': 'has not',
        "haven't": 'have not', 'haven,t': 'have not', 'haven;t': 'have not', 'haven´t': 'have not',
        'haven’t': 'have not', "he'd": 'he would',
        "he'd've": 'he would have', "he'll": 'he will',
        "he's": 'he is', 'he,d': 'he would', 'he,d,ve': 'he would have', 'he,ll': 'he will', 'he,s': 'he is',
        'he;d': 'he would',
        'he;d;ve': 'he would have', 'he;ll': 'he will', 'he;s': 'he is', 'he´d': 'he would', 'he´d´ve': 'he would have',
        'he´ll': 'he will',
        'he´s': 'he is', 'he’d': 'he would', 'he’d’ve': 'he would have', 'he’ll': 'he will', 'he’s': 'he is',
        "how'd": 'how did', "how'll": 'how will',
        "how's": 'how is', 'how,d': 'how did', 'how,ll': 'how will', 'how,s': 'how is', 'how;d': 'how did',
        'how;ll': 'how will',
        'how;s': 'how is', 'how´d': 'how did', 'how´ll': 'how will', 'how´s': 'how is', 'how’d': 'how did',
        'how’ll': 'how will',
        'how’s': 'how is', "i'd": 'i would', "i'll": 'i will', "i'm": 'i am', "i've": 'i have', 'i,d': 'i would',
        'i,ll': 'i will',
        'i,m': 'i am', 'i,ve': 'i have', 'i;d': 'i would', 'i;ll': 'i will', 'i;m': 'i am', 'i;ve': 'i have',
        "isn't": 'is not',
        'isn,t': 'is not', 'isn;t': 'is not', 'isn´t': 'is not', 'isn’t': 'is not', "it'd": 'it would',
        "it'll": 'it will', "It's": 'it is',
        "it's": 'it is', 'it,d': 'it would', 'it,ll': 'it will', 'it,s': 'it is', 'it;d': 'it would',
        'it;ll': 'it will', 'it;s': 'it is', 'it´d': 'it would', 'it´ll': 'it will', 'it´s': 'it is',
        'it’d': 'it would', 'it’ll': 'it will', 'it’s': 'it is',
        'i´d': 'i would', 'i´ll': 'i will', 'i´m': 'i am', 'i´ve': 'i have', 'i’d': 'i would', 'i’ll': 'i will',
        'i’m': 'i am',
        'i’ve': 'i have', "let's": 'let us', 'let,s': 'let us', 'let;s': 'let us', 'let´s': 'let us',
        'let’s': 'let us', "ma'am": 'madam', 'ma,am': 'madam', 'ma;am': 'madam', "mayn't": 'may not',
        'mayn,t': 'may not', 'mayn;t': 'may not',
        'mayn´t': 'may not', 'mayn’t': 'may not', 'ma´am': 'madam', 'ma’am': 'madam', "might've": 'might have',
        'might,ve': 'might have', 'might;ve': 'might have', "mightn't": 'might not', 'mightn,t': 'might not',
        'mightn;t': 'might not', 'mightn´t': 'might not',
        'mightn’t': 'might not', 'might´ve': 'might have', 'might’ve': 'might have', "must've": 'must have',
        'must,ve': 'must have', 'must;ve': 'must have',
        "mustn't": 'must not', 'mustn,t': 'must not', 'mustn;t': 'must not', 'mustn´t': 'must not',
        'mustn’t': 'must not', 'must´ve': 'must have',
        'must’ve': 'must have', "needn't": 'need not', 'needn,t': 'need not', 'needn;t': 'need not',
        'needn´t': 'need not', 'needn’t': 'need not', "oughtn't": 'ought not', 'oughtn,t': 'ought not',
        'oughtn;t': 'ought not',
        'oughtn´t': 'ought not', 'oughtn’t': 'ought not', "sha'n't": 'shall not', 'sha,n,t': 'shall not',
        'sha;n;t': 'shall not', "shan't": 'shall not',
        'shan,t': 'shall not', 'shan;t': 'shall not', 'shan´t': 'shall not', 'shan’t': 'shall not',
        'sha´n´t': 'shall not', 'sha’n’t': 'shall not',
        "she'd": 'she would', "she'll": 'she will', "she's": 'she is', 'she,d': 'she would', 'she,ll': 'she will',
        'she,s': 'she is', 'she;d': 'she would', 'she;ll': 'she will', 'she;s': 'she is', 'she´d': 'she would',
        'she´ll': 'she will',
        'she´s': 'she is', 'she’d': 'she would', 'she’ll': 'she will', 'she’s': 'she is', "should've": 'should have',
        'should,ve': 'should have', 'should;ve': 'should have',
        "shouldn't": 'should not', 'shouldn,t': 'should not', 'shouldn;t': 'should not', 'shouldn´t': 'should not',
        'shouldn’t': 'should not', 'should´ve': 'should have',
        'should’ve': 'should have', "that'd": 'that would', "that's": 'that is', 'that,d': 'that would',
        'that,s': 'that is', 'that;d': 'that would',
        'that;s': 'that is', 'that´d': 'that would', 'that´s': 'that is', 'that’d': 'that would', 'that’s': 'that is',
        "there'd": 'there had',
        "there's": 'there is', 'there,d': 'there had', 'there,s': 'there is', 'there;d': 'there had',
        'there;s': 'there is',
        'there´d': 'there had', 'there´s': 'there is', 'there’d': 'there had', 'there’s': 'there is',
        "they'd": 'they would', "they'll": 'they will', "they're": 'they are', "they've": 'they have',
        'they,d': 'they would', 'they,ll': 'they will', 'they,re': 'they are', 'they,ve': 'they have',
        'they;d': 'they would', 'they;ll': 'they will', 'they;re': 'they are',
        'they;ve': 'they have', 'they´d': 'they would', 'they´ll': 'they will', 'they´re': 'they are',
        'they´ve': 'they have', 'they’d': 'they would', 'they’ll': 'they will',
        'they’re': 'they are', 'they’ve': 'they have', "wasn't": 'was not', 'wasn,t': 'was not', 'wasn;t': 'was not',
        'wasn´t': 'was not',
        'wasn’t': 'was not', "we'd": 'we would', "we'll": 'we will', "we're": 'we are', "we've": 'we have',
        'we,d': 'we would', 'we,ll': 'we will',
        'we,re': 'we are', 'we,ve': 'we have', 'we;d': 'we would', 'we;ll': 'we will', 'we;re': 'we are',
        'we;ve': 'we have',
        "weren't": 'were not', 'weren,t': 'were not', 'weren;t': 'were not', 'weren´t': 'were not',
        'weren’t': 'were not', 'we´d': 'we would', 'we´ll': 'we will',
        'we´re': 'we are', 'we´ve': 'we have', 'we’d': 'we would', 'we’ll': 'we will', 'we’re': 'we are',
        'we’ve': 'we have', "what'll": 'what will', "what're": 'what are', "what's": 'what is',
        "what've": 'what have', 'what,ll': 'what will', 'what,re': 'what are', 'what,s': 'what is',
        'what,ve': 'what have', 'what;ll': 'what will', 'what;re': 'what are',
        'what;s': 'what is', 'what;ve': 'what have', 'what´ll': 'what will',
        'what´re': 'what are', 'what´s': 'what is', 'what´ve': 'what have', 'what’ll': 'what will',
        'what’re': 'what are', 'what’s': 'what is',
        'what’ve': 'what have', "where'd": 'where did', "where's": 'where is', 'where,d': 'where did',
        'where,s': 'where is', 'where;d': 'where did',
        'where;s': 'where is', 'where´d': 'where did', 'where´s': 'where is', 'where’d': 'where did',
        'where’s': 'where is',
        "who'll": 'who will', "who's": 'who is', 'who,ll': 'who will', 'who,s': 'who is', 'who;ll': 'who will',
        'who;s': 'who is',
        'who´ll': 'who will', 'who´s': 'who is', 'who’ll': 'who will', 'who’s': 'who is', "won't": 'will not',
        'won,t': 'will not', 'won;t': 'will not',
        'won´t': 'will not', 'won’t': 'will not', "wouldn't": 'would not', 'wouldn,t': 'would not',
        'wouldn;t': 'would not', 'wouldn´t': 'would not',
        'wouldn’t': 'would not', "you'd": 'you would', "you'll": 'you will', "you're": 'you are', 'you,d': 'you would',
        'you,ll': 'you will',
        'you,re': 'you are', 'you;d': 'you would', 'you;ll': 'you will',
        'you;re': 'you are', 'you´d': 'you would', 'you´ll': 'you will', 'you´re': 'you are', 'you’d': 'you would',
        'you’ll': 'you will', 'you’re': 'you are',
        '´cause': 'because', '’cause': 'because', "you've": "you have", "could'nt": 'could not',
        "havn't": 'have not', "here’s": "here is", 'i""m': 'i am', "i'am": 'i am', "i'l": "i will", "i'v": 'i have',
        "wan't": 'want', "was'nt": "was not", "who'd": "who would",
        "who're": "who are", "who've": "who have", "why'd": "why would", "would've": "would have", "y'all": "you all",
        "y'know": "you know", "you.i": "you i",
        "your'e": "you are", "arn't": "are not", "agains't": "against", "c'mon": "common", "doens't": "does not",
        'don""t': "do not", "dosen't": "does not",
        "dosn't": "does not", "shoudn't": "should not", "that'll": "that will", "there'll": "there will",
        "there're": "there are",
        "this'll": "this all", "u're": "you are", "ya'll": "you all", "you'r": "you are", "you’ve": "you have",
        "d'int": "did not", "did'nt": "did not", "din't": "did not", "dont't": "do not", "gov't": "government",
        "i'ma": "i am", "is'nt": "is not", "‘I": 'I',
        'ᴀɴᴅ': 'and', 'ᴛʜᴇ': 'the', 'ʜᴏᴍᴇ': 'home', 'ᴜᴘ': 'up', 'ʙʏ': 'by', 'ᴀᴛ': 'at', '…and': 'and',
        'civilbeat': 'civil beat', \
        'TrumpCare': 'Trump care', 'Trumpcare': 'Trump care', 'OBAMAcare': 'Obama care', 'ᴄʜᴇᴄᴋ': 'check', 'ғᴏʀ': 'for',
        'ᴛʜɪs': 'this', 'ᴄᴏᴍᴘᴜᴛᴇʀ': 'computer', \
        'ᴍᴏɴᴛʜ': 'month', 'ᴡᴏʀᴋɪɴɢ': 'working', 'ᴊᴏʙ': 'job', 'ғʀᴏᴍ': 'from', 'Sᴛᴀʀᴛ': 'start', 'gubmit': 'submit',
        'CO₂': 'carbon dioxide', 'ғɪʀsᴛ': 'first', \
        'ᴇɴᴅ': 'end', 'ᴄᴀɴ': 'can', 'ʜᴀᴠᴇ': 'have', 'ᴛᴏ': 'to', 'ʟɪɴᴋ': 'link', 'ᴏғ': 'of', 'ʜᴏᴜʀʟʏ': 'hourly',
        'ᴡᴇᴇᴋ': 'week', 'ᴇɴᴅ': 'end', 'ᴇxᴛʀᴀ': 'extra', \
        'Gʀᴇᴀᴛ': 'great', 'sᴛᴜᴅᴇɴᴛs': 'student', 'sᴛᴀʏ': 'stay', 'ᴍᴏᴍs': 'mother', 'ᴏʀ': 'or', 'ᴀɴʏᴏɴᴇ': 'anyone',
        'ɴᴇᴇᴅɪɴɢ': 'needing', 'ᴀɴ': 'an', 'ɪɴᴄᴏᴍᴇ': 'income', \
        'ʀᴇʟɪᴀʙʟᴇ': 'reliable', 'ғɪʀsᴛ': 'first', 'ʏᴏᴜʀ': 'your', 'sɪɢɴɪɴɢ': 'signing', 'ʙᴏᴛᴛᴏᴍ': 'bottom',
        'ғᴏʟʟᴏᴡɪɴɢ': 'following', 'Mᴀᴋᴇ': 'make', \
        'ᴄᴏɴɴᴇᴄᴛɪᴏɴ': 'connection', 'ɪɴᴛᴇʀɴᴇᴛ': 'internet', 'financialpost': 'financial post', 'ʜaᴠᴇ': ' have ',
        'ᴄaɴ': ' can ', 'Maᴋᴇ': ' make ', 'ʀᴇʟɪaʙʟᴇ': ' reliable ', 'ɴᴇᴇᴅ': ' need ',
        'ᴏɴʟʏ': ' only ', 'ᴇxᴛʀa': ' extra ', 'aɴ': ' an ', 'aɴʏᴏɴᴇ': ' anyone ', 'sᴛaʏ': ' stay ', 'Sᴛaʀᴛ': ' start',
        'SHOPO': 'shop',
    }
    punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
    punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2",
                     "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e",
                     '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta',
                     '∅': '', '³': '3', 'π': 'pi', }
    mispell_dict = {'SB91': 'senate bill', 'tRump': 'trump', 'utmterm': 'utm term', 'FakeNews': 'fake news',
                    'Gʀᴇat': 'great', 'ʙᴏᴛtoᴍ': 'bottom', 'washingtontimes': 'washington times',
                    'garycrum': 'gary crum', 'htmlutmterm': 'html utm term', 'RangerMC': 'car',
                    'TFWs': 'tuition fee waiver', 'SJWs': 'social justice warrior', 'Koncerned': 'concerned',
                    'Vinis': 'vinys', 'Yᴏᴜ': 'you', 'Trumpsters': 'trump', 'Trumpian': 'trump', 'bigly': 'big league',
                    'Trumpism': 'trump', 'Yoyou': 'you', 'Auwe': 'wonder', 'Drumpf': 'trump', 'utmterm': 'utm term',
                    'Brexit': 'british exit', 'utilitas': 'utilities', 'ᴀ': 'a', '😉': 'wink', '😂': 'joy',
                    '😀': 'stuck out tongue', 'theguardian': 'the guardian', 'deplorables': 'deplorable',
                    'theglobeandmail': 'the globe and mail', 'justiciaries': 'justiciary',
                    'creditdation': 'Accreditation', 'doctrne': 'doctrine', 'fentayal': 'fentanyl',
                    'designation-': 'designation', 'CONartist': 'con-artist', 'Mutilitated': 'Mutilated',
                    'Obumblers': 'bumblers', 'negotiatiations': 'negotiations', 'dood-': 'dood', 'irakis': 'iraki',
                    'cooerate': 'cooperate', 'COx': 'cox', 'racistcomments': 'racist comments',
                    'envirnmetalists': 'environmentalists', }
    avg_glove_vector_840b300d = [0.22418134, -0.28881392, 0.13854356, 0.00365387, -0.12870757, 0.10243822, 0.061626635,
                                 0.07318011, -0.061350107, -1.3477012, 0.42037755, -0.063593924, -0.09683349,
                                 0.18086134, 0.23704372, 0.014126852, 0.170096, -1.1491593, 0.31497982, 0.06622181,
                                 0.024687296, 0.076693475, 0.13851812, 0.021302193, -0.06640582, -0.010336159,
                                 0.13523154, -0.042144544, -0.11938788, 0.006948221, 0.13333307, -0.18276379,
                                 0.052385733, 0.008943111, -0.23957317, 0.08500333, -0.006894406, 0.0015864656,
                                 0.063391194, 0.19177166, -0.13113557, -0.11295479, -0.14276934, 0.03413971,
                                 -0.034278486, -0.051366422, 0.18891625, -0.16673574, -0.057783455, 0.036823478,
                                 0.08078679, 0.022949161, 0.033298038, 0.011784158, 0.05643189, -0.042776518,
                                 0.011959623, 0.011552498, -0.0007971594, 0.11300405, -0.031369694, -0.0061559738,
                                 -0.009043574, -0.415336, -0.18870236, 0.13708843, 0.005911723, -0.113035575,
                                 -0.030096142, -0.23908928, -0.05354085, -0.044904727, -0.20228513, 0.0065645403,
                                 -0.09578946, -0.07391877, -0.06487607, 0.111740574, -0.048649278, -0.16565254,
                                 -0.052037314, -0.078968436, 0.13684988, 0.0757494, -0.006275573, 0.28693774,
                                 0.52017444, -0.0877165, -0.33010918, -0.1359622, 0.114895485, -0.09744406, 0.06269521,
                                 0.12118575, -0.08026362, 0.35256687, -0.060017522, -0.04889904, -0.06828978,
                                 0.088740796, 0.003964443, -0.0766291, 0.1263925, 0.07809314, -0.023164088, -0.5680669,
                                 -0.037892066, -0.1350967, -0.11351585, -0.111434504, -0.0905027, 0.25174105,
                                 -0.14841858, 0.034635577, -0.07334565, 0.06320108, -0.038343467, -0.05413284,
                                 0.042197507, -0.090380974, -0.070528865, -0.009174437, 0.009069661, 0.1405178,
                                 0.02958134, -0.036431845, -0.08625681, 0.042951006, 0.08230793, 0.0903314, -0.12279937,
                                 -0.013899368, 0.048119213, 0.08678239, -0.14450377, -0.04424887, 0.018319942,
                                 0.015026873, -0.100526, 0.06021201, 0.74059093, -0.0016333034, -0.24960588,
                                 -0.023739101, 0.016396184, 0.11928964, 0.13950661, -0.031624354, -0.01645025,
                                 0.14079992, -0.0002824564, -0.08052984, -0.0021310581, -0.025350995, 0.086938225,
                                 0.14308536, 0.17146006, -0.13943303, 0.048792403, 0.09274929, -0.053167373,
                                 0.031103406, 0.012354865, 0.21057427, 0.32618305, 0.18015954, -0.15881181, 0.15322933,
                                 -0.22558987, -0.04200665, 0.0084689725, 0.038156632, 0.15188617, 0.13274793,
                                 0.113756925, -0.095273495, -0.049490947, -0.10265804, -0.27064866, -0.034567792,
                                 -0.018810693, -0.0010360252, 0.10340131, 0.13883452, 0.21131058, -0.01981019,
                                 0.1833468, -0.10751636, -0.03128868, 0.02518242, 0.23232952, 0.042052146, 0.11731903,
                                 -0.15506615, 0.0063580726, -0.15429358, 0.1511722, 0.12745973, 0.2576985, -0.25486213,
                                 -0.0709463, 0.17983761, 0.054027, -0.09884228, -0.24595179, -0.093028545, -0.028203879,
                                 0.094398156, 0.09233813, 0.029291354, 0.13110267, 0.15682974, -0.016919162, 0.23927948,
                                 -0.1343307, -0.22422817, 0.14634751, -0.064993896, 0.4703685, -0.027190214, 0.06224946,
                                 -0.091360025, 0.21490277, -0.19562101, -0.10032754, -0.09056772, -0.06203493,
                                 -0.18876675, -0.10963594, -0.27734384, 0.12616494, -0.02217992, -0.16058226,
                                 -0.080475815, 0.026953284, 0.110732645, 0.014894041, 0.09416802, 0.14299914,
                                 -0.1594008, -0.066080004, -0.007995227, -0.11668856, -0.13081996, -0.09237365,
                                 0.14741232, 0.09180138, 0.081735, 0.3211204, -0.0036552632, -0.047030564, -0.02311798,
                                 0.048961394, 0.08669574, -0.06766279, -0.50028914, -0.048515294, 0.14144728,
                                 -0.032994404, -0.11954345, -0.14929578, -0.2388355, -0.019883996, -0.15917352,
                                 -0.052084364, 0.2801028, -0.0029121689, -0.054581646, -0.47385484, 0.17112483,
                                 -0.12066923, -0.042173345, 0.1395337, 0.26115036, 0.012869649, 0.009291686,
                                 -0.0026459037, -0.075331464, 0.017840583, -0.26869613, -0.21820338, -0.17084768,
                                 -0.1022808, -0.055290595, 0.13513643, 0.12362477, -0.10980586, 0.13980341, -0.20233242,
                                 0.08813751, 0.3849736, -0.10653763, -0.06199595, 0.028849555, 0.03230154, 0.023856193,
                                 0.069950655, 0.19310954, -0.077677034, -0.144811]
    avg_glove_vector_6b50d = [
        -0.12920076, -0.28866628, -0.01224866, -0.05676644, -0.20210965, -0.08389011,
        0.33359843, 0.16045167, 0.03867431, 0.17833012, 0.04696583, -0.00285802,
        0.29099807, 0.04613704, -0.20923874, -0.06613114, -0.06822549, 0.07665912,
        0.3134014, 0.17848536, -0.1225775, -0.09916984, -0.07495987, 0.06413227,
        0.14441176, 0.60894334, 0.17463093, 0.05335403, -0.01273871, 0.03474107,
        -0.8123879, -0.04688699, 0.20193407, 0.2031118, -0.03935686, 0.06967544,
        -0.01553638, -0.03405238, -0.06528071, 0.12250231, 0.13991883, -0.17446303,
        -0.08011883, 0.0849521, -0.01041659, -0.13705009, 0.20127155, 0.10069408,
        0.00653003, 0.01685157]
    avg_fasttext_2m300d = [-1.79720020e-02, 4.59612617e-02, 2.44339478e-03, 1.14453539e-01, -2.44256777e-02,
                           2.95815698e-03, 1.03025117e-01, -8.68739766e-02, -7.52062656e-02, 3.33940391e-02,
                           -2.02820195e-02, 1.04620800e+00, -8.15427891e-02, -2.37318223e-02, 1.08383076e-02,
                           1.23609346e-02, -7.79699414e-03, -2.23746934e-02, -7.49582109e-02, -3.71978281e-02,
                           8.45225293e-03, 1.36429609e-02, 1.13141182e-02, 1.16137305e-02, 1.78923281e-02,
                           2.93113906e-02, 2.33791348e-02, 4.45499180e-02, 3.13387656e-02, -5.82022891e-02,
                           1.28034033e-02, 6.50548906e-02, -5.06797070e-02, 7.72173047e-02, -2.29646797e-02,
                           3.88520117e-02, 1.71532629e-03, 1.70338457e-02, 5.56611836e-02, -1.01905477e-01,
                           3.67041523e-02, 5.74241641e-02, -1.77972734e-02, 6.12034492e-02, -4.36416680e-02,
                           6.36411289e-02, -8.27315547e-02, 3.75490508e-02, -1.78396484e-02, -1.28496953e-01,
                           6.64232891e-02, 1.05493500e-01, 9.04765500e-01, -7.41661865e-03, -4.32927813e-02,
                           -2.54867422e-02, -8.73220410e-03, 2.10639160e-02, 3.87381523e-02, -3.23720020e-02,
                           2.68976562e-02, 9.85046641e-02, 8.94975879e-03, 7.13236797e-02, 6.74019141e-02,
                           -4.23013594e-02, -3.24311895e-02, -2.18652187e-02, 1.22457446e-03, -1.69734023e-02,
                           5.71789961e-02, 1.75010547e-02, 5.41422656e-02, -1.03576953e-01, 2.51064453e-02,
                           5.84627305e-02, -2.65568812e-01, 6.46930391e-02, -1.34845484e-01, -1.91715762e-02,
                           3.80878375e-01, 2.69076562e-02, -3.52429258e-02, -1.69530273e-02, 1.07793914e-01,
                           -1.86758438e-02, -2.67136758e-02, -3.22055469e-02, 5.86570469e-02, 7.08526406e-02,
                           -2.68723008e-02, 1.92197484e-01, 2.70301465e-02, -1.94913164e-02, 4.47472500e-02,
                           4.61226680e-02, -4.06695801e-03, 9.14075156e-02, 2.77345449e-02, 4.14191602e-03,
                           2.13715762e-02, -2.92259395e-02, -4.92994687e-02, -1.45244111e-02, 8.72082344e-02,
                           7.49558438e-02, 2.61175366e-03, -6.05494961e-02, 1.71665586e-02, 1.59877451e-02,
                           -1.20828984e-01, -7.45510400e-03, 6.95515313e-02, -3.91321211e-02, -7.08964625e-01,
                           2.03243242e-02, -8.41571250e-02, 4.52808047e-02, -9.51013281e-02, 3.80058750e-02,
                           -2.27888809e-02, -1.18743838e-02, 2.64994678e-03, 5.20781016e-02, -5.60906328e-02,
                           4.11016953e-02, 2.95097583e-03, -5.99409297e-02, 2.78802063e-01, -4.49910273e-02,
                           1.25225615e-02, 4.84114180e-02, -8.44276270e-03, 6.06409375e-02, -6.56986563e-02,
                           2.60665020e-02, 6.63916563e-02, 2.79978613e-02, 1.53083525e-02, -8.38284750e-01,
                           -9.80946797e-02, 7.07481953e-02, -1.79018281e-02, -1.96852203e-01, -4.31327344e-02,
                           9.00138281e-02, 6.33505508e-02, 8.09894727e-03, -5.04032266e-02, 6.08266172e-02,
                           -2.99341836e-02, 3.85930625e-02, -6.72227813e-02, 1.06172568e-02, 3.76806562e-02,
                           -2.82583340e-02, -7.97553162e-04, -4.15865273e-02, -4.67399883e-02, -1.20315713e-02,
                           6.93710000e-01, -1.15665117e-01, 7.59228271e-03, 2.35386504e-02, -4.73535625e-02,
                           -2.91041562e-02, 6.39238232e-03, -2.03673437e-02, -7.01801953e-02, -6.97110547e-02,
                           2.89453047e-02, -8.13781885e-03, -4.22333047e-02, 2.23304082e-02, 1.35199590e-02,
                           1.04268930e-01, 4.66349570e-02, -5.46767969e-02, 1.17977109e-02, -2.35034980e-02,
                           -6.54309766e-02, -6.37768789e-02, -3.06396738e-02, 3.84648682e-03, 1.32344512e-02,
                           5.18146641e-02, 2.33360156e-02, 6.76717187e-02, 4.41813789e-02, -9.02206797e-02,
                           2.00402237e-04, 1.68000547e-02, 7.26895703e-02, 5.82857695e-02, 7.91773535e-03,
                           7.33317109e-02, -8.23000625e-02, 3.43291914e-02, 2.13939355e-02, -8.97733582e-04,
                           1.62947686e-02, -3.40215308e-03, -3.65187622e-03, 1.13290557e-02, -2.23267852e-02,
                           7.43288281e-02, 2.72833008e-03, 3.15662461e-02, -4.44898008e-02, 1.59659607e-03,
                           -9.52254785e-03, 5.41178562e-01, -8.59935000e-02, -7.34877588e-03, -5.13457383e-02,
                           3.51231367e-02, -5.37677578e-02, 8.89518203e-02, -1.24034219e-02, -2.90076309e-02,
                           -4.45301445e-02, -1.53866531e-01, 1.51414297e-02, 3.40245996e-03, 4.36298945e-02,
                           -4.71652852e-02, -2.37308716e-03, -6.84343689e-04, -1.08431689e-02, 1.52163955e-02,
                           -6.94941641e-02, 5.52748096e-03, -4.93338906e-02, -5.39452246e-03, -3.05559512e-02,
                           -7.84589563e-01, 2.31532129e-02, 3.92418633e-02, 7.04897578e-02, 3.05945742e-02,
                           2.05342793e-02, -2.50711963e-03, -1.07908672e-02, -2.40677129e-02, 8.24847656e-02,
                           5.70925703e-02, -9.27704375e-02, -4.90966602e-03, -3.39621758e-02, -2.83528887e-02,
                           -4.09463789e-02, 3.32492944e-03, 2.37715723e-02, -4.22930117e-02, 4.90904443e-03,
                           4.00825508e-02, 5.94003555e-02, 3.44932539e-02, 7.00529297e-02, -2.54125977e-02,
                           -1.90498359e-02, -2.21944668e-02, -7.29946875e-02, 3.34532617e-02, 4.51727422e-02,
                           -3.81429956e-03, 5.02195352e-02, -5.68943359e-03, 5.14126758e-02, -5.58987656e-02,
                           2.53255840e-02, -5.08024570e-02, 2.10721797e-02, 1.08110758e-01, -5.11616162e-03,
                           -3.50598164e-02, 2.98414863e-02, 5.91751641e-02, 9.42860840e-03, -4.89523594e-02,
                           -2.19117832e-02, -1.22236461e-01, -2.76476836e-02, -1.73402070e-02, -9.76283750e-02,
                           7.31549531e-02, -4.90989141e-02, 1.08870596e-02, -2.87949980e-02, -4.20323789e-02,
                           -2.64264844e-03, -3.77317969e-03, -3.13873340e-02, 1.04166969e-01, 8.90471562e-02,
                           8.65275000e-02, -4.02050195e-02, 6.80396250e-02, -2.37387500e-02, -1.00368828e-02]

    def __init__(self, prepare_emb=False):
        self.MAX_LEN = MAX_LEN  # check hist (log), you can know, most data is < 180 (99.1%), < 160 (95.2%), < 170 (97.4%), 140 (91.8%)
        self.embedding_matrix = None
        self.embedding_index = None
        self.vocab = None
        self.INPUT_DATA_DIR = '../input/jigsaw-unintended-bias-in-toxicity-classification/'

        self.test_df = None
        self.train_df = None
        self.df = None
        self._contraction_handled = False
        self._special_chars_handled = False
        self._text_preprocessed = False
        self.tokenizer = None

        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_aux_train = None

        self.test_df_id = None  # only id series is needed for generating submission csv file

        self.do_emb_matrix_preparation = prepare_emb

        #self.BIN_FOLDER="/proc/driver/nvidia/"
        self.BIN_FOLDER = "./"
        # test_df_id = None # for generating result, need id
        #self.E_M_FILE = self.BIN_FOLDER+"embedding.mat"
        #self.DATA_TRAIN_FILE = self.BIN_FOLDER+"emb_train_features.bin"
        #self.DATA_TEST_FILE = self.BIN_FOLDER+"emb_test_features.bin"

    @property
    def DATA_TEST_FILE(self):
        return self.BIN_FOLDER+"emb_test_features.bin"
    @property
    def DATA_TRAIN_FILE(self):
        return self.BIN_FOLDER+"emb_train_features.bin"
    @property
    def E_M_FILE(self):
        return self.BIN_FOLDER+"embedding.mat"

    def read_train_test(self, train_only=False):
        if self.train_df is None:
            try:
                self.train_df = pd.read_csv(self.INPUT_DATA_DIR + 'train.csv')
            except FileNotFoundError:
                if not os.path.isdir(self.INPUT_DATA_DIR):
                    self.INPUT_DATA_DIR = '/home/pengyu/works/input/jigsaw-unintended-bias-in-toxicity-classification/'
                if not os.path.isdir(self.INPUT_DATA_DIR):
                    self.INPUT_DATA_DIR = self.BIN_FOLDER
                if not os.path.isdir(self.INPUT_DATA_DIR):
                    self.INPUT_DATA_DIR = '/content/gdrivedata/My Drive/'

                self.train_df = pd.read_csv(self.INPUT_DATA_DIR + 'train.csv')
        if not train_only:
            self.test_df = pd.read_csv(self.INPUT_DATA_DIR + 'test.csv')
            self.test_df_id = self.test_df.id  # only id series is needed for generating submission csv file
            self.df = pd.concat([self.train_df.iloc[:, [0, 2]], self.test_df.iloc[:, :2]])

    def build_vocab(self, texts):
        if self.vocab:
            return  # no need to rebuild
        sentences = texts.apply(lambda x: x.split()).values  # for pandas data
        vocab = {}
        for sentence in sentences:
            for word in sentence:
                try:
                    vocab[word] += 1
                except KeyError:
                    vocab[word] = 1
        self.vocab = vocab

    def test_coverage(self):
        self.build_vocab(self.df[TEXT_COLUMN])  # test data text is known, too
        self.embedding_index = EmbeddingHandler.load_embeddings(EMBEDDING_FILES[0])
        EmbeddingHandler.check_coverage(self.vocab, self.embedding_index)

    def add_lower_to_embedding(self, embedding, vocab):
        count = 0
        for word in vocab:
            if word in embedding and word.lower() not in embedding:
                embedding[word.lower()] = embedding[word]
                count += 1
        lstm.logger.debug(f"Added {count} words to embedding")

    def contraction_normalize(self):
        def clean_contractions(text, mapping):
            specials = ["’", "‘", "´", "`"]
            for s in specials:
                text = text.replace(s, "'")
            text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])
            return text

        if not self._contraction_handled:
            if 'lowered_comment' not in self.df.columns:
                self.df['lowered_comment'] = self.df[TEXT_COLUMN].apply(lambda x: x.lower())
            self.df['treated_comment'] = self.df['lowered_comment'].apply(lambda x: clean_contractions(x,
                                                                                                       EmbeddingHandler.contraction_mapping))
            self._contraction_handled = True
        # vocab = build_vocab(df['treated_comment']) # about 13.51 coverage

    def special_chars_normalize(self):
        def clean_special_chars(text, punct, mapping):
            for p in mapping:
                text = text.replace(p, mapping[p])
            for p in punct:
                text = text.replace(p, f' {p} ')
            specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '',
                        'है': ''}  # Other special characters that I have to deal with in last
            for s in specials:
                text = text.replace(s, specials[s])
            return text

        if not self._special_chars_handled:
            if not self._contraction_handled:
                self.contraction_normalize()
            self.df['treated_comment'] = self.df['treated_comment'].apply(
                lambda x: clean_special_chars(x, EmbeddingHandler.punct, EmbeddingHandler.punct_mapping))
            self._special_chars_handled = True

    def spelling_normalize(self):
        def correct_spelling(x, dic):
            for word in dic.keys():
                x = x.replace(word, dic[word])
            return x

        if not self._special_chars_handled:
            self.special_chars_normalize()
        self.df['treated_comment'] = self.df['treated_comment'].apply(
            lambda x: correct_spelling(x, EmbeddingHandler.mispell_dict))

    # vocab = build_vocab(df['treated_comment'])
    # print("Glove : ")
    # oov_glove = check_coverage(vocab, embed_glove)

    @staticmethod
    def check_coverage(vocab, embeddings_index):
        import operator
        known_words = {}
        unknown_words = {}
        nb_known_words = 0
        nb_unknown_words = 0
        for word in vocab.keys():
            try:
                known_words[word] = embeddings_index[word]
                nb_known_words += vocab[word]
            except KeyError:
                unknown_words[word] = vocab[word]
                nb_unknown_words += vocab[word]
                pass

        print('Found embeddings for {:.2%} of vocab'.format(len(known_words) / len(vocab)))
        print('Found embeddings for  {:.2%} of all text'.format(nb_known_words / (nb_known_words + nb_unknown_words)))
        unknown_words = sorted(unknown_words.items(), key=operator.itemgetter(1))[::-1]

        return unknown_words

    def get_identity_df(self):
        if not self._text_preprocessed: # then we might be restore from numpy pickle file, so still need to read csv
            self.read_train_test()
            #for column in IDENTITY_COLUMNS :
            #    # it seems the .values will make a copy out, so it won't infect above sef.y_train
            #    self.train_df[column] = np.where(self.train_df[column] >= 0.5, True, False)
            #    # todo analyze >= 0.5 or not, what is the difference
            # refer to the paper, "This includes 450,000 comments annotated with the identities..."
        train_y_identity_df = self.train_df[IDENTITY_COLUMNS].dropna(how='all').fillna(0).astype(np.float32)
        return train_y_identity_df

    def get_identity_train_data_df_idx(self):
        train_y_identity_df = self.get_identity_df()
        return self.x_train[train_y_identity_df.index], train_y_identity_df.values, train_y_identity_df.index# non-binary

    def text_preprocess(self, target_binarize=True):
        lstm.logger.debug("Text preprocessing")
        if self._text_preprocessed:
            return  # just run once.... not a good flag
        self.spelling_normalize()  # will add 'treated_comment' column to df
        train_len = self.train_df.shape[0]
        self.train_df[TEXT_COLUMN] = self.df['treated_comment'][:train_len]
        self.test_df[TEXT_COLUMN] = self.df['treated_comment'][train_len:]

        if target_binarize:
            # todo consider multiple bin-split (one~ten star) for different subgroup
            for column in IDENTITY_COLUMNS + AUX_COLUMNS + [TARGET_COLUMN]:
                self.train_df[column] = np.where(self.train_df[column] >= 0.5, True, False)

        # handle binary target data
        self.y_train = self.train_df[TARGET_COLUMN].values
        self.y_aux_train = self.train_df[AUX_COLUMNS].values

        x_train = self.train_df[TEXT_COLUMN].astype(str)
        x_test = self.test_df[TEXT_COLUMN].astype(str)

        # tokenizer = text.Tokenizer(filters=CHARS_TO_REMOVE)
        tokenizer = text.Tokenizer()
        tokenizer.fit_on_texts(list(x_train) + list(x_test))
        x_train = tokenizer.texts_to_sequences(x_train)
        x_test = tokenizer.texts_to_sequences(x_test)
        self.x_train = sequence.pad_sequences(x_train, maxlen=self.MAX_LEN)  # longer will be truncated
        self.x_test = sequence.pad_sequences(x_test, maxlen=self.MAX_LEN)

        self.tokenizer = tokenizer
        self._text_preprocessed = True
        lstm.logger.debug("Text preprocessing Done")

    def dump_obj(self, obj, filename, fullpath=False, force=False):
        if not fullpath:
            path = self.BIN_FOLDER+filename
        else:
            path = filename
        if not force and os.path.isfile(path):
            print(f"{path} already existed, not dumping")
        else:
            print(f"Overwrite {path}!")
            pickle.dump(obj, open(path, 'wb'))

    def get_obj_and_save(self, filename, default=None, fullpath=False):
        if not fullpath:
            path = self.BIN_FOLDER+filename
        else:
            path = filename
        if os.path.isfile(path):
            return pickle.load(open(path, 'rb'))
        else:
            self.dump_obj(default, filename)
            return default

    def file_exist(self, filename, fullpath=False):
        if not fullpath:
            path = self.BIN_FOLDER+filename
        else:
            path = filename
        return os.path.isfile(path)

    def build_matrix_modify_comment(self, path, emb_matrix_exsited):
        """
        build embedding matrix given tokenizer word_index and pre-trained embedding file

        :param word_index: word_index from tokenizer
        :param path: path to load pre-trained embedding
        :return: embedding matrix
        """
        lstm.logger.debug(f'{path} is being processed')
        if emb_matrix_exsited:
            lstm.logger.debug("Start cooking embedding matrix and train/test data: only train/test data")
            self.text_preprocess()
            return  # only need process text

        lstm.logger.debug("Start cooking embedding matrix and train/test data")

        if path.find("840B.300d") > 0:
            emb_save_filename = 'matrix_840b'
        if path.find("300d-2M") > 0:
            emb_save_filename = 'matrix_crawl'

        emb_from_file = self.get_obj_and_save(emb_save_filename)
        if emb_from_file is not None:
            return emb_from_file


        if not self.file_exist('word_index'):
            self.build_vocab(self.df[TEXT_COLUMN])
            vocab = self.get_obj_and_save("vocab", self.vocab)  # word to integer value index
            self.text_preprocess()  # tokenizer processed in this function, not related to embedding
            lstm.logger.debug('Text processed')

            embedding_index = EmbeddingHandler.load_embeddings(path)  # embedding_index is an dict, value is the feature vector
            lstm.logger.debug(f"loading embedding from {path} done")
            self.add_lower_to_embedding(embedding_index, vocab)  # will change embedding_index, add lower words in the vocab to this embedding

            word_index = self.get_obj_and_save("word_index", self.tokenizer.word_index)  # word to integer value index
            embedding_matrix = np.zeros((len(word_index) + 1, 300))  # last one for unknown?
        else:
            lstm.logger.debug('Restore word index from files')
            word_index = self.get_obj_and_save("word_index")  # word to integer value index
            try:
                vocab = self.get_obj_and_save("vocab", self.vocab)  # word to integer value index
            except FileNotFoundError:
                vocab = self.vocab
                if vocab is None:
                    raise RuntimeError('vocab shoule be None, process embedding_index need it')

            embedding_index = EmbeddingHandler.load_embeddings(path)  # embedding_index is an dict, value is the feature vector
            lstm.logger.debug(f"loading embedding from {path} done")
            self.add_lower_to_embedding(embedding_index, vocab)  # will change embedding_index, add lower words in the vocab to this embedding

            embedding_matrix = np.zeros((len(word_index) + 1, 300))  # last one for unknown?

        if path.find("840B.300d") > 0:
            avg_vector = EmbeddingHandler.avg_glove_vector_840b300d
        if path.find("300d-2M") > 0:
            avg_vector = EmbeddingHandler.avg_fasttext_2m300d

        for word, i in word_index.items():
            try:
                embedding_matrix[i] = embedding_index[word]
            except KeyError:
                # for unk
                # https://stackoverflow.com/questions/49239941/what-is-unk-in-the-pretrained-glove-vector-files-e-g-glove-6b-50d-txt
                embedding_matrix[i] = avg_vector

        lstm.logger.debug(f'Done cooking embedding matrix for {path} with train/test words')
        self.dump_obj(embedding_matrix, emb_save_filename, force=True)

        return embedding_matrix

    @staticmethod
    def get_coefs(word, *arr):
        ## word embedding related
        return word, np.asarray(arr, dtype='float32')

    @staticmethod
    def load_embeddings(path):
        with open(path, 'r') as f:
            return dict(EmbeddingHandler.get_coefs(*line.strip().split(' ')) for line in f)

    def prepare_tfrecord_data(self, dump=True, train_test_data=True, embedding=True, action=None):  # 和上一级有耦合，先这样吧
        if action is not None and action == lstm.CONVERT_DATA_Y_NOT_BINARY:  # unpicker, change y
            self.read_train_test(train_only=True)
            if not os.path.isfile(DATA_FILE_FLAG):
                raise FileNotFoundError("Pickle files should be present")

            self.y_train = self.train_df[TARGET_COLUMN].values  # not preprocessed, so still be float value

            if dump:
                train_data = zip(self.x_train, self.y_train, self.y_aux_train)
                pickle.dump(train_data, open(self.DATA_TRAIN_FILE, 'wb'))  # overwrite with new

                # file flag
                with open(DATA_FILE_FLAG, 'wb') as f:
                    f.write(bytes("", 'utf-8'))
            return self.x_train, self.y_train, self.y_aux_train, self.x_test, self.embedding_matrix

        self.read_train_test()

        if os.path.isfile(DATA_FILE_FLAG) and not self.do_emb_matrix_preparation and not train_test_data and not embedding:
            # just recover from record file
            raise Exception('tfrecord should already existed, please check')

        if embedding:
            lstm.logger.debug("Still need build embedding matrix")
            self.embedding_matrix = np.concatenate(
                [self.build_matrix_modify_comment(f, emb_matrix_exsited=False) for f in EMBEDDING_FILES], axis=-1)
            del self.tokenizer
            if dump:
                if embedding:
                    pickle.dump(self.embedding_matrix, open(self.E_M_FILE, 'wb'))
            gc.collect()
            self.tokenizer = None
        elif train_test_data:  # no need to rebuild emb matrix, only need train test data
            f = EMBEDDING_FILES[0]
            self.build_matrix_modify_comment(f, emb_matrix_exsited=True)

        # create an embedding_matrix
        # after this, embedding_matrix is a matrix of size
        # len(tokenizer.word_index)+1   x 600
        # we concatenate two matrices, 600 = 300+300 (crawl+glove)
        train_data = zip(self.x_train, self.y_train, self.y_aux_train)  # this need text_preprocess being called
        # global embedding_matrix

        if dump:
            if train_test_data:
                pickle.dump(train_data, open(self.DATA_TRAIN_FILE, 'wb'))
                pickle.dump(self.x_test, open(self.DATA_TEST_FILE, 'wb'))

            # file flag
            with open(DATA_FILE_FLAG, 'wb') as f:
                f.write(bytes("", 'utf-8'))
        return self.x_train, self.y_train, self.y_aux_train, self.x_test, self.embedding_matrix

    def read_emb_data_from_input(self):
        emb_path='../input/jigsaw-embedding-matrix/embedding.mat'
        data_path='../input/jigsaw-vectors/emb_train_features.bin'
        test_data_path='../input/jigsaw-vectors/emb_test_features.bin'
        emb = self.get_obj_and_save(emb_path, fullpath=True)
        data = self.get_obj_and_save(data_path, fullpath=True)
        test_data = self.get_obj_and_save(test_data_path, fullpath=True)
        return emb, data, test_data

    def data_prepare(self, action=None):
        """Returns the iris dataset as (train_x, train_y), (test_x, test_y).
        we load this from the tfrecord, maybe save the ones just after embedding, so it can be faster
        """
        if action is not None: lstm.logger.debug("{} in data preparation".format(action))

        try:
        # just recover from record file
            emb, data_train, test_data = self.read_emb_data_from_input()
            self.x_train, self.y_train, self.y_aux_train = zip(*data_train)
            self.x_train, self.y_train, self.y_aux_train = np.array(self.x_train), np.array(self.y_train), np.array(self.y_aux_train)
            lstm.logger.debug("restored data from files for training")
            return self.x_train, self.y_train, self.y_aux_train, test_data, emb
        except FileNotFoundError:
            lstm.logger.debug('cannot restore emb, trainX from jigsaw kaggle file data')

        #if os.path.isfile(DATA_FILE_FLAG) and not self.do_emb_matrix_preparation:  # in final stage, no need to check this...
        if not self.do_emb_matrix_preparation:
            # global embedding_matrix
            if action is not None and action == lstm.DATA_ACTION_NO_NEED_LOAD_EMB_M:
                self.embedding_matrix = None
            else:
                try:
                    self.embedding_matrix = pickle.load(open(self.E_M_FILE, "rb"))
                except FileNotFoundError:
                    self.BIN_FOLDER = '/content/gdrivedata/My Drive/'
                    if not os.path.isdir(self.BIN_FOLDER):
                        self.BIN_FOLDER = './'
                        if not self.file_exist(self.E_M_FILE, fullpath=True):
                            self.BIN_FOLDER = '/proc/driver/nvidia/'
                    self.embedding_matrix = pickle.load(open(self.E_M_FILE, "rb"))


            if action is not None:  # exist data, need to convert data
                if action == lstm.CONVERT_TRAIN_DATA:
                    self.prepare_tfrecord_data(train_test_data=True, embedding=False, action=action) # train data will rebuild, so we put it before read from pickle


            try:
                data_train = pickle.load(open(self.DATA_TRAIN_FILE, "rb"))  # (None, 2048)
            except FileNotFoundError:
                self.BIN_FOLDER = '/content/gdrivedata/My Drive/'
                if not os.path.isdir(self.BIN_FOLDER):
                    self.BIN_FOLDER = './'
                data_train = pickle.load(open(self.DATA_TRAIN_FILE, "rb"))  # (None, 2048)

            self.x_test = pickle.load(open(self.DATA_TEST_FILE, "rb"))  # (None, 2048) 2048 features from xception net

            self.x_train, self.y_train, self.y_aux_train = zip(*data_train)
            self.x_train, self.y_train, self.y_aux_train = np.array(self.x_train), np.array(self.y_train), np.array(
                self.y_aux_train)

            # global test_df_id
            # test_df_id = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv').id

            try:
                self.test_df_id = pd.read_csv(self.INPUT_DATA_DIR + 'test.csv').id  # only id series is needed for generating submission csv file
            except FileNotFoundError:
                self.INPUT_DATA_DIR = '../input/'
                if not os.path.isdir(self.INPUT_DATA_DIR):
                    self.INPUT_DATA_DIR = '/home/pengyu/works/input/jigsaw-unintended-bias-in-toxicity-classification/'
                if not os.path.isdir(self.INPUT_DATA_DIR):
                    self.INPUT_DATA_DIR = self.BIN_FOLDER  # put same folder in google drive
                self.test_df_id = pd.read_csv(self.INPUT_DATA_DIR + 'test.csv').id  # only id series is needed for generating submission csv file

            if action is not None:  # exist data, need to convert data, so put after read from pickle
                if action == lstm.CONVERT_DATA_Y_NOT_BINARY:
                    self.prepare_tfrecord_data(train_test_data=False, embedding=False, action=action)  # train_test_data=False just not rebuild words, the y still need to change

            return self.x_train, self.y_train, self.y_aux_train, self.x_test, self.embedding_matrix
        else:
            # (x_train, y_train, y_aux_train), x_test = prepare_tfrecord_data()
            if action is not None and action == lstm.CONVERT_TRAIN_DATA:
                self.embedding_matrix = pickle.load(open(self.E_M_FILE, "rb"))
                lstm.logger.debug("Only build train test data, embedding loaded from pickle")
                return self.prepare_tfrecord_data(embedding=False, action=action)
            else:
                return self.prepare_tfrecord_data(embedding=True)
