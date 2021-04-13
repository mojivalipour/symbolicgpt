"""
# Inspired by https://github.com/imcaspar/gpt2-ml
Turn a merged corpus into tfrecord files.

NOTE: You will want to do this using several processes. I did this on an AWS machine with 72 CPUs using GNU parallel
as that's where I had the deduplicated RealNews dataset.
"""
import argparse
import ujson as json
# from sample.encoder import get_encoder, tokenize_for_grover_training, detokenize, sliding_window, create_int_feature
import random
import tensorflow.compat.v1 as tf
import collections
import os
from tempfile import TemporaryDirectory

#from tokenization import tokenization
from tokenizers import Tokenizer, ByteLevelBPETokenizer
from glob import glob

parser = argparse.ArgumentParser(description='SCRAPE!')
parser.add_argument(
    '-fold',
    dest='fold',
    default=0,
    type=int,
    help='which fold we are on'
)
parser.add_argument(
    '-num_folds',
    dest='num_folds',
    default=1,
    type=int,
    help='Number of folds (corresponding to both the number of training files and the number of testing files)',
)
parser.add_argument(
    '-seed',
    dest='seed',
    default=1337,
    type=int,
    help='which seed to use'
)
parser.add_argument(
    '-base_fn',
    dest='base_fn',
    default='news2016zh_',
    type=str,
    help='We will output files that are like {base_fn}_{n}.tfrecord for n in 0, ..., 1023'
)

parser.add_argument(
    '-input_fn',
    dest='input_fn',
    default='realnews.jsonl',
    type=str,
    help='Base filename to use. THIS MUST BE A LOCAL FILE.'
)
parser.add_argument(
    '-max_seq_length',
    dest='max_seq_length',
    default=1025,
    type=int,
    help='Max sequence length',
)

parser.add_argument(
    '-max_num_points',
    dest='max_num_points',
    default=30,
    type=int,
    help='Max number of points for the pointNET',
)

parser.add_argument(
    '-max_num_vars',
    dest='max_num_vars',
    default=5,
    type=int,
    help='Max number of variables in the input x',
)

parser.add_argument(
    '-modelType',
    dest='modelType',
    default='GPT2',
    type=str,
    help='The type of the model that use the data. GPT2/PT'
)

args = parser.parse_args()
random.seed(args.seed + args.fold)

tokenizer = Tokenizer.from_file("./bpe.tokenizer.json")

class TFRecordWriter(object):
    def __init__(self, fn):
        self.fn = fn
        if fn.startswith('gs://'):
            from google.cloud import storage
            self.s3client = None
            self.gclient = storage.Client()
            self.storage_dir = TemporaryDirectory()
            self.writer = tf.python_io.TFRecordWriter(
                os.path.join(self.storage_dir.name, 'temp.tfrecord'))
            self.bucket_name, self.file_name = self.fn.split(
                'gs://', 1)[1].split('/', 1)

        else:
            self.s3client = None
            self.gclient = None
            self.bucket_name = None
            self.file_name = None
            self.storage_dir = None
            self.writer = tf.python_io.TFRecordWriter(fn)

    def write(self, x):
        self.writer.write(x)

    def close(self):
        self.writer.close()

        if self.gclient is not None:
            bucket = self.gclient.get_bucket(self.bucket_name)
            blob = bucket.blob(self.file_name)
            blob.upload_from_filename(os.path.join(
                self.storage_dir.name, 'temp.tfrecord'))
            self.storage_dir.cleanup()

    def __enter__(self):
        # Called when entering "with" context.
        return self

    def __exit__(self, *_):
        # Called when exiting "with" context.
        # Upload shit
        print("CALLING CLOSE")
        self.close()


def article_iterator(tokenizer, final_desired_size=1025):
    """ Iterate through the provided filename + tokenize"""
    assert os.path.exists(args.input_fn)
    for (dirpath, dirnames, filenames) in os.walk(args.input_fn):
        for filename in filenames:
            with open(os.path.join(dirpath, filename), 'r') as f:
                for l_no, l in enumerate(f):
                    if l_no % args.num_folds == args.fold:
                        if '[]' in l or '}' != l[-2]: # ignore samples without x information
                            continue

                        # NaN/Infinity
                        l = l.replace('Infinity','0')
                        l = l.replace('NaN','0')

                        try:
                            article = json.loads(l)
                        except Exception as error:
                            print('\n-->', l, '\n', error)

                        if args.modelType == 'PT':
                            x = [e+[0]*(args.max_num_vars-len(e)) for e in article.pop("X")] # [[1..n],[1..n]]
                            y = article.pop("Y") # [1,2]
                            # Convert X and Y to [[1..n+1],[1..n+1]]
                            article['input_points'] = list(map(lambda x,y:x+[y],x,y)) + [[0]*len(x[0]+[y[0]])]*args.max_num_points #list(zip(x,y))

                        tokens, article['input_ids'] = tokenize_for_grover_training(tokenizer, article, desired_size=final_desired_size,
                                                                    unconditional_prob=.35)

                        article['inst_index'] = (l_no // args.num_folds)
                        if article['inst_index'] < 100:
                            print('---\nFilename{}INPUT{}. {}\n---\nTokens: {}\n'.format(
                                                                            filename,
                                                                            article['inst_index'],
                                                                            tokens,
                                                                            article['input_ids']
                                                                            ), flush=True)
                        if len(article['input_ids']) <= 10:  # min size of article
                            continue

                        yield article

def _tokenize_article_pieces(tokenizer, item):
    """
    Turn the article into tokens

    :param item: Contains things that need to be tokenized

    fields are {
    "X":"", "Y":"", "EQ":""
    }
    :return: dict
    """
    #article_pieces_ids = {'X':[]}
    #article_pieces_tokens = {'X':[]}
    article_pieces_ids = {}
    article_pieces_tokens = {}
    for itm in item:
        if len(item[itm]) != 0:
            line = tokenizer.encode(str(item[itm]))
            article_pieces_ids[itm] = [tokenizer.token_to_id('<SOS_'+itm+'>')] + line.ids + [tokenizer.token_to_id('<EOS_'+itm+'>')]
            article_pieces_tokens[itm] = ['<SOS_'+itm+'>'] + line.tokens + ['<EOS_'+itm+'>']
    return article_pieces_tokens, article_pieces_ids

memory = {}
memoryLimit = 1000

def tokenize_for_grover_training(tokenizer, item, desired_size=1024, unconditional_prob=0.35, metadata_dropout_prob=0.1,cut_prob=0.2, stratified=False):
    """
    Not only will we tokenize an item with a BPE encoder, but we'll also put it in a nice format for language modeling.
    The goal is to MINIMIZE PADDING. If we don't fill up the desired size of 1024 tokens then we're wasting compute.

    The canonical order is

    DOMAIN DATE AUTHORS TITLE ARTICLE SUMMARY

    :param encoder:
    :param item: Contains things like
          {"url": "https://www.advocate.com/node/1010911",
          "timestamp": "20180118211607",
           "url_used": "https://web.archive.org/web/20180118211607id_/https://www.advocate.com/node/1010911",
           "domain": "advocate.com",
           "title": "Report: One-Third of Trump's Judicial Picks Are Anti-LGBT",
           "text": ....
           "summary": ....
           "authors": list
           "publish_date": ...
           }
    {
    "X":"", "Y":"", "EQ":""
    }

    :param desired_size: the goal for how long the span will be
    :param unconditional_prob: The probability that we will generate JUST THE TEXT first.
    :param metadata_dropout_prob: The probability that we will drop out each item of metadata
    :param cut_prob: The probability that, if we're already over the desired size, we'll cut the article and start
                    predicting metadata before the desired_size window ends.
    :return:
    """
    # Get all the bits and pieces
    article_pieces_tokens, article_pieces_ids = _tokenize_article_pieces(tokenizer, item)
    #canonical_metadata_order = ['X', 'Y', 'EQ'] #list(article_pieces_ids) if article_pieces_ids is not None else []
    canonical_metadata_order = ['Y', 'EQ']

    chunk_a = article_pieces_ids.pop('Y')
    chunk_b = article_pieces_ids.pop('EQ')

    # remove those keys that are not in our interest
    keysList = list(article_pieces_tokens.keys())
    for key in keysList:
        if key not in canonical_metadata_order:
            article_pieces_tokens.pop(key)

    # if stratified:
    #     #TODO: add a stratified strategy
    #     # keep the last 200 samples and sample one variation of that specific bucket
    #     # if item['type'] in memory.keys():
    #     #     if item['domain'] in memory[item['type']]['samples'].keys():

    #     #     else:
    #     #         memory[item['type']]['samples'][item['domain']] = [(article_pieces_tokens, article_pieces_ids)]

    #     # else:
    #     #     # this is the first item
    #     #     memory[item['type']] = {'samples':{},'size':0}
    #     #     memory[item['type']]['samples'][item['domain']] = [(article_pieces_tokens, article_pieces_ids)]
    #     #     memory[item['type']]['size'] += 1
    #     pass
    
    # unconditional_prob is probability we only generate the text first, without any metadata
    # switch = random.random()
    # if switch < unconditional_prob:
    #     assignments = {'EQ': 'a'}
    #     canonical_metadata_order_copy = canonical_metadata_order.copy()
    #     chunk_a = article_pieces_ids.pop('EQ')
    #     chunk_b = []
    #     for x in canonical_metadata_order_copy.pop(2): # 2 is the index for "EQ"
    #         if random.random() > metadata_dropout_prob:
    #             chunk_b.extend(article_pieces_ids.pop(x, []))
    #             assignments[x] = 'b'
    # elif switch < 0.5:
    #     # Put everything in chunk_a, without dropout
    #     assignments = {}
    #     chunk_a = []
    #     chunk_b = []
    #     for x in canonical_metadata_order:
    #         chunk_a.extend(article_pieces_ids.pop(x, []))
    #         assignments[x] = 'a'
    # else:
    #     assignments = {}
    #     chunk_a = []
    #     chunk_b = []
    #     for k in canonical_metadata_order:
    #         if random.random() < metadata_dropout_prob and k not in ('EQ'):
    #             pass
    #         elif random.random() < 0.5:
    #             # if k != 'summary':
    #             chunk_a.extend(article_pieces_ids.pop(k, []))
    #             assignments[k] = 'a'
    #         else:
    #             chunk_b.extend(article_pieces_ids.pop(k, []))
    #             assignments[k] = 'b'
    
    if (len(chunk_a) + len(chunk_b)) <= desired_size:
        return article_pieces_tokens,chunk_a + chunk_b
    
    if (len(chunk_b) > 0) and (random.random() < cut_prob): # (assignments.get('EQ', '') == 'a') and 
       return article_pieces_tokens,_cut_tokens_to_add_stuff(chunk_a, chunk_b, desired_size, tokenizer.token_to_id('<PAD>'))
    
    tokens_ids = chunk_a + chunk_b
    return article_pieces_tokens, tokens_ids

def _cut_tokens_to_add_stuff(tokens, stuff_to_add, desired_size, padding_token):
    """
    The idea behind this function is to take away tokens from `tokens' such that tokens[:LENGTH] + stuff_to_add becomes
    exactly at the right size (desired_size).

    :param tokens:
    :param stuff_to_add:
    :param desired_size:
    :return:
    """
    if len(tokens) >= desired_size:
        return tokens

    # no way we can add this stuff
    if len(stuff_to_add) >= desired_size:
        return tokens

    if (len(tokens) + len(stuff_to_add)) <= desired_size:
        return tokens + stuff_to_add

    # Otherwise we'll have to actually cut
    tokens = tokens[:(desired_size - len(stuff_to_add) - 1)]
    tokens.append(padding_token)
    tokens.extend(stuff_to_add)
    return tokens

def create_int_feature(values):
    feature = tf.train.Feature(
        int64_list=tf.train.Int64List(value=list(values)))
    return feature

def create_float_feature(values):
    # The list has been flatten to be able to use FloatList protocol
    feature = tf.train.Feature(
        float_list=tf.train.FloatList(value=[item for l in values for item in l])) # TODO: remember to reshape it back to 2D ([numPoints, numVars+1]) in the modelling code
    return feature

def buffered_and_sliding_window_article_iterator(tokenizer, final_desired_size=1025, final_desired_points=30):
    """ We apply a sliding window to fix long sequences, and use a buffer that combines short sequences."""
    for article in article_iterator(tokenizer):
        if len(article['input_ids']) >= final_desired_size:
            article['input_ids'] = article['input_ids'][0:final_desired_size-1]
        while len(article['input_ids']) < final_desired_size:
            article['input_ids'].append(0)

        if args.modelType == 'PT': # if we are using pointNET
            if len(article['input_points']) >= final_desired_points:
                article['input_points'] = article['input_points'][0:final_desired_points-1]
            while len(article['input_points']) < final_desired_points:
                article['input_points'].append([0]*len(article['input_points'][0])) 

        yield article

# OK now write the tfrecord file
total_written = 0
train_file = args.base_fn + 'train_{:04d}.tfrecord'.format(args.fold)
with TFRecordWriter(train_file) as train_writer:
    for article in buffered_and_sliding_window_article_iterator(tokenizer,
                                                                final_desired_size=args.max_seq_length + 1,
                                                                final_desired_points=args.max_num_points + 1):
        writer2use = train_writer
        assert len(article['input_ids']) == (args.max_seq_length + 1)

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(article['input_ids'])

        if args.modelType == 'PT': # if we are using pointNET
            assert len(article['input_points']) == (args.max_num_points + 1)
            features["input_points"] = create_float_feature(article['input_points'])

        tf_example = tf.train.Example(
            features=tf.train.Features(feature=features))

        writer2use.write(tf_example.SerializeToString())
        total_written += 1

        # DEBUG
        if article['inst_index'] < 5:
            # print("~~~\nIndex {}. ARTICLE: {}\n---\nTokens: {}\n\n".format(article['inst_index'],
            #                                                                tokenizer.convert_ids_to_tokens(
            #                                                                    article['input_ids']),
            #                                                                article['input_ids']
            #                                                                ), flush=True)
            print("~~~\nIndex {}. ARTICLE: {}\n---\nTokens: {}\nPoints: {}\n\n".format(article['inst_index'],
                                                                           tokenizer.decode(
                                                                           article['input_ids'], skip_special_tokens=False),
                                                                           article['input_ids'],
                                                                           article['input_points'] if args.modelType == 'PT' else ''
                                                                           ), flush=True)
        if article['inst_index'] % 1000 == 0:
            print("{} articles, {} written".format(
                article['inst_index'], total_written), flush=True)
print("DONE UPLOADING", flush=True)
