#!/usr/bin/env python
# -*- coding: utf-8

def main():
    import sys
    import json
    import os
    import argparse
    import json
    import re
    import tensorflow.compat.v1 as tf
    import numpy as np
    from utils import printable_text, convert_to_unicode
    #from lm.modeling import GroverModel, GroverConfig, sample
    from modeling import GroverModel, GroverConfig, sample
    #from tokenization import tokenization
    from tokenizers import Tokenizer, ByteLevelBPETokenizer
    from tensorflow.python.util import deprecation
    try:
        from tensorflow.python.util import module_wrapper as deprecation
    except ImportError:
        from tensorflow.python.util import deprecation_wrapper as deprecation

    sys.path.append(os.getcwd())
    ##### ignore tf deprecated warning temporarily
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.logging.set_verbosity(tf.logging.DEBUG)
    deprecation._PRINT_DEPRECATION_WARNINGS = False
    deprecation._PER_MODULE_WARNING_LIMIT = 0
    #####

    parser = argparse.ArgumentParser(description='Contextual generation (aka given some metadata we will generate articles)')
    # parser.add_argument(
    #     '-metadata_fn',
    #     dest='metadata_fn',
    #     type=str,
    #     help='Path to a JSONL containing metadata',
    # )
    # parser.add_argument(
    #     '-out_fn',
    #     dest='out_fn',
    #     type=str,
    #     help='Out jsonl, which will contain the completed jsons',
    # )
    # parser.add_argument(
    #     '-input',
    #     dest='input',
    #     type=str,
    #     help='Text to complete',
    # )
    parser.add_argument(
        '-config_fn',
        dest='config_fn',
        default='configs/mega.json',
        type=str,
        help='Configuration JSON for the model',
    )
    parser.add_argument(
        '-ckpt_fn',
        dest='ckpt_fn',
        default='./models/mega/model.ckpt',
        type=str,
        help='checkpoint file for the model',
    )
    # parser.add_argument(
    #     '-target',
    #     dest='target',
    #     default='article',
    #     type=str,
    #     help='What to generate for each item in metadata_fn. can be article (body), title, etc.',
    # )
    parser.add_argument(
        '-batch_size',
        dest='batch_size',
        default=1,
        type=int,
        help='How many things to generate per context. will split into chunks if need be',
    )
    parser.add_argument(
        '-num_folds',
        dest='num_folds',
        default=1,
        type=int,
        help='Number of folds. useful if we want to split up a big file into multiple jobs.',
    )
    parser.add_argument(
        '-fold',
        dest='fold',
        default=0,
        type=int,
        help='which fold we are on. useful if we want to split up a big file into multiple jobs.'
    )
    parser.add_argument(
        '-max_batch_size',
        dest='max_batch_size',
        default=None,
        type=int,
        help='max batch size. You can leave this out and we will infer one based on the number of hidden layers',
    )
    parser.add_argument(
        '-top_p',
        dest='top_p',
        default=0.95,
        type=float,
        help='p to use for top p sampling. if this isn\'t none, use this for everthing'
    )
    parser.add_argument(
        '-min_len',
        dest='min_len',
        default=1024,
        type=int,
        help='min length of sample',
    )
    parser.add_argument(
        '-eos_token',
        dest='eos_token',
        default=-100000,
        type=int,
        help='eos token id',
    )
    parser.add_argument(
        '-samples',
        dest='samples',
        default=5,
        type=int,
        help='num_samples',
    )
    parser.add_argument(
        '-filters',
        dest='filters',
        default='',
        type=str,
        help='list of the filters separated by ;',
    )
    parser.add_argument(
        '-saveOutput',
        dest='saveOutput',
        action='store_true',
        help='weather save output or not',
    )

    parser.add_argument(
        '-context',
        dest='context',
        default='user',
        type=str,
        help='if we want to get input from the user or just generate data based on a context input',
    )

    def extract_generated_target(output_tokens, tokenizer):
        """
        Given some tokens that were generated, extract the target
        :param output_tokens: [num_tokens] thing that was generated
        :param encoder: how they were encoded
        :param target: the piece of metadata we wanted to generate!
        :return:
        """
        # Filter out first instance of start token
        assert output_tokens.ndim == 1

        start_ind = 0
        end_ind = output_tokens.shape[0]

        return {
            'extraction': printable_text(''.join(tokenizer.decode(output_tokens, skip_special_tokens=False))),
            'start_ind': start_ind,
            'end_ind': end_ind,
        }

    args = parser.parse_args()
    proj_root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    vocab_file_path = os.path.join(proj_root_path, "tokenization/clue-vocab.txt")
    #tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file_path , do_lower_case=True)
    tokenizer = Tokenizer.from_file("bpe.tokenizer.json")
    news_config = GroverConfig.from_json_file(args.config_fn)

    # We might have to split the batch into multiple chunks if the batch size is too large
    default_mbs = {12: 32, 24: 16, 48: 3}
    max_batch_size = args.max_batch_size if args.max_batch_size is not None else default_mbs[news_config.num_hidden_layers]

    # factorize args.batch_size = (num_chunks * batch_size_per_chunk) s.t. batch_size_per_chunk < max_batch_size
    num_chunks = int(np.ceil(args.batch_size / max_batch_size))
    batch_size_per_chunk = int(np.ceil(args.batch_size / num_chunks))

    # This controls the top p for each generation.
    top_p = np.ones((num_chunks, batch_size_per_chunk), dtype=np.float32) * args.top_p

    tf_config = tf.ConfigProto(allow_soft_placement=True)

    filterList = args.filters.split(';') if args.filters != '' else None
    saveFlag = args.saveOutput

    def runSample(text, num_chunks, sess, tokens, probs, batch_size_per_chunk, args, top_p, tokenizer, filterList):
        #line = tokenization.convert_to_unicode(text)
        #bert_tokens = tokenizer.tokenize(line)
        #encoded = tokenizer.convert_tokens_to_ids(bert_tokens)
        line = convert_to_unicode(text)
        encodedLine = tokenizer.encode(line)
        bert_tokens = encodedLine.tokens
        encodedIDS = encodedLine.ids 
        context_formatted = []
        context_formatted.extend(encodedIDS)

        gens = []
        gens_raw = []
        gen_probs = []

        for chunk_i in range(num_chunks):
            tokens_out, probs_out = sess.run([tokens, probs],
                                            feed_dict={initial_context: [context_formatted] * batch_size_per_chunk,
                                                        eos_token: args.eos_token, min_len: args.min_len,
                                                        p_for_topp: top_p[chunk_i]})

            for t_i, p_i in zip(tokens_out, probs_out):
                extraction = extract_generated_target(output_tokens=t_i, tokenizer=tokenizer)
                gens.append(extraction['extraction'])

        l = re.findall('.{1,70}', gens[0].replace('[UNK]', '').replace('##', ''))
        l = "\n".join(l)
        if filterList:
            lf = ''
            for f in filterList:
                startT = '<SOS_{}>'.format(f)
                startF = l.find(startT)
                endT = '<EOS_{}>'.format(f)
                endF = l.find(endT)
                #print('Debug: ',startT, endT, startF, endF)
                if startF != -1 and endF != -1 and startF < endF:
                    lf += l[startF+len(startT):endF]
                elif endF != -1:
                    lf += l[:endF] + '\n'
            l = lf  
        return l

    g = tf.Graph()
    sess = tf.Session(config=tf_config, graph=g)
    #with tf.Session(config=tf_config, graph=tf.Graph()) as sess:
    with g.as_default() as graph: 
        with sess.as_default() as sess:
            initial_context = tf.placeholder(tf.int32, [batch_size_per_chunk, None])
            p_for_topp = tf.placeholder(tf.float32, [batch_size_per_chunk])
            eos_token = tf.placeholder(tf.int32, [])
            min_len = tf.placeholder(tf.int32, [])
            tokens, probs = sample(news_config=news_config, initial_context=initial_context,
                                eos_token=eos_token, min_len=min_len, ignore_ids=None, p_for_topp=p_for_topp,
                                do_topk=False)

            saver = tf.train.Saver()
            saver.restore(sess, args.ckpt_fn)
            print('🍺Model loaded. \n')
            #text =  sys.stdin.readlines() #input()

            if args.context == 'user':
                print('Input something please:⬇️')
                prompt = [] 
                orignalInput = line = input("Input prompt ending with an empty line: ")
                while line:
                    prompt.append(line)
                    line = input() 
                text = "\n".join(prompt)
                text = text.replace('\\n', '\n')

                # ask if we want to save the output 
                if saveFlag:
                    i = 0 # args.samples

                    from datetime import datetime
                    now = datetime.now()
                    fileName = 'results_{}.txt'.format(now.strftime("%d%m%Y_%H%M%S"))
                    
                    # save in the output file
                    with open(fileName, 'w') as f:
                        f.write('Parameters:\n')
                        json.dump(args.__dict__, f, indent=2)
                        f.write('\nOriginal Input: {}\n'.format(orignalInput)) # is the orignal input 
                        f.write('Cherry Picked Results:\n')

                    print("--> Sample,", i + 1, " of ", args.samples)
                    lf = runSample(text, num_chunks, sess, tokens, probs, batch_size_per_chunk, args, top_p, tokenizer, filterList)
                    print(lf)

                    # Src: https://stackoverflow.com/questions/45188464/return-output-of-the-function-executed-on-click
                    from ipywidgets import widgets, Button, HBox
                    from IPython.display import display, clear_output, Javascript
                    from google.colab import files
                    from traitlets import traitlets
                    from IPython import get_ipython
                    import time
                    import io
                    import nbformat

                    class button(widgets.Button):
                        """A button that can holds a value as a attribute."""
                        def __init__(self, value=None, info=None, *args, **kwargs):
                            super(button, self).__init__(*args, **kwargs)
                            # Create the value attribute.
                            self.add_traits(value=traitlets.Any(value))
                            self.add_traits(info=traitlets.Any(info))

                    def clicked(ex):
                        clear_output()
                        lf, text, num_chunks, sess, tokens, probs, batch_size_per_chunk, args, top_p, tokenizer, filterList, i, orignalInput, fileName = ex.info
                        if ex.value: # User like the result
                            # save in a file
                            #lf is the generate sample
                            with open(fileName, 'a') as f:
                                f.write('\n --- \n')
                                f.write('{}'.format(lf.replace(';','\n')))

                        print("--> Sample,", i + 1, " of ", args.samples)
                        lf = runSample(text, num_chunks, sess, tokens, probs, batch_size_per_chunk, args, top_p, tokenizer, filterList)
                        print(lf)
                        i += 1
                    
                        #clear_output(wait=True)
                        likeButton = button(description="Save", value=True, info=[lf, text, num_chunks, sess, tokens, probs, batch_size_per_chunk, args, top_p, tokenizer, filterList, i, orignalInput, fileName])
                        disLikeButton = button(description="Ignore", value=False, info=[lf, text, num_chunks, sess, tokens, probs, batch_size_per_chunk, args, top_p, tokenizer, filterList, i, orignalInput, fileName])
                        likeButton.on_click(clicked)
                        disLikeButton.on_click(clicked)
                        display(HBox([likeButton,disLikeButton]))

                    likeButton = button(description="Save", value=True, info=[lf, text, num_chunks, sess, tokens, probs, batch_size_per_chunk, args, top_p, tokenizer, filterList, i, orignalInput, fileName])
                    disLikeButton = button(description="Ignore", value=False, info=[lf, text, num_chunks, sess, tokens, probs, batch_size_per_chunk, args, top_p, tokenizer, filterList, i, orignalInput, fileName])
                    likeButton.on_click(clicked)
                    disLikeButton.on_click(clicked)
                    #clear_output(wait=True)
                    display(HBox([likeButton,disLikeButton]))
                else:
                    while text != "":
                        for i in range(args.samples):
                            print("Sample,", i + 1, " of ", args.samples)
                            lf = runSample(text, num_chunks, sess, tokens, probs, batch_size_per_chunk, args, top_p, tokenizer, filterList)
                            print(lf)

                        print('Next try:⬇️')
                        #text =  sys.stdin.readlines() #input()
                        prompt = [] 
                        line = input("Input prompt ending with an empty line: ")
                        while line:
                            prompt.append(line)
                            line = input()

                        text = "\n".join(prompt)
                        text = text.replace('\\n', '\n')
            else:
                result = []
                text = args.context
                if text[0] == '[':
                    text = eval(text) # it should be a list now
                    for t in text:
                        res = runSample(t, num_chunks, sess, tokens, probs, batch_size_per_chunk, args, top_p, tokenizer, filterList)
                        result.append(res)
                else: # only single sample
                    res = runSample(text, num_chunks, sess, tokens, probs, batch_size_per_chunk, args, top_p, tokenizer, filterList)
                    result.append(res)
                return result
def wraper(top_p, config_fn, ckpt_fn, min_len, sample_num, saveFlag, filters, context='user'):
    import sys
    sys.argv = ['', '-context', '{}'.format(context), '-config_fn', '{}'.format(config_fn) ,'-ckpt_fn', '{}'.format(ckpt_fn), '-min_len', '{}'.format(min_len), '-samples', '{}'.format(sample_num), '-eos_token', '{}'.format(-10000000), '-top_p', '{}'.format(top_p)] 
    if saveFlag:
        sys.argv += ['-saveOutput']
    if filters != '':
        sys.argv += ['-filters', '{}'.format(filters)]
    return main()

if __name__ == "__main__":
    main()
