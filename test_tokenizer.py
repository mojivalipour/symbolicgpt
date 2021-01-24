import os
import sys
from glob import glob
from tokenizers import ByteLevelBPETokenizer
from tokenizers import Tokenizer

inputDIR = sys.argv[0]

tokenizer = Tokenizer.from_file("./bpe.tokenizer.json")

sentence = '<SOS_X>[[0.1], [0.2], [0.31], [0.41], [0.51], [0.62], [0.72], [0.82], [0.93], [1.03], [1.13], [1.24], [1.34], [1.44], [1.55], [1.65], [1.76], [1.86], [1.96], [2.07], [2.17], [2.27], [2.38], [2.48], [2.58], [2.69], [2.79], [2.89], [3.0], [3.1]]<EOS_X><SOS_Y>[-4.61, -3.22, -2.34, -1.78, -1.35, -0.96, -0.66, -0.4, -0.15, 0.06, 0.24, 0.43, 0.59, 0.73, 0.88, 1.0, 1.13, 1.24, 1.35, 1.46, 1.55, 1.64, 1.73, 1.82, 1.9, 1.98, 2.05, 2.12, 2.2, 2.26]<EOS_Y><SOS_EQ>2*log(x1)<EOS_EQ>'
print(sentence)
encoded = tokenizer.encode(sentence)
print(encoded.ids)
print(encoded.tokens)
print(tokenizer.decode(encoded.ids))


sentence = '<SOS_EQ>-sin(1.96/x1)<EOS_EQ>'
print(sentence)
encoded = tokenizer.encode(sentence)
print(encoded.ids)
print(encoded.tokens)
print(tokenizer.decode(encoded.ids))

print('--> <EOS>: ',tokenizer.token_to_id('<EOS>'))

specialTokens = ['<PAD>','<SOS>','<EOS>','<BLANK>']
fields = {
    "X":"", "Y":"", "EQ":""
}
for field in fields:
    specialTokens.append('<SOS_'+field+'>')
    specialTokens.append('<EOS_'+field+'>')

for token in specialTokens:
    print('--> {}:{} \n'.format(token,tokenizer.token_to_id(token)))
