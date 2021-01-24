import os
import sys
from glob import glob
from tokenizers import ByteLevelBPETokenizer

inputDIR = sys.argv[1]

tokenizer = ByteLevelBPETokenizer()
#tokenizer.add_tokens(['<PAD>','<SOS>','<EOS>','<BLANK>'])
print('path to files ->',inputDIR)
assert os.path.exists(inputDIR)
specialTokens = ['<PAD>','<SOS>','<EOS>','<BLANK>']

fields = {
    "X":"", "Y":"", "EQ":""
}

for field in fields:
    specialTokens.append('<SOS_'+field+'>')
    specialTokens.append('<EOS_'+field+'>')
print('The list of special tokens: ',specialTokens)
tokenizer.train(glob(inputDIR+'/*.json'), vocab_size=50000, special_tokens=specialTokens)
tokenizer.save("./bpe.tokenizer.json")
print('Tokenizer saved correctly!!')