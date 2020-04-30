import os,sys,time
import tensorflow as tf
import pickle
import argparse
sys.path.insert(0, './src')
import e2e_model_100emb_frame as nn_model_foreval
import Feature_extraction as fe

# import embdedding_object as nobj

###########################
# Global Variable Initialization

# Variable Initialization DID-5
FEAT_TYPE = 'logmel'
N_FFT = 400
HOP = 160
VAD = True
CMVN = 'mv'
EXCLUDE_SHORT=0
IS_BATCHNORM = False
IS_TRAINING = False
INPUT_DIM = 40

softmax_num = 5


#init placeholder
x = tf.placeholder(tf.float32, [None,None,40])
y = tf.placeholder(tf.int32, [None])
s = tf.placeholder(tf.int32, [None,2])


###########################
# Model Architechture

emnet_validation = nn_model_foreval.nn(x,y,y,s,softmax_num,IS_TRAINING,INPUT_DIM,IS_BATCHNORM);
sess = tf.InteractiveSession()
saver = tf.train.Saver()
tf.initialize_all_variables().run()

### Loading neural network 
saver.restore(sess,'./pretrained_models/model3996000.ckpt-3996000')

### IO

parser = argparse.ArgumentParser(description="Extract frame-level embeddings from wav list. The pipeline outputs an embedding object with wavid and corresponding frame", add_help=True)
parser.add_argument("--wavlist", type=str, default="data/test.txt",help="wav list filename")
# parser.add_argument("--outputlayer", action='store_true', help="(option) use if you want to extract output also")
args = parser.parse_known_args()[0]

lines = open(args.wavlist).readlines()
wavfile, _ = os.path.splitext(os.path.basename(args.wavlist))
### Feature and embedding extration
wavlist = []
for line in lines:
    wavlist.append(line.rstrip().split()[0])

embeddings = {}

for filename in wavlist:


    start_time = time.time()
    base, _ = os.path.splitext(os.path.basename(filename))
    print(base)
    # nn_info = nobj.netinfo(base)

    feat, utt_label, utt_shape, tffilename = fe.feat_extract([filename],FEAT_TYPE,N_FFT,HOP,VAD,CMVN,EXCLUDE_SHORT)
    # print('features:', type(feat), len(feat[0]))
    embeddings [base] = emnet_validation.ac2.eval({x:feat, s:utt_shape})
    print((embeddings [base]).shape)

    # if args.outputlayer:
    #     outputlayer.append(emnet_validation.o1.eval({x:feat, s:utt_shape}))
    #
    elapsed_time = time.time() - start_time
    print (format(elapsed_time) + ' seconds elapsed for ' + filename.split('/')[-1])
# embeddings = np.array(embeddings)
# np.save(args.wavlist.split('/')[-1].split('.')[0]+'_embeddings',embeddings)

# if args.outputlayer:
#     outputlayer = np.array(outputlayer)
#     np.save(args.wavlist.split('/')[-1].split('.')[0]+'_outputlayer',outputlayer)
# print('Total Wav Proccessed: ', len(embeddings))

with open(wavfile+'.pickle', 'wb') as handle:
    pickle.dump(embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)
