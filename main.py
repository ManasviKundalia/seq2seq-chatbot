'''
author - manasvi kundalia

code to develop chatbot using seq2seq models
'''

import numpy as np 
from nce import NCELoss
import torch
import torch.nn as nn
from data import Data_cb
from os import listdir
from models import EncoderRNN,DecoderRNN
from torch.autograd import Variable
from torch import optim
import random
import time
import math

MAX_LENGTH = 20
use_cuda = torch.cuda.is_available()
SOS_token = 0
EOS_token = 1
num_files = 15

pairs = []
dataobj = Data_cb()
'''
#in case of open subtitles dataset uncomment this block

data_path = "./data/"
filenames = listdir(data_path)
print("number of files: ", len(filenames))


for num in range(num_files):
	file = open(data_path+filenames[num])
	text = file.read().split('\n')
	for count in range(len(text)):
		#dataobj.addSentence(text[count])
		if(count<len(text)-1):
			pairs.append([text[count],text[count+1]])

'''

#in case of ubuntu dataset uncomment this block
data_path = './ubuntu_csvfiles/testset.csv'
import pandas as pd
dataframe = pd.read_csv(data_path,names = ['context','utterance','label'])
dataframe = dataframe[dataframe['label']==1]

for row in dataframe.iterrows():
	pairs.append([row[1]['context'],row[1]['utterance']])

data_path2 = './ubuntu_csvfiles/valset.csv'
dataframe = pd.read_csv(data_path2,names = ['context','utterance','label'])
dataframe = dataframe[dataframe['label']==1]

for row in dataframe.iterrows():
	sentences = row[1]['context'].split('__EOS__')
	sentences.extend(row[1]['utterance'].split('__EOS__'))
	for i in range(len(sentences)-1):
		pairs.append([sentences[i],sentences[i+1]])
	#pairs.append([row[1]['context'],row[1]['utterance']])


print("Number of pairs: ",len(pairs))


def trim_dataset(pairs,max_len = MAX_LENGTH):
	pairs2 = []
	for p in pairs:
		if len(p[0].strip().split(' '))<=max_len and len(p[1].strip().split(' '))<=max_len:
			dataobj.addSentence(p[0])
			dataobj.addSentence(p[1])
			pairs2.append(p)
	return pairs2

print("Trimming dataset...")

pairs = trim_dataset(pairs)

print("Number of training pairs after trimming = ", len(pairs))
print("Number of words: ", dataobj.n_words)
hidden_size = 256
encoder_rnn = EncoderRNN(dataobj.n_words,hidden_size)
decoder_rnn = DecoderRNN(hidden_size,dataobj.n_words)

def indexesFromSentence(dataobj, sentence):
    return [dataobj.word2index[word] for word in sentence.split(' ')]


def variableFromSentence(dataobj, sentence):
    indexes = indexesFromSentence(dataobj, sentence)
    indexes.append(EOS_token)
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    if use_cuda:
        return result.cuda()
    else:
        return result


def variablesFromPair(pair):
    input_variable = variableFromSentence(dataobj, pair[0])
    target_variable = variableFromSentence(dataobj, pair[1])
    return (input_variable, target_variable)

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def train(input_variable,target_variable,encoder_rnn,decoder_rnn,criterion, encoder_optimizer,decoder_optimizer, max_len = MAX_LENGTH):
	encoder_hidden = encoder_rnn.initHidden()
	encoder_optimizer.zero_grad()
	decoder_optimizer.zero_grad()
	input_length = input_variable.size()[0]
	target_length = target_variable.size()[0]

	loss = 0

	
	for ei in range(input_length):
		encoder_op, encoder_hidden = encoder_rnn(input_variable[ei],encoder_hidden)

	decoder_input = Variable(torch.LongTensor([[SOS_token]]))
	decoder_input = decoder_input.cuda() if use_cuda else decoder_input
	decoder_hidden = encoder_hidden
	
	for di in range(target_length):
		
		decoder_output,decoder_hidden = decoder_rnn(decoder_input,decoder_hidden)
		loss += criterion.forward(decoder_output,target_variable[di])
		#not using teacher forcing
		topv,topi = decoder_output.data.topk(1)
		ni = topi[0][0]
		decoder_input = Variable(torch.LongTensor([[ni]]))
		decoder_input = decoder_input.cuda() if use_cuda else decoder_input
		if ni==EOS_token:
			break

	loss.backward()
	encoder_optimizer.step()
	decoder_optimizer.step()

	return loss.data[0]/target_length

def trainiters(pairs,encoder_rnn,decoder_rnn,num,criterion_name = 'NLL',print_every = 1000,learning_rate = 0.01):
	start = time.time()
	print_loss_total = 0 # reset at every print_every
	encoder_optimizer = optim.SGD(encoder_rnn.parameters(), lr=learning_rate)
	decoder_optimizer = optim.SGD(decoder_rnn.parameters(), lr=learning_rate)
	training_pairs = [variablesFromPair(random.choice(pairs)) for i in range(num)]
	if criterion_name=='NCE':
		criterion = NCELoss(noise)
	else:
		criterion = nn.NLLLoss()
	for i in range(num):
		loss = train(training_pairs[i][0],training_pairs[i][1],encoder_rnn,decoder_rnn,criterion,encoder_optimizer,decoder_optimizer)
		print_loss_total +=loss 
		if i%print_every==0 and i!=0:
			print_loss = print_loss_total/print_every
			print_loss_total = 0
			print('%s (%d %d%%) %.4f' %(timeSince(start, i / num),i, i / num * 100, print_loss))

def evaluate(pairs, encoder_rnn,decoder_rnn,num = 25):
	pairs_s = [random.choice(pairs) for i in range(num)]
	ev_pairs  = [variablesFromPair(pairs_s[i]) for i in range(num)]

	for i in range(num):
		input_var = ev_pairs[i][0]
		target_var = ev_pairs[i][1]
		input_length = input_var.size()[0]
		target_length = target_var.size()[0]

		encoder_hidden = encoder_rnn.initHidden()

		for ei in range(input_length):
			encoder_output,encoder_hidden = encoder_rnn(input_var[ei],encoder_hidden)

		decoder_hidden = encoder_hidden
		decoder_input = Variable(torch.LongTensor([[SOS_token]]))
		decoder_input = decoder_input.cuda() if use_cuda else decoder_input

		chatbot_op = ""

		for di in range(target_length):
			decoder_output,decoder_hidden = decoder_rnn(decoder_input,decoder_hidden)
			topv,topi = decoder_output.data.topk(1)
			ni = topi[0][0]
			chatbot_op += dataobj.index2word[ni]+" "
			decoder_input = Variable(torch.LongTensor([[ni]]))
			decoder_input = decoder_input.cuda() if use_cuda else decoder_input
			if ni==EOS_token:
				break

		print(">>> ", pairs_s[i][0])
		print("<<< ", chatbot_op)
		print()

noise = torch.Tensor(list(range(dataobj.n_words)))
trainiters(pairs,encoder_rnn,decoder_rnn,int(len(pairs)/2),'NCE',50)
evaluate(pairs,encoder_rnn,decoder_rnn,20)


















