import torch
import torch.optim as optim 
import math
import random 
from model import EM_BiLSTM_CRF 

import sys 

from option import Options

def list_2_tensor(vec):
	return torch.tensor(vec, dtype=torch.long)

def next_batch(trian_data , step , batch):
	return list_2_tensor(trian_data[step*batch:(step+1)*batch])

EMBEDDING_DIM = 32
HIDDEN_DIM = 256


def main():
	args = Options().parse()

	print(args)
	if args.subcommand is None :
		raise ValueError("ERROR:no detail info ")

	dic_word = torch.load(args.word_dic)



	if args.subcommand == 'train':
		train(args , dic_word )
	elif args.subcommand =='sentence':
		test_sentence(args ,  dic_word)
	elif args.subcommand =='test':
		test(args,  dic_word )


def test(args , dic):


	dic_word = dic	
	dic_div = torch.load(args.div_dic)
	dic_ch = torch.load(args.ch_dic)

	model = EM_BiLSTM_CRF(len(dic_word),len(dic_div),len(dic_ch) , EMBEDDING_DIM ,HIDDEN_DIM)

	if args.model :
		model.load_state_dict(torch.load(args.model))

	test_model(args,model,args.testdata)



def train(args , dic):

	#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	#device = torch.device('cpu')
	#print(device)
	dic_word = dic
	
	dic_div = torch.load(args.div_dic)
	dic_ch = torch.load(args.ch_dic)

	batch_size = args.batch_size

	model = EM_BiLSTM_CRF(len(dic_word),len(dic_div),len(dic_ch) , EMBEDDING_DIM ,HIDDEN_DIM)
	#model.to(device)

	if args.pre_model :
		model.load_state_dict(torch.load(args.pre_model))
	train_data = torch.load(args.dataset)
	#train_data = list_2_tensor(train_data)
	#train_data.to(device)

	#optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=1e-4)

	optimizer = optim.Adam(model.parameters(), lr=args.lr)
	data_len = len(train_data)
	total_step = math.ceil(data_len / batch_size)


	#test_setence_out(args,dic,model,args.dataset)

	for epoch in range(args.epoch):
		random.shuffle(train_data)
		#train_data = train_data[torch.randperm(train_data.size(0))]
		#test_model(args,model , args.testdata)
		for step in range(total_step):
			
			data = next_batch(train_data,step , batch_size)
			#data = data.to(device)

			model.zero_grad()

			sentens_in = data[:,0]
			divs_in = data[:,1]
			chs_in = data[:,2]

			loss , loss_div , loss_ch = model.loss(sentens_in ,divs_in,chs_in,args.weight)


			loss.backward()
			optimizer.step()

			if step%100 ==0:
				print("step {} / {} , loss :{} div_loss:{} ch_loss:{}".format(step,total_step,loss.item(),loss_div.item(),loss_ch.item()))


		torch.save(model.state_dict(),'test.model-'+str(epoch))
		test_model(args,model,args.dataset)
		test_model(args,model , args.testdata)
	return 

def test_sentence(args ,dic ):
	
	dic_word = dic	
	dic_div = torch.load(args.div_dic)
	dic_ch = torch.load(args.ch_dic)

	senten = [dic_word[w] for w in args.senten]
	senten_in = list_2_tensor(senten)


	div_map_r = {}
	for i , j in dic_div.items() :
		div_map_r[j] = i

	ch_map_r ={}
	for i , j in dic_ch.items():
		ch_map_r[j]=i

	model = EM_BiLSTM_CRF(len(dic_word),len(dic_div),len(dic_ch) , EMBEDDING_DIM ,HIDDEN_DIM)
	if args.model :
		model.load_state_dict(torch.load(args.model))
	
	#model = EM_BiLSTM_CRF(len(dic),label_map , EMBEDDING_DIM ,HIDDEN_DIM)
	#model.load_state_dict(torch.load(args.model))

	tags_div , tags_ch = model.test_sentence(senten_in)

	print(tags_div ,tags_ch )

	predict_div  = [div_map_r[w] for w in tags_div ]
	predict_ch = [ch_map_r[w] for w in tags_ch]


	final_senten = []
	local_senten =""
	for i , label in enumerate(predict_div):
		local_senten+=args.senten[i]
		if label  in ('S','E'):
			
			final_senten.append((local_senten,[predict_ch[i]]))
			local_senten =""

	if local_senten !="":
		print('error type ')
		final_senten.append(local_senten)

	print('predict list is {}'.format(final_senten))
	
	return 

def test_setence_out(args,dic ,model, dataset):

	dic_div = torch.load(args.div_dic)
	dic_ch = torch.load(args.ch_dic)

	dic_div_t = {}
	dic_ch_t ={}

	for i,j in dic_div.items():
		dic_div_t[j] = i
	
	for i, j in dic_ch.items():
		dic_ch_t[j] = i 
	
	dic_word_t = {}
	for i , j in dic.items():
		dic_word_t[j] = i
	
	dataset = torch.load(dataset)
	dataset = dataset[:100]

	with torch.no_grad():
		for step in range(len(dataset)):
			data = next_batch(dataset,step,1)
			#data = list_2_tensor(data)
			senten_in = data[:,0]
			tag_div = data[0,1]
			tag_ch = data[0,2]

			predict_div , predict_ch = model(senten_in)

			words = ''
			length = 0 
			for i in senten_in[0]:
				if i.item() == 0:
					break
				length +=1
				words += dic_word_t[i.item()]
			
			print(length)
			words = words[:length]
			predict_ch = predict_ch[0][:length]
			predict_div = predict_div[0][:length]
			# print('senten in is :',words)
			# print('predict div :',predict_div)
			# print('predict chL: ' ,predict_ch )

			print('senten in is :',words)
			print('predict div :',[dic_div_t[w] for w in predict_div])
			print('predict chL',[dic_ch_t[w] for w in predict_ch])

			# print('tags div :',[dic_div_t[w] for w in tag_div])
			# print('tags chL',[dic_ch_t[w] for w in tag_ch])


	return


def test_model ( args, model ,dataset ,batch_size= 128) :
	test_data = torch.load(dataset)

	random.shuffle(test_data)
	test_data = test_data[:20000]
	print('test  start !!!!!!!!!!')
	data_len = len(test_data)
	total_step = math.ceil(data_len / batch_size)

	total = 0
	correct_div = 0 
	correct_ch = 0
	with torch.no_grad():
		
		for step in range(total_step):

			data = next_batch(test_data ,step , batch_size)
			
			sentens_in = data[:,0]
			divs_in = data[:,1]
			chs_in = data[:,2]

			#(sentens_in.size())

			predict = model(sentens_in)

			predict_div = list_2_tensor(predict[0])
			predict_ch = list_2_tensor(predict[1])
			
			data_zero = torch.zeros_like(divs_in)
			zero_num = (data_zero == divs_in).sum().item()

			predict_div = predict_div.clone().view(-1)
			predict_ch = predict_ch.clone().view(-1)
			divs_in = divs_in.clone().view(-1)
			chs_in = chs_in.clone().view(-1)


			total += chs_in.size(0) - zero_num
			
			
			correct_div +=(predict_div == divs_in).sum().item() -zero_num
			correct_ch +=(predict_ch == chs_in).sum().item() - zero_num

			#if step %10 ==0 :
	print('current {}/{}  div_acc :{}/{} --{}% , ch_acc :{}/{}  --{}%'.format(step,total_step,correct_div,total,correct_div/total*100,correct_ch,total,correct_ch/total*100))		

	return



if __name__ == '__main__':
	main()