import torch
import os 
from collections import OrderedDict 



data = []
	

def setupData(fileName ,senten_len_low, senten_len ,dic_word  = None , dic_div =None , dic_ch = None):
	if dic_word is None :
		dic_word = OrderedDict()
		dic_word['EMPTY'] = len(dic_word)
	if dic_div is None :
		dic_div = OrderedDict()
		dic_div['EMPTY'] = len(dic_div)
	if dic_ch is None :
		dic_ch = OrderedDict()
		dic_ch['EMPTY'] = len(dic_ch)
	
	fileDic = open(fileName,'r', encoding='UTF-8')
	arr = fileDic.readlines()
	div = []
	ch = []
	senten = []
	data = []
	count = 0
	flag = False 


	for no,a in enumerate(arr) :
		a = a.strip('\r\n\t')
		if flag and len(senten):
			add_len = senten_len - len(senten)
			if add_len > 0 :
				senten = senten + [0]*add_len
				div = div + [0]*add_len
				ch = ch + [0]*add_len
			if add_len <0:
				
				re_dic = {}
				for i , j in dic_word.items():
					re_dic[j]=i
				
				trans_senten = ''
				for w in senten :
					trans_senten += re_dic[w]
			
				errorno = 'too long {}:{},{}'.format(no,senten_len-add_len,trans_senten)
				raise ValueError(errorno)
			
			data.append((senten,div,ch))
			senten = []
			div = []
			ch = []
			count = 0 
			flag = False


		if a =='':
			#flag=True
			continue 

		word = a.split('\t')
		if word[0] not in dic_word :
			dic_word[word[0]] = len(dic_word)
		
		if word[1].find(']') != -1:
			word[1] = word[1][:word[1].find(']')]

		if word[1] not in dic_ch :
			dic_ch[word[1]] = len(dic_ch)
		if word[2][0] not in dic_div :
			dic_div[word[2][0]] = len(dic_div)
		
		senten.append(dic_word[word[0]])
		div.append(dic_div[word[2][0]])
		ch.append(dic_ch[word[1]])

		if len(senten) > senten_len_low and word[2][0] in ('S','E'):
			flag = True

		if word[0] in ('。','！','？'):
			flag = True

		if no %10000 == 0:
			print (no)

		# if word[0] in ('。','！','？','，','；'):
		# 	add_len = senten_len - len(senten)
		# 	if add_len > 0 :
		# 		senten = senten + [0]*add_len
		# 		div = div + [0]*add_len
		# 		ch = ch + [0]*add_len
		# 	if add_len <0:
				
		# 		raise ValueError('too long {}'.format(senten_len-add_len))
			
		# 	data.append((senten,div,ch))
		# 	senten = []
		# 	div = []
		# 	ch = []
	
	return data , dic_word , dic_div , dic_ch

if __name__ == '__main__':
	dic_word = torch.load('word.dic')
	dic_div = torch.load('div.dic')
	dic_ch = torch.load('ch.dic')
	print(len(dic_word))
	data, dic_word , dic_div , dic_ch = setupData('6.test.data',32,50,dic_word,dic_div,dic_ch)
	#print(len(dic_word))
	torch.save(data,'test.data')
	
	torch.save(dic_word , 'word.dic')
	torch.save(dic_div , 'div.dic')
	torch.save(dic_ch , 'ch.dic')
	# dic_word = torch.load('word.dic')
	# dic_ch  = torch.load('ch.dic')
	# dic_div  = torch.load('div.dic')
	# print (dic_div)
	# print(dic_ch)
	# print(len(dic_word),len(dic_ch),len(dic_div))
