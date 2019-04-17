import argparse
import os

class Options():
	def __init__(self):
		self.parser = argparse.ArgumentParser(description="parser for Msg-Net")
		subparsers = self.parser.add_subparsers(title="subcommands", dest="subcommand")


		train_arg = subparsers.add_parser('train' , help = 'trian model ')
		train_arg.add_argument('--dataset',type=str ,default= 'train.data', help ='traindata for train')
		train_arg.add_argument('--word_dic' ,default= 'word.dic',type=str , help='word dic for train ')
		train_arg.add_argument('--div_dic' ,default= 'div.dic',type=str , help='div dic for train ')
		train_arg.add_argument('--ch_dic' ,default= 'ch.dic',type=str , help='ch dic for train ')
		train_arg.add_argument('--weight' ,default=0.5 , type=float , help='')		
	
		train_arg.add_argument('--batch_size',type = int , default = 64, help='batch size of train')
		train_arg.add_argument('--epoch' , type = int ,default=5 ,help='epoch for train')
		train_arg.add_argument('--lr' ,type = float , default =1e-2 , help = 'learning rate for train ')
		train_arg.add_argument('--pre_model' , type = str , default='' , help = 'pre trained model for train')
		train_arg.add_argument('--testdata',default = 'test.data',type=str , help ='test data')

		
		senten_arg = subparsers.add_parser('sentence' , help = 'test sentence ')
		senten_arg.add_argument('--senten',type=str , help ='senten for test , chinese')
		senten_arg.add_argument('--model',default= 'test.model2-0',type=str , help ='traindata for train')
		senten_arg.add_argument('--word_dic' ,default= 'word.dic',type=str , help='word dic for train ')
		senten_arg.add_argument('--div_dic' ,default= 'div.dic',type=str , help='div dic for train ')
		senten_arg.add_argument('--ch_dic' ,default= 'ch.dic',type=str , help='ch dic for train ')

		test_arg = subparsers.add_parser('test' , help = 'test model on test-dataset ')
		test_arg.add_argument('--testdata',default = 'test.data',type=str , help ='test data')
		test_arg.add_argument('--model',default= 'test.model2-0',type=str , help ='traindata for train')
		test_arg.add_argument('--word_dic' ,default= 'word.dic',type=str , help='word dic for train ')
		test_arg.add_argument('--div_dic' ,default= 'div.dic',type=str , help='div dic for train ')
		test_arg.add_argument('--ch_dic' ,default= 'ch.dic',type=str , help='ch dic for train ')

	def parse(self):
		return self.parser.parse_args()
