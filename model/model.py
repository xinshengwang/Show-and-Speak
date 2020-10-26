import torch
import os
from torch import nn
from math import sqrt
from hparams import hparams as hps
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import models
from model.layers import ConvNorm, LinearNorm
from utils.util import to_var, get_mask_from_lengths
import pdb

class I2SLoss(nn.Module):
	def __init__(self,args):
		super(I2SLoss, self).__init__()
		self.args = args
	def forward(self, model_output, targets, iteration):
		mel_target, gate_target = targets[0], targets[1]
		mel_target.requires_grad = False
		gate_target.requires_grad = False
		slice = torch.arange(0, gate_target.size(1), hps.n_frames_per_step)
		gate_target = gate_target[:, slice].view(-1, 1)

		mel_out, mel_out_postnet, gate_out, _, image_vector,mel_vector = model_output
		gate_out = gate_out.view(-1, 1)
		p = hps.p
		mel_loss = nn.MSELoss()(p*mel_out, p*mel_target) + \
			nn.MSELoss()(p*mel_out_postnet, p*mel_target)
		gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)
		embedding_constraint_loss = Semantic_consistent_Loss(image_vector,mel_vector)*self.args.gamma1
		return mel_loss+gate_loss + embedding_constraint_loss, (mel_loss/(p**2)+gate_loss + embedding_constraint_loss).item()

def Semantic_consistent_Loss(image_outputs, audio_outputs):
    
    batch_size = image_outputs.shape[0]
    labels = Variable(torch.LongTensor(range(batch_size))).cuda()
    S = torch.mm(image_outputs, audio_outputs.transpose(1, 0))
    # S = scores.squeeze()
    m = nn.LogSoftmax() 
    loss = -torch.sum(m(S).diag())-torch.sum(m(S.T).diag())
    loss = loss / batch_size
    return loss

class LocationLayer(nn.Module):
	def __init__(self, attention_n_filters, attention_kernel_size,
				 attention_dim):
		super(LocationLayer, self).__init__()
		padding = int((attention_kernel_size - 1) / 2)
		self.location_conv = ConvNorm(2, attention_n_filters,
									  kernel_size=attention_kernel_size,
									  padding=padding, bias=False, stride=1,
									  dilation=1)
		self.location_dense = LinearNorm(attention_n_filters, attention_dim,
										 bias=False, w_init_gain='tanh')

	def forward(self, attention_weights_cat):
		processed_attention = self.location_conv(attention_weights_cat)
		processed_attention = processed_attention.transpose(1, 2)
		processed_attention = self.location_dense(processed_attention)
		return processed_attention


class Attention(nn.Module):
	def __init__(self, attention_rnn_dim, embedding_dim, attention_dim,
				 attention_location_n_filters, attention_location_kernel_size):
		super(Attention, self).__init__()
		self.query_layer = LinearNorm(attention_rnn_dim, attention_dim,
									  bias=False, w_init_gain='tanh')
		self.memory_layer = LinearNorm(embedding_dim, attention_dim, bias=False,
									   w_init_gain='tanh')
		self.v = LinearNorm(attention_dim, 1, bias=False)
		self.location_layer = LocationLayer(attention_location_n_filters,
											attention_location_kernel_size,
											attention_dim)
		self.score_mask_value = -float('inf')

	def get_alignment_energies(self, query, processed_memory,
							   attention_weights_cat):
		'''
		PARAMS
		------
		query: decoder output (batch, num_mels * n_frames_per_step)
		processed_memory: processed encoder outputs (B, T_in, attention_dim)
		attention_weights_cat: cumulative and prev. att weights (B, 2, max_time)

		RETURNS
		-------
		alignment (batch, max_time)
		'''

		processed_query = self.query_layer(query.unsqueeze(1))
		processed_attention_weights = self.location_layer(attention_weights_cat)
		energies = self.v(torch.tanh(
			processed_query + processed_attention_weights + processed_memory))

		energies = energies.squeeze(-1)
		return energies

	def forward(self, attention_hidden_state, memory, processed_memory,
				attention_weights_cat, mask):
		'''
		PARAMS
		------
		attention_hidden_state: attention rnn last output
		memory: encoder outputs
		processed_memory: processed encoder outputs
		attention_weights_cat: previous and cummulative attention weights
		mask: binary mask for padded data
		'''
		alignment = self.get_alignment_energies(
			attention_hidden_state, processed_memory, attention_weights_cat)

		if mask is not None:
			alignment.data.masked_fill_(mask, self.score_mask_value)

		attention_weights = F.softmax(alignment, dim=1)
		attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
		attention_context = attention_context.squeeze(1)

		return attention_context, attention_weights


class Prenet(nn.Module):
	def __init__(self, in_dim, sizes):
		super(Prenet, self).__init__()
		in_sizes = [in_dim] + sizes[:-1]
		self.layers = nn.ModuleList(
			[LinearNorm(in_size, out_size, bias=False)
			 for (in_size, out_size) in zip(in_sizes, sizes)])

	def forward(self, x):
		for linear in self.layers:
			x = F.dropout(F.relu(linear(x)), p=0.5, training=True)
		return x


class Postnet(nn.Module):
	'''Postnet
		- Five 1-d convolution with 512 channels and kernel size 5
	'''

	def __init__(self):
		super(Postnet, self).__init__()
		self.convolutions = nn.ModuleList()

		self.convolutions.append(
			nn.Sequential(
				ConvNorm(hps.num_mels, hps.postnet_embedding_dim,
						 kernel_size=hps.postnet_kernel_size, stride=1,
						 padding=int((hps.postnet_kernel_size - 1) / 2),
						 dilation=1, w_init_gain='tanh'),
				nn.BatchNorm1d(hps.postnet_embedding_dim))
		)

		for i in range(1, hps.postnet_n_convolutions - 1):
			self.convolutions.append(
				nn.Sequential(
					ConvNorm(hps.postnet_embedding_dim,
							 hps.postnet_embedding_dim,
							 kernel_size=hps.postnet_kernel_size, stride=1,
							 padding=int((hps.postnet_kernel_size - 1) / 2),
							 dilation=1, w_init_gain='tanh'),
					nn.BatchNorm1d(hps.postnet_embedding_dim))
			)

		self.convolutions.append(
			nn.Sequential(
				ConvNorm(hps.postnet_embedding_dim, hps.num_mels,
						 kernel_size=hps.postnet_kernel_size, stride=1,
						 padding=int((hps.postnet_kernel_size - 1) / 2),
						 dilation=1, w_init_gain='linear'),
				nn.BatchNorm1d(hps.num_mels))
			)

	def forward(self, x):
		for i in range(len(self.convolutions) - 1):
			x = F.dropout(torch.tanh(self.convolutions[i](x)), 0.5, self.training)
		x = F.dropout(self.convolutions[-1](x), 0.5, self.training)

		return x


class Image_Encoder(nn.Module):
	'''Encoder module:
		- two layer FC
		- Bidirectional LSTM
	'''
	def __init__(self,in_dim):
		super(Image_Encoder, self).__init__()
		in_sizes = [in_dim] + [hps.encoder_embedding_dim*2]
		sizes = [hps.encoder_embedding_dim*2] + [hps.encoder_embedding_dim]
		self.layers = nn.ModuleList(
			[LinearNorm(in_size, out_size, bias=False)
			 for (in_size, out_size) in zip(in_sizes, sizes)])

		self.lstm = nn.LSTM(hps.encoder_embedding_dim,
							int(hps.encoder_embedding_dim / 2), 1,
							batch_first=True, bidirectional=True)

	def forward(self, x):
		for linear in self.layers:
			x = F.dropout(F.relu(linear(x)), p=0.2, training=True)
		# outputs, _ = self.lstm(x)
		return x

	def inference(self, x):
		for linear in self.layers:
			x = F.dropout(F.relu(linear(x)), p=0.2)
		# outputs, _ = self.lstm(x)
		return outputs

class Decoder(nn.Module):
	def __init__(self):
		super(Decoder, self).__init__()
		self.num_mels = hps.num_mels
		self.n_frames_per_step = hps.n_frames_per_step
		self.encoder_embedding_dim = hps.encoder_embedding_dim
		self.attention_rnn_dim = hps.attention_rnn_dim
		self.decoder_rnn_dim = hps.decoder_rnn_dim
		self.prenet_dim = hps.prenet_dim
		self.max_decoder_steps = hps.max_decoder_steps
		self.gate_threshold = hps.gate_threshold
		self.p_attention_dropout = hps.p_attention_dropout
		self.p_decoder_dropout = hps.p_decoder_dropout
		self.prenet = Prenet(
			hps.num_mels * hps.n_frames_per_step,
			[hps.prenet_dim, hps.prenet_dim])

		self.attention_rnn = nn.LSTMCell(
			hps.prenet_dim + hps.encoder_embedding_dim,
			hps.attention_rnn_dim)

		self.attention_layer = Attention(
			hps.attention_rnn_dim, hps.encoder_embedding_dim,
			hps.attention_dim, hps.attention_location_n_filters,
			hps.attention_location_kernel_size)

		self.decoder_rnn = nn.LSTMCell(
			hps.attention_rnn_dim + hps.encoder_embedding_dim,
			hps.decoder_rnn_dim, 1)

		self.linear_projection = LinearNorm(
			hps.decoder_rnn_dim + hps.encoder_embedding_dim,
			hps.num_mels * hps.n_frames_per_step)

		self.gate_layer = LinearNorm(
			hps.decoder_rnn_dim + hps.encoder_embedding_dim, 1,
			bias=True, w_init_gain='sigmoid')

	def get_go_frame(self, memory):
		''' Gets all zeros frames to use as first decoder input
		PARAMS
		------
		memory: decoder outputs

		RETURNS
		-------
		decoder_input: all zeros frames
		'''
		B = memory.size(0)
		decoder_input = Variable(memory.data.new(
			B, self.num_mels * self.n_frames_per_step).zero_())
		return decoder_input

	def initialize_decoder_states(self, memory, mask=None):
		''' Initializes attention rnn states, decoder rnn states, attention
		weights, attention cumulative weights, attention context, stores memory
		and stores processed memory
		PARAMS
		------
		memory: Encoder outputs
		mask: Mask for padded data if training, expects None for inference
		'''
		B = memory.size(0)
		MAX_TIME = memory.size(1)

		self.attention_hidden = Variable(memory.data.new(
			B, self.attention_rnn_dim).zero_())
		self.attention_cell = Variable(memory.data.new(
			B, self.attention_rnn_dim).zero_())

		self.decoder_hidden = Variable(memory.data.new(
			B, self.decoder_rnn_dim).zero_())
		self.decoder_cell = Variable(memory.data.new(
			B, self.decoder_rnn_dim).zero_())

		self.attention_weights = Variable(memory.data.new(
			B, MAX_TIME).zero_())
		self.attention_weights_cum = Variable(memory.data.new(
			B, MAX_TIME).zero_())
		self.attention_context = Variable(memory.data.new(
			B, self.encoder_embedding_dim).zero_())

		self.memory = memory
		self.processed_memory = self.attention_layer.memory_layer(memory)
		self.mask = mask

	def parse_decoder_inputs(self, decoder_inputs):
		''' Prepares decoder inputs, i.e. mel outputs
		PARAMS
		------
		decoder_inputs: inputs used for teacher-forced training, i.e. mel-specs

		RETURNS
		-------
		inputs: processed decoder inputs

		'''
		# (B, num_mels, T_out) -> (B, T_out, num_mels)
		decoder_inputs = decoder_inputs.transpose(1, 2).contiguous()
		decoder_inputs = decoder_inputs.view(
			decoder_inputs.size(0),
			int(decoder_inputs.size(1)/self.n_frames_per_step), -1)
		# (B, T_out, num_mels) -> (T_out, B, num_mels)
		decoder_inputs = decoder_inputs.transpose(0, 1)
		return decoder_inputs

	def parse_decoder_outputs(self, mel_outputs, gate_outputs, alignments):
		''' Prepares decoder outputs for output
		PARAMS
		------
		mel_outputs:
		gate_outputs: gate output energies
		alignments:

		RETURNS
		-------
		mel_outputs:
		gate_outpust: gate output energies
		alignments:
		'''
		# (T_out, B) -> (B, T_out)
		alignments = torch.stack(alignments).transpose(0, 1)
		# (T_out, B) -> (B, T_out)
		gate_outputs = torch.stack(gate_outputs).transpose(0, 1)
		gate_outputs = gate_outputs.contiguous()
		# (T_out, B, num_mels) -> (B, T_out, num_mels)
		mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous()
		# decouple frames per step
		mel_outputs = mel_outputs.view(
			mel_outputs.size(0), -1, self.num_mels)
		# (B, T_out, num_mels) -> (B, num_mels, T_out)
		mel_outputs = mel_outputs.transpose(1, 2)

		return mel_outputs, gate_outputs, alignments

	def decode(self, decoder_input):
		''' Decoder step using stored states, attention and memory
		PARAMS
		------
		decoder_input: previous mel output

		RETURNS
		-------
		mel_output:
		gate_output: gate output energies
		attention_weights:
		'''
		cell_input = torch.cat((decoder_input, self.attention_context), -1)
		self.attention_hidden, self.attention_cell = self.attention_rnn(
			cell_input, (self.attention_hidden, self.attention_cell))
		self.attention_hidden = F.dropout(
			self.attention_hidden, self.p_attention_dropout, self.training)

		attention_weights_cat = torch.cat(
			(self.attention_weights.unsqueeze(1),
				self.attention_weights_cum.unsqueeze(1)), dim=1)
		self.attention_context, self.attention_weights = self.attention_layer(
			self.attention_hidden, self.memory, self.processed_memory,
			attention_weights_cat, self.mask)

		self.attention_weights_cum += self.attention_weights
		decoder_input = torch.cat(
			(self.attention_hidden, self.attention_context), -1)
		self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
			decoder_input, (self.decoder_hidden, self.decoder_cell))
		self.decoder_hidden = F.dropout(
			self.decoder_hidden, self.p_decoder_dropout, self.training)

		decoder_hidden_attention_context = torch.cat(
			(self.decoder_hidden, self.attention_context), dim=1)
		decoder_output = self.linear_projection(
			decoder_hidden_attention_context)

		gate_prediction = self.gate_layer(decoder_hidden_attention_context)
		return decoder_output, gate_prediction, self.attention_weights

	def forward(self, memory, decoder_inputs,ss_prob,memory_lengths):
		''' Decoder forward pass for training
		PARAMS
		------
		memory: Encoder outputs
		decoder_inputs: Decoder inputs for teacher forcing. i.e. mel-specs
		memory_lengths: Encoder output lengths for attention masking.

		RETURNS
		-------
		mel_outputs: mel outputs from the decoder
		gate_outputs: gate outputs from the decoder
		alignments: sequence of attention weights from the decoder
		'''
		decoder_input = self.get_go_frame(memory).unsqueeze(0)
		decoder_inputs = self.parse_decoder_inputs(decoder_inputs)
		decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0)
		decoder_inputs = self.prenet(decoder_inputs)

		self.initialize_decoder_states(
			memory, mask=~get_mask_from_lengths(memory_lengths))

		mel_outputs, gate_outputs, alignments = [], [], []
		while len(mel_outputs) < decoder_inputs.size(0) - 1:
			if hps.Scheduled_Sampling and len(mel_outputs) != 0:
				sample_prob = memory.new(memory.shape[0]).uniform_(0, 1)
				sample_mask = sample_prob < ss_prob
				pre_pred = self.prenet(mel_output.detach())
				if sample_mask.sum() == 0:
					decoder_input = decoder_inputs[len(mel_outputs)]
				elif sample_mask.sum() == memory.shape[0]:
					decoder_input = pre_pred
				else:
					sample_ind = sample_mask.nonzero().view(-1)
					decoder_input = decoder_inputs[len(mel_outputs)].clone()
					decoder_input.index_copy_(0, sample_ind, pre_pred.index_select(0, sample_ind))
			else:
				decoder_input = decoder_inputs[len(mel_outputs)]
		
			mel_output, gate_output, attention_weights = self.decode(
				decoder_input)
			mel_outputs += [mel_output.squeeze(1)]
			gate_outputs += [gate_output.squeeze()]
			alignments += [attention_weights]
		mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
			mel_outputs, gate_outputs, alignments)

		return mel_outputs, gate_outputs, alignments

	def inference(self, memory):
		''' Decoder inference
		PARAMS
		------
		memory: Encoder outputs

		RETURNS
		-------
		mel_outputs: mel outputs from the decoder
		gate_outputs: gate outputs from the decoder
		alignments: sequence of attention weights from the decoder
		'''
		decoder_input = self.get_go_frame(memory)

		self.initialize_decoder_states(memory, mask=None)
		mel_lengths = torch.zeros([memory.size(0)], dtype=torch.int32)
		not_finished = torch.ones([memory.size(0)], dtype=torch.int32)
		
		if torch.cuda.is_available():
			mel_lengths = mel_lengths.cuda()
			not_finished = not_finished.cuda()

		mel_outputs, gate_outputs, alignments = [], [], []
		while True:
			decoder_input = self.prenet(decoder_input)
			mel_output, gate_output, alignment = self.decode(decoder_input)
			dec = torch.le(torch.sigmoid(gate_output.data),
							self.gate_threshold).to(torch.int32).squeeze(1)
			not_finished = not_finished*dec
			mel_lengths += not_finished
			# pdb.set_trace()
			if torch.sum(not_finished) == 0 and torch.sum(mel_lengths) > 1:
				break
			mel_outputs += [mel_output.squeeze(1)]
			gate_outputs += [gate_output]
			alignments += [alignment]

			if len(mel_outputs) == self.max_decoder_steps:
				print('Warning: Reached max decoder steps.')
				break

			decoder_input = mel_output

		mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
			mel_outputs, gate_outputs, alignments)
		return mel_outputs, gate_outputs, alignments,mel_lengths

def is_end_of_frames(output, eps = 0.2):
    return (output.data <= eps).all()

class I2SModel(nn.Module):
	def __init__(self,args):
		super(I2SModel, self).__init__()
		self.args = args
		self.num_mels = hps.num_mels
		self.mask_padding = hps.mask_padding
		self.n_frames_per_step = hps.n_frames_per_step
		if self.args.img_format == 'BU':
			in_dim = 2048+1024
		else:
			in_dim = 2048
		self.Linear_vis_info = LinearNorm(1607, 1024, bias=False)
		self.encoder = Image_Encoder(in_dim)
		
		self.decoder = Decoder()
		self.postnet = Postnet()

		self.image_encoder =  LinearNorm(
			hps.encoder_embedding_dim,
			hps.encoder_embedding_dim)
		self.mel_encoder = Mel_encoder(hps.encoder_embedding_dim)

	def parse_batch(self, batch):
		imgs,vis_info,input_lengths, mel_padded, gate_padded, output_lengths, keys = batch
		input_lengths = input_lengths.long().cuda()
		imgs = imgs.float().cuda()
		vis_info = vis_info.float().cuda()
		if self.args.img_format == 'vector':
			imgs = imgs.unsqueeze(1)
		elif self.args.img_format == 'tensor':
			imgs = imgs.view(imgs.shape[0],2048,-1)
			imgs = imgs.transpose(2,1)
			
		mel_padded = mel_padded.float().cuda()
		gate_padded = gate_padded.float().cuda()
		output_lengths = output_lengths.long().cuda()
		return (
			(imgs,vis_info,input_lengths, mel_padded, output_lengths),
			(mel_padded, gate_padded))

	def parse_output(self, outputs, output_lengths=None):
		if self.mask_padding and output_lengths is not None:
			mask = ~get_mask_from_lengths(output_lengths, True) # (B, T)
			mask = mask.expand(self.num_mels, mask.size(0), mask.size(1)) # (80, B, T)
			mask = mask.permute(1, 0, 2) # (B, 80, T)
			
			outputs[0].data.masked_fill_(mask, 0.0) # (B, 80, T)
			outputs[1].data.masked_fill_(mask, 0.0) # (B, 80, T)
			slice = torch.arange(0, mask.size(2), self.n_frames_per_step)
			outputs[2].data.masked_fill_(mask[:, 0, slice], 1e3)  # gate energies (B, T//n_frames_per_step)

		return outputs
	def parse_decoder_inputs(self, decoder_inputs):
		''' Prepares decoder inputs, i.e. mel outputs
		PARAMS
		------
		decoder_inputs: inputs used for teacher-forced training, i.e. mel-specs

		RETURNS
		-------
		inputs: processed decoder inputs

		'''
		# (B, num_mels, T_out) -> (B, T_out, num_mels)
		decoder_inputs = decoder_inputs.transpose(1, 2).contiguous()
		decoder_inputs = decoder_inputs.view(
			decoder_inputs.size(0),
			int(decoder_inputs.size(1)/self.n_frames_per_step), -1)
		# (B, T_out, num_mels) -> (T_out, B, num_mels)
		decoder_inputs = decoder_inputs.transpose(0, 1)
		return decoder_inputs
	def forward(self, inputs,ss_prob):
		imgs,vis_info,input_lenghts, mels, output_lengths = inputs
		vis_info_emb = F.dropout(F.relu(self.Linear_vis_info(vis_info)), p=0.2, training=True)
		img_embeddings = torch.cat((imgs,vis_info_emb),dim=-1)
		encoder_outputs = self.encoder(img_embeddings)
		image_vector = self.image_encoder(encoder_outputs.mean(1))
		mel_vector = self.mel_encoder(self.parse_decoder_inputs(mels))
		mel_outputs, gate_outputs, alignments = self.decoder(encoder_outputs, mels,ss_prob,memory_lengths=input_lenghts)

		mel_outputs_postnet = self.postnet(mel_outputs)
		mel_outputs_postnet = mel_outputs + mel_outputs_postnet

		return self.parse_output(
			[mel_outputs, mel_outputs_postnet, gate_outputs, alignments,image_vector,mel_vector],
			output_lengths)

	def inference(self, imgs,vis_info):
		vis_info_emb = F.dropout(F.relu(self.Linear_vis_info(vis_info)), p=0.2, training=True)
		img_embeddings = torch.cat((imgs,vis_info_emb),dim=-1)
		encoder_outputs = self.encoder(img_embeddings)


		mel_outputs, gate_outputs, alignments,mel_lengths = self.decoder.inference(
			encoder_outputs)

		mel_outputs_postnet = self.postnet(mel_outputs)
		mel_outputs_postnet = mel_outputs + mel_outputs_postnet

		outputs = self.parse_output(
			[mel_outputs, mel_outputs_postnet, gate_outputs, alignments,mel_lengths])

		return outputs

	def teacher_infer(self, inputs, mels):
		il, _ =  torch.sort(torch.LongTensor([len(x) for x in inputs]),
							dim = 0, descending = True)
		text_lengths = to_var(il)

		embedded_inputs = self.embedding(inputs).transpose(1, 2)

		encoder_outputs = self.encoder(embedded_inputs, text_lengths)

		mel_outputs, gate_outputs, alignments = self.decoder(
			encoder_outputs, mels, memory_lengths=text_lengths)
		
		mel_outputs_postnet = self.postnet(mel_outputs)
		mel_outputs_postnet = mel_outputs + mel_outputs_postnet

		return self.parse_output(
			[mel_outputs, mel_outputs_postnet, gate_outputs, alignments])




class pBLSTMLayer(nn.Module):
    def __init__(self,input_feature_dim,hidden_dim,rnn_unit='LSTM',dropout_rate=0.0):
        super(pBLSTMLayer, self).__init__()
        self.rnn_unit = getattr(nn,rnn_unit.upper())
        # feature dimension will be doubled since time resolution reduction
        self.BLSTM = self.rnn_unit(input_feature_dim*2,hidden_dim,1, bidirectional=True, 
                                   dropout=dropout_rate,batch_first=True)
    
    def forward(self,input_x):
        batch_size = input_x.size(0)
        timestep = input_x.size(1)
        feature_dim = input_x.size(2)
        # Reduce time resolution
        input_x = input_x.contiguous().view(batch_size,int(timestep/2),feature_dim*2)
        # Bidirectional RNN
        output,hidden = self.BLSTM(input_x)
        return output,hidden


class Mel_encoder(nn.Module):
    def __init__(self,embedding_dim=512,n_layer=2):
        super(Mel_encoder,self).__init__()
        self.Conv = nn.Conv1d(in_channels=240,out_channels=256,
                              kernel_size=5,stride=2,
                              padding=0)
        self.bnorm = nn.BatchNorm1d(256)
        self.rnn = nn.GRU(256, int(embedding_dim/2), n_layer, batch_first=True, dropout=0.5,
                        bidirectional=True)  
    
    def forward(self, input):
            input = (input.transpose(1,0)).transpose(2,1)
            x = self.Conv(input)
            x = self.bnorm(x)
            x, hx = self.rnn(x.transpose(2,1))
            x = x.mean(1)
            return x