class hparams:
	################################
	# Data Parameters              #
	################################
	# Image                        #
	################################
	img_size = 224
	RNN_TYPE = 'LSTM'
	################################
	# Audio                        #
	################################
	num_mels = 80
	################################
	# Model Parameters             #
	################################
	# Encoder parameters
	encoder_embedding_dim = 512 #1024

	# Decoder parameters
	n_frames_per_step = 3
	decoder_rnn_dim = 1024
	prenet_dim = 256
	max_decoder_steps = 200
	gate_threshold = 0.5
	p_attention_dropout = 0.1
	p_decoder_dropout = 0.1

	# Attention parameters
	attention_rnn_dim = 1024
	attention_dim = 256

	# Location Layer parameters
	attention_location_n_filters = 32
	attention_location_kernel_size = 31

	# Mel-post processing network parameters
	postnet_embedding_dim = 512
	postnet_kernel_size = 5
	postnet_n_convolutions = 5

	################################
	# Train                        #
	################################
	is_cuda = True
	pin_mem = True
	n_workers = 8 #8
	lr = 2e-3
	betas = (0.9, 0.999)
	eps = 1e-6
	sch = True
	sch_step = 4000
	max_iter = 2e5
	batch_size = 32  #32
	weight_decay = 1e-6
	grad_clip_thresh = 1.0
	mask_padding = True
	p = 10 # mel spec loss penalty
	Scheduled_Sampling = True
	################################
	# Infer                        #
	################################
	sigma_infer = 0.9
	denoising_strength = 0.01
	seft_hop_length = 256

	
