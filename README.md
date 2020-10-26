# Show-and-Speak
This is the pytorch implement for our paper ["SHOW AND SPEAK: DIRECTLY SYNTHESIZE SPOKEN DESCRIPTION OF IMAGES"](https://arxiv.org/abs/2010.12267). More details can be seen in the [project page](https://xinshengwang.github.io/projects/SAS/).

### Requirements
python 3.6
pytorch 1.4.0
scipy==1.2.1

### train the code
#### Download database

You can download our processed database from [Flickr8k_SAS](https://zenodo.org/record/4126934/files/Flickr8k_SAS.tar.gz?download=1). Then unzip the file in the root directory of the code. You can get the directory tree as:

```
├── Data_for_SAS
│   ├── bottom_up_features_36_info
│   ├── images
│   ├── mel_80
│   ├── wavs
│   ├── train
│   │   ├── filenames.pickle
│   ├── val
│   │   ├── filenames.pickle
│   ├── test
│   │   ├── filenames.pickle
```
Among them, "bottom_up_features_36_info" contains the extracted bottom-up features of images; "images" contains all raw images of Flickr8k; "mel_80" contains the mel spectrogram of audio files; "wavs" constains all the speech synthesized by TTS system.

#### Train the code 

run
```
python train --data_dir Data_for_SAS --save_path outputs 
```

### Inference 
Download the [pre-trained waveglow model](https://drive.google.com/file/d/1DDxqWr7m44e7BXeu5w84zwYNkcnYaNQ-/view?usp=sharing) and put it in the root directory of this code.

run 
```
python train --data_dir Data_for_SAS --save_path outputs --only_val
```

#### Cite
@article{wang2020show,  
  title={Show and Speak: Directly Synthesize Spoken Description of Images},  
  author={Xinsheng Wang, Siyuan Feng, Jihua Zhu, Mark Hasegawa-Johnson, Odette Scharenborg},  
  journal={arXiv preprint arXiv:arXiv:2010.12267},  
  year={2020}  
}