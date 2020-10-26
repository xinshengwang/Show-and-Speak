data_path=/tudelft.net/staff-bulk/ewi/insy/MMC/xinsheng/data/Flickr8k
save_root=output
result_file=image2speech_bu_iec0.5_k160.text
start_epoch=0
max_epoch=1100
img_format=BU
gamma1=0.5
k=160
m=0.05
python train.py --data_dir $data_path \
              --save_path $save_root \
      			  --start_epoch $start_epoch\
      			  --result_file $result_file\
      			  --max_epoch $max_epoch\
				  --img_format $img_format \
				  --gamma1 $gamma1 \
				  --k $k \
				  --m $m
      			
                               
