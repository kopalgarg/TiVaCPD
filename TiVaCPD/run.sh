dataset='constant_var_mean'
data_path='./data/constant_var_mean'

python3 simulate.py --constant_mean True --constant_corr False  --n_samples 1 --constant_var True --exp changing_correlation
python3 simulate.py --constant_mean False --constant_corr True  --n_samples 20 --constant_var True --exp jumping_mean
python3 simulate.py --constant_mean True --constant_corr True  --n_samples 20 --constant_var False --exp changing_variance

python main.py --model_type 'MMDATVGL_CPD' --data_path  $data_path --exp $dataset 
echo 'MMDATVGL_CPD done'
python main.py --model_type 'KLCPD' --data_path $data_path --exp $dataset
echo 'KLCPD done'
python main.py --model_type 'GRAPHTIME_CPD' --data_path $data_path --exp $dataset
echo 'GRAPHTIME_CPD done'
python main.py --model_type 'KSTBTVGL_CPD' --data_path $data_path --exp $dataset
echo 'KSTBTVGL_CPD done'
python main.py --model_type 'KSTB_CPD' --data_path $data_path --exp $dataset
echo 'KSTB_CPD done'
python main.py --model_type 'MMDA_CPD' --data_path $data_path --exp $dataset
echo 'MMDA_CPD done'
