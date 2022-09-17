dataset='constant_var_mean'
data_path='./data/constant_var_mean'

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
