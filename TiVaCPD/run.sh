
# simulate datasets 
python3 simulate.py --constant_mean True --constant_corr False  --n_samples 10 --constant_var True --exp changing_correlation --num_cp 5
python3 simulate.py --constant_mean False --constant_corr True  --n_samples 10 --constant_var True --exp jumping_mean --num_cp 5
python3 simulate.py --constant_mean True --constant_corr True  --n_samples 10 --constant_var False --exp changing_variance --num_cp 5
python3 simulate.py --constant_mean False --constant_corr False  --n_samples 10 --constant_var False --exp changing_mean_variance_correlation --num_cp 5
python3 simulate2.py --n_samples 10 --num_cp 3

