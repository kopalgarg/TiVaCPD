for d in /repos/Delivery-Readiness/data/*/; do
  e="$(basename -- $d)"
 for f in $d/*; do
   p="$(basename -- $f)"
   p2=${p%_0*}
   python run.py --exp $e --data_path $d --out_path ./out --model_type MMDA_CPD --prefix $p2 --threshold 0.005
 done
done

# python run.py --exp bump_test --data_path /repos/Delivery-Readiness/data/bump_/ --out_path ./out --model_type MMDA_CPD --prefix rmssd --threshold 0.005