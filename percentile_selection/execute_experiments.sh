# python -u percentile_experiment.py hepth 50 | tee results/hepth_50.dat
# python -u percentile_experiment.py hepth 90 | tee results/hepth_90.dat
# python -u percentile_experiment.py hepth 99 | tee results/hepth_99.dat
# python -u percentile_experiment.py patent 50 | tee results/patent_50.dat
# python -u percentile_experiment.py patent 90 | tee results/patent_90.dat
# python -u percentile_experiment.py patent 99 | tee results/patent_99.dat
# python -u percentile_experiment.py income 50 | tee results/income_50.dat
# python -u percentile_experiment.py income 90 | tee results/income_90.dat
# python -u percentile_experiment.py income 99 | tee results/income_99.dat

python -u err_percentile_experiment.py hepth 50 | tee results/err_beg_hepth_50.dat
python -u err_percentile_experiment.py hepth 90 | tee results/err_beg_hepth_90.dat
python -u err_percentile_experiment.py hepth 99 | tee results/err_beg_hepth_99.dat
python -u err_percentile_experiment.py patent 50 | tee results/err_beg_patent_50.dat
python -u err_percentile_experiment.py patent 90 | tee results/err_beg_patent_90.dat
python -u err_percentile_experiment.py patent 99 | tee results/err_beg_patent_99.dat
python -u err_percentile_experiment.py income 50 | tee results/err_beg_income_50.dat
python -u err_percentile_experiment.py income 90 | tee results/err_beg_income_90.dat
python -u err_percentile_experiment.py income 99 | tee results/err_beg_income_99.dat