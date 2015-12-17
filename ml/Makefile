clean_pyc:
	rm -rf ./pde/*.pyc

clean_data:
	rm -rf ./data/solutions 
	python ./utils/run.py -mode original -action solve
	
	rm -rf ./data/cv_results
