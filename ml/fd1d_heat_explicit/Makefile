

all: run clean

run:
	python fd1d_heat_explicit_test.py > fd1d_heat_explicit_test_output.txt
	@echo "Test program output written to fd1d_heat_explicit_test_output.txt."

clean:
	@echo "Cleaning up the pyc, created by imports"
	rm -rf *txt *.png *pyc
