all: dense_distances
	echo "compiling all"

%:
	python setup_dense_distances.py build_ext --inplace
	mv dense_distances.so ../

clean:
	rm -rf build

.PHONY: clean
