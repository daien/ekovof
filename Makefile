BASE_DIR   = $(CURDIR)/ekovof
SPARSE_DIR = $(BASE_DIR)/sparse/sparse_distances/c_functions
DENSE_DIR  = $(BASE_DIR)/dense/cython_distances


all: sparse dense

sparse:
	@(cd $(SPARSE_DIR) && $(MAKE))

dense:
	@(cd $(DENSE_DIR) && $(MAKE))

clean:
	@(cd $(SPARSE_DIR) && $(MAKE) $@)
	@(cd $(DENSE_DIR) && $(MAKE) $@)

tests:
	nosetests -s ekovof

testscov:
	nosetests -s --with-coverage --cover-package=ekovof ekovof

.PHONY: all sparse dense clean tests testscov
