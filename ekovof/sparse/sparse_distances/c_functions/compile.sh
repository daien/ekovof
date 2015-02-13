#!/usr/bin/env bash

base_dists=$* # sequence of base distance identifier
# 0 -> linear,
# 1 -> intersection,
# 2 -> total variation,
# 3 -> chi-square,
# 4 -> l2,

bdir="build"
odir=".."

if [ "$1" = "clean" ]; then
  clean_cmd="rm -rf $bdir"
  echo $clean_cmd
  $clean_cmd
  exit 0
fi

if [ "$base_dists" = "" ]; then
  base_dists="0 1 2 3 4"
fi
mkdir -p "$bdir"
#touch $odir/__init__.py

# XXX hacks to avoid harcoding or asking user...
PY_INC_DIR=$(python -c "import os; import sys; print os.path.join(sys.prefix, 'include', 'python{}.{}'.format(sys.version_info.major, sys.version_info.minor))")
NPY_INC_DIR=$(python -c "import numpy; print numpy.get_include()")

CFLAGS="-pthread -fno-strict-aliasing -DNDEBUG -O2 -g -pipe -Wall -Wp,-D_FORTIFY_SOURCE=2 -fexceptions -fstack-protector --param=ssp-buffer-size=4 -m64 -mtune=generic -D_GNU_SOURCE -fPIC -fopenmp -I. -I$PY_INC_DIR -I$NPY_INC_DIR $CFLAGS"

for bd in $base_dists; do
  echo "-----------------------------------"
  if [ "$bd" = "0" ]; then
    echo "Compiling with linear"
    dname="linear"
  elif [ "$bd" = "1" ]; then
    echo "Compiling with intersection"
    dname="intersection"
  elif [ "$bd" = "2" ]; then
    echo "Compiling with total variation"
    dname="totvar"
  elif [ "$bd" = "3" ]; then
    echo "Compiling with chi-square"
    dname="chisquare"
  elif [ "$bd" = "4" ]; then
    echo "Compiling with l2"
    dname="l2"
  else
    echo "Error: unknown BASE_DIST ($bd)"
    exit -1
  fi
  echo "-----------------------------------"
  # generate swig wrapper
  wrap_c="$bdir/_sparse_${dname}_wrap.c"
  swig -python -DBASE_DIST=$bd -o $wrap_c sparse_distance.i
  # move back python wrapper to output directory
  mv $bdir/sparse_$dname.py $odir/
  # compile wrapper
  wrap_o="${wrap_c/.c/.o}"
  gcc $CFLAGS -DBASE_DIST=$bd -c $wrap_c -o $wrap_o
  # compile original source
  orig_o="$bdir/_sparse_${dname}.o"
  gcc $CFLAGS -DBASE_DIST=$bd -c sparse_distance.c -o $orig_o
  # compile final shared library (trailing '_' is necessary!)
  gcc -pthread -shared $wrap_o $orig_o -lgomp -o $odir/_sparse_$dname.so
done
