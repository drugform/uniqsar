#!/bin/sh

usage()
{
  echo "Usage: bin/run_train.sh
  		[ -c | --config-file]   Path to config file, which describes
		       			  the training procedure (see examples).
                [ -g | --gpus]         NVidia device ids to run on,
		       		         comma-separated list of ids.
				       	 Optional, will run on CPU if not set
					 (probably thats not what you want).
                [ -d | --dev-mode]     Flag to run in development (interactive) mode.
		       		       	 Optional, non-intractive by default.

	Run from project root (i.e. ./bin/run_train.sh ...).
	Other paths may be relative."
}


opts=$(getopt -o c:g:d \
	      -l config-file:,gpus:,dev-mode \
	      -n run_train.sh \
	      -- $@)

if test $? -ne 0; then
    echo 'Failed to parse command line args:'
    usage
    exit 1
fi

eval set -- "$opts"

dev="no"
gpus=""
while true; do
    case $1 in
    -c|--config-file)  config_file=$2 ;shift;;
    -g|--gpus)        gpus=$2 ;shift;;
    -d|--dev-mode)    dev="yes";;
    --) shift; break;;
    *) usage; exit 1;; #?)
    esac
    shift
done   

check_arg ()
{
    name=$1
    eval "val=\$$name"
    if [ -z $val ]; then
	echo Argument is not set: $name
	usage
	exit 1
    fi
    echo Using $name = $val
}

check_arg config-file
set -eu

realpath $config_file

if test "$dev" = "yes"
then
    echo "Interactive mode"
    iflag="-i"
    dflag=""
    docker build -t uniqsar .
else
    iflag=""
    dflag="-d"
fi

command="docker run $dflag
     -v `realpath models`:/app/models:rw
     -v `realpath data`:/app/data:rw
     -v `realpath lib`:/app/lib:rw
     -v `realpath $config_file`:/app/src/train.py:rw
     -e MKL_NUM_THREADS=4
     -u $(id -u):$(id -g)
     --gpus \"device=${gpus}\"
     -it uniqsar
     python $iflag -u train.py"

printf "$command\n"
if test "$dev" = "yes"; then
    $command
else
    cid=$($command)
    echo "Process launched in docker container with id:" $(echo $cid | cut -c1-8)
fi

