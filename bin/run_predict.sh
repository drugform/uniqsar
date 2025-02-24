#!/bin/sh

usage()
{
  echo "Usage: bin/run_predict.sh
  		[ -b | --batch-size]   Batch size.
		[ -m | --model-name]   Which model to use in prediction.
                [ -i | --input-file]   Path to csv file with data to predict.
                [ -o | --output-file]  Where to write prediction values,
		       		         will be created if not existing.
                [ -g | --gpus]         NVidia device ids to run on,
		       		         comma-separated list of ids.
				       	 Optional, will run on CPU if not set
                [ -d | --dev-mode]     Flag to run in development (interactive) mode.
		       		       	 Optional, non-interactive by default.

	Run from project root (i.e. ./bin/run_predict.sh ...).
	Other paths may be relative."
}

opts=$(getopt -o b:g:m:i:o:d \
	      -l batch-size:,gpus:,model-name:,input-file:,output-file:,dev-mode \
	      -n run_predict.sh \
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
    -b|--batch-size)  batch_size=$2 ;shift;;
    -g|--gpus)        gpus=$2 ;shift;;
    -m|--model-name)  model_name=$2 ;shift;;
    -i|--input-file)  input_file=$2 ;shift;;
    -o|--output-file) output_file=$2 ;shift;;
    -d|--dev-mode)    dev="yes";;
    --) shift; break;;
    *) usage; exit 1;;
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

check_arg batch_size
check_arg model_name
check_arg input_file
check_arg output_file

set -eu

if test "$dev" = "yes"
then
    echo "Interactive mode"
    iflag="-i"
    dflag=""
    docker build -t uniqsar_standalone .
else
    iflag=""
    dflag="-d"
fi

if [ -e $output_file ] ; then
    echo Output file exists: $output_file. Delete to continue? \(y/n\)
    rm -ri $output_file
    if [ -e $output_file ]; then
	echo 'Not going further until you manage the output file'
	exit 1
    fi
fi

touch $output_file

command="docker run $dflag
     -v `realpath models`:/app/models:rw
     -v `realpath data`:/app/data:rw
     -v `realpath lib`:/app/lib:rw
     -v `realpath $input_file`:/tmp/predict_input.csv:rw
     -v `realpath $output_file`:/tmp/predict_output.csv:rw
     -e MKL_NUM_THREADS=4
     -u $(id -u):$(id -g)
     --gpus \"device=${gpus}\"
     -it uniqsar_standalone
     python $iflag -u predict.py
         $model_name
         /tmp/predict_input.csv
         /tmp/predict_output.csv
         $batch_size"

printf "$command\n"
if test "$dev" = "yes"; then
    $command
else
    cid=$($command)
    echo "Process launched in docker container with id:" $(echo $cid | cut -c1-8)
fi
