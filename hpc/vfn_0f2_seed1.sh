#PBS -lselect=1:ncpus=24:mem=24gb:ngpus=1:gpu_type=RTX6000
#PBS -lwalltime=72:00:00
ls
pwd

module load anaconda3/personal

nvidia-smi

echo "starting script."

python $HOME/laplace-sparse-convolutions/

echo "script done."

pwd 

ls

ls checkpoints

cp -R checkpoints/* $HOME/continuous-filters/checkpoints/

echo "save done."
