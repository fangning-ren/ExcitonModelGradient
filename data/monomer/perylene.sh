#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --partition=day-long
#SBATCH --nodes=1
#SBATCH --mem=1G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --exclude=node[1,2]
echo $HOSTNAME
module load TeraChem/build2022
echo $CUDA_VISIBLE_DEVICES

terachem perylene.inp > perylene.out
cp scr-perylene/perylene.molden ./
terachem perylene-ca.inp > perylene-ca.out
terachem perylene-an.inp > perylene-an.out
