HOME_DIR='/network/abhinav/code/cerebras_llama2_7B_sparse50_retrained'
for SPARSITY in 0.7
do
    CUDA_VISIBLE_DEVICES=6,7 ./eval_nosamples.sh ${HOME_DIR}/${SPARSITY}sp_m5_l5/obcq_deployment &
    CUDA_VISIBLE_DEVICES=4,5 ./eval_nosamples.sh ${HOME_DIR}/${SPARSITY}sp_m5_l8/obcq_deployment
done
