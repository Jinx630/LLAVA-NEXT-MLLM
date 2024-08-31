export PYTHONPATH=.
model_path=/mnt/workspace/workgroup/jinmu/ckpts/llava_weights/llama3-llava-next-8b
question_path=/mnt/workspace/workgroup/jinmu/ai_competition/LLaVA-NeXT/playground/data/test_a/questions_for_pred.json
base_answer_path=output/model_test
image_folder=/mnt/workspace/workgroup/jinmu/ai_competition/LLaVA-NeXT/playground/data/test_a
temperature=0
# 几张卡推理
N=4
# Loop over each chunk/process
for (( chunk_id=0; chunk_id<N; chunk_id++ ))
do
    # Define the answer path for each chunk
    answer_path="${base_answer_path}/result_${chunk_id}.jsonl"
    if [ -f "$answer_path" ]; then
        rm "$answer_path"
    fi
    echo "$answer_path"
    # Run the Python program in the background
    CUDA_VISIBLE_DEVICES="$chunk_id" python3 llava/eval/question_inference.py --model-path "$model_path" --question-file "$question_path" --answers-file "$answer_path" --num-chunks "$N" --chunk-idx "$chunk_id" --image-folder "$image_folder" --temperature "$temperature" &

    # Uncomment below if you need a slight delay between starting each process
    # sleep 0.1
done

# Wait for all background processes to finish
wait

merged_file="${base_answer_path}/result.jsonl"
submit_file="${base_answer_path}/submit.jsonl"
if [ -f "$merged_file" ]; then
    rm "$merged_file"
fi
# Merge all the JSONL files into one
#cat "${base_answer_path}"_*.jsonl > "${base_answer_path}.jsonl"
for ((i=0; i<N; i++)); do
  input_file="${base_answer_path}/result_${i}.jsonl"
  cat "$input_file" >> "${base_answer_path}/result.jsonl"
done
# remove the unmerged files
for (( chunk_id=0; chunk_id<N; chunk_id++ ))
do
    # Define the answer path for each chunk
    answer_path="${base_answer_path}/result_${chunk_id}.jsonl"
    if [ -f "$answer_path" ]; then
        rm "$answer_path"
    fi
done

# 整理为提交格式
python reformat_to_submit.py $merged_file $submit_file
