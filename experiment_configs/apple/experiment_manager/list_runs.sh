EXPERIMENT_CONFIGS_DIR="$HOME/TimeSeAD/experiment_configs/apple"
RUN_FILE="$HOME/TimeSeAD/timesead_experiments/grid_search.py"
IGNORE_EXPS=(
)

# Replace 'generative.' with '' in each element of IGNORE_EXPS
for i in "${!IGNORE_EXPS[@]}"; do
    IGNORE_EXPS[$i]="${IGNORE_EXPS[$i]//generative./}"
done

experiment_configs=()
for folder in "$EXPERIMENT_CONFIGS_DIR"/*/; do
    # Check if it's a directory and doesn't meet the exclusion criteria
    if [ -d "$folder" ]; then
        folder_name=$(basename "$folder")  # Extract the folder name
        if [[ "$folder_name" == "__pycache__" || "$folder_name" == "experiment_manager" ]]; then
            continue  # Skip these folders
        fi
        # echo "$folder"  # Print the folder path

        # Loop through all .yml files in the current folder
        for file in "$folder"*.yml; do
            filename=$(basename "$file" .yml)  # Get the filename without the extension
            
            # Iterate through the IGNORE_EXPS array to check for matches
            skip_file=false
            for ignore_exp in "${IGNORE_EXPS[@]}"; do
                ignore_parent="${ignore_exp%%.*}"  # Get the part before the dot (folder)
                ignore_child="${ignore_exp#*.}"    # Get the part after the dot (filename)

                # Check if the current folder and filename match the ignore pattern
                if [[ "$folder_name" == "$ignore_parent" && "$filename" == *"$ignore_child"* ]]; then
                    skip_file=true
                    break  # Stop checking further if we found a match
                fi
            done

            # Skip the file if it matches the ignore pattern
            if $skip_file; then
                continue  # Skip this file
            fi

            # Check if a matching .yml file exists
            if [ -f "$file" ]; then
                experiment_configs+=("python $RUN_FILE with $file training_param_updates.training.device=cuda seed=123")
            fi
        done
    fi
done

cmds=()
for config_file in "${experiment_configs[@]}"; do
    cmds+=("$config_file ")
done

for i in "${!cmds[@]}"; do
    if [[ $i -eq $((${#cmds[@]} - 1)) ]]; then
        echo "${cmds[$i]}"
    else
        echo "${cmds[$i]} \\"
    fi
done
