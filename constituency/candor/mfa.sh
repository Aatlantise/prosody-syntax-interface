#!/bin/bash


# --- 2. Define Directories ---
CORPUS_BASE="/home/scratch/jm3743/candor/mfa/pre_alignment"
OUTPUTS_BASE="/home/scratch/jm3743/candor/mfa/post_alignment"

# Ensure the main output directory exists
mkdir -p "$OUTPUTS_BASE"

echo "Starting batch MFA alignment using $SLURM_CPUS_PER_TASK cores..."

# --- 3. The Alignment Loop ---
# Iterate over every conversation folder inside pre_alignment
for convo_dir in "$CORPUS_BASE"/*; do

    # Check if it's actually a directory (skips random files like .DS_Store)
    if [ -d "$convo_dir" ]; then

        # Extract just the folder name (e.g., "convo_123")
        convo_id=$(basename "$convo_dir")

        # Define where this specific conversation should be saved
        out_dir="$OUTPUTS_BASE/$convo_id"
        mkdir -p "$out_dir"

        echo "---------------------------------------------------"
        echo "Aligning Conversation: $convo_id"

        # Run MFA on just this one folder
        mfa align \
            "$convo_dir" \
            english_us_arpa \
            english_us_arpa \
            "$out_dir" \
            -j $SLURM_CPUS_PER_TASK \
            --clean \
            --overwrite \
            --textgrid_cleanup \
            --output_format csv

        # Basic error catching: $? is the exit status of the last command (MFA)
        if [ $? -eq 0 ]; then
            echo "SUCCESS: $convo_id"
        else
            echo "FAILED: $convo_id (Check the MFA logs for details)"
        fi

    fi
done

echo "---------------------------------------------------"
echo "All conversations processed."