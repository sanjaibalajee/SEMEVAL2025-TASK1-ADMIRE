import pandas as pd
import time
import os
import logging
from langchain_community.llms import Ollama


# Setup logging
log_file = "cot_generation.log"
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console.setFormatter(formatter)
logging.getLogger().addHandler(console)

# Define output CSV file
output_file = "train_with_COT.csv"
llm=Ollama(model="llama3.1")



# Check if the file exists to resume progress
if os.path.exists(output_file):
    train_df = pd.read_csv(output_file)
    logging.info(f"✅ Resuming from saved file: {output_file}")
else:
    logging.info("🔄 Starting fresh... No saved file found.")
    train_df=pd.read_csv('../data/Train/subtask_a_train.tsv', sep='\t')

# Ensure "COT" column exists in the dataframe
if "COT" not in train_df.columns:
    train_df["COT"] = ""

# Logging total rows to process
total_rows = len(train_df)
logging.info(f"📌 Total rows to process: {total_rows}")

# Start processing rows
start_time = time.time()

for index, row in train_df.iterrows():
    try:
        # Skip rows where COT is already generated
        if pd.notna(row["COT"]) and row["COT"].strip() != "":
            continue  # Skip already processed rows

        text = row['sentence']  # Sentence column
        phrase = row['compound']  # Phrase column
        label = row['sentence_type']  # Sentence type column (Yes/No)

        # Generate Chain of Thought (CoT) prompt
        cot_prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context.  
Write a response that appropriately completes the request.  
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.  

### Instruction:  
You are a linguistic expert with advanced knowledge in semantics, idiomatic expressions, and contextual analysis.  
Your task is to analyze whether the given phrase in the sentence is used figuratively, idiomatically, or literally.  
You must first break down the meaning of the phrase, analyze how it is used in context, and then determine whether it conveys a figurative or literal meaning.  

### Input:  
Sentence: "{text}"  
Phrase: "{phrase}"  
Correct Label: "{label}"  

### Response:  
<think> Let's analyze this step by step:

1. **Phrase Meaning:** Define the phrase’s standard dictionary meaning.
2. **Possible Figurative Use:** Check if the phrase has an idiomatic or non-literal meaning in common usage.
3. **Contextual Analysis:** Analyze how the phrase is used in the sentence. Does it match its literal meaning, or does it carry an alternate figurative meaning?
4. **Final Decision:** Compare the phrase’s usage in the sentence with its common idiomatic meaning. If it aligns with its literal meaning, label it as "Literal" If it takes on a non-literal, figurative meaning, label it as "Idiomatic"

Based on this reasoning, the phrase "{phrase}" in the given sentence is classified as: "{label}".  
</think>  
"""

        # Invoke the model

        cot_response = llm.invoke(cot_prompt)

        # Store the generated CoT response in the DataFrame
        train_df.at[index, "COT"] = cot_response

        # Save the DataFrame incrementally (after every row)
        train_df.to_csv(output_file, index=False)

        # Logging progress
        elapsed_time = time.time() - start_time
        logging.info(f"✅ Processed {index+1}/{total_rows} rows - Time elapsed: {elapsed_time:.2f}s - Saved progress.")

    except Exception as e:
        logging.error(f"❌ Error processing row {index}: {e}")
        continue  # Move to the next row without stopping the script

# Final log message
logging.info(f"\n🎉 Finished generating CoT for all rows. Final dataset saved to {output_file}")
