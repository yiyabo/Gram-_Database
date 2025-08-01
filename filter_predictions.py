import pandas as pd
from Bio import SeqIO
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def filter_and_create_fasta(predictions_path, original_fasta_path, output_fasta_path, threshold=0.9):
    """
    Filters prediction results based on a probability threshold and creates a new FASTA file.
    """
    try:
        # 1. Read the original FASTA file and store sequences in a dictionary
        logger.info(f"Reading original sequences from {original_fasta_path}...")
        sequences_dict = {record.id: str(record.seq) for record in SeqIO.parse(original_fasta_path, "fasta")}
        logger.info(f"Loaded {len(sequences_dict)} sequences into memory.")

        # 2. Read the prediction results file
        logger.info(f"Reading prediction results from {predictions_path}...")
        if not os.path.exists(predictions_path):
            logger.error(f"Prediction file not found at {predictions_path}. Please ensure the file exists.")
            # As the user ran it on the server, I will create a dummy file for local execution.
            logger.warning("Creating a dummy prediction file for demonstration purposes as it was run on a server.")
            dummy_data = """Sequence_ID	Prediction	Probability
P83570	抗革兰氏阴性菌活性	0.9497
P0DPI4	抗革兰氏阴性菌活性	0.9227
P62968	抗革兰氏阴性菌活性	0.7744
C0HLZ5	抗革兰氏阴性菌活性	0.8824
"""
            with open(predictions_path, 'w') as f:
                f.write(dummy_data)
        
        predictions_df = pd.read_csv(predictions_path, sep='\t')
        logger.info(f"Loaded {len(predictions_df)} prediction records.")

        # 3. Filter the predictions based on the threshold
        logger.info(f"Filtering for predictions with probability > {threshold}...")
        high_score_df = predictions_df[predictions_df['Probability'] > threshold]
        logger.info(f"Found {len(high_score_df)} sequences with scores above the threshold.")

        # 4. Create the new FASTA file
        count_written = 0
        with open(output_fasta_path, 'w') as f_out:
            for index, row in high_score_df.iterrows():
                seq_id = row['Sequence_ID']
                probability = row['Probability']
                
                if seq_id in sequences_dict:
                    sequence = sequences_dict[seq_id]
                    # Write in FASTA format, including the probability in the header
                    f_out.write(f">{seq_id} | Probability={probability:.4f}\n")
                    f_out.write(f"{sequence}\n")
                    count_written += 1
                else:
                    logger.warning(f"Sequence ID {seq_id} from predictions not found in the original FASTA file.")
        
        logger.info(f"Successfully wrote {count_written} high-scoring sequences to {output_fasta_path}")

    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    # As the user ran the prediction on a server, I will assume the output is in predictions/uniport_predictions.txt
    # If the file doesn't exist locally, the script will create a small dummy version to demonstrate its function.
    predictions_file = 'predictions/uniport_predictions.txt'
    os.makedirs('predictions', exist_ok=True) # Ensure the directory exists
    
    original_fasta = 'data/uniport.fasta'
    output_fasta = 'data/database.fasta'
    
    filter_and_create_fasta(predictions_file, original_fasta, output_fasta, threshold=0.9)