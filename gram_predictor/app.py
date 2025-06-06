#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Gram-Negative Bacteria Prediction Service
Provides a web interface to predict if peptide sequences are active against Gram-negative bacteria.
Uses a hybrid Keras model (LSTM+MLP).
"""

import os
import sys
import uuid
import json
import tempfile
import tensorflow as tf
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
from io import StringIO, BytesIO
from Bio import SeqIO
from peptides import Peptide 
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences #type: ignore
from collections import Counter
import pickle
import traceback # For detailed error logging

# 导入生成服务
from generation_service import get_generation_service

print("!!!!!!!!!! DEBUG: app.py IS LOADING FRESHLY (Hybrid Model Version) !!!!!!!!!!")

# --- Constants and functions copied/adapted from hybrid_classifier.py ---
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
VOCAB_DICT = {aa: i+2 for i, aa in enumerate(AMINO_ACIDS)}
VOCAB_DICT['<PAD>'] = 0
VOCAB_DICT['<UNK>'] = 1
VOCAB_SIZE = len(VOCAB_DICT)

MAX_SEQUENCE_LENGTH = 32 
PAD_TOKEN_ID = VOCAB_DICT['<PAD>']
UNK_TOKEN_ID = VOCAB_DICT['<UNK>']

def tokenize_and_pad_sequences_app(sequences, vocab_dict, max_len, pad_token_id, unk_token_id):
    """Converts a list of raw amino acid sequence strings to tokenized and padded integer ID sequences."""
    tokenized_sequences = []
    for seq_str in sequences:
        seq_str = str(seq_str).upper()
        tokens = [vocab_dict.get(aa, unk_token_id) for aa in seq_str]
        tokenized_sequences.append(tokens)
    
    padded_sequences = pad_sequences(
        tokenized_sequences,
        maxlen=max_len,
        dtype='int32',
        padding='post',
        truncating='post',
        value=pad_token_id
    )
    return padded_sequences
# --- End of copied/adapted section ---


def load_keras_model(model_path):
    """Loads a trained Keras model."""
    print(f"Loading Keras model from {model_path}...")
    try:
        model = tf.keras.models.load_model(model_path)
        print("Keras model loaded successfully!")
        return model
    except FileNotFoundError:
        print(f"Error: Keras model file not found at {model_path}")
        raise
    except Exception as e:
        print(f"Unknown error occurred while loading Keras model: {e}")
        raise

def load_app_scaler(scaler_path):
    """Loads the StandardScaler for global features."""
    print(f"Attempting to load StandardScaler from: {scaler_path}")
    if not os.path.exists(scaler_path):
        error_msg = f"Error: Required StandardScaler file '{scaler_path}' not found. Cannot proceed with predictions."
        print(error_msg)
        raise FileNotFoundError(error_msg)
    try:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print(f"Successfully loaded StandardScaler from {scaler_path}.")
        return scaler
    except Exception as e:
        print(f"Error loading StandardScaler from {scaler_path}: {e}")
        raise RuntimeError(f"Could not load StandardScaler file '{scaler_path}': {e}")

def extract_features_from_fasta(fasta_file):
    """Extracts peptide features and raw sequences from a FASTA file."""
    print(f"Extracting features from {fasta_file}...")
    records_data = [] 
    
    feature_names_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'feature_names.txt')
    expected_feature_order = []
    if os.path.exists(feature_names_path):
        with open(feature_names_path, 'r') as f_names:
            expected_feature_order = [line.strip() for line in f_names if line.strip()]
        if len(expected_feature_order) != 28:
            print(f"Warning: Number of features in feature_names.txt ({len(expected_feature_order)}) is not 28. Will use default order.")
            expected_feature_order = [] 
    else:
        print(f"Warning: {feature_names_path} not found. Will use default feature order.")

    if not expected_feature_order: 
        expected_feature_order = [
            'Length', 'Charge', 'Hydrophobicity', 'Hydrophobic_Moment',
            'Instability_Index', 'Isoelectric_Point', 'Aliphatic_Index',
            'Hydrophilicity', 'AA_A', 'AA_C', 'AA_D', 'AA_E', 'AA_F', 'AA_G',
            'AA_H', 'AA_I', 'AA_K', 'AA_L', 'AA_M', 'AA_N', 'AA_P', 'AA_Q',
            'AA_R', 'AA_S', 'AA_T', 'AA_V', 'AA_W', 'AA_Y'
        ]

    for record in SeqIO.parse(fasta_file, "fasta"):
        peptide_id = record.id
        sequence = str(record.seq).upper()
        
        if not sequence or any(aa not in AMINO_ACIDS for aa in sequence):
            print(f"Skipping invalid sequence {peptide_id}: {sequence}")
            continue
            
        try:
            pep = Peptide(sequence)
            length = len(sequence)
            
            feature_values_dict = {
                'ID': peptide_id,
                'Sequence': sequence,
                'Length': float(length),
                'Charge': float(pep.charge(pH=7.4)),
                'Hydrophobicity': float(pep.hydrophobicity(scale="Eisenberg")),
                'Hydrophobic_Moment': float(pep.hydrophobic_moment(window=min(11,length)) or 0.0),
                'Instability_Index': float(pep.instability_index()),
                'Isoelectric_Point': float(pep.isoelectric_point()),
                'Aliphatic_Index': float(pep.aliphatic_index()),
                'Hydrophilicity': float(pep.hydrophobicity(scale="HoppWoods"))
            }
            aa_comp = {f'AA_{aa}': sequence.count(aa) / length for aa in AMINO_ACIDS}
            feature_values_dict.update(aa_comp)
            
            ordered_feature_dict = {'ID': peptide_id, 'Sequence': sequence}
            for fname in expected_feature_order:
                ordered_feature_dict[fname] = feature_values_dict.get(fname, 0.0)
            
            records_data.append(ordered_feature_dict)
            
        except Exception as e:
            print(f"Error processing sequence {peptide_id}: {e}")
    
    df = pd.DataFrame(records_data)
    if not df.empty:
        cols_to_keep = ['ID', 'Sequence'] + [col for col in expected_feature_order if col in df.columns]
        df = df[cols_to_keep]

    print(f"Successfully extracted features and raw sequences for {len(df)} sequences.")
    return df

def predict_with_hybrid_model_app(k_model, scaler, sequence_ids_list, sequence_strings_list, global_features_df, threshold=0.5):
    """Makes predictions using the loaded Keras hybrid model and scaler."""
    print("!!!!!!!!!! DEBUG: predict_with_hybrid_model_app FUNCTION HAS BEEN CALLED !!!!!!!!!!")
    if not sequence_ids_list:
        print("No sequences to predict.")
        return pd.DataFrame(columns=['ID', 'Sequence', 'Probability', 'Prediction', 'Label'])

    print(f"Processing {len(sequence_strings_list)} sequences for LSTM branch...")
    tokenized_padded_sequences = tokenize_and_pad_sequences_app(
        sequence_strings_list, VOCAB_DICT, MAX_SEQUENCE_LENGTH, PAD_TOKEN_ID, UNK_TOKEN_ID
    )
    print(f"Sequence data processed, shape: {tokenized_padded_sequences.shape}")

    feature_column_names = [col for col in global_features_df.columns if col not in ['ID', 'Sequence']]
    global_features_np = global_features_df[feature_column_names].values.astype(np.float32)
    print(f"Global features extracted, shape: {global_features_np.shape}")
    
    scaled_global_features_np = global_features_np
    if scaler:
        try:
            if global_features_np.shape[1] != scaler.n_features_in_:
                 print(f"Error: Number of features for scaler ({global_features_np.shape[1]}) does not match scaler's expected features ({scaler.n_features_in_}).")
                 error_results = [{'ID': sid, 'Sequence': sstr, 'Probability': np.nan, 'Prediction': -1, 'Label': "Feature dimension error"} for sid, sstr in zip(sequence_ids_list, sequence_strings_list)]
                 return pd.DataFrame(error_results)
            scaled_global_features_np = scaler.transform(global_features_np)
            print("Global features successfully standardized.")
        except Exception as e:
            print(f"Error applying scaler: {e}. Using non-standardized features.")
    else:
        print("Warning: Scaler not provided. Using raw global feature values for prediction.")

    print(f"Predicting for {len(sequence_ids_list)} sequences...")
    if tokenized_padded_sequences.shape[0] == 0 or scaled_global_features_np.shape[0] == 0:
        print("Warning: Processed sequence data or global feature data is empty. Cannot predict.")
        return pd.DataFrame(columns=['ID', 'Sequence', 'Probability', 'Prediction', 'Label'])
    
    if tokenized_padded_sequences.shape[0] != scaled_global_features_np.shape[0]:
        print(f"Error: Mismatch in sample count between sequence data ({tokenized_padded_sequences.shape[0]}) and global feature data ({scaled_global_features_np.shape[0]}).")
        error_results = [{'ID': sid, 'Sequence': sstr, 'Probability': np.nan, 'Prediction': -1, 'Label': "Data mismatch"} for sid, sstr in zip(sequence_ids_list, sequence_strings_list)]
        return pd.DataFrame(error_results)

    try:
        logits = k_model.predict([tokenized_padded_sequences, scaled_global_features_np])
        probabilities = tf.sigmoid(logits).numpy().flatten()
        predictions = (probabilities >= threshold).astype(int)
    except Exception as e:
        print(f"Error during Keras model prediction: {e}")
        error_results = [{'ID': sid, 'Sequence': sstr, 'Probability': np.nan, 'Prediction': -1, 'Label': "Prediction failed"} for sid, sstr in zip(sequence_ids_list, sequence_strings_list)]
        return pd.DataFrame(error_results)

    results_list = []
    for i, seq_id in enumerate(sequence_ids_list):
        results_list.append({
            'ID': seq_id,
            'Sequence': sequence_strings_list[i],
            'Probability': float(probabilities[i]),
            'Prediction': int(predictions[i]),
            'Label': "Anti-Gram-Negative" if predictions[i] == 1 else "Non-Anti-Gram-Negative"
        })
    results_df = pd.DataFrame(results_list)
    
    if not results_df.empty:
        prediction_counts = Counter(results_df['Label'])
        total = len(results_df)
        print("Prediction summary:")
        for label, count in prediction_counts.items():
            percentage = count / total * 100
            print(f"  {label}: {count} ({percentage:.2f}%)")
    return results_df

# Initialize Flask application
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# Global variables
APP_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_DIR = os.path.dirname(APP_ROOT_DIR)
KERAS_MODEL_PATH = os.path.join(PROJECT_ROOT_DIR, 'model', 'hybrid_classifier_best_tuned.keras')
SCALER_PATH_APP = os.path.join(PROJECT_ROOT_DIR, 'model', 'hybrid_model_scaler.pkl')

keras_model_global = None 
global_feature_scaler_app = None 

def load_app_dependencies():
    """Loads the Keras model and scaler required by the app."""
    global keras_model_global, global_feature_scaler_app
    if keras_model_global is None:
        keras_model_global = load_keras_model(KERAS_MODEL_PATH)
    if global_feature_scaler_app is None:
        global_feature_scaler_app = load_app_scaler(SCALER_PATH_APP)
    print("Keras model and Scaler loaded for the app!")

@app.before_request
def before_first_request_func():
    if keras_model_global is None or global_feature_scaler_app is None:
        load_app_dependencies()

@app.route('/')
def predict_submit_page(): # Renamed from index, will serve the new submit page
    return render_template('predict_submit.html')

@app.route('/predict/results')
def predict_results_page(): # New route for displaying results
    return render_template('predict_results.html')

@app.route('/about')
def about_page(): # Renamed for consistency with url_for in new templates
    return render_template('about.html')

@app.route('/generate')
def generate_page(): # Renamed for consistency with url_for in new templates
    return render_template('generate.html')

@app.route('/predict', methods=['POST'])
def predict_sequence_api(): # Renamed to clarify it's an API endpoint
    global keras_model_global, global_feature_scaler_app 
    try:
        temp_fasta = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4()}.fasta")
        
        if 'fasta_file' in request.files and request.files['fasta_file'].filename:
            file = request.files['fasta_file']
            if not file.filename.endswith(('.fasta', '.fa', '.txt')):
                return jsonify({'error': 'Please upload a FASTA format file (.fasta, .fa, .txt).'}), 400
            file.save(temp_fasta)
        elif 'fasta_text' in request.form and request.form['fasta_text'].strip():
            fasta_text = request.form['fasta_text']
            if '>' in fasta_text:
                with open(temp_fasta, 'w') as f: f.write(fasta_text)
            else:
                with open(temp_fasta, 'w') as f:
                    lines = fasta_text.strip().split('\n')
                    for i, line in enumerate(lines):
                        if line.strip(): f.write(f">Seq_{i+1}\n{line.strip()}\n")
        else:
            return jsonify({'error': 'Please upload a FASTA file or input sequence data.'}), 400
        
        if not os.path.exists(temp_fasta) or os.path.getsize(temp_fasta) == 0:
            return jsonify({'error': 'Failed to create temporary FASTA file or file is empty.'}), 400
            
        sequence_count = sum(1 for _ in SeqIO.parse(temp_fasta, "fasta"))
        if sequence_count == 0:
            if os.path.exists(temp_fasta): os.remove(temp_fasta)
            return jsonify({'error': 'No valid sequences detected.'}), 400
        
        MAX_SEQUENCES_ALLOWED = 20000 
        if sequence_count > MAX_SEQUENCES_ALLOWED: 
            if os.path.exists(temp_fasta): os.remove(temp_fasta)
            return jsonify({'error': f'Too many sequences ({sequence_count}). Please limit to {MAX_SEQUENCES_ALLOWED} or less.'}), 400
        
        all_data_df = extract_features_from_fasta(temp_fasta) 
        
        if all_data_df.empty:
            if os.path.exists(temp_fasta): os.remove(temp_fasta)
            return jsonify({'error': 'Could not extract valid features from the input sequences.'}), 400
        
        sequence_ids_list = all_data_df['ID'].tolist()
        sequence_strings_list = all_data_df['Sequence'].tolist()
        global_feature_cols = [col for col in all_data_df.columns if col not in ['ID', 'Sequence']]
        global_features_input_df = all_data_df[global_feature_cols]

        results_df = predict_with_hybrid_model_app(
            k_model=keras_model_global, 
            scaler=global_feature_scaler_app, 
            sequence_ids_list=sequence_ids_list,
            sequence_strings_list=sequence_strings_list,
            global_features_df=global_features_input_df,
            threshold=0.5 
        )
        
        if os.path.exists(temp_fasta):
            try:
                os.remove(temp_fasta)
            except Exception as e_rm:
                print(f"Error deleting temporary FASTA file {temp_fasta}: {e_rm}")

        if results_df.empty and not sequence_ids_list:
             return jsonify({'error': 'No valid sequences for prediction.'}), 400
        if results_df.empty and sequence_ids_list: # Check if prediction failed for existing sequences
             # Check if all predictions are -1 (error state)
            if all(results_df['Prediction'] == -1):
                # Try to get a more specific error message if available from the Label column
                specific_error = "An error occurred during prediction"
                if 'Label' in results_df.columns and results_df['Label'].nunique() == 1:
                    specific_error = results_df['Label'].iloc[0] # e.g., "Feature dimension error"
                return jsonify({'error': f'{specific_error}, failed to generate valid results.'}), 500
            # If not all failed, proceed to format results

        results_for_json = []
        for _, row in results_df.iterrows():
            results_for_json.append({
                'id': row['ID'],
                'sequence': row['Sequence'],
                'probability': float(row['Probability']) if pd.notna(row['Probability']) else -1.0,
                'prediction': int(row['Prediction']) if pd.notna(row['Prediction']) else -1,
                'label': row['Label']
            })
        
        valid_results_count = len(results_for_json) - sum(1 for r in results_for_json if r['prediction'] == -1)
        stats = {
            'total': len(results_for_json),
            'positive': sum(1 for r in results_for_json if r['prediction'] == 1),
            'negative': sum(1 for r in results_for_json if r['prediction'] == 0),
            'failed': sum(1 for r in results_for_json if r['prediction'] == -1),
            'avg_probability': float(np.mean([r['probability'] for r in results_for_json if r['probability'] != -1.0])) if any(r['probability'] != -1.0 for r in results_for_json) else 0.0,
            'positive_percentage': round(sum(1 for r in results_for_json if r['prediction'] == 1) / valid_results_count * 100, 1) if valid_results_count > 0 else 0
        }
        
        return jsonify({
            'success': True,
            'results': results_for_json,
            'stats': stats
        })
    
    except Exception as e:
        print(f"Critical error in prediction route (/predict): {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'An internal server error occurred while processing your request. Please try again later or contact an administrator.'}), 500

@app.route('/export', methods=['POST'])
def export_results():
    """Export prediction results"""
    try:
        data = request.json
        format_type = data.get('format', 'csv')
        results = data.get('results', [])
        
        if not results:
            return jsonify({'error': 'No results to export.'}), 400
        
        df = pd.DataFrame(results)
        
        if format_type == 'csv':
            output = BytesIO()
            df.to_csv(output, index=False)
            output.seek(0)
            return send_file(
                output,
                mimetype='text/csv',
                as_attachment=True,
                download_name=f'gram_prediction_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            )
        elif format_type == 'fasta':
            output = BytesIO()
            for result in results:
                if result['prediction'] == 1:
                    output.write(f">{result['id']} | Probability={result['probability']:.4f}\n".encode('utf-8'))
                    output.write(f"{result['sequence']}\n".encode('utf-8'))
            output.seek(0)
            return send_file(
                output,
                mimetype='text/plain',
                as_attachment=True,
                download_name=f'gram_positive_{datetime.now().strftime("%Y%m%d_%H%M%S")}.fasta'
            )
        else:
            return jsonify({'error': 'Unsupported export format.'}), 400
    
    except Exception as e:
        print(f"Export error: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'Error exporting results: {str(e)}'}), 500

@app.route('/example')
def get_example():
    """Get example data"""
    example_fasta = """>AP00001
GLWSKIKEVGKEAAKAAAKAAGKAALGAVSEAV
>AP00002
YVPLPNVPQPGRRPFPTFPGQGPFNPKIKWPQGY
>AP00004
NLCERASLTWTGNCGNTGHCDTQCRNWESAKHGACHKRGNWKCFCYFDC
>AP00005
VFIDILDKVENAIHNAAQVGIGFAKPFEKLINPK
>AP00006
GNNRPVYIPQPRPPHPRI"""
    return jsonify({'fasta': example_fasta})

@app.route('/generate_sequences', methods=['POST'])
def generate_sequences_route():
    """API route for generating antimicrobial peptide sequences."""
    try:
        # Get generation service
        gen_service = get_generation_service()
        
        # Parse request parameters
        data = request.json
        num_sequences = int(data.get('num_sequences', 5))
        seq_length = int(data.get('seq_length', 40))
        sampling_method = data.get('sampling_method', 'diverse')
        temperature = float(data.get('temperature', 1.0))
        reference_sequences = data.get('reference_sequences', [])
        
        # Parameter validation
        if num_sequences < 1 or num_sequences > 50:
            return jsonify({'error': 'Number of sequences must be between 1 and 50.'}), 400
        
        if seq_length < 10 or seq_length > 100:
            return jsonify({'error': 'Sequence length must be between 10 and 100.'}), 400
        
        if sampling_method not in ['basic', 'diverse', 'top_k', 'nucleus']:
            return jsonify({'error': 'Invalid sampling method.'}), 400
        
        if temperature < 0.1 or temperature > 3.0:
            return jsonify({'error': 'Temperature must be between 0.1 and 3.0.'}), 400
        
        # Generate sequences
        result = gen_service.generate_sequences(
            num_sequences=num_sequences,
            seq_length=seq_length,
            sampling_method=sampling_method,
            temperature=temperature,
            reference_sequences=reference_sequences if reference_sequences else None,
            k=int(data.get('k', 10)),  # top_k parameter
            p=float(data.get('p', 0.9)),  # nucleus parameter
            diversity_strength=float(data.get('diversity_strength', 0.3))  # diversity parameter
        )
        
        if result['success']:
            return jsonify({
                'success': True,
                'sequences': result['sequences'],
                'parameters': result['parameters'],
                'model_info': gen_service.get_model_info()
            })
        else:
            return jsonify({
                'success': False,
                'error': result['error']
            }), 500
    
    except Exception as e:
        print(f"Error during sequence generation: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'Internal server error during sequence generation: {str(e)}'}), 500

@app.route('/model_status')
def model_status():
    """Get model status."""
    try:
        gen_service = get_generation_service()
        return jsonify(gen_service.get_model_info())
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
