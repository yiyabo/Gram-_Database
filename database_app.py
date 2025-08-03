#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Antimicrobial Peptide Database Viewer
Database viewer for antimicrobial peptides, display only
"""

import os
import pandas as pd
from flask import Flask, render_template, jsonify, request
from Bio import SeqIO

# Simplified amino acid property calculation
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'

# Amino acid properties table
AA_PROPERTIES = {
    'A': {'hydrophobicity': 1.8, 'charge': 0, 'mw': 89.1},
    'C': {'hydrophobicity': 2.5, 'charge': 0, 'mw': 121.0},
    'D': {'hydrophobicity': -3.5, 'charge': -1, 'mw': 133.1},
    'E': {'hydrophobicity': -3.5, 'charge': -1, 'mw': 147.1},
    'F': {'hydrophobicity': 2.8, 'charge': 0, 'mw': 165.2},
    'G': {'hydrophobicity': -0.4, 'charge': 0, 'mw': 75.1},
    'H': {'hydrophobicity': -3.2, 'charge': 0.1, 'mw': 155.2},
    'I': {'hydrophobicity': 4.5, 'charge': 0, 'mw': 131.2},
    'K': {'hydrophobicity': -3.9, 'charge': 1, 'mw': 146.2},
    'L': {'hydrophobicity': 3.8, 'charge': 0, 'mw': 131.2},
    'M': {'hydrophobicity': 1.9, 'charge': 0, 'mw': 149.2},
    'N': {'hydrophobicity': -3.5, 'charge': 0, 'mw': 132.1},
    'P': {'hydrophobicity': -1.6, 'charge': 0, 'mw': 115.1},
    'Q': {'hydrophobicity': -3.5, 'charge': 0, 'mw': 146.2},
    'R': {'hydrophobicity': -4.5, 'charge': 1, 'mw': 174.2},
    'S': {'hydrophobicity': -0.8, 'charge': 0, 'mw': 105.1},
    'T': {'hydrophobicity': -0.7, 'charge': 0, 'mw': 119.1},
    'V': {'hydrophobicity': 4.2, 'charge': 0, 'mw': 117.1},
    'W': {'hydrophobicity': -0.9, 'charge': 0, 'mw': 204.2},
    'Y': {'hydrophobicity': -1.3, 'charge': 0, 'mw': 181.2}
}

def calculate_simple_properties(sequence):
    """Calculate basic physicochemical properties of sequence"""
    if not sequence or not all(aa in AA_PROPERTIES for aa in sequence):
        return None
    
    length = len(sequence)
    total_charge = sum(AA_PROPERTIES[aa]['charge'] for aa in sequence)
    avg_hydrophobicity = sum(AA_PROPERTIES[aa]['hydrophobicity'] for aa in sequence) / length
    molecular_weight = sum(AA_PROPERTIES[aa]['mw'] for aa in sequence) - (length - 1) * 18.0  # subtract water molecules
    
    # Amino acid composition
    aa_counts = {aa: sequence.count(aa) for aa in AMINO_ACIDS}
    aa_frequencies = {f'{aa}': count / length * 100 for aa, count in aa_counts.items()}
    
    # Basic properties
    polar_count = sum(1 for aa in sequence if aa in 'DEHKNQRSTYW')
    hydrophobic_count = sum(1 for aa in sequence if aa in 'AILVFWM')
    
    properties = {
        'length': length,
        'charge': round(total_charge, 2),
        'hydrophobicity': round(avg_hydrophobicity, 3),
        'molecular_weight': round(molecular_weight, 1),
        'polar_residues': polar_count,
        'hydrophobic_residues': hydrophobic_count,
        'polar_percentage': round(polar_count / length * 100, 1),
        'hydrophobic_percentage': round(hydrophobic_count / length * 100, 1)
    }
    
    # Add amino acid frequencies
    properties.update(aa_frequencies)
    
    return properties

def load_database_sequences():
    """Load database sequences"""
    # Look for possible FASTA files
    possible_paths = [
        '../data/Gram+-.fasta',
        '../data/database.fasta',
        'data/Gram+-.fasta',
        'data/database.fasta',
        '../data/Gram+.fasta',
        'data/Gram+.fasta'
    ]
    
    database_file = None
    for path in possible_paths:
        if os.path.exists(path):
            database_file = path
            break
    
    if not database_file:
        print("Warning: No database file found. Creating sample data.")
        return create_sample_data()
    
    sequences = []
    try:
        for record in SeqIO.parse(database_file, "fasta"):
            sequence = str(record.seq).upper()
            if sequence and all(aa in AMINO_ACIDS for aa in sequence):
                properties = calculate_simple_properties(sequence)
                if properties:
                    sequences.append({
                        'id': record.id,
                        'sequence': sequence,
                        'description': record.description,
                        **properties
                    })
        print(f"Loaded {len(sequences)} sequences from {database_file}")
        return sequences
    except Exception as e:
        print(f"Error loading database: {e}")
        return create_sample_data()

def create_sample_data():
    """创建示例数据"""
    sample_sequences = [
        {'id': 'AMP001', 'seq': 'GLWSKIKEVGKEAAKAAAKAAGKAALGAVSEAV', 'desc': 'Antimicrobial peptide 1'},
        {'id': 'AMP002', 'seq': 'YVPLPNVPQPGRRPFPTFPGQGPFNPKIKWPQGY', 'desc': 'Antimicrobial peptide 2'},
        {'id': 'AMP003', 'seq': 'NLCERASLTWTGNCGNTGHCDTQCRNWESAKHGACHKRGNWKCFCYFDC', 'desc': 'Antimicrobial peptide 3'},
        {'id': 'AMP004', 'seq': 'VFIDILDKVENAIHNAAQVGIGFAKPFEKLINPK', 'desc': 'Antimicrobial peptide 4'},
        {'id': 'AMP005', 'seq': 'GNNRPVYIPQPRPPHPRI', 'desc': 'Antimicrobial peptide 5'},
    ]
    
    sequences = []
    for item in sample_sequences:
        properties = calculate_simple_properties(item['seq'])
        if properties:
            sequences.append({
                'id': item['id'],
                'sequence': item['seq'],
                'description': item['desc'],
                **properties
            })
    
    print(f"Created {len(sequences)} sample sequences")
    return sequences

# Flask应用
app = Flask(__name__)

# 全局变量存储数据库数据
database_data = None

def init_database():
    """初始化数据库"""
    global database_data
    if database_data is None:
        database_data = load_database_sequences()

# Flask 2.2+ compatibility: use before_request instead of before_first_request
@app.before_request
def before_first_request():
    if database_data is None:
        init_database()

@app.route('/')
def index():
    """Main page"""
    return render_template('database_simple.html')

@app.route('/api/database')
def get_database():
    """Get database data API"""
    try:
        if database_data is None:
            init_database()
        
        # Calculate statistics
        if database_data:
            total_sequences = len(database_data)
            avg_length = sum(seq['length'] for seq in database_data) / total_sequences
            length_range = (
                min(seq['length'] for seq in database_data),
                max(seq['length'] for seq in database_data)
            )
            avg_charge = sum(seq['charge'] for seq in database_data) / total_sequences
            avg_hydrophobicity = sum(seq['hydrophobicity'] for seq in database_data) / total_sequences
            
            stats = {
                'total_sequences': total_sequences,
                'average_length': round(avg_length, 1),
                'length_range': length_range,
                'average_charge': round(avg_charge, 2),
                'average_hydrophobicity': round(avg_hydrophobicity, 3)
            }
        else:
            stats = {}
        
        return jsonify({
            'success': True,
            'sequences': database_data,
            'stats': stats
        })
    
    except Exception as e:
        print(f"Database API error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/search')
def search_database():
    """Search database"""
    try:
        if database_data is None:
            init_database()
        
        # Get search parameters
        query = request.args.get('q', '').strip().upper()
        min_length = request.args.get('min_length', type=int)
        max_length = request.args.get('max_length', type=int)
        min_charge = request.args.get('min_charge', type=float)
        max_charge = request.args.get('max_charge', type=float)
        
        filtered_sequences = database_data
        
        # Apply filters
        if query:
            filtered_sequences = [
                seq for seq in filtered_sequences
                if query in seq['id'].upper() or query in seq['sequence'] or query in seq['description'].upper()
            ]
        
        if min_length is not None:
            filtered_sequences = [seq for seq in filtered_sequences if seq['length'] >= min_length]
        
        if max_length is not None:
            filtered_sequences = [seq for seq in filtered_sequences if seq['length'] <= max_length]
        
        if min_charge is not None:
            filtered_sequences = [seq for seq in filtered_sequences if seq['charge'] >= min_charge]
        
        if max_charge is not None:
            filtered_sequences = [seq for seq in filtered_sequences if seq['charge'] <= max_charge]
        
        return jsonify({
            'success': True,
            'sequences': filtered_sequences,
            'total': len(filtered_sequences)
        })
    
    except Exception as e:
        print(f"Search error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/export/fasta')
def export_fasta():
    """Export sequences as FASTA file"""
    try:
        if database_data is None:
            init_database()
        
        # Get filter parameters (same as search)
        query = request.args.get('q', '').strip().upper()
        min_length = request.args.get('min_length', type=int)
        max_length = request.args.get('max_length', type=int)
        min_charge = request.args.get('min_charge', type=float)
        max_charge = request.args.get('max_charge', type=float)
        
        filtered_sequences = database_data
        
        # Apply filters
        if query:
            filtered_sequences = [
                seq for seq in filtered_sequences
                if query in seq['id'].upper() or query in seq['sequence'] or query in seq['description'].upper()
            ]
        
        if min_length is not None:
            filtered_sequences = [seq for seq in filtered_sequences if seq['length'] >= min_length]
        
        if max_length is not None:
            filtered_sequences = [seq for seq in filtered_sequences if seq['length'] <= max_length]
        
        if min_charge is not None:
            filtered_sequences = [seq for seq in filtered_sequences if seq['charge'] >= min_charge]
        
        if max_charge is not None:
            filtered_sequences = [seq for seq in filtered_sequences if seq['charge'] <= max_charge]
        
        # Generate FASTA content
        fasta_content = ""
        for seq in filtered_sequences:
            # Create detailed description with properties
            desc = f"{seq['description']} | Length: {seq['length']} | Charge: {seq['charge']} | Hydrophobicity: {seq['hydrophobicity']}"
            fasta_content += f">{seq['id']} {desc}\n{seq['sequence']}\n"
        
        if not fasta_content:
            return jsonify({'success': False, 'error': 'No sequences found with current filters'}), 400
        
        # Create response with file download
        from flask import Response
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"peptide_database_{timestamp}.fasta"
        
        return Response(
            fasta_content,
            mimetype='text/plain',
            headers={
                'Content-Disposition': f'attachment; filename={filename}',
                'Content-Type': 'application/octet-stream'
            }
        )
        
    except Exception as e:
        print(f"Export error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'version': 'database-only'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
