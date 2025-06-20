import bz2
import os

def extract_bz2(bz2_path, output_path):
    with open(bz2_path, 'rb') as source, open(output_path, 'wb') as dest:
        dest.write(bz2.decompress(source.read()))
    print(f"Extracted {bz2_path} to {output_path}")

if __name__ == "__main__":
    bz2_file = "models/shape_predictor_68_face_landmarks.dat.bz2"
    output_file = "models/shape_predictor_68_face_landmarks.dat"
    
    if os.path.exists(bz2_file):
        extract_bz2(bz2_file, output_file)
    else:
        print(f"Error: {bz2_file} not found") 