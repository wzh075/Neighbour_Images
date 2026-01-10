import os
import h5py
import json
import glob
import numpy as np
import torch

class PointCloudDebugLoader:
    """
    Debug loader for ModelNet40 point cloud data
    """
    def __init__(self, pointcloud_root):
        self.pointcloud_root = pointcloud_root
        self.pointcloud_data = {}
        self.id2file_mapping = {}
        
        # Load all HDF5 files
        self._load_pointcloud_data()
        
        # Load all id2file mappings
        self._load_id2file_mappings()
        
        # Print debug information
        self._print_debug_info()
    
    def _load_pointcloud_data(self):
        """Load point cloud data from HDF5 files"""
        h5_files = glob.glob(os.path.join(self.pointcloud_root, '*.h5'))
        
        for h5_file in h5_files:
            with h5py.File(h5_file, 'r') as f:
                file_name = os.path.basename(h5_file).split('.')[0]
                
                # Fix file name format to match JSON mapping
                # Convert from 'ply_data_test0' to 'ply_data_test_0'
                if 'test' in file_name:
                    fixed_name = file_name.replace('test', 'test_')
                elif 'train' in file_name:
                    fixed_name = file_name.replace('train', 'train_')
                else:
                    fixed_name = file_name
                
                self.pointcloud_data[fixed_name] = {
                    'points': f['data'][:],  # Shape: (N, 2048, 3)
                    'labels': f['label'][:]   # Shape: (N,)
                }
                print(f"Loaded HDF5 file: {file_name}, containing {len(self.pointcloud_data[fixed_name]['points'])} samples")
    
    def _load_id2file_mappings(self):
        """Load id2file mappings from JSON files"""
        json_files = glob.glob(os.path.join(self.pointcloud_root, '*_id2file.json'))
        
        for json_file in json_files:
            with open(json_file, 'r') as f:
                file_paths = json.load(f)
            
            file_name = os.path.basename(json_file).split('_id2file.json')[0]
            
            # Build mapping
            for idx, file_path in enumerate(file_paths):
                # Extract key from file path
                # Example file path: "airplane/airplane_0627.ply"
                parts = file_path.split('/')
                if len(parts) != 2:
                    continue
                
                category, filename = parts
                obj_id = filename.split('.')[0]  # "airplane_0627"
                
                # Store multiple potential keys for flexibility
                self.id2file_mapping[obj_id] = {
                    'file_name': file_name,
                    'index': idx,
                    'full_path': file_path
                }
                
                # Also store with category + numeric ID
                numeric_id = obj_id.split('_')[-1]  # "0627"
                category_numeric_key = f"{category}_{numeric_id}"
                if category_numeric_key not in self.id2file_mapping:
                    self.id2file_mapping[category_numeric_key] = {
                        'file_name': file_name,
                        'index': idx,
                        'full_path': file_path
                    }
        
        print(f"Loaded {len(self.id2file_mapping)} mappings from JSON files")
    
    def _print_debug_info(self):
        """Print debug information"""
        print("\n=== DEBUG INFORMATION ===")
        print(f"Number of HDF5 files: {len(self.pointcloud_data)}")
        print(f"Number of mappings: {len(self.id2file_mapping)}")
        
        # Print first few mappings
        print("\nFirst 5 mappings:")
        for i, (key, value) in enumerate(list(self.id2file_mapping.items())[:5]):
            print(f"  {i+1}. {key} -> {value}")
    
    def find_pointcloud(self, category, obj_id):
        """Find point cloud for given category and object ID"""
        print(f"\n=== Looking for point cloud ===")
        print(f"Category: {category}")
        print(f"Object ID: {obj_id}")
        
        # Try different key formats
        potential_keys = [
            obj_id,  # Full object ID (e.g., "airplane_0627")
            f"{category}_{obj_id.split('_')[-1]}",  # Category + numeric ID
        ]
        
        for key in potential_keys:
            if key in self.id2file_mapping:
                mapping = self.id2file_mapping[key]
                print(f"✓ Found key: {key}")
                print(f"  Mapping: {mapping}")
                
                # Load point cloud data
                if mapping['file_name'] in self.pointcloud_data:
                    pc_data = self.pointcloud_data[mapping['file_name']]
                    points = pc_data['points'][mapping['index']]
                    label = pc_data['labels'][mapping['index']]
                    
                    print(f"✓ Successfully loaded point cloud")
                    print(f"  Point cloud shape: {points.shape}")
                    print(f"  Label: {label}")
                    
                    return {
                        'points': points,
                        'label': label
                    }
                else:
                    print(f"✗ File name {mapping['file_name']} not found in point cloud data")
            else:
                print(f"✗ Key not found: {key}")
        
        return None
    
    def get_random_sample(self):
        """Get a random sample from the point cloud data"""
        # Get a random key
        keys = list(self.id2file_mapping.keys())
        if not keys:
            return None
        
        import random
        random_key = random.choice(keys)
        mapping = self.id2file_mapping[random_key]
        
        print(f"\n=== Random Sample ===")
        print(f"Key: {random_key}")
        print(f"Mapping: {mapping}")
        
        if mapping['file_name'] in self.pointcloud_data:
            pc_data = self.pointcloud_data[mapping['file_name']]
            points = pc_data['points'][mapping['index']]
            label = pc_data['labels'][mapping['index']]
            
            print(f"Point cloud shape: {points.shape}")
            print(f"Label: {label}")
            
            return {
                'key': random_key,
                'points': points,
                'label': label
            }
        
        return None


def main():
    # Point cloud root directory
    pointcloud_root = '../Dataset/modelnet40_ply_hdf5_2048'
    
    # Create debug loader
    loader = PointCloudDebugLoader(pointcloud_root)
    
    # Try to find a specific point cloud
    # Example from the test output: airplane_0627
    print("\n\n=== Testing with airplane_0627 ===")
    result = loader.find_pointcloud('airplane', 'airplane_0627')
    
    if result:
        print("\n✓ Point cloud found successfully!")
    else:
        print("\n✗ Point cloud not found")
    
    # Try a random sample
    print("\n\n=== Getting random sample ===")
    random_sample = loader.get_random_sample()
    
    if random_sample:
        print("\n✓ Random sample retrieved successfully!")
    else:
        print("\n✗ Failed to retrieve random sample")


if __name__ == "__main__":
    main()
