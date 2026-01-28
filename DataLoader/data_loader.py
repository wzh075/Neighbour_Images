import os
import glob
import yaml
import h5py
import json
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np


def load_config(config_path):
    """
    Load configuration from YAML file
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


class PointCloudLoader:
    """
    Loader for ModelNet40 point cloud data
    """
    def __init__(self, pointcloud_root):
        self.pointcloud_root = pointcloud_root
        self.pointcloud_data = {}
        self.id2file_mapping = {}
        self.category_mapping = {}
        
        # Load all HDF5 files and build mappings
        self._load_pointcloud_data()
        self._build_id2file_mapping()
        
    def _load_pointcloud_data(self):
        """
        Load all HDF5 files containing point cloud data
        """
        # Find all HDF5 files
        h5_files = glob.glob(os.path.join(self.pointcloud_root, '*.h5'))
        
        # Load each HDF5 file
        for h5_file in h5_files:
            with h5py.File(h5_file, 'r') as f:
                # Get file name without extension
                file_name = os.path.basename(h5_file).split('.')[0]
                
                # Fix file name format to match JSON mapping
                # Convert from 'ply_data_test0' to 'ply_data_test_0'
                if 'test' in file_name:
                    fixed_name = file_name.replace('test', 'test_')
                elif 'train' in file_name:
                    fixed_name = file_name.replace('train', 'train_')
                else:
                    fixed_name = file_name
                
                # Store point cloud data
                self.pointcloud_data[fixed_name] = {
                    'points': f['data'][:],  # Shape: (N, 2048, 3)
                    'labels': f['label'][:]   # Shape: (N,)
                }
    
    def _build_id2file_mapping(self):
        """
        Build mapping from obj_id to point cloud data index
        """
        # Find all id2file JSON files
        json_files = glob.glob(os.path.join(self.pointcloud_root, '*_id2file.json'))
        
        # Load each JSON file
        for json_file in json_files:
            with open(json_file, 'r') as f:
                file_paths = json.load(f)
            
            # Get file name without extension
            file_name = os.path.basename(json_file).split('_id2file.json')[0]
            
            # Build mapping from obj_id to index
            for idx, file_path in enumerate(file_paths):
                # Extract category and object_id
                parts = file_path.split('/')
                category = parts[0]
                obj_filename = parts[1].split('.')[0]
                
                # Use the full filename without extension as obj_id
                # This matches the format used in the image dataset (e.g., "airplane_0627")
                obj_id = obj_filename
                
                # Create unique key (just the obj_id since it already contains category)
                key = obj_id
                
                # Store mapping
                self.id2file_mapping[key] = {
                    'file_name': file_name,
                    'index': idx
                }
                
                # Update category mapping
                if category not in self.category_mapping:
                    self.category_mapping[category] = []
                self.category_mapping[category].append(key)
    
    def get_pointcloud(self, category, obj_id):
        """
        Get point cloud data for a given category and object_id
        """
        # Debug: Print input parameters
        # print(f"DEBUG: Looking for point cloud for category={category}, obj_id={obj_id}")
        
        # Try different key formats
        potential_keys = [
            obj_id,  # Full object ID (e.g., "airplane_0627")
            f'{category}_{obj_id.split("_")[-1]}',  # Category + numeric ID
            f'{category}/{obj_id}.ply'  # Full path format
        ]
        
        found_key = None
        for key in potential_keys:
            if key in self.id2file_mapping:
                found_key = key
                # print(f"DEBUG: Found key: {key}")
                break
        
        if found_key is None:
            # print(f"DEBUG: No mapping found for any potential key: {potential_keys}")
            return None
        
        # Get file name and index
        mapping = self.id2file_mapping[found_key]
        file_name = mapping['file_name']
        index = mapping['index']
        
        # Debug: Print mapping info
        # print(f"DEBUG: Mapping - file_name={file_name}, index={index}")
        
        # Get point cloud data
        pointcloud_data = self.pointcloud_data[file_name]
        points = pointcloud_data['points'][index]
        label = pointcloud_data['labels'][index]
        
        # Debug: Print point cloud info
        # print(f"DEBUG: Point cloud loaded successfully, shape={points.shape}")
        
        return {
            'points': points,
            'label': label
        }

class ModelNet40NeighbourDataset(Dataset):
    """
    Dataset loader for ModelNet40 with Neighbour Images.
    
    Data Structure Concept:
    Unlike standard multi-view datasets, this loader expects a specific folder structure 
    where each view folder contains a group of 'Neighbour Images'.
    
    Structure: Object -> View_X -> [img_0, img_1, img_2, img_3, img_4]
    
    Where:
    - img_0: Center view (standard rendering).
    - img_1~4: Neighbour views (rendered with slight camera deviations: up, down, left, right).
    
    These images provide complementary local geometry information for the specific viewpoint.
    """
    def __init__(self, root_dir, transform=None, expected_images_per_view=5, pointcloud_root=None):
        self.root_dir = root_dir
        self.transform = transform
        self.expected_images_per_view = expected_images_per_view
        self.pointcloud_root = pointcloud_root
        self.data = []
        self.category_counts = {}
        self.pointcloud_load_failures = 0  # 新增：统计点云加载失败的数量
        self.total_pointcloud_attempts = 0  # 新增：统计点云加载尝试的总数量
        
        # Initialize point cloud loader if root is provided
        self.pointcloud_loader = PointCloudLoader(pointcloud_root) if pointcloud_root else None
        
        self._build_index()
        self._print_category_counts()
        # 移除：初始化时不打印点云统计，因为此时还没有实际加载点云数据
    
    def _build_index(self):
        categories = os.listdir(self.root_dir)
        categories = [c for c in categories if os.path.isdir(os.path.join(self.root_dir, c))]
        
        total_objects = 0
        # 计算总对象数用于进度条
        for category in categories:
            category_path = os.path.join(self.root_dir, category)
            splits = os.listdir(category_path)
            for split in splits:
                split_path = os.path.join(category_path, split)
                objects = os.listdir(split_path)
                total_objects += len(objects)
        
        progress_bar = tqdm(total=total_objects, desc="Loading dataset")
        
        for category in categories:
            category_path = os.path.join(self.root_dir, category)
            splits = os.listdir(category_path)
            
            for split in splits:
                split_path = os.path.join(category_path, split)
                objects = os.listdir(split_path)
                
                for obj in objects:
                    obj_path = os.path.join(split_path, obj)
                    views = sorted(os.listdir(obj_path), key=lambda v: (''.join([c for c in v if not c.isdigit()]), int(''.join([c for c in v if c.isdigit()]) or -1)))
                    
                    # 检查视点数量是否一致（根据数据集自适应）
                    view_paths = []
                    for view in views:
                        view_path = os.path.join(obj_path, view)
                        if os.path.isdir(view_path):
                            view_paths.append(view_path)
                    
                    # 收集每个视点下的邻域图 (Neighbour Images)
                    view_images = {}
                    for view_path in view_paths:
                        view_name = os.path.basename(view_path)
                        images = glob.glob(os.path.join(view_path, "*.png"))
                        
                        # 检查图片数量是否符合预期 (应为 5 张: 1中心 + 4邻域)
                        if len(images) != self.expected_images_per_view:
                            raise ValueError(f"Error: Object {obj} in category {category} has {len(images)} images in {view_name}, expected {self.expected_images_per_view}")
                        
                        # 按索引排序图片以保持邻域的结构顺序 (e.g., 0=center, 1=up, 2=down...)
                        images.sort(key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))
                        view_images[view_name] = images
                    
                    # 检查所有视点的图片数量是否一致
                    num_views = len(view_images)
                    if num_views == 0:
                        raise ValueError(f"Error: Object {obj} in category {category} has no views")
                    
                    # 存储对象信息
                    self.data.append({
                        'category': category,
                        'object_id': obj,
                        'view_images': view_images
                    })
                    
                    # 更新类别计数
                    if category in self.category_counts:
                        self.category_counts[category] += 1
                    else:
                        self.category_counts[category] = 1
                    
                    progress_bar.update(1)
        
        progress_bar.close()
        
        # 检查是否有类别没有数据
        for category in categories:
            if category not in self.category_counts:
                raise ValueError(f"Error: Category {category} has no data")
    
    def _print_category_counts(self):
        print("\nCategory Sample Counts:")
        print("-" * 30)
        total_samples = 0
        for category, count in sorted(self.category_counts.items()):
            print(f"{category}: {count} samples")
            total_samples += count
        print("-" * 30)
        print(f"Total samples: {total_samples}")
    
    def _print_pointcloud_load_stats(self):
        """打印点云加载统计信息"""
        if self.pointcloud_loader:
            print("\nPointCloud Load Statistics:")
            print("-" * 30)
            print(f"Total pointcloud attempts: {self.total_pointcloud_attempts}")
            print(f"Failed pointcloud loads: {self.pointcloud_load_failures}")
            if self.total_pointcloud_attempts > 0:
                success_rate = ((self.total_pointcloud_attempts - self.pointcloud_load_failures) / self.total_pointcloud_attempts) * 100
                print(f"Pointcloud load success rate: {success_rate:.2f}%")
            print("-" * 30)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        category = sample['category']
        object_id = sample['object_id']
        view_images = sample['view_images']
        
        # 加载点云数据（先检查点云是否有效，避免后续处理无效数据）
        pointcloud = None
        if self.pointcloud_loader:
            self.total_pointcloud_attempts += 1  # 增加点云加载尝试计数
            pointcloud_data = self.pointcloud_loader.get_pointcloud(category, object_id)
            if pointcloud_data:
                pointcloud = torch.from_numpy(pointcloud_data['points'])
            else:
                # 如果点云加载失败，增加失败计数并重新获取下一个样本
                self.pointcloud_load_failures += 1
                return self.__getitem__((idx + 1) % len(self))
        
        # 加载所有视点的图片
        loaded_views = {}
        for view_name, image_paths in view_images.items():
            images = []
            for img_path in image_paths:
                img = Image.open(img_path).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                images.append(img)
            # 将同一视点的邻域图堆叠在一起: (5, C, H, W)
            # 这是一个 "Neighbour Group"，而非随机 Batch
            loaded_views[view_name] = torch.stack(images)
        
        # 构建返回字典
        result = {
            'category': category,
            'object_id': object_id,
            'views': loaded_views
        }
        
        # 只有当pointcloud_loader存在且pointcloud不为None时，才添加pointcloud键
        if self.pointcloud_loader and pointcloud is not None:
            result['pointcloud'] = pointcloud
        
        return result

def main():
    # 从配置文件加载配置
    config = load_config("./config.yaml")
    
    # 简单的图像转换
    from torchvision import transforms
    transform_list = [
        transforms.Resize(tuple(config['transform']['resize'])),
        transforms.ToTensor()
    ]
    
    # 添加归一化（如果配置中存在）
    if 'normalize' in config['transform']:
        norm_config = config['transform']['normalize']
        transform_list.append(transforms.Normalize(mean=norm_config['mean'], std=norm_config['std']))
    
    transform = transforms.Compose(transform_list)
    
    try:
        # Create dataset with point cloud support if pointcloud_root is provided
        dataset = ModelNet40NeighbourDataset(
            root_dir=config['dataset']['root_dir'],
            transform=transform,
            expected_images_per_view=config['dataset']['expected_images_per_view'],
            pointcloud_root=config.get('pointcloud', {}).get('root_dir')
        )
        print(f"\nDataset loaded successfully with {len(dataset)} objects")
        
        # 创建数据加载器
        dataloader = DataLoader(
            dataset, 
            batch_size=config['dataloader']['batch_size'],
            shuffle=config['dataloader']['shuffle'],
            num_workers=config['dataloader']['num_workers'],
            drop_last=config['dataloader']['drop_last']
        )
        
        # 测试加载一个批次
        print("\nTesting data loading...")
        for batch in dataloader:
            print(f"Category: {batch['category'][0]}")
            print(f"Object ID: {batch['object_id'][0]}")
            print(f"Number of views: {len(batch['views'])}")
            for view_name, view_tensor in batch['views'].items():
                print(f"View {view_name}: {view_tensor.shape}")  # 期望形状: (batch_size, 5, 3, 224, 224)
            
            # 检查点云数据
            if 'pointcloud' in batch and batch['pointcloud'] is not None:
                print(f"Point cloud shape: {batch['pointcloud'].shape}")  # 期望形状: (batch_size, 2048, 3)
            else:
                print("No point cloud data available")
            
            break  # 只测试一个批次
        
        print("\nData loader test completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
