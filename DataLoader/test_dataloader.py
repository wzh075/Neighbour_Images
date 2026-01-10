import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
from torchvision import transforms
from PIL import Image
import time

# Import custom modules
from data_loader import ModelNet40NeighbourDataset, load_config


def visualize_images(views, category, object_id):
    """
    Visualize images from all views and crops
    """
    view_names = list(views.keys())
    num_views = len(view_names)
    num_crops = views[view_names[0]].shape[0]
    
    # Create inverse transform to convert tensor back to image
    inv_transform = transforms.Compose([
        transforms.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
        transforms.ToPILImage()
    ])
    
    # Create figure
    fig, axes = plt.subplots(num_views, num_crops, figsize=(15, num_views * 4))
    fig.suptitle(f'{category} - {object_id} Views and Crops', fontsize=16)
    
    # Plot each view and crop
    for i, view_name in enumerate(view_names):
        for j in range(num_crops):
            # Get image tensor
            img_tensor = views[view_name][j]
            
            # Convert to PIL image
            img = inv_transform(img_tensor)
            
            # Plot
            ax = axes[i, j] if num_views > 1 else axes[j]
            ax.imshow(img)
            ax.set_title(f'{view_name} - Crop {j+1}')
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()


def visualize_pointcloud(pointcloud, category, object_id):
    """
    Visualize point cloud data
    """
    if pointcloud is None:
        print('No point cloud data available for visualization')
        return
    
    # Convert to numpy array
    if isinstance(pointcloud, torch.Tensor):
        pointcloud = pointcloud.numpy()
    
    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot point cloud
    ax.scatter(pointcloud[:, 0], pointcloud[:, 1], pointcloud[:, 2], s=1, alpha=0.5)
    
    # Set plot properties
    ax.set_title(f'{category} - {object_id} Point Cloud')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Set equal aspect ratio
    ax.set_aspect('equal')
    
    plt.show()


def test_single_sample(dataset, sample_idx=0):
    """
    Test loading and visualizing a single sample
    """
    print(f'Testing single sample with index {sample_idx}')
    
    # Get sample
    sample = dataset[sample_idx]
    
    # Print sample information
    print(f'Category: {sample["category"]}')
    print(f'Object ID: {sample["object_id"]}')
    print(f'Number of views: {len(sample["views"])}')
    
    # Check point cloud
    if sample['pointcloud'] is not None:
        print(f'Point cloud shape: {sample["pointcloud"].shape}')
    else:
        print('No point cloud data available')
    
    # Visualize images
    visualize_images(sample['views'], sample['category'], sample['object_id'])
    
    # Visualize point cloud if available
    if sample['pointcloud'] is not None:
        visualize_pointcloud(sample['pointcloud'], sample['category'], sample['object_id'])


def test_batch(dataloader, num_batches=1):
    """
    Test loading and processing batches of data
    """
    print(f'Testing {num_batches} batches of data')
    
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_batches:
            break
        
        print(f'\nBatch {batch_idx + 1}:')
        print(f'  Batch size: {batch["views"][list(batch["views"].keys())[0]].shape[0]}')
        print(f'  Categories: {batch["category"]}')
        print(f'  Object IDs: {batch["object_id"]}')
        print(f'  Number of views: {len(batch["views"])}')
        
        # Get view names
        view_names = list(batch['views'].keys())
        
        # Print shape of one view
        print(f'  View shape: {batch["views"][view_names[0]].shape}')
        
        # Check point cloud data
        if batch['pointcloud'] is not None:
            print(f'  Point cloud shape: {batch["pointcloud"].shape}')
        else:
            print('  No point cloud data in batch')
        
        # Visualize first sample in batch
        first_sample = {
            'category': batch['category'][0],
            'object_id': batch['object_id'][0],
            'views': {view_name: batch['views'][view_name][0] for view_name in view_names},
            'pointcloud': batch['pointcloud'][0] if batch['pointcloud'] is not None else None
        }
        
        print(f'\nVisualizing first sample: {first_sample["category"]} - {first_sample["object_id"]}')
        
        # Visualize images
        visualize_images(first_sample['views'], first_sample['category'], first_sample['object_id'])
        
        # Visualize point cloud if available
        if first_sample['pointcloud'] is not None:
            visualize_pointcloud(first_sample['pointcloud'], first_sample['category'], first_sample['object_id'])


def test_performance(dataloader, num_batches=5):
    """
    Test the performance of the dataloader
    """
    print(f'Measuring performance for {num_batches} batches')
    
    total_time = 0
    
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_batches:
            break
        
        start_time = time.time()
        
        # Simulate processing (forward pass)
        for view_name, view_tensor in batch['views'].items():
            _ = view_tensor.mean()  # Simple operation
        
        if batch['pointcloud'] is not None:
            _ = batch['pointcloud'].mean()  # Simple operation
        
        end_time = time.time()
        batch_time = end_time - start_time
        total_time += batch_time
        
        print(f'Batch {batch_idx + 1} processed in {batch_time:.4f} seconds')
    
    print(f'\nAverage time per batch: {total_time / num_batches:.4f} seconds')
    print(f'Total time for {num_batches} batches: {total_time:.4f} seconds')


def custom_collate(batch):
    """
    Custom collate function to handle None values in point cloud data
    """
    # Extract keys from first sample
    keys = batch[0].keys()
    
    # Initialize result dictionary
    result = {}
    
    for key in keys:
        if key == 'pointcloud':
            # Handle pointcloud separately since it might contain None values
            pointclouds = [sample[key] for sample in batch]
            
            # Check if all are None
            if all(pc is None for pc in pointclouds):
                result[key] = None
            else:
                # Filter out None values and stack
                # Note: This approach discards samples with no point cloud
                # For a better approach, you could use a placeholder tensor
                valid_pcs = [pc for pc in pointclouds if pc is not None]
                if valid_pcs:
                    result[key] = torch.stack(valid_pcs)
                else:
                    result[key] = None
        elif key == 'views':
            # Handle views dictionary
            views_dict = {}
            view_names = batch[0][key].keys()
            
            for view_name in view_names:
                # Stack tensors for each view
                view_tensors = [sample[key][view_name] for sample in batch]
                views_dict[view_name] = torch.stack(view_tensors)
            
            result[key] = views_dict
        else:
            # Handle other keys (category, object_id)
            result[key] = [sample[key] for sample in batch]
    
    return result


def main():
    """
    Main function to test the dataloader
    """
    print('=== ModelNet40 DataLoader Test ===')
    
    # Load configuration
    config = load_config('./config.yaml')
    
    # Print configuration summary
    print('\nConfiguration Summary:')
    print(f'Dataset Root: {config["dataset"]["root_dir"]}')
    print(f'Expected Images per View: {config["dataset"]["expected_images_per_view"]}')
    if 'pointcloud' in config:
        print(f'Point Cloud Root: {config["pointcloud"]["root_dir"]}')
    print(f'Batch Size: {config["dataloader"]["batch_size"]}')
    print(f'Num Workers: {config["dataloader"]["num_workers"]}')
    
    # Create image transform
    transform_list = [
        transforms.Resize(tuple(config['transform']['resize'])),
        transforms.ToTensor()
    ]
    
    # Add normalization if specified
    if 'normalize' in config['transform']:
        norm_config = config['transform']['normalize']
        transform_list.append(transforms.Normalize(mean=norm_config['mean'], std=norm_config['std']))
    
    transform = transforms.Compose(transform_list)
    
    # Initialize dataset
    print('\nInitializing dataset...')
    dataset = ModelNet40NeighbourDataset(
        root_dir=config['dataset']['root_dir'],
        transform=transform,
        expected_images_per_view=config['dataset']['expected_images_per_view'],
        pointcloud_root=config.get('pointcloud', {}).get('root_dir')
    )
    
    print(f'Dataset initialized with {len(dataset)} objects')
    
    # Initialize dataloader with custom collate function
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config['dataloader']['batch_size'],
        shuffle=config['dataloader']['shuffle'],
        num_workers=config['dataloader']['num_workers'],
        drop_last=config['dataloader']['drop_last'],
        collate_fn=custom_collate
    )
    
    # Test single sample
    print('\n=== Single Sample Test ===')
    test_single_sample(dataset, sample_idx=0)
    
    # Test batch processing
    print('\n=== Batch Processing Test ===')
    test_batch(dataloader, num_batches=1)
    
    # Test performance
    print('\n=== Performance Test ===')
    test_performance(dataloader, num_batches=5)
    
    print('\n=== Test Complete ===')


if __name__ == "__main__":
    main()
