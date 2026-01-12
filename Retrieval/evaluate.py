import os
import sys
import argparse
import logging
import h5py
import numpy as np
import torch
from tqdm import tqdm

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class RetrievalEngine:
    """检索引擎类，用于执行各种检索任务"""
    
    def __init__(self, feature_file, device='cpu', batch_size=128):
        """
        初始化检索引擎
        
        :param feature_file: 特征文件路径
        :param device: 计算设备
        :param batch_size: 检索计算时的批次大小
        """
        self.feature_file = feature_file
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.features = None
        self.object_ids = None
        self.categories = None
        
        # 加载特征
        self.load_features()
        
    def load_features(self):
        """加载特征文件"""
        logging.info(f"加载特征文件: {self.feature_file}")
        
        with h5py.File(self.feature_file, 'r') as f:
            # 加载特征
            self.refined_view_feats = torch.from_numpy(f['refined_view_feats'][:]).to(self.device)
            self.global_image_feats = torch.from_numpy(f['global_image_feats'][:]).to(self.device)
            self.global_point_feats = torch.from_numpy(f['global_point_feats'][:]).to(self.device)
            
            # 加载元数据
            self.object_ids = [obj_id.decode('utf-8') for obj_id in f['object_ids'][:]]
            self.categories = [cat.decode('utf-8') for cat in f['categories'][:]]
        
        # 展平refined_view_feats，用于view-based检索
        # 原始形状: (Total_Samples, N_Views, 1024)
        # 展平后形状: (Total_Samples * N_Views, 1024)
        self.flattened_view_feats = self.refined_view_feats.view(-1, 1024)
        
        # 生成展平后的object_ids和categories
        # 每个样本的N_Views个view都对应相同的object_id和category
        N_Views = self.refined_view_feats.shape[1]
        self.flattened_object_ids = []
        self.flattened_categories = []
        for obj_id, cat in zip(self.object_ids, self.categories):
            self.flattened_object_ids.extend([obj_id] * N_Views)
            self.flattened_categories.extend([cat] * N_Views)
        
        logging.info(f"特征加载完成:")
        logging.info(f"  - 总样本数: {len(self.object_ids)}")
        logging.info(f"  - 总View数: {len(self.flattened_view_feats)} (每个样本{int(len(self.flattened_view_feats)/len(self.object_ids))}个view)")
        logging.info(f"  - 特征维度: {self.global_image_feats.shape[1]}")
    
    def compute_cosine_similarity(self, query_feats, gallery_feats):
        """
        计算查询特征和图库特征之间的余弦相似度
        
        :param query_feats: 查询特征，形状为 (num_queries, feat_dim)
        :param gallery_feats: 图库特征，形状为 (num_gallery, feat_dim)
        :return: 相似度矩阵，形状为 (num_queries, num_gallery)
        """
        # 验证输入张量的形状
        logging.info(f"计算相似度 - 查询特征形状: {query_feats.shape}, 图库特征形状: {gallery_feats.shape}")
        
        # 确保特征维度匹配
        if query_feats.shape[1] != gallery_feats.shape[1]:
            raise ValueError(f"特征维度不匹配: 查询特征维度={query_feats.shape[1]}, 图库特征维度={gallery_feats.shape[1]}")
        
        # 归一化特征
        query_feats = query_feats / query_feats.norm(dim=1, keepdim=True)
        gallery_feats = gallery_feats / gallery_feats.norm(dim=1, keepdim=True)
        
        # 批次计算相似度，避免显存溢出
        num_queries = query_feats.shape[0]
        num_gallery = gallery_feats.shape[0]
        similarity_matrix = torch.zeros((num_queries, num_gallery), device=self.device)
        
        # 调整批次大小以适应GPU内存
        adjusted_batch_size = self.batch_size
        if self.device.type == 'cuda':
            # 估计每个批次需要的显存并调整批次大小
            estimated_memory_per_batch = (query_feats.element_size() * query_feats.shape[1] * num_gallery * 2) / (1024 ** 3)
            logging.info(f"估计每个批次需要显存: {estimated_memory_per_batch:.2f} GB")
            
            # 获取当前GPU的可用显存
            available_memory = torch.cuda.get_device_properties(self.device).total_memory - torch.cuda.memory_allocated(self.device)
            available_memory_gb = available_memory / (1024 ** 3)
            logging.info(f"GPU可用显存: {available_memory_gb:.2f} GB")
            
            # 根据可用显存调整批次大小
            if estimated_memory_per_batch > available_memory_gb * 0.8:
                adjusted_batch_size = max(1, int(self.batch_size * (available_memory_gb * 0.8 / estimated_memory_per_batch)))
                logging.info(f"批次大小调整为: {adjusted_batch_size}")
        
        try:
            for i in tqdm(range(0, num_queries, adjusted_batch_size), desc="计算相似度"):
                query_batch = query_feats[i:i+adjusted_batch_size]
                # 确保查询批次的形状正确
                if query_batch.dim() != 2:
                    query_batch = query_batch.view(-1, query_feats.shape[1])
                sim_batch = torch.matmul(query_batch, gallery_feats.t())
                similarity_matrix[i:i+adjusted_batch_size] = sim_batch
                
                # 释放中间张量的内存
                torch.cuda.empty_cache() if self.device.type == 'cuda' else None
        except Exception as e:
            logging.error(f"计算相似度时出错: {e}")
            logging.error(f"查询批次形状: {query_batch.shape}")
            logging.error(f"图库特征转置后形状: {gallery_feats.t().shape}")
            # 尝试在CPU上计算
            logging.info("尝试在CPU上计算相似度...")
            query_feats_cpu = query_feats.cpu()
            gallery_feats_cpu = gallery_feats.cpu()
            similarity_matrix_cpu = torch.zeros((num_queries, num_gallery))
            
            for i in tqdm(range(0, num_queries, adjusted_batch_size), desc="计算相似度 (CPU)"):
                query_batch_cpu = query_feats_cpu[i:i+adjusted_batch_size]
                if query_batch_cpu.dim() != 2:
                    query_batch_cpu = query_batch_cpu.view(-1, query_feats_cpu.shape[1])
                sim_batch_cpu = torch.matmul(query_batch_cpu, gallery_feats_cpu.t())
                similarity_matrix_cpu[i:i+adjusted_batch_size] = sim_batch_cpu
            
            return similarity_matrix_cpu.to(self.device)
        
        return similarity_matrix
    
    def evaluate_recall(self, similarity_matrix, query_object_ids, gallery_object_ids, exclude_self=False):
        """
        计算Recall@K指标
        
        :param similarity_matrix: 相似度矩阵，形状为 (num_queries, num_gallery)
        :param query_object_ids: 查询的object_id列表
        :param gallery_object_ids: 图库的object_id列表
        :param exclude_self: 是否排除自身匹配（用于View->View检索）
        :return: recall@1, recall@5, recall@10
        """
        num_queries = similarity_matrix.shape[0]
        num_gallery = similarity_matrix.shape[1]
        list_gallery_length = len(gallery_object_ids)
        
        # 确保图库大小匹配
        if num_gallery != list_gallery_length:
            logging.warning(f"图库大小不匹配: similarity_matrix显示{num_gallery}个样本，但gallery_object_ids列表长度为{list_gallery_length}")
            num_gallery = min(num_gallery, list_gallery_length)
        
        # 计算每个查询的top-k索引
        top_k = 10  # 计算到R@10
        _, indices = similarity_matrix.topk(top_k, dim=1, largest=True, sorted=True)
        
        # 将indices移动到CPU上，并转换为numpy数组以便安全处理
        indices = indices.cpu().numpy()
        
        # 过滤掉超出范围的索引值
        # 异常大的索引值（如内存地址）通常大于gallery长度的100倍
        max_valid_idx = list_gallery_length - 1
        indices = np.where(indices > max_valid_idx, -1, indices)  # 将无效索引设为-1
        
        # 计算Recall
        recall_1 = 0
        recall_5 = 0
        recall_10 = 0
        
        for i in tqdm(range(num_queries), desc="计算Recall"):
            query_obj_id = query_object_ids[i]
            
            # 安全地获取检索结果，确保索引在有效范围内
            retrieved_obj_ids = []
            for idx in indices[i]:
                if idx < 0 or idx >= list_gallery_length:
                    continue
                retrieved_obj_ids.append(gallery_object_ids[idx])
            
            # 排除自身匹配（如果需要）
            if exclude_self:
                # 对于View->View检索，每个query_view对应一个gallery_view
                # 自身位置为 i，需要从检索结果中排除
                if i in indices[i]:
                    # 创建一个新的列表，排除自身匹配
                    new_retrieved_obj_ids = []
                    for idx, obj_id in zip(indices[i], retrieved_obj_ids):
                        if idx != i:
                            new_retrieved_obj_ids.append(obj_id)
                    retrieved_obj_ids = new_retrieved_obj_ids
            
            # 检查是否在top-k中找到相同的object_id
            if query_obj_id in retrieved_obj_ids[:1]:
                recall_1 += 1
            if query_obj_id in retrieved_obj_ids[:5]:
                recall_5 += 1
            if query_obj_id in retrieved_obj_ids[:10]:
                recall_10 += 1
        
        recall_1 /= num_queries
        recall_5 /= num_queries
        recall_10 /= num_queries
        
        return recall_1, recall_5, recall_10
    
    def evaluate_per_category(self, similarity_matrix, query_object_ids, gallery_object_ids, query_categories, exclude_self=False):
        """
        计算分类别的Recall@1指标
        
        :param similarity_matrix: 相似度矩阵，形状为 (num_queries, num_gallery)
        :param query_object_ids: 查询的object_id列表
        :param gallery_object_ids: 图库的object_id列表
        :param query_categories: 查询的类别列表
        :param exclude_self: 是否排除自身匹配
        :return: 分类别的R@1字典
        """
        num_queries = similarity_matrix.shape[0]
        num_gallery = len(gallery_object_ids)
        
        # 计算每个查询的top-1索引
        _, indices = similarity_matrix.topk(1, dim=1, largest=True, sorted=True)
        
        # 将indices移动到CPU上，并转换为numpy数组以便安全处理
        indices = indices.cpu().numpy()
        
        # 过滤掉超出范围的索引值
        max_valid_idx = num_gallery - 1
        indices = np.where(indices > max_valid_idx, -1, indices)
        
        # 统计每个类别的查询数量和正确召回数量
        category_stats = {}
        
        for i in range(num_queries):
            query_obj_id = query_object_ids[i]
            query_cat = query_categories[i]
            retrieved_idx = indices[i][0]
            
            # 添加防御性检查：确保retrieved_idx在有效范围内
            if retrieved_idx < 0 or retrieved_idx >= num_gallery:
                retrieved_obj_id = None
            else:
                retrieved_obj_id = gallery_object_ids[retrieved_idx]
            
            # 更新统计信息
            if query_cat not in category_stats:
                category_stats[query_cat] = {'total': 0, 'correct': 0}
            
            category_stats[query_cat]['total'] += 1
            
            # 检查是否召回成功，需要考虑排除自身匹配的情况
            # 只有当retrieved_obj_id有效且不等于None，并且（不需要排除自身或不是自身匹配）时，才认为召回成功
            if retrieved_obj_id is not None and retrieved_obj_id == query_obj_id:
                if not exclude_self or retrieved_idx != i:
                    category_stats[query_cat]['correct'] += 1
        
        # 计算每个类别的R@1
        per_category_r1 = {}
        for category, stats in category_stats.items():
            per_category_r1[category] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
        
        return per_category_r1
    
    def run_view_to_global_image(self):
        """执行View -> Global Image检索任务"""
        logging.info("\n=== 执行 View -> Global Image 检索任务 ===")
        
        # 计算相似度矩阵
        similarity_matrix = self.compute_cosine_similarity(
            self.flattened_view_feats,
            self.global_image_feats
        )
        
        # 计算整体Recall
        recall_1, recall_5, recall_10 = self.evaluate_recall(
            similarity_matrix,
            self.flattened_object_ids,
            self.object_ids
        )
        
        # 计算分类别Recall@1
        per_category_r1 = self.evaluate_per_category(
            similarity_matrix,
            self.flattened_object_ids,
            self.object_ids,
            self.flattened_categories
        )
        
        return {
            'name': 'View -> Global Image',
            'overall': (recall_1, recall_5, recall_10),
            'per_category': per_category_r1
        }
    
    def run_view_to_view(self):
        """执行View -> View检索任务"""
        logging.info("\n=== 执行 View -> View 检索任务 ===")
        
        # 计算相似度矩阵
        similarity_matrix = self.compute_cosine_similarity(
            self.flattened_view_feats,
            self.flattened_view_feats
        )
        
        # 计算整体Recall（排除自身匹配）
        recall_1, recall_5, recall_10 = self.evaluate_recall(
            similarity_matrix,
            self.flattened_object_ids,
            self.flattened_object_ids,
            exclude_self=True
        )
        
        # 计算分类别Recall@1（排除自身匹配）
        per_category_r1 = self.evaluate_per_category(
            similarity_matrix,
            self.flattened_object_ids,
            self.flattened_object_ids,
            self.flattened_categories,
            exclude_self=True
        )
        
        return {
            'name': 'View -> View',
            'overall': (recall_1, recall_5, recall_10),
            'per_category': per_category_r1
        }
    
    def run_view_to_point_cloud(self):
        """执行View -> Point Cloud检索任务"""
        logging.info("\n=== 执行 View -> Point Cloud 检索任务 ===")
        
        # 计算相似度矩阵
        similarity_matrix = self.compute_cosine_similarity(
            self.flattened_view_feats,
            self.global_point_feats
        )
        
        # 计算整体Recall
        recall_1, recall_5, recall_10 = self.evaluate_recall(
            similarity_matrix,
            self.flattened_object_ids,
            self.object_ids
        )
        
        # 计算分类别Recall@1
        per_category_r1 = self.evaluate_per_category(
            similarity_matrix,
            self.flattened_object_ids,
            self.object_ids,
            self.flattened_categories
        )
        
        return {
            'name': 'View -> Point Cloud',
            'overall': (recall_1, recall_5, recall_10),
            'per_category': per_category_r1
        }
    
    def run_global_image_to_point_cloud(self):
        """执行Global Image -> Point Cloud检索任务"""
        logging.info("\n=== 执行 Global Image -> Point Cloud 检索任务 ===")
        
        # 计算相似度矩阵
        similarity_matrix = self.compute_cosine_similarity(
            self.global_image_feats,
            self.global_point_feats
        )
        
        # 计算整体Recall
        recall_1, recall_5, recall_10 = self.evaluate_recall(
            similarity_matrix,
            self.object_ids,
            self.object_ids
        )
        
        # 计算分类别Recall@1
        per_category_r1 = self.evaluate_per_category(
            similarity_matrix,
            self.object_ids,
            self.object_ids,
            self.categories
        )
        
        return {
            'name': 'Global Image -> Point Cloud',
            'overall': (recall_1, recall_5, recall_10),
            'per_category': per_category_r1
        }
    
    def run_all_tasks(self):
        """执行所有检索任务"""
        results = []
        
        results.append(self.run_view_to_global_image())
        results.append(self.run_view_to_view())
        results.append(self.run_view_to_point_cloud())
        results.append(self.run_global_image_to_point_cloud())
        
        return results
    
    def print_report(self, results):
        """打印检索性能报告"""
        print("\n" + "="*80)
        print("\t\t\t检索性能评估报告")
        print("="*80)
        
        # 打印整体召回率
        print("\n=== 整体召回率 (Overall Recall) ===")
        print(f"{'任务':<30} {'R@1':>8} {'R@5':>8} {'R@10':>8}")
        print("-" * 60)
        
        for result in results:
            task_name = result['name']
            r1, r5, r10 = result['overall']
            print(f"{task_name:<30} {r1*100:>7.2f}% {r5*100:>7.2f}% {r10*100:>7.2f}%")
        
        # 打印分类别召回率
        print("\n=== 分类别召回率@1 (Per-Category R@1) ===")
        
        # 获取所有唯一类别
        all_categories = sorted(list(set(self.categories)))
        
        # 打印表头
        headers = ["类别"] + [result['name'] for result in results]
        print(f"{'类别':<20}" + "".join([f"{name:<15}" for name in ["R@1"]*len(results)]))
        print("-" * (20 + 15*len(results)))
        
        # 打印每个类别的结果
        for category in all_categories:
            row = [category]
            for result in results:
                r1 = result['per_category'].get(category, 0.0)
                row.append(f"{r1*100:>14.2f}%")
            print(f"{row[0]:<20}" + "".join(row[1:]))
        
        print("\n" + "="*80)

def setup_logging():
    """设置日志系统"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='检索性能评估脚本')
    parser.add_argument('--feature_file', type=str, default='../Embedding/features_all.h5', help='特征文件路径')
    parser.add_argument('--batch_size', type=int, default=128, help='检索计算时的批次大小')
    parser.add_argument('--device', type=str, default='cuda', help='计算设备')
    
    args = parser.parse_args()
    
    setup_logging()
    logging.info("开始检索性能评估流程")
    
    # 创建检索引擎
    engine = RetrievalEngine(
        feature_file=args.feature_file,
        device=args.device,
        batch_size=args.batch_size
    )
    
    # 执行所有检索任务
    results = engine.run_all_tasks()
    
    # 打印报告
    engine.print_report(results)
    
    logging.info("检索性能评估完成")

if __name__ == '__main__':
    main()