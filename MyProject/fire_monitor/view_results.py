import os
import sqlite3
import argparse
from database import ImageDatabase

def search_and_view_images(query=None, category=None, page=1, page_size=20):
    """搜索并查看图像，支持分页
    Args:
        query: 搜索关键词
        category: 图像类别
        page: 页码
        page_size: 每页显示数量
    """
    try:
        # 创建数据库连接
        db = ImageDatabase()

        # 搜索图像
        images, total_pages = db.search_images(query, category, page, page_size)

        if not images:
            print("没有找到匹配的图像")
            return

        # 打印结果表头
        print("\n搜索结果 (第 {page}/{total_pages} 页):\n".format(page=page, total_pages=total_pages))
        print(f"{'ID':<5} {'文件名':<20} {'类别':<10} {'预测结果':<12} {'时间戳'}")
        print("-" * 70)  # 分隔线
        
        # 打印结果行
        for row in images:
            image_id, filename, timestamp, category, prediction = row
            prediction = prediction if prediction is not None else "未处理"
            print(f"{image_id:<5} {filename:<20} {category:<10} {prediction:<12} {timestamp}")

        print(f"\n共找到 {len(images)} 条记录，总页数: {total_pages}")

    except Exception as e:
        print(f"查询错误: {e}")
    finally:
        # 关闭数据库连接
        if 'db' in locals():
            db.close()


def view_processed_results():
    """查看所有已处理的图像结果"""
    search_and_view_images(category=None, page=1, page_size=100)

if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='搜索和查看数据库中的图像')
    parser.add_argument('-q', '--query', type=str, help='搜索关键词(文件名)')
    parser.add_argument('-c', '--category', type=str, choices=['fire', 'non_fire'], help='图像类别')
    parser.add_argument('-p', '--page', type=int, default=1, help='页码(默认1)')
    parser.add_argument('-s', '--page_size', type=int, default=20, help='每页显示数量(默认20)')
    parser.add_argument('--all', action='store_true', help='查看所有图像')
    
    # 解析参数
    args = parser.parse_args()
    
    if args.all:
        # 查看所有图像
        search_and_view_images(page=args.page, page_size=args.page_size)
    else:
        # 按条件搜索
        search_and_view_images(
            query=args.query,
            category=args.category,
            page=args.page,
            page_size=args.page_size
        )