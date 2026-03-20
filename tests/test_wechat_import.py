import pytest
import os
import tempfile
import csv
from src.loaders.csv_loader import WeChatCSVLoader
from src.loaders.base import DataLoader

def test_wechat_csv_loader_filtering():
    """测试微信加载器的过滤逻辑"""
    content = [
        {"msg": "你好", "talker": "user1", "is_sender": "0"},
        {"msg": "[表情]", "talker": "user1", "is_sender": "0"}, # 应该过滤
        {"msg": "嘿嘿", "talker": "user1", "is_sender": "1"},
        {"msg": "a", "talker": "user2", "is_sender": "0"},     # 应该过滤 (长度 <= 1)
        {"msg": "这是一个合法的消息", "talker": "user2", "is_sender": "0", "room_name": "group1"}
    ]
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["msg", "talker", "is_sender", "room_name"])
        writer.writeheader()
        writer.writerows(content)
        temp_path = f.name

    try:
        loader = WeChatCSVLoader(temp_path)
        docs = loader.load()
        
        # 应该只剩下 "你好", "嘿嘿", "这是一个合法的消息"
        assert len(docs) == 3
        assert "你好" in docs[0].content
        assert "嘿嘿" in docs[1].content
        assert "group1" in docs[2].content
        assert docs[1].metadata["is_sender"] == 1
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def test_dataloader_batch_and_incremental():
    """测试基类的批量加载和增量追踪能力"""
    # 创建两个临时 CSV
    with tempfile.TemporaryDirectory() as tmpdir:
        f1_path = os.path.join(tmpdir, "chat1.csv")
        f2_path = os.path.join(tmpdir, "chat2.csv")
        tracking_path = os.path.join(tmpdir, "tracking.json")
        
        with open(f1_path, 'w', encoding='utf-8') as f:
            f.write("msg,talker,is_sender\n消息1,user1,0\n消息2,user1,0\n")
        
        # 1. 第一轮加载
        docs = WeChatCSVLoader.load_batch(os.path.join(tmpdir, "*.csv"), incremental=True, tracking_file=tracking_path)
        assert len(docs) == 2
        
        # 2. 第二轮加载 (无新文件)
        docs2 = WeChatCSVLoader.load_batch(os.path.join(tmpdir, "*.csv"), incremental=True, tracking_file=tracking_path)
        assert len(docs2) == 0
        
        # 3. 新增文件
        with open(f2_path, 'w', encoding='utf-8') as f:
            f.write("msg,talker,is_sender\n消息3,user2,0\n")
        
        docs3 = WeChatCSVLoader.load_batch(os.path.join(tmpdir, "*.csv"), incremental=True, tracking_file=tracking_path)
        assert len(docs3) == 1
        assert "消息3" in docs3[0].content
