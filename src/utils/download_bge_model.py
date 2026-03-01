#!/usr/bin/env python3
"""
下载并测试中文 Embedding 模型 (BAAI/bge-large-zh)
"""
import os
import sys

def download_model():
    """下载 BGE 中文模型"""
    print("=" * 60)
    print("下载中文 Embedding 模型: BAAI/bge-large-zh")
    print("=" * 60)
    
    try:
        from transformers import AutoTokenizer, AutoModel
        import torch
        
        model_name = "BAAI/bge-large-zh"
        cache_dir = "/root/.cache/huggingface/hub"
        
        print(f"\n1. 下载 Tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        print("✓ Tokenizer 下载完成")
        
        print(f"\n2. 下载 Model...")
        model = AutoModel.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        print("✓ Model 下载完成")
        
        print(f"\n3. 测试模型...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        model.eval()
        
        # 测试编码
        test_texts = [
            "这是一个测试句子",
            "自然语言处理",
            "北京大学"
        ]
        
        instruction = "为这个句子生成表示以用于检索相关文章："
        
        with torch.no_grad():
            for text in test_texts:
                inputs = tokenizer(
                    instruction + text,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(device)
                
                outputs = model(**inputs)
                vec = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                print(f"  '{text}' -> 向量维度: {vec.shape}")
        
        print("\n✓ 模型测试通过!")
        print(f"\n模型缓存位置: {cache_dir}")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_similarity():
    """测试语义相似度"""
    print("\n" + "=" * 60)
    print("测试中文语义相似度")
    print("=" * 60)
    
    try:
        from transformers import AutoTokenizer, AutoModel
        from sklearn.metrics.pairwise import cosine_similarity
        import torch
        import numpy as np
        
        model_name = "BAAI/bge-large-zh"
        cache_dir = "/root/.cache/huggingface/hub"
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"加载模型到 {device}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir).to(device)
        model.eval()
        
        # 测试查询和文档
        query = "人工智能的发展历史"
        documents = [
            "人工智能是计算机科学的一个分支",
            "机器学习是AI的核心技术",
            "北京故宫是中国古代皇宫",
            "深度学习在图像识别中应用广泛",
            "Python是一种编程语言"
        ]
        
        print(f"\n查询: '{query}'")
        print("\n文档:")
        for i, doc in enumerate(documents, 1):
            print(f"  {i}. {doc}")
        
        # 编码
        instruction = "为这个句子生成表示以用于检索相关文章："
        
        def encode(texts, is_query=False):
            vectors = []
            with torch.no_grad():
                for text in texts:
                    if is_query:
                        text = instruction + text
                    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
                    outputs = model(**inputs)
                    vec = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                    vectors.append(vec[0])
            return np.array(vectors)
        
        print("\n编码中...")
        query_vec = encode([query], is_query=True)
        doc_vecs = encode(documents, is_query=False)
        
        # 计算相似度
        similarities = cosine_similarity(query_vec, doc_vecs)[0]
        
        print("\n相似度排名:")
        ranked = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)
        for idx, sim in ranked:
            print(f"  {documents[idx]}: {sim:.4f}")
        
        print("\n✓ 相似度测试完成!")
        return True
        
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("BAAI/bge-large-zh 中文 Embedding 模型下载与测试\n")
    
    # 下载模型
    if download_model():
        # 测试相似度
        test_similarity()
        
        print("\n" + "=" * 60)
        print("下一步: 运行 vector_retriever_chinese.py 构建向量索引")
        print("=" * 60)
    else:
        print("\n模型下载失败，请检查网络连接")
        sys.exit(1)
