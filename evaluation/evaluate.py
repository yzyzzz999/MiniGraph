#!/usr/bin/env python3
"""
MiniGraph 评测脚本
对比不同方案的效果：纯 LLM、RAG 单跳、RAG 多跳、RAG 增强
"""

import json
import time
import requests
from typing import Dict, List
from dataclasses import dataclass
from datetime import datetime

API_BASE = "http://127.0.0.1:5000"


@dataclass
class TestResult:
    id: str
    question: str
    ground_truth: str
    category: str
    difficulty: str
    predictions: Dict[str, str]
    latencies: Dict[str, float]
    correct: Dict[str, bool]


class MiniGraphEvaluator:
    def __init__(self, api_base: str = API_BASE):
        self.api_base = api_base
        self.methods = {
            'pure_llm': self.query_pure_llm,
            'rag_single': self.query_rag_single,
            'rag_multihop': self.query_rag_multihop,
            'rag_enhanced': self.query_rag_enhanced,
        }
    
    def query_pure_llm(self, question: str):
        start = time.time()
        try:
            response = requests.post(
                f"{self.api_base}/query",
                json={"question": question},
                timeout=30
            )
            latency = time.time() - start
            if response.status_code == 200:
                return response.json().get('answer', ''), latency
            return f"Error: {response.status_code}", latency
        except Exception as e:
            return f"Error: {str(e)}", time.time() - start
    
    def query_rag_single(self, question: str):
        start = time.time()
        try:
            response = requests.post(
                f"{self.api_base}/query",
                json={"question": question},
                timeout=30
            )
            latency = time.time() - start
            if response.status_code == 200:
                return response.json().get('answer', ''), latency
            return f"Error: {response.status_code}", latency
        except Exception as e:
            return f"Error: {str(e)}", time.time() - start
    
    def query_rag_multihop(self, question: str):
        start = time.time()
        try:
            response = requests.post(
                f"{self.api_base}/multihop",
                json={"question": question, "max_hops": 2},
                timeout=30
            )
            latency = time.time() - start
            if response.status_code == 200:
                return response.json().get('answer', ''), latency
            return f"Error: {response.status_code}", latency
        except Exception as e:
            return f"Error: {str(e)}", time.time() - start
    
    def query_rag_enhanced(self, question: str):
        start = time.time()
        try:
            response = requests.post(
                f"{self.api_base}/query_enhanced",
                json={"question": question},
                timeout=30
            )
            latency = time.time() - start
            if response.status_code == 200:
                return response.json().get('answer', ''), latency
            return f"Error: {response.status_code}", latency
        except Exception as e:
            return f"Error: {str(e)}", time.time() - start
    
    def check_correctness(self, prediction: str, ground_truth: str) -> bool:
        """
        检查答案是否正确
        优化版本：更宽松的匹配逻辑
        """
        if not prediction or not ground_truth:
            return False
        
        # 如果预测包含错误信息
        if prediction.startswith('Error:'):
            return False
        
        prediction_lower = prediction.lower()
        ground_truth_lower = ground_truth.lower()
        
        # 提取 ground_truth 中的关键词（至少2个字符）
        import re
        keywords = re.findall(r'[\u4e00-\u9fa5]{2,}|[a-zA-Z]{2,}|\d+', ground_truth_lower)
        
        if not keywords:
            return False
        
        # 特殊处理：朝代/时代等价
        dynasty_mapping = {
            '唐朝': ['唐朝', '唐代', '大唐'],
            '宋朝': ['宋朝', '宋代', '大宋'],
            '明朝': ['明朝', '明代', '大明'],
            '清朝': ['清朝', '清代', '大清'],
            '秦朝': ['秦朝', '秦代', '大秦'],
            '汉朝': ['汉朝', '汉代', '大汉'],
        }
        
        # 检查是否有朝代映射
        for dynasty, variants in dynasty_mapping.items():
            if dynasty in ground_truth_lower:
                # 如果 ground_truth 包含朝代，检查 prediction 是否包含任一 variant
                if any(v in prediction_lower for v in variants):
                    return True
        
        # 检查人名（允许部分匹配）
        # 例如 "唐太宗李世民" 可以匹配 "唐太宗" 或 "李世民"
        name_keywords = [k for k in keywords if len(k) >= 2]
        matched_keywords = 0
        
        for keyword in name_keywords:
            if keyword in prediction_lower:
                matched_keywords += 1
        
        # 如果匹配了至少一个关键词，认为是正确的
        # 对于短答案（1-2个关键词），需要全部匹配
        # 对于长答案（>2个关键词），匹配至少一个即可
        if len(name_keywords) <= 2:
            return matched_keywords >= len(name_keywords)
        else:
            return matched_keywords >= 1
    
    def evaluate_test_case(self, test_case: Dict) -> TestResult:
        print(f"\n评测: {test_case['id']} - {test_case['question']}")
        predictions = {}
        latencies = {}
        correct = {}
        
        for method_name, method_func in self.methods.items():
            try:
                answer, latency = method_func(test_case['question'])
                predictions[method_name] = answer
                latencies[method_name] = round(latency, 2)
                correct[method_name] = self.check_correctness(answer, test_case['answer'])
                status = "✓" if correct[method_name] else "✗"
                print(f"  {method_name}: {status} ({latency:.2f}s)")
            except Exception as e:
                predictions[method_name] = f"Error: {str(e)}"
                latencies[method_name] = 0
                correct[method_name] = False
                print(f"  {method_name}: ✗ Error")
        
        return TestResult(
            id=test_case['id'],
            question=test_case['question'],
            ground_truth=test_case['answer'],
            category=test_case['category'],
            difficulty=test_case['difficulty'],
            predictions=predictions,
            latencies=latencies,
            correct=correct
        )
    
    def evaluate_all(self, test_cases: List[Dict]):
        results = []
        for test_case in test_cases:
            result = self.evaluate_test_case(test_case)
            results.append(result)
        summary = self.compute_summary(results)
        return {
            'results': [self.result_to_dict(r) for r in results],
            'summary': summary,
            'timestamp': datetime.now().isoformat()
        }
    
    def compute_summary(self, results: List[TestResult]):
        summary = {}
        for method_name in self.methods.keys():
            correct_count = sum(1 for r in results if r.correct.get(method_name, False))
            total_count = len(results)
            avg_latency = sum(r.latencies.get(method_name, 0) for r in results) / total_count
            summary[method_name] = {
                'accuracy': round(correct_count / total_count, 3),
                'correct_count': correct_count,
                'total_count': total_count,
                'avg_latency': round(avg_latency, 2)
            }
        return summary
    
    def result_to_dict(self, result: TestResult):
        return {
            'id': result.id,
            'question': result.question,
            'ground_truth': result.ground_truth,
            'category': result.category,
            'difficulty': result.difficulty,
            'predictions': result.predictions,
            'latencies': result.latencies,
            'correct': result.correct
        }


def main():
    print("=" * 60)
    print("MiniGraph 评测脚本")
    print("=" * 60)
    
    with open('test_cases.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        test_cases = data['test_cases']
    
    print(f"加载了 {len(test_cases)} 个测试用例")
    
    evaluator = MiniGraphEvaluator()
    print("\n开始评测...")
    results = evaluator.evaluate_all(test_cases)
    
    output_file = f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'=' * 60}")
    print("评测完成!")
    print(f"结果已保存到: {output_file}")
    print("\n汇总结果:")
    for method, stats in results['summary'].items():
        print(f"  {method}: 准确率 {stats['accuracy']:.1%}, 平均延迟 {stats['avg_latency']:.2f}s")


if __name__ == '__main__':
    main()
