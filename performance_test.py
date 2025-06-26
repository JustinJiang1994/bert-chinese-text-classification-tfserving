#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BERT中文文本分类API性能测试脚本
测试维度：响应时间、并发能力、吞吐量、错误率
"""

import time
import json
import requests
import threading
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

class PerformanceTester:
    def __init__(self, base_url="http://localhost:5001"):
        self.base_url = base_url
        self.test_texts = [
            "这手机拍照真好看，我很喜欢！",
            "质量太差了，不推荐购买。",
            "这个产品的功能比较齐全，价格也合理。",
            "服务态度很好，物流也很快。",
            "价格有点贵，但是质量确实不错。",
            "不错",
            "这款手机的设计非常精美，外观时尚大气，手感也很好。",
            "屏幕显示效果清晰，色彩鲜艳，观看视频和玩游戏都很享受。",
            "拍照功能强大，夜景模式特别出色，能够拍出很清晰的照片。",
            "电池续航能力也不错，正常使用一天没问题。"
        ]
        
    def single_request_test(self, text):
        """单次请求测试"""
        start_time = time.time()
        try:
            response = requests.post(
                f"{self.base_url}/predict",
                json={"text": text},
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "success": True,
                    "response_time": (end_time - start_time) * 1000,  # 转换为毫秒
                    "status_code": response.status_code,
                    "confidence": result.get("confidence", 0),
                    "text_length": len(text)
                }
            else:
                return {
                    "success": False,
                    "response_time": (end_time - start_time) * 1000,
                    "status_code": response.status_code,
                    "error": response.text
                }
        except Exception as e:
            end_time = time.time()
            return {
                "success": False,
                "response_time": (end_time - start_time) * 1000,
                "error": str(e)
            }
    
    def baseline_test(self, num_requests=10):
        """基线测试：顺序请求"""
        print(f"🔍 基线测试：{num_requests}个顺序请求")
        print("=" * 50)
        
        results = []
        for i in range(num_requests):
            text = self.test_texts[i % len(self.test_texts)]
            result = self.single_request_test(text)
            results.append(result)
            
            if result["success"]:
                print(f"请求 {i+1:2d}: {result['response_time']:6.2f}ms | "
                      f"置信度: {result['confidence']:.3f} | "
                      f"文本长度: {result['text_length']}")
            else:
                print(f"请求 {i+1:2d}: 失败 - {result.get('error', 'Unknown error')}")
        
        self._analyze_results(results, "基线测试")
    
    def concurrent_test(self, num_requests=50, max_workers=10):
        """并发测试"""
        print(f"\n🚀 并发测试：{num_requests}个请求，{max_workers}个并发线程")
        print("=" * 50)
        
        results = []
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_text = {
                executor.submit(self.single_request_test, self.test_texts[i % len(self.test_texts)]): i 
                for i in range(num_requests)
            }
            
            # 收集结果
            for future in as_completed(future_to_text):
                result = future.result()
                results.append(result)
        
        total_time = time.time() - start_time
        self._analyze_results(results, "并发测试", total_time)
    
    def load_test(self, duration=60, requests_per_second=10):
        """负载测试：持续一段时间的高频请求"""
        print(f"\n⚡ 负载测试：{duration}秒，每秒{requests_per_second}个请求")
        print("=" * 50)
        
        results = []
        start_time = time.time()
        request_count = 0
        
        while time.time() - start_time < duration:
            batch_start = time.time()
            
            # 发送一批请求
            with ThreadPoolExecutor(max_workers=requests_per_second) as executor:
                batch_futures = [
                    executor.submit(self.single_request_test, self.test_texts[i % len(self.test_texts)])
                    for i in range(requests_per_second)
                ]
                
                for future in as_completed(batch_futures):
                    result = future.result()
                    results.append(result)
                    request_count += 1
            
            # 控制请求频率
            batch_time = time.time() - batch_start
            if batch_time < 1.0:
                time.sleep(1.0 - batch_time)
        
        total_time = time.time() - start_time
        self._analyze_results(results, "负载测试", total_time)
    
    def stress_test(self, max_concurrent=50):
        """压力测试：逐步增加并发数"""
        print(f"\n💥 压力测试：最大{max_concurrent}个并发")
        print("=" * 50)
        
        concurrent_levels = [1, 5, 10, 20, 30, 50]
        stress_results = {}
        
        for concurrency in concurrent_levels:
            if concurrency > max_concurrent:
                break
                
            print(f"\n测试并发数: {concurrency}")
            results = []
            start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [
                    executor.submit(self.single_request_test, self.test_texts[i % len(self.test_texts)])
                    for i in range(concurrency * 2)  # 每个并发级别发送2倍请求
                ]
                
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
            
            total_time = time.time() - start_time
            analysis = self._analyze_results(results, f"压力测试({concurrency}并发)", total_time, verbose=False)
            stress_results[concurrency] = analysis
            
            # 如果错误率过高，停止测试
            if analysis['error_rate'] > 0.1:  # 10%错误率
                print(f"⚠️  错误率过高({analysis['error_rate']:.1%})，停止压力测试")
                break
        
        return stress_results
    
    def _analyze_results(self, results, test_name, total_time=None, verbose=True):
        """分析测试结果"""
        successful_results = [r for r in results if r["success"]]
        failed_results = [r for r in results if not r["success"]]
        
        total_requests = len(results)
        successful_requests = len(successful_results)
        failed_requests = len(failed_results)
        
        if successful_requests == 0:
            print(f"❌ {test_name}: 所有请求都失败了！")
            return {
                "total_requests": total_requests,
                "successful_requests": 0,
                "failed_requests": failed_requests,
                "error_rate": 1.0,
                "avg_response_time": 0,
                "min_response_time": 0,
                "max_response_time": 0,
                "throughput": 0
            }
        
        # 响应时间统计
        response_times = [r["response_time"] for r in successful_results]
        avg_response_time = statistics.mean(response_times)
        min_response_time = min(response_times)
        max_response_time = max(response_times)
        median_response_time = statistics.median(response_times)
        p95_response_time = np.percentile(response_times, 95)
        
        # 吞吐量计算
        if total_time:
            throughput = successful_requests / total_time
        else:
            throughput = 0
        
        # 错误率
        error_rate = failed_requests / total_requests if total_requests > 0 else 0
        
        if verbose:
            print(f"\n📊 {test_name} 结果分析:")
            print(f"   总请求数: {total_requests}")
            print(f"   成功请求: {successful_requests}")
            print(f"   失败请求: {failed_requests}")
            print(f"   错误率: {error_rate:.2%}")
            print(f"   平均响应时间: {avg_response_time:.2f}ms")
            print(f"   中位数响应时间: {median_response_time:.2f}ms")
            print(f"   95%分位响应时间: {p95_response_time:.2f}ms")
            print(f"   最小响应时间: {min_response_time:.2f}ms")
            print(f"   最大响应时间: {max_response_time:.2f}ms")
            if throughput > 0:
                print(f"   吞吐量: {throughput:.2f} 请求/秒")
        
        return {
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "failed_requests": failed_requests,
            "error_rate": error_rate,
            "avg_response_time": avg_response_time,
            "median_response_time": median_response_time,
            "p95_response_time": p95_response_time,
            "min_response_time": min_response_time,
            "max_response_time": max_response_time,
            "throughput": throughput
        }
    
    def run_full_test_suite(self):
        """运行完整的性能测试套件"""
        print("🎯 BERT中文文本分类API性能测试")
        print("=" * 60)
        print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"目标服务: {self.base_url}")
        print("=" * 60)
        
        # 1. 基线测试
        self.baseline_test(num_requests=10)
        
        # 2. 并发测试
        self.concurrent_test(num_requests=50, max_workers=10)
        
        # 3. 负载测试
        self.load_test(duration=30, requests_per_second=5)  # 缩短测试时间
        
        # 4. 压力测试
        stress_results = self.stress_test(max_concurrent=30)
        
        # 5. 总结报告
        self._print_summary_report(stress_results)
    
    def _print_summary_report(self, stress_results):
        """打印总结报告"""
        print("\n" + "=" * 60)
        print("📋 性能测试总结报告")
        print("=" * 60)
        
        if not stress_results:
            print("❌ 压力测试未完成")
            return
        
        print("并发性能分析:")
        print(f"{'并发数':<8} {'平均响应时间(ms)':<18} {'95%分位(ms)':<15} {'吞吐量(请求/秒)':<18} {'错误率':<8}")
        print("-" * 70)
        
        for concurrency, result in stress_results.items():
            print(f"{concurrency:<8} {result['avg_response_time']:<18.2f} "
                  f"{result['p95_response_time']:<15.2f} {result['throughput']:<18.2f} "
                  f"{result['error_rate']:<8.2%}")
        
        # 性能建议
        print("\n💡 性能优化建议:")
        
        # 找到最佳并发数
        best_concurrency = max(stress_results.keys(), 
                              key=lambda x: stress_results[x]['throughput'])
        best_result = stress_results[best_concurrency]
        
        print(f"   1. 推荐并发数: {best_concurrency} (吞吐量: {best_result['throughput']:.2f} 请求/秒)")
        
        if best_result['avg_response_time'] > 1000:
            print("   2. 响应时间较高，建议优化模型推理性能")
        
        if best_result['error_rate'] > 0.05:
            print("   3. 错误率较高，建议检查服务稳定性")
        
        print("   4. 考虑增加Gunicorn worker数量提升并发能力")
        print("   5. 可以部署多个推理服务实例进行负载均衡")

if __name__ == "__main__":
    # 创建测试器实例
    tester = PerformanceTester()
    
    # 运行完整测试套件
    tester.run_full_test_suite() 