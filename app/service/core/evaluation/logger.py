# app/service/core/evaluation/logger.py
"""
评估日志模块 - 将评估结果写入日志文件
"""

import os
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class EvaluationLogger:
    """评估日志记录器"""

    def __init__(self):
        self.logs_dir = self._get_logs_dir()
        self._init_file_handlers()

    def _get_logs_dir(self) -> Path:
        """获取日志目录"""
        # 获取项目根目录
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent.parent.parent
        logs_dir = project_root / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        return logs_dir

    def _init_file_handlers(self):
        """初始化文件处理器"""
        # 评估日志专用文件（按日期分割）
        self.evaluation_log_file = self.logs_dir / f"evaluation_{datetime.now().strftime('%Y%m%d')}.log"

        # 创建专用日志器
        self.evaluation_logger = logging.getLogger("rag.evaluation")
        self.evaluation_logger.setLevel(logging.INFO)

        # 避免重复添加处理器
        if not self.evaluation_logger.handlers:
            file_handler = logging.FileHandler(
                self.evaluation_log_file,
                encoding='utf-8'
            )
            file_handler.setLevel(logging.INFO)

            formatter = logging.Formatter(
                '%(asctime)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(formatter)
            self.evaluation_logger.addHandler(file_handler)
            self.evaluation_logger.propagate = False

    def log_evaluation(
        self,
        question: str,
        answer: str,
        retrieval_metrics: Dict[str, Any],
        generation_metrics: Dict[str, Any],
        session_id: Optional[str] = None
    ):
        """
        记录单次评估结果

        Args:
            question: 用户问题
            answer: Agent 回答
            retrieval_metrics: 检索层指标 (Hit@K, MRR)
            generation_metrics: 生成层指标 (Faithfulness, Answer Relevancy, Context Recall, Context Precision)
            session_id: 会话ID
        """
        # 格式化输出
        log_entry = self._format_log_entry(
            question=question,
            answer=answer,
            retrieval_metrics=retrieval_metrics,
            generation_metrics=generation_metrics,
            session_id=session_id
        )

        self.evaluation_logger.info(log_entry)
        logger.debug(f"评估记录已写入: {self.evaluation_log_file}")

    def _format_log_entry(
        self,
        question: str,
        answer: str,
        retrieval_metrics: Dict[str, Any],
        generation_metrics: Dict[str, Any],
        session_id: Optional[str] = None
    ) -> str:
        """格式化日志条目"""
        import json

        # 截断过长的内容
        question_short = question[:200] + "..." if len(question) > 200 else question
        answer_short = answer[:500] + "..." if answer and len(answer) > 500 else answer

        # 构建日志行
        lines = []
        lines.append("=" * 80)
        lines.append(f"[SESSION] {session_id or 'N/A'}")
        lines.append(f"[QUESTION] {question_short}")
        lines.append(f"[ANSWER] {answer_short or 'N/A'}")

        # 检索层指标
        lines.append(f"[RETRIEVAL_METRICS]")
        lines.append(f"  Hit@K: {retrieval_metrics.get('hit_at_k', 0):.4f}")
        lines.append(f"  MRR: {retrieval_metrics.get('mrr', 0):.4f}")

        # 生成层指标
        lines.append(f"[GENERATION_METRICS]")
        lines.append(f"  Faithfulness: {generation_metrics.get('faithfulness', 0):.4f}")
        lines.append(f"  Answer Relevancy: {generation_metrics.get('answer_relevancy', 0):.4f}")
        lines.append(f"  Context Recall: {generation_metrics.get('context_recall', 0):.4f}")
        lines.append(f"  Context Precision: {generation_metrics.get('context_precision', 0):.4f}")

        lines.append("=" * 80)

        return "\n".join(lines)

    def log_batch_summary(
        self,
        total_cases: int,
        avg_retrieval_metrics: Dict[str, Any],
        avg_generation_metrics: Dict[str, Any]
    ):
        """
        记录批量评估汇总

        Args:
            total_cases: 总测试用例数
            avg_retrieval_metrics: 平均检索指标
            avg_generation_metrics: 平均生成指标
        """
        lines = []
        lines.append("=" * 80)
        lines.append(f"[BATCH_EVALUATION_SUMMARY] Total Cases: {total_cases}")
        lines.append("")
        lines.append("[AVG_RETRIEVAL_METRICS]")
        lines.append(f"  Hit@K: {avg_retrieval_metrics.get('hit_at_k', 0):.4f}")
        lines.append(f"  MRR: {avg_retrieval_metrics.get('mrr', 0):.4f}")
        lines.append("")
        lines.append("[AVG_GENERATION_METRICS]")
        lines.append(f"  Faithfulness: {avg_generation_metrics.get('faithfulness', 0):.4f}")
        lines.append(f"  Answer Relevancy: {avg_generation_metrics.get('answer_relevancy', 0):.4f}")
        lines.append(f"  Context Recall: {avg_generation_metrics.get('context_recall', 0):.4f}")
        lines.append(f"  Context Precision: {avg_generation_metrics.get('context_precision', 0):.4f}")
        lines.append("=" * 80)

        self.evaluation_logger.info("\n".join(lines))
        logger.info(f"批量评估汇总已记录: {total_cases} 个用例")

    def get_log_file_path(self) -> Path:
        """获取当前日志文件路径"""
        return self.evaluation_log_file

    def get_evaluation_stats(self, days: int = 7) -> Dict[str, Any]:
        """
        获取评估统计信息

        Args:
            days: 统计最近几天的日志

        Returns:
            统计信息字典
        """
        import glob
        import re

        stats = {
            "total_entries": 0,
            "avg_metrics": {
                "hit_at_k": 0.0,
                "mrr": 0.0,
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
                "context_recall": 0.0,
                "context_precision": 0.0
            }
        }

        # 获取最近几天的日志文件
        log_files = []
        for i in range(days):
            date_str = (datetime.now() - __import__('datetime').timedelta(days=i)).strftime('%Y%m%d')
            log_file = self.logs_dir / f"evaluation_{date_str}.log"
            if log_file.exists():
                log_files.append(log_file)

        # 解析日志文件
        metrics_list = []
        for log_file in log_files:
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 提取指标
                hit_pattern = r"Hit@K: ([\d.]+)"
                mrr_pattern = r"MRR: ([\d.]+)"
                faithfulness_pattern = r"Faithfulness: ([\d.]+)"
                relevancy_pattern = r"Answer Relevancy: ([\d.]+)"
                recall_pattern = r"Context Recall: ([\d.]+)"
                precision_pattern = r"Context Precision: ([\d.]+)"

                hits = re.findall(hit_pattern, content)
                mrrs = re.findall(mrr_pattern, content)
                faithfulness = re.findall(faithfulness_pattern, content)
                relevancy = re.findall(relevancy_pattern, content)
                recall = re.findall(recall_pattern, content)
                precision = re.findall(precision_pattern, content)

                for i in range(max(len(hits), len(mrrs), len(faithfulness), len(relevancy), len(recall), len(precision))):
                    metrics_list.append({
                        "hit_at_k": float(hits[i]) if i < len(hits) else 0,
                        "mrr": float(mrrs[i]) if i < len(mrrs) else 0,
                        "faithfulness": float(faithfulness[i]) if i < len(faithfulness) else 0,
                        "answer_relevancy": float(relevancy[i]) if i < len(relevancy) else 0,
                        "context_recall": float(recall[i]) if i < len(recall) else 0,
                        "context_precision": float(precision[i]) if i < len(precision) else 0
                    })

            except Exception as e:
                logger.error(f"解析日志文件失败 {log_file}: {e}")

        stats["total_entries"] = len(metrics_list)

        if metrics_list:
            for key in stats["avg_metrics"].keys():
                values = [m[key] for m in metrics_list if key in m]
                if values:
                    stats["avg_metrics"][key] = sum(values) / len(values)

        return stats


__all__ = ['EvaluationLogger']