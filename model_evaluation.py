# model_evaluation.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (roc_auc_score, roc_curve, precision_recall_curve, auc,
                             confusion_matrix, classification_report)
from sklearn.calibration import calibration_curve
import joblib
import os


class SpineModelEvaluator:
    """脊柱螺钉预测模型评估类"""

    def __init__(self, model_path, scaler_path, selector_path, feature_names):
        """
        初始化评估器

        Args:
            model_path: 模型文件路径
            scaler_path: 标准化器路径
            selector_path: 特征选择器路径
            feature_names: 特征名称列表
        """
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.selector = joblib.load(selector_path)
        self.feature_names = feature_names
        self.results = {}

    def evaluate_model(self, X_test, y_test, X_sel=None, y=None, selected_names=None):
        """
        综合模型评估

        Args:
            X_test: 测试集特征
            y_test: 测试集标签
            X_sel: 筛选后的特征（用于逻辑回归分析）
            y: 完整标签（用于逻辑回归分析）
            selected_names: 筛选后的特征名
        """
        print("=" * 60)
        print("开始综合模型评估")
        print("=" * 60)

        # 1. 基础预测
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        # 2. 执行各项评估
        self._calculate_basic_metrics(y_test, y_pred, y_pred_proba)
        self._plot_roc_curve(y_test, y_pred_proba)
        self._plot_pr_curve(y_test, y_pred_proba)
        self._plot_calibration_curve(y_test, y_pred_proba)
        self._plot_confusion_matrix(y_test, y_pred)

        # 3. 如果有逻辑回归数据，执行DCA
        if X_sel is not None and y is not None:
            self._decision_curve_analysis(y_test, y_pred_proba)

        # 4. 保存所有结果
        self._save_results()

        return self.results

    def _calculate_basic_metrics(self, y_test, y_pred, y_pred_proba):
        """计算基础性能指标"""
        print("计算基础性能指标...")

        # AUC
        auc_score = roc_auc_score(y_test, y_pred_proba)

        # 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # 关键指标
        sensitivity = tp / (tp + fn)  # 召回率
        specificity = tn / (tn + fp)
        ppv = tp / (tp + fp)  # 精确率
        npv = tn / (tn + fn)  # 阴性预测值
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        f1_score = 2 * (ppv * sensitivity) / (ppv + sensitivity)

        # PR-AUC
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall, precision)

        self.results['basic_metrics'] = {
            'AUC': auc_score,
            'PR_AUC': pr_auc,
            'Sensitivity': sensitivity,
            'Specificity': specificity,
            'PPV': ppv,
            'NPV': npv,
            'Accuracy': accuracy,
            'F1_Score': f1_score,
            'Confusion_Matrix': cm
        }

        print("✓ 基础指标计算完成")
        print(f"  AUC: {auc_score:.4f}, PR-AUC: {pr_auc:.4f}")
        print(f"  准确率: {accuracy:.4f}, F1分数: {f1_score:.4f}")

    def _plot_roc_curve(self, y_test, y_pred_proba):
        """绘制ROC曲线"""
        print("绘制ROC曲线...")

        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc_score = roc_auc_score(y_test, y_pred_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'XGBoost (AUC = {auc_score:.3f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Spine Screw Prediction')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("✓ ROC曲线已保存")

    def _plot_pr_curve(self, y_test, y_pred_proba):
        """绘制精确率-召回率曲线"""
        print("绘制PR曲线...")

        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall, precision)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'XGBoost (PR-AUC = {pr_auc:.3f})', linewidth=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('pr_curve.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("✓ PR曲线已保存")

    def _plot_calibration_curve(self, y_test, y_pred_proba):
        """绘制校准曲线"""
        print("绘制校准曲线...")

        fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_pred_proba, n_bins=10)

        plt.figure(figsize=(8, 6))
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="XGBoost", linewidth=2, markersize=6)
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated", linewidth=2)
        plt.xlabel("Mean predicted probability")
        plt.ylabel("Fraction of positives")
        plt.title("Calibration Plot")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('calibration_curve.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("✓ 校准曲线已保存")

    def _plot_confusion_matrix(self, y_test, y_pred):
        """绘制混淆矩阵"""
        print("绘制混淆矩阵...")

        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No Screw', 'Need Screw'],
                    yticklabels=['No Screw', 'Need Screw'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("✓ 混淆矩阵已保存")

    def _decision_curve_analysis(self, y_test, y_pred_proba):
        """决策曲线分析"""
        print("执行决策曲线分析...")

        thresholds = np.linspace(0.01, 0.99, 100)
        net_benefits = []
        n = len(y_test)
        prevalence = np.mean(y_test)

        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            tp = np.sum((y_pred == 1) & (y_test == 1))
            fp = np.sum((y_pred == 1) & (y_test == 0))

            net_benefit = (tp / n) - (fp / n) * (threshold / (1 - threshold))
            net_benefits.append(net_benefit)

        # 基准线
        treat_all_benefit = [prevalence - (1 - prevalence) * (t / (1 - t)) for t in thresholds]
        treat_none_benefit = [0] * len(thresholds)

        # 绘制DCA
        plt.figure(figsize=(8, 6))
        plt.plot(thresholds, net_benefits, label='XGBoost Model', linewidth=2)
        plt.plot(thresholds, treat_none_benefit, 'k--', label='Treat None', linewidth=2)
        plt.plot(thresholds, treat_all_benefit, 'r--', label='Treat All', linewidth=2)
        plt.xlabel('Threshold Probability')
        plt.ylabel('Net Benefit')
        plt.title('Decision Curve Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([-0.1, 0.5])
        plt.tight_layout()
        plt.savefig('dca_curve.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 计算最优阈值
        max_net_benefit_idx = np.argmax(net_benefits)
        optimal_threshold = thresholds[max_net_benefit_idx]
        max_net_benefit = net_benefits[max_net_benefit_idx]

        self.results['dca'] = {
            'optimal_threshold': optimal_threshold,
            'max_net_benefit': max_net_benefit
        }

        print("✓ DCA分析完成")
        print(f"  最优阈值: {optimal_threshold:.3f}, 最大净收益: {max_net_benefit:.4f}")

    def _save_results(self):
        """保存所有结果"""
        # 保存基础指标
        basic_df = pd.DataFrame([self.results['basic_metrics']])
        basic_df.to_csv('model_evaluation_metrics.csv', index=False)

        # 如果有DCA结果也保存
        if 'dca' in self.results:
            dca_df = pd.DataFrame([self.results['dca']])
            dca_df.to_csv('dca_results.csv', index=False)

        print("✓ 所有结果已保存")
        print("✓ 评估完成！")



def main():
    """使用示例"""
    evaluator = SpineModelEvaluator(
        model_path='spine_nail_xgb.pkl',
        scaler_path='spine_nail_scaler.pkl',
        selector_path='spine_nail_feature_selector.pkl',
        feature_names=selected_names  
    )

    # 执行评估
    results = evaluator.evaluate_model(X_test, y_test)


if __name__ == "__main__":

    main()
