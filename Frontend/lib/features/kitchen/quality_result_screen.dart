import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../core/theme/app_colors.dart';
import '../../data/dtos/models.dart';
import '../../data/services/surplus_service.dart';
import '../../shared/widgets/common_widgets.dart';

class QualityResultScreen extends ConsumerStatefulWidget {
  final String batchId;
  const QualityResultScreen({super.key, required this.batchId});

  @override
  ConsumerState<QualityResultScreen> createState() => _QualityResultScreenState();
}

class _QualityResultScreenState extends ConsumerState<QualityResultScreen> {
  QualityResult? _result;
  bool _loading = true;

  @override
  void initState() {
    super.initState();
    _loadResult();
  }

  Future<void> _loadResult() async {
    final service = ref.read(surplusServiceProvider);
    final result = await service.checkQuality(widget.batchId);
    result.when(
      success: (data) => setState(() { _result = data; _loading = false; }),
      failure: (e) => setState(() => _loading = false),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Quality Check')),
      body: _loading
          ? const Center(child: CircularProgressIndicator())
          : _result == null
              ? const ErrorState(message: 'Failed to load quality data')
              : _buildContent(_result!),
    );
  }

  Widget _buildContent(QualityResult r) {
    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        _buildVisualCard(r.visual),
        const SizedBox(height: 16),
        _buildSafetyCard(r.safetyDecision),
        if (r.requiresHumanApproval) ...[
          const SizedBox(height: 16),
          _buildApprovalCard(r),
        ],
      ],
    );
  }

  Widget _buildVisualCard(VisualBlock v) {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppColors.border, width: 0.5),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                padding: const EdgeInsets.all(10),
                decoration: BoxDecoration(
                  color: AppColors.info.withOpacity(0.15),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: const Icon(Icons.visibility, size: 20, color: AppColors.info),
              ),
              const SizedBox(width: 12),
              const Text('Visual Analysis', style: TextStyle(fontSize: 17, fontWeight: FontWeight.w600, color: AppColors.textPrimary)),
              const Spacer(),
              StatusBadge(label: v.status),
            ],
          ),
          const SizedBox(height: 20),
          _metricRow('Risk Level', v.riskLevel, AppColors.statusColor(v.riskLevel)),
          _metricRow('Confidence', '${(v.confidence * 100).round()}%', AppColors.textPrimary),
          _metricRow('Reason', v.reason, AppColors.textSecondary),
          const SizedBox(height: 8),
          Text('Model: ${v.modelVersion}', style: const TextStyle(fontSize: 11, color: AppColors.textMuted)),
        ],
      ),
    );
  }

  Widget _buildSafetyCard(SafetyBlock s) {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppColors.border, width: 0.5),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                padding: const EdgeInsets.all(10),
                decoration: BoxDecoration(
                  color: AppColors.good.withOpacity(0.15),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: const Icon(Icons.health_and_safety, size: 20, color: AppColors.good),
              ),
              const SizedBox(width: 12),
              const Text('Safety Decision', style: TextStyle(fontSize: 17, fontWeight: FontWeight.w600, color: AppColors.textPrimary)),
              const Spacer(),
              StatusBadge(label: s.status),
            ],
          ),
          const SizedBox(height: 20),
          _metricRow('Danger Zone', '${s.dangerZoneMinutes} min', s.dangerZoneMinutes > 30 ? AppColors.danger : AppColors.good),
          _metricRow('Hours to Expiry', '${s.hoursToExpiry.toStringAsFixed(1)}h', s.hoursToExpiry < 2 ? AppColors.danger : AppColors.textPrimary),
          const SizedBox(height: 12),
          if (s.reasons.isNotEmpty) ...[
            const Text('Reasons:', style: TextStyle(fontSize: 13, color: AppColors.textSecondary)),
            const SizedBox(height: 8),
            Wrap(
              spacing: 6,
              runSpacing: 6,
              children: s.reasons.map((r) => StatusBadge(label: r, color: AppColors.accent, outlined: true)).toList(),
            ),
          ],
        ],
      ),
    );
  }

  Widget _buildApprovalCard(QualityResult r) {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        gradient: const LinearGradient(colors: [Color(0xFF1A1A2E), Color(0xFF16213E)]),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppColors.accent.withOpacity(0.3)),
      ),
      child: Column(
        children: [
          Row(
            children: [
              const Icon(Icons.gavel, color: AppColors.accent, size: 22),
              const SizedBox(width: 10),
              const Expanded(child: Text('Human Approval Required', style: TextStyle(fontSize: 15, fontWeight: FontWeight.w600, color: AppColors.accent))),
            ],
          ),
          const SizedBox(height: 16),
          Row(
            children: [
              Expanded(
                child: OutlinedButton.icon(
                  onPressed: () async {
                    final service = ref.read(surplusServiceProvider);
                    await service.approveSurplus(r.batchId, decision: 'REJECTED');
                    if (mounted) Navigator.pop(context);
                  },
                  icon: const Icon(Icons.close, color: AppColors.danger),
                  label: const Text('Reject', style: TextStyle(color: AppColors.danger)),
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: ElevatedButton.icon(
                  onPressed: () async {
                    final service = ref.read(surplusServiceProvider);
                    await service.approveSurplus(r.batchId, decision: 'APPROVED');
                    if (mounted) Navigator.pop(context);
                  },
                  icon: const Icon(Icons.check),
                  label: const Text('Approve'),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _metricRow(String label, String value, Color color) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 10),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label, style: const TextStyle(fontSize: 14, color: AppColors.textSecondary)),
          Text(value, style: TextStyle(fontSize: 14, fontWeight: FontWeight.w600, color: color)),
        ],
      ),
    );
  }
}
