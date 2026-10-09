import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:qr_flutter/qr_flutter.dart';
import '../../core/enums.dart';
import '../../core/theme/app_colors.dart';
import '../../core/api/api_client.dart';
import '../../core/result/result.dart';
import '../../data/dtos/models.dart';
import '../../data/services/surplus_service.dart';
import '../../shared/widgets/common_widgets.dart';

final surplusListProvider = FutureProvider<List<Surplus>>((ref) async {
  final service = ref.read(surplusServiceProvider);
  final result = await service.getSurplusList();
  return result.when(success: (data) => data, failure: (e) => throw Exception(e.message));
});

class SurplusListScreen extends ConsumerWidget {
  const SurplusListScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final surplusAsync = ref.watch(surplusListProvider);

    return Scaffold(
      appBar: AppBar(
        title: const Text('Surplus Batches'),
        actions: [
          IconButton(icon: const Icon(Icons.filter_list), onPressed: () {}),
        ],
      ),
      body: RefreshIndicator(
        onRefresh: () async => ref.invalidate(surplusListProvider),
        child: surplusAsync.when(
          loading: () => const Center(child: CircularProgressIndicator()),
          error: (e, _) => ErrorState(message: e.toString(), onRetry: () => ref.invalidate(surplusListProvider)),
          data: (list) => list.isEmpty
              ? const EmptyState(icon: Icons.inventory_2_outlined, title: 'No surplus batches', subtitle: 'Create your first surplus entry')
              : ListView.builder(
                  padding: const EdgeInsets.fromLTRB(16, 8, 16, 100),
                  itemCount: list.length,
                  itemBuilder: (context, i) => _SurplusCard(surplus: list[i]),
                ),
        ),
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: () => context.push('/kitchen/surplus/create'),
        tooltip: 'Create Surplus',
        child: const Icon(Icons.add),
      ),
    );
  }
}

class _SurplusCard extends ConsumerWidget {
  final Surplus surplus;
  const _SurplusCard({required this.surplus});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final expiry = surplus.timeToExpiry;
    final expiryText = expiry.isNegative
        ? 'Expired'
        : expiry.inHours > 0
            ? '${expiry.inHours}h ${expiry.inMinutes % 60}m left'
            : '${expiry.inMinutes}m left';

    return GestureDetector(
      child: Container(
        margin: const EdgeInsets.only(bottom: 12),
        padding: const EdgeInsets.all(16),
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
                    color: AppColors.primary.withOpacity(0.15),
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: const Icon(Icons.restaurant, size: 20, color: AppColors.primary),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(surplus.foodName, style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w600, color: AppColors.textPrimary)),
                      const SizedBox(height: 2),
                      Text(surplus.batchCode, style: const TextStyle(fontSize: 12, color: AppColors.textMuted, fontFamily: 'monospace')),
                    ],
                  ),
                ),
                StatusBadge(label: surplus.status.label),
              ],
            ),
            const SizedBox(height: 14),
            Wrap(
              spacing: 16,
              runSpacing: 8,
              children: [
                _infoItem(Icons.scale, '${surplus.quantityKg} ${surplus.quantityUnit}'),
                _infoItem(Icons.timer_outlined, expiryText, color: surplus.isExpiringSoon ? AppColors.danger : null),
                if (surplus.foodCategory != null)
                  _infoItem(Icons.category_outlined, surplus.foodCategory!),
              ],
            ),
            const SizedBox(height: 12),
            _buildStepper(surplus.status),
            if (surplus.status == SurplusStatus.available) ...[
              const SizedBox(height: 16),
              const Divider(height: 1),
              Row(
                mainAxisAlignment: MainAxisAlignment.end,
                children: [
                  TextButton.icon(
                    onPressed: () => _triggerMatch(context, ref, surplus.batchId),
                    icon: const Icon(Icons.people, color: AppColors.primary),
                    label: const Text('Find NGOs', style: TextStyle(color: AppColors.primary)),
                  ),
                ],
              ),
            ],
            if (surplus.status == SurplusStatus.matched || surplus.status == SurplusStatus.inTransit) ...[
              const SizedBox(height: 16),
              const Divider(height: 1),
              Row(
                mainAxisAlignment: MainAxisAlignment.end,
                children: [
                  TextButton.icon(
                    onPressed: () => _showQrDialog(context, surplus),
                    icon: const Icon(Icons.qr_code, color: AppColors.accent),
                    label: const Text('Show QR', style: TextStyle(color: AppColors.accent)),
                  ),
                ],
              ),
            ],
          ],
        ),
      ),
    );
  }

  Future<void> _triggerMatch(BuildContext context, WidgetRef ref, String batchId) async {
    try {
      final apiClient = ref.read(apiClientProvider);
      var response = await apiClient.post('/matches/$batchId', data: {}, parser: (d) => true);
      if (response is Failure) {
        response = await apiClient.post('/surplus/$batchId/match', data: {}, parser: (d) => true);
      }
      response.when(
        success: (_) {
          ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Notified nearby NGOs!')));
          ref.invalidate(surplusListProvider);
        },
        failure: (e) {
          ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Failed: ${e.message}')));
        }
      );
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Error: $e')));
    }
  }

  void _showQrDialog(BuildContext context, Surplus surplus) {
    final manualCode = surplus.batchCode.isNotEmpty ? surplus.batchCode : surplus.batchId;
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        backgroundColor: AppColors.surfaceElevated,
        title: Text('Batch $manualCode'),
        content: SingleChildScrollView(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Text('Show this QR code to the driver for pickup scanning.', style: TextStyle(color: AppColors.textSecondary, fontSize: 13), textAlign: TextAlign.center),
              const SizedBox(height: 16),
              Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(12),
                ),
                child: QrImageView(
                  data: surplus.batchId, // The driver will scan this batch ID
                  version: QrVersions.auto,
                  size: 200.0,
                  backgroundColor: Colors.white,
                ),
              ),
              const SizedBox(height: 16),
              Container(
                width: double.infinity,
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: AppColors.surface,
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(color: AppColors.border),
                ),
                child: Column(
                  children: [
                    const Text(
                      'If QR scanner does not work, use Batch Number:',
                      textAlign: TextAlign.center,
                      style: TextStyle(fontSize: 12, color: AppColors.textSecondary),
                    ),
                    const SizedBox(height: 6),
                    SelectableText(
                      manualCode,
                      style: const TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                        letterSpacing: 1.5,
                        fontFamily: 'monospace',
                        color: AppColors.primary,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Close'),
          ),
        ],
      ),
    );
  }

  Widget _infoItem(IconData icon, String text, {Color? color}) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Icon(icon, size: 14, color: color ?? AppColors.textMuted),
        const SizedBox(width: 4),
        Text(text, style: TextStyle(fontSize: 12, color: color ?? AppColors.textSecondary)),
      ],
    );
  }

  Widget _buildStepper(SurplusStatus status) {
    final steps = ['Created', 'Quality', 'Approve', 'Match', 'Transit', 'Delivered'];
    final currentIndex = switch (status) {
      SurplusStatus.pendingSafety => 1,
      SurplusStatus.available || SurplusStatus.hold => 2,
      SurplusStatus.matched => 3,
      SurplusStatus.inTransit => 4,
      SurplusStatus.delivered => 5,
      _ => 0,
    };

    return Row(
      children: List.generate(steps.length * 2 - 1, (i) {
        if (i.isOdd) {
          final stepIndex = i ~/ 2;
          return Expanded(
            child: Container(
              height: 2,
              color: stepIndex < currentIndex ? AppColors.primary : AppColors.surfaceElevated,
            ),
          );
        }
        final stepIndex = i ~/ 2;
        final isComplete = stepIndex < currentIndex;
        final isCurrent = stepIndex == currentIndex;
        return Container(
          width: 8,
          height: 8,
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            color: isComplete ? AppColors.primary : isCurrent ? AppColors.accent : AppColors.surfaceElevated,
            border: isCurrent ? Border.all(color: AppColors.accent, width: 2) : null,
          ),
        );
      }),
    );
  }
}
