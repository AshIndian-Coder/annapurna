import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import '../../core/enums.dart';
import '../../core/theme/app_colors.dart';
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
      floatingActionButton: FloatingActionButton.extended(
        onPressed: () => context.push('/kitchen/surplus/create'),
        icon: const Icon(Icons.add),
        label: const Text('Create'),
      ),
    );
  }
}

class _SurplusCard extends StatelessWidget {
  final Surplus surplus;
  const _SurplusCard({required this.surplus});

  @override
  Widget build(BuildContext context) {
    final expiry = surplus.timeToExpiry;
    final expiryText = expiry.isNegative
        ? 'Expired'
        : expiry.inHours > 0
            ? '${expiry.inHours}h ${expiry.inMinutes % 60}m left'
            : '${expiry.inMinutes}m left';

    return GestureDetector(
      onTap: () => context.push('/kitchen/quality/result/${surplus.batchId}'),
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
                    color: AppColors.primary.withValues(alpha: 0.15),
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
            Row(
              children: [
                _infoItem(Icons.scale, '${surplus.quantityKg} kg'),
                const SizedBox(width: 20),
                _infoItem(Icons.timer_outlined, expiryText, color: surplus.isExpiringSoon ? AppColors.danger : null),
                if (surplus.foodCategory != null) ...[
                  const SizedBox(width: 20),
                  _infoItem(Icons.category_outlined, surplus.foodCategory!),
                ],
              ],
            ),
            const SizedBox(height: 12),
            _buildStepper(surplus.status),
          ],
        ),
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
