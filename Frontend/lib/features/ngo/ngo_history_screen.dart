import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:qr_flutter/qr_flutter.dart';
import '../../core/theme/app_colors.dart';
import '../../data/services/ngo_service.dart';
import '../../shared/widgets/common_widgets.dart';
import '../../shared/widgets/profile_drawer.dart';
import 'package:intl/intl.dart';

final ngoHistoryProvider = FutureProvider.autoDispose<List<NgoHistory>>((ref) async {
  final service = ref.read(ngoServiceProvider);
  final result = await service.getHistory();
  return result.when(
    success: (data) => data,
    failure: (error) => throw error,
  );
});

class NgoHistoryScreen extends ConsumerWidget {
  const NgoHistoryScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final historyAsync = ref.watch(ngoHistoryProvider);
    final dateFormat = DateFormat('MMM d, yyyy • HH:mm');

    return Scaffold(
      appBar: AppBar(
        title: const Text('Delivery History'),
      ),
      drawer: const ProfileDrawer(),
      body: historyAsync.when(
        loading: () => const Center(child: CircularProgressIndicator()),
        error: (err, _) => ErrorState(
          message: err.toString(),
          onRetry: () => ref.refresh(ngoHistoryProvider),
        ),
        data: (history) {
          if (history.isEmpty) {
            return const EmptyState(
              icon: Icons.history,
              title: 'No history yet',
              subtitle: 'Past deliveries will appear here.',
            );
          }
          return RefreshIndicator(
            onRefresh: () async => ref.refresh(ngoHistoryProvider.future),
            child: ListView.separated(
              padding: const EdgeInsets.all(16),
              itemCount: history.length,
              separatorBuilder: (_, __) => const SizedBox(height: 12),
              itemBuilder: (context, index) {
                final item = history[index];
                return Card(
                  child: ListTile(
                    title: Text(item.foodName),
                    subtitle: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const SizedBox(height: 4),
                        Text('${item.quantityKg} ${item.quantityUnit} • ${item.batchCode}'),
                        const SizedBox(height: 4),
                        Text('Expiry: ${dateFormat.format(item.expiryAt)}'),
                      ],
                    ),
                      trailing: Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          StatusBadge(label: item.status),
                          if (item.status == 'ACCEPTED' || item.status == 'IN_TRANSIT')
                            IconButton(
                              icon: const Icon(Icons.qr_code, color: AppColors.accent),
                              onPressed: () => _showQrDialog(context, item.batchId, item.batchCode),
                              padding: EdgeInsets.zero,
                              constraints: const BoxConstraints(),
                            ),
                        ],
                      ),
                    isThreeLine: true,
                  ),
                );
              },
            ),
          );
        },
      ),
    );
  }

  void _showQrDialog(BuildContext context, String batchId, String batchCode) {
    final manualCode = batchCode.isNotEmpty ? batchCode : batchId;
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        backgroundColor: AppColors.surfaceElevated,
        title: Text('Incoming: $manualCode'),
        content: SingleChildScrollView(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Text('Show this QR to the driver when they arrive to hand off the food.', style: TextStyle(color: AppColors.textSecondary, fontSize: 13)),
              const SizedBox(height: 16),
              Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(12),
                ),
                child: QrImageView(
                  data: batchId,
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
}

