import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
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
                        Text('${item.quantityKg} kg • ${item.batchCode}'),
                        const SizedBox(height: 4),
                        Text('Expiry: ${dateFormat.format(item.expiryAt)}'),
                      ],
                    ),
                    trailing: StatusBadge(label: item.status),
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
}
