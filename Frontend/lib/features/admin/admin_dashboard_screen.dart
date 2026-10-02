import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../data/services/admin_service.dart';
import '../../shared/widgets/common_widgets.dart';

final systemHealthProvider = FutureProvider.autoDispose<List<SystemHealth>>((ref) async {
  final service = ref.read(adminServiceProvider);
  final result = await service.getSystemHealth();
  return result.when(
    success: (data) => data,
    failure: (error) => throw error,
  );
});

class AdminDashboardScreen extends ConsumerWidget {
  const AdminDashboardScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final healthAsync = ref.watch(systemHealthProvider);

    return Scaffold(
      appBar: AppBar(
        title: const Text('System Operations'),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: () => ref.refresh(systemHealthProvider),
          ),
        ],
      ),
      body: healthAsync.when(
        loading: () => const Center(child: CircularProgressIndicator()),
        error: (err, _) => ErrorState(
          message: err.toString(),
          onRetry: () => ref.refresh(systemHealthProvider),
        ),
        data: (healthItems) {
          return RefreshIndicator(
            onRefresh: () async => ref.refresh(systemHealthProvider.future),
            child: ListView.separated(
              padding: const EdgeInsets.all(16),
              itemCount: healthItems.length,
              separatorBuilder: (_, __) => const SizedBox(height: 12),
              itemBuilder: (context, index) {
                final item = healthItems[index];
                return Card(
                  child: ListTile(
                    leading: const Icon(Icons.dns),
                    title: Text(item.service),
                    subtitle: Text('Uptime: ${item.uptime}\nSeen: ${item.lastSeen}'),
                    isThreeLine: true,
                    trailing: StatusBadge(status: item.status),
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
