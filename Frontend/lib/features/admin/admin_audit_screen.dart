import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../data/services/admin_service.dart';
import '../../shared/widgets/common_widgets.dart';
import 'package:intl/intl.dart';

final auditLogsProvider = FutureProvider.autoDispose<List<AuditLog>>((ref) async {
  final service = ref.read(adminServiceProvider);
  final result = await service.getAuditLogs();
  return result.when(
    success: (data) => data,
    failure: (error) => throw error,
  );
});

class AdminAuditScreen extends ConsumerWidget {
  const AdminAuditScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final logsAsync = ref.watch(auditLogsProvider);

    return Scaffold(
      appBar: AppBar(
        title: const Text('Audit Logs'),
      ),
      body: logsAsync.when(
        loading: () => const Center(child: CircularProgressIndicator()),
        error: (err, _) => ErrorState(
          message: err.toString(),
          onRetry: () => ref.refresh(auditLogsProvider),
        ),
        data: (logs) {
          return RefreshIndicator(
            onRefresh: () async => ref.refresh(auditLogsProvider.future),
            child: ListView.separated(
              padding: const EdgeInsets.all(16),
              itemCount: logs.length,
              separatorBuilder: (_, __) => const Divider(),
              itemBuilder: (context, index) {
                final log = logs[index];
                final date = DateTime.parse(log.timestamp);
                final formattedDate = DateFormat('MMM d, HH:mm').format(date);
                
                return ListTile(
                  title: Text(log.action),
                  subtitle: Text('${log.actor} on ${log.entityType}'),
                  trailing: Text(formattedDate, style: Theme.of(context).textTheme.bodySmall),
                  contentPadding: EdgeInsets.zero,
                );
              },
            ),
          );
        },
      ),
    );
  }
}
