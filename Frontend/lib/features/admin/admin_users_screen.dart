import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../data/services/admin_service.dart';
import '../../shared/widgets/common_widgets.dart';

final systemUsersProvider = FutureProvider.autoDispose<List<SystemUser>>((ref) async {
  final service = ref.read(adminServiceProvider);
  final result = await service.getUsers();
  return result.when(
    success: (data) => data,
    failure: (error) => throw error,
  );
});

class AdminUsersScreen extends ConsumerWidget {
  const AdminUsersScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final usersAsync = ref.watch(systemUsersProvider);

    return Scaffold(
      appBar: AppBar(
        title: const Text('User Management'),
      ),
      body: usersAsync.when(
        loading: () => const Center(child: CircularProgressIndicator()),
        error: (err, _) => ErrorState(
          message: err.toString(),
          onRetry: () => ref.refresh(systemUsersProvider),
        ),
        data: (users) {
          return RefreshIndicator(
            onRefresh: () async => ref.refresh(systemUsersProvider.future),
            child: ListView.separated(
              padding: const EdgeInsets.all(16),
              itemCount: users.length,
              separatorBuilder: (_, __) => const SizedBox(height: 12),
              itemBuilder: (context, index) {
                final user = users[index];
                return Card(
                  child: ListTile(
                    leading: CircleAvatar(
                      child: Text(user.name.substring(0, 1)),
                    ),
                    title: Text(user.name),
                    subtitle: Text('Role: ${user.role}\nLast login: ${user.lastLogin}'),
                    isThreeLine: true,
                    trailing: StatusBadge(status: user.status == 'ACTIVE' ? 'GOOD' : 'HOLD'),
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
