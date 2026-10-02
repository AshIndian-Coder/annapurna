import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../data/services/admin_service.dart';
import '../../shared/widgets/common_widgets.dart';

final mlModelsProvider = FutureProvider.autoDispose<List<MlModel>>((ref) async {
  final service = ref.read(adminServiceProvider);
  final result = await service.getModels();
  return result.when(
    success: (data) => data,
    failure: (error) => throw error,
  );
});

class AdminModelsScreen extends ConsumerWidget {
  const AdminModelsScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final modelsAsync = ref.watch(mlModelsProvider);

    return Scaffold(
      appBar: AppBar(
        title: const Text('ML Models'),
      ),
      body: modelsAsync.when(
        loading: () => const Center(child: CircularProgressIndicator()),
        error: (err, _) => ErrorState(
          message: err.toString(),
          onRetry: () => ref.refresh(mlModelsProvider),
        ),
        data: (models) {
          return RefreshIndicator(
            onRefresh: () async => ref.refresh(mlModelsProvider.future),
            child: ListView.separated(
              padding: const EdgeInsets.all(16),
              itemCount: models.length,
              separatorBuilder: (_, __) => const SizedBox(height: 12),
              itemBuilder: (context, index) {
                final model = models[index];
                return Card(
                  child: Padding(
                    padding: const EdgeInsets.all(16),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          children: [
                            Text(model.version, style: Theme.of(context).textTheme.titleLarge),
                            StatusBadge(status: model.status == 'ACTIVE' ? 'GOOD' : (model.status == 'SHADOW' ? 'INFO' : 'UNKNOWN')),
                          ],
                        ),
                        const SizedBox(height: 16),
                        Row(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          children: [
                            _Stat('Accuracy', '${model.accuracy}%'),
                            _Stat('Inferences', '${model.inferenceCount}'),
                            _Stat('Data Drift', '${model.dataDriftScore}'),
                          ],
                        ),
                        const SizedBox(height: 16),
                        if (model.status == 'SHADOW')
                          SizedBox(
                            width: double.infinity,
                            child: FilledButton(
                              onPressed: () {},
                              child: const Text('Promote to Active'),
                            ),
                          ),
                      ],
                    ),
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

class _Stat extends StatelessWidget {
  final String label;
  final String value;
  const _Stat(this.label, this.value);

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(label, style: Theme.of(context).textTheme.bodySmall),
        Text(value, style: Theme.of(context).textTheme.titleMedium),
      ],
    );
  }
}
