import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import '../../core/theme/app_colors.dart';
import '../../data/dtos/models.dart';
import '../../data/services/dataset_service.dart';
import '../../data/services/kitchen_service.dart';
import '../../shared/widgets/stat_card.dart';
import '../../shared/widgets/common_widgets.dart';
import '../auth/auth_provider.dart';

final overviewProvider = FutureProvider<KitchenOverview>((ref) async {
  final service = ref.read(kitchenServiceProvider);
  final result = await service.getOverview();
  return result.when(
    success: (data) => data,
    failure: (error) => throw Exception(error.message),
  );
});

final alertsProvider = FutureProvider<List<Alert>>((ref) async {
  final service = ref.read(kitchenServiceProvider);
  final result = await service.getAlerts();
  return result.when(
    success: (data) => data,
    failure: (error) => throw Exception(error.message),
  );
});

/// Datasets the kitchen has uploaded, used to nudge the first upload.
final datasetCountProvider = FutureProvider<int>((ref) async {
  final result = await ref.read(datasetServiceProvider).list();
  return result.when(
    success: (data) => data.length,
    failure: (_) => 0,
  );
});

class DashboardScreen extends ConsumerWidget {
  const DashboardScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final overview = ref.watch(overviewProvider);
    final alerts = ref.watch(alertsProvider);
    final datasetCount = ref.watch(datasetCountProvider);
    final user = ref.watch(currentUserProvider);

    return Scaffold(
      appBar: AppBar(
        title: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Hello, ${user?.displayName.split(' ').first ?? 'Chef'} 👋', style: const TextStyle(fontSize: 16, color: AppColors.textSecondary, fontWeight: FontWeight.w400)),
            const Text('Dashboard', style: TextStyle(fontSize: 22, fontWeight: FontWeight.w700)),
          ],
        ),
        toolbarHeight: 72,
        actions: [
          IconButton(
            icon: const Badge(
              smallSize: 8,
              child: Icon(Icons.notifications_outlined),
            ),
            onPressed: () => context.push('/kitchen/alerts'),
          ),
          const SizedBox(width: 8),
        ],
      ),
      body: RefreshIndicator(
        onRefresh: () async {
          ref.invalidate(overviewProvider);
          ref.invalidate(alertsProvider);
          ref.invalidate(datasetCountProvider);
        },
        child: overview.when(
          loading: () => _buildSkeleton(),
          error: (error, _) => ErrorState(message: error.toString(), onRetry: () => ref.invalidate(overviewProvider)),
          data: (data) => _buildContent(context, data, alerts, datasetCount, ref),
        ),
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: () => context.push('/kitchen/surplus/create'),
        tooltip: 'New Surplus',
        child: const Icon(Icons.add_rounded),
      ),
    );
  }

  Widget _buildContent(
    BuildContext context,
    KitchenOverview data,
    AsyncValue<List<Alert>> alertsAsync,
    AsyncValue<int> datasetCount,
    WidgetRef ref,
  ) {
    return ListView(
      padding: const EdgeInsets.fromLTRB(16, 8, 16, 100),
      children: [
        alertsAsync.whenOrNull(
          data: (alerts) {
            final critical = alerts.where((a) => a.severity == 'CRITICAL' && !a.acknowledged).toList();
            if (critical.isEmpty) return const SizedBox.shrink();
            return Container(
              margin: const EdgeInsets.only(bottom: 16),
              padding: const EdgeInsets.all(14),
              decoration: BoxDecoration(
                gradient: AppColors.dangerGradient,
                borderRadius: BorderRadius.circular(14),
              ),
              child: Row(
                children: [
                  const Icon(Icons.warning_amber_rounded, color: Colors.white, size: 22),
                  const SizedBox(width: 10),
                  Expanded(
                    child: Text(critical.first.message, style: const TextStyle(color: Colors.white, fontSize: 13, fontWeight: FontWeight.w500)),
                  ),
                  TextButton(
                    onPressed: () => context.push('/kitchen/alerts'),
                    child: Text('View', style: TextStyle(color: Colors.white.withOpacity(0.9), fontWeight: FontWeight.w700)),
                  ),
                ],
              ),
            );
          },
        ) ?? const SizedBox.shrink(),
        GridView.count(
          crossAxisCount: 2,
          mainAxisSpacing: 12,
          crossAxisSpacing: 12,
          childAspectRatio: 1.3,
          shrinkWrap: true,
          physics: const NeverScrollableScrollPhysics(),
          children: [
            StatCard(title: 'Expected Diners', value: '${data.expectedDiners}', icon: Icons.people_outline, color: AppColors.info),
            StatCard(title: 'Forecast (p50)', value: '${data.forecastP50.round()}', unit: 'kg', icon: Icons.show_chart, color: AppColors.primary),
            StatCard(title: 'Recommended', value: '${data.recommendedProduction.round()}', unit: 'kg', icon: Icons.precision_manufacturing_outlined, color: AppColors.accent),
            StatCard(title: 'Surplus Risk', value: data.surplusRisk, icon: Icons.security, color: AppColors.statusColor(data.surplusRisk)),
          ],
        ),
        const SizedBox(height: 20),
        _sectionTitle('Today\'s Activity'),
        const SizedBox(height: 12),
        GridView.count(
          crossAxisCount: 3,
          mainAxisSpacing: 12,
          crossAxisSpacing: 12,
          childAspectRatio: 0.95,
          shrinkWrap: true,
          physics: const NeverScrollableScrollPhysics(),
          children: [
            _miniCard('Waste', '${data.wasteToday}', 'kg', Icons.delete_outline, AppColors.danger),
            _miniCard('Available', '${data.availableKg}', 'kg', Icons.inventory_2_outlined, AppColors.primary),
            _miniCard('Redistributed', '${data.redistributedToday}', 'kg', Icons.volunteer_activism, AppColors.info),
          ],
        ),
        const SizedBox(height: 24),
        _sectionTitle('Quick Actions'),
        const SizedBox(height: 12),
        Wrap(
          spacing: 10,
          runSpacing: 10,
          children: [
            _actionChip(context, 'Predict Demand', Icons.analytics, () => context.push('/kitchen/predict')),
            _actionChip(context, 'Log Waste', Icons.delete_sweep, () => context.push('/kitchen/waste')),
            _actionChip(context, 'What-If', Icons.tune, () => context.push('/kitchen/what-if')),
            _actionChip(context, 'Sensors', Icons.sensors, () => context.push('/kitchen/sensors')),
            _actionChip(context, 'Impact', Icons.eco, () => context.push('/kitchen/impact')),
          ],
        ),

        // Nudge the upload until there is history to predict from, since a
        // prediction is only as good as the data behind it.
        if (datasetCount.valueOrNull == 0) ...[
          const SizedBox(height: 24),
          _buildUploadPrompt(context),
        ],
      ],
    );
  }

  Widget _sectionTitle(String title) {
    return Text(title, style: const TextStyle(fontSize: 17, fontWeight: FontWeight.w600, color: AppColors.textPrimary));
  }

  Widget _buildUploadPrompt(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: AppColors.primarySurface,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppColors.primary.withOpacity(0.35)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Icon(Icons.upload_file, color: AppColors.primary, size: 20),
              const SizedBox(width: 8),
              const Expanded(
                child: Text(
                  'Add your history for accurate predictions',
                  style: TextStyle(fontSize: 15, fontWeight: FontWeight.w600, color: AppColors.textPrimary),
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          const Text(
            'Upload a CSV, TSV, TXT or Excel file of your previous footfall and '
            'orders data so demand can be forecast from your own numbers.',
            style: TextStyle(fontSize: 12, color: AppColors.textSecondary),
          ),
          const SizedBox(height: 14),
          SizedBox(
            width: double.infinity,
            height: 46,
            child: ElevatedButton.icon(
              onPressed: () => context.push('/kitchen/predict'),
              icon: const Icon(Icons.add_chart, size: 18),
              label: const Text('Upload History & Predict'),
            ),
          ),
        ],
      ),
    );
  }

  Widget _miniCard(String label, String value, String unit, IconData icon, Color color) {
    return Container(
      padding: const EdgeInsets.all(10),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: AppColors.border, width: 0.5),
      ),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 20, color: color),
          const SizedBox(height: 6),
          FittedBox(
            fit: BoxFit.scaleDown,
            child: RichText(text: TextSpan(children: [
              TextSpan(text: value, style: const TextStyle(fontSize: 18, fontWeight: FontWeight.w700, color: AppColors.textPrimary)),
              TextSpan(text: ' $unit', style: const TextStyle(fontSize: 11, color: AppColors.textSecondary)),
            ])),
          ),
          const SizedBox(height: 3),
          Text(label, style: const TextStyle(fontSize: 10, color: AppColors.textMuted), overflow: TextOverflow.ellipsis),
        ],
      ),
    );
  }

  Widget _actionChip(BuildContext context, String label, IconData icon, VoidCallback onTap) {
    return ActionChip(
      avatar: Icon(icon, size: 18),
      label: Text(label),
      onPressed: onTap,
    );
  }

  Widget _buildSkeleton() {
    return Padding(
      padding: const EdgeInsets.all(16),
      child: Column(
        children: List.generate(4, (_) => Padding(
          padding: const EdgeInsets.only(bottom: 12),
          child: SkeletonLoader(height: 100, borderRadius: 16),
        )),
      ),
    );
  }
}
