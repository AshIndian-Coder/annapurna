import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../core/theme/app_colors.dart';
import '../../data/services/logistics_service.dart';
import '../../shared/widgets/common_widgets.dart';

final routesProvider = FutureProvider.autoDispose<List<DriverRoute>>((ref) async {
  final service = ref.read(logisticsServiceProvider);
  final result = await service.getTodayRoutes();
  return result.when(
    success: (data) => data,
    failure: (error) => throw error,
  );
});

class LogisticsRoutesScreen extends ConsumerWidget {
  const LogisticsRoutesScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final routesAsync = ref.watch(routesProvider);

    return Scaffold(
      appBar: AppBar(
        title: const Text('Today\'s Routes'),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: () => ref.refresh(routesProvider),
          ),
        ],
      ),
      body: routesAsync.when(
        loading: () => const Center(child: CircularProgressIndicator()),
        error: (err, _) => ErrorState(
          message: err.toString(),
          onRetry: () => ref.refresh(routesProvider),
        ),
        data: (routes) {
          if (routes.isEmpty) {
            return const EmptyState(
              icon: Icons.done_all,
              title: 'All done for today',
              subtitle: 'You have no assigned routes right now.',
            );
          }
          return RefreshIndicator(
            onRefresh: () async => ref.refresh(routesProvider.future),
            child: ListView.separated(
              padding: const EdgeInsets.all(16),
              itemCount: routes.length,
              separatorBuilder: (_, __) => const SizedBox(height: 16),
              itemBuilder: (context, index) => _RouteCard(route: routes[index]),
            ),
          );
        },
      ),
    );
  }
}

class _RouteCard extends StatelessWidget {
  final DriverRoute route;
  const _RouteCard({required this.route});

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Container(
            padding: const EdgeInsets.all(16),
            decoration: const BoxDecoration(
              color: AppColors.surfaceElevated,
              borderRadius: BorderRadius.vertical(top: Radius.circular(12)),
            ),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text(
                  'Route ${route.routeId}',
                  style: Theme.of(context).textTheme.titleMedium,
                ),
                Text(
                  '${route.completedStops} / ${route.totalStops} stops',
                  style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                    color: route.completedStops == route.totalStops ? AppColors.good : AppColors.primary,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ],
            ),
          ),
          Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              children: [
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceAround,
                  children: [
                    _Metric(icon: Icons.social_distance, label: '${route.distanceKm} km'),
                    _Metric(icon: Icons.timer, label: '${route.etaMinutes} mins'),
                  ],
                ),
                const SizedBox(height: 16),
                const Divider(),
                ListView.separated(
                  shrinkWrap: true,
                  physics: const NeverScrollableScrollPhysics(),
                  itemCount: route.stops.length,
                  separatorBuilder: (_, __) => const SizedBox(height: 8),
                  itemBuilder: (context, i) {
                    final stop = route.stops[i];
                    return ListTile(
                      leading: CircleAvatar(
                        backgroundColor: stop.status == 'DELIVERED' ? AppColors.good : AppColors.surfaceHighlight,
                        child: Text('${stop.sequence}'),
                      ),
                      title: Text(stop.recipientName),
                      subtitle: Text(stop.address),
                      trailing: StatusBadge(status: stop.status),
                      contentPadding: EdgeInsets.zero,
                    );
                  },
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _Metric extends StatelessWidget {
  final IconData icon;
  final String label;

  const _Metric({required this.icon, required this.label});

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Icon(icon, size: 20, color: AppColors.textSecondary),
        const SizedBox(width: 8),
        Text(label, style: Theme.of(context).textTheme.bodyLarge),
      ],
    );
  }
}
