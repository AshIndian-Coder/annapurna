import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:latlong2/latlong.dart';
import 'package:go_router/go_router.dart';
import '../../core/theme/app_colors.dart';
import '../../data/services/logistics_service.dart';
import '../../shared/widgets/common_widgets.dart';
import '../../shared/widgets/profile_drawer.dart';

final availableRoutesProvider = FutureProvider.autoDispose<List<dynamic>>((ref) async {
  final service = ref.read(logisticsServiceProvider);
  final result = await service.getAvailableRoutes();
  return result.when(
    success: (data) => data,
    failure: (error) => throw error,
  );
});

final assignedRoutesProvider = FutureProvider.autoDispose<List<dynamic>>((ref) async {
  final service = ref.read(logisticsServiceProvider);
  final result = await service.getAssignedRoutes();
  return result.when(
    success: (data) => data,
    failure: (error) => throw error,
  );
});

class LogisticsRoutesScreen extends ConsumerStatefulWidget {
  const LogisticsRoutesScreen({super.key});

  @override
  ConsumerState<LogisticsRoutesScreen> createState() => _LogisticsRoutesScreenState();
}

class _LogisticsRoutesScreenState extends ConsumerState<LogisticsRoutesScreen> {
  final MapController _mapController = MapController();

  Future<void> _claimRoute(String routeId) async {
    try {
      final result = await ref.read(logisticsServiceProvider).assignRoute(routeId);
      result.when(
        success: (_) {
          ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Route claimed successfully!')));
          ref.refresh(availableRoutesProvider);
          ref.refresh(assignedRoutesProvider);
        },
        failure: (e) {
          ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Failed to claim route: $e')));
        },
      );
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Error: $e')));
    }
  }

  @override
  Widget build(BuildContext context) {
    return DefaultTabController(
      length: 2,
      child: Scaffold(
        appBar: AppBar(
          title: const Text('Deliveries'),
          actions: [
            IconButton(
              icon: const Icon(Icons.refresh),
              onPressed: () {
                ref.refresh(availableRoutesProvider);
                ref.refresh(assignedRoutesProvider);
              },
            ),
          ],
          bottom: const TabBar(
            tabs: [
              Tab(text: 'Available'),
              Tab(text: 'My Deliveries'),
            ],
          ),
        ),
        drawer: const ProfileDrawer(),
        body: TabBarView(
          children: [
            _buildAvailableTab(),
            _buildAssignedTab(),
          ],
        ),
      ),
    );
  }

  Widget _buildAvailableTab() {
    final routesAsync = ref.watch(availableRoutesProvider);
    return routesAsync.when(
        loading: () => const Center(child: CircularProgressIndicator()),
        error: (err, _) => ErrorState(
          message: err.toString(),
          onRetry: () => ref.refresh(availableRoutesProvider),
        ),
        data: (deliveries) {
          if (deliveries.isEmpty) {
            return const EmptyState(
              icon: Icons.done_all,
              title: 'No pending deliveries',
              subtitle: 'Check back later for new deliveries.',
            );
          }
          
          double defaultLat = 28.6139;
          double defaultLng = 77.2090;
          if (deliveries.isNotEmpty) {
            final firstKLat = (deliveries.first['kitchen_lat'] as num?)?.toDouble();
            final firstKLng = (deliveries.first['kitchen_lng'] as num?)?.toDouble();
            if (firstKLat != null && firstKLng != null && firstKLat != 0.0) {
              defaultLat = firstKLat;
              defaultLng = firstKLng;
            }
          }
          final center = LatLng(defaultLat, defaultLng);

          return Column(
            children: [
              Expanded(
                flex: 2,
                child: Stack(
                  children: [
                    FlutterMap(
                      mapController: _mapController,
                      options: MapOptions(
                        initialCenter: center,
                        initialZoom: 12.0,
                        minZoom: 4.0,
                        maxZoom: 18.0,
                      ),
                      children: [
                        TileLayer(
                          urlTemplate: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
                          fallbackUrl: 'https://a.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png',
                          userAgentPackageName: 'com.annapurna.foodrescue',
                          maxZoom: 19,
                        ),
                        PolylineLayer(
                          polylines: deliveries.map<Polyline>((d) {
                            final kLat = (d['kitchen_lat'] as num?)?.toDouble() ?? 28.6139;
                            final kLng = (d['kitchen_lng'] as num?)?.toDouble() ?? 77.2090;
                            final rLat = (d['recipient_lat'] as num?)?.toDouble() ?? 28.6200;
                            final rLng = (d['recipient_lng'] as num?)?.toDouble() ?? 77.2150;
                            return Polyline(
                              points: [LatLng(kLat, kLng), LatLng(rLat, rLng)],
                              color: AppColors.primary.withOpacity(0.85),
                              strokeWidth: 3.5,
                            );
                          }).toList(),
                        ),
                        MarkerLayer(
                          markers: deliveries.expand<Marker>((d) {
                            final kLat = (d['kitchen_lat'] as num?)?.toDouble() ?? 28.6139;
                            final kLng = (d['kitchen_lng'] as num?)?.toDouble() ?? 77.2090;
                            final rLat = (d['recipient_lat'] as num?)?.toDouble() ?? 28.6200;
                            final rLng = (d['recipient_lng'] as num?)?.toDouble() ?? 77.2150;
                            final kName = d['kitchen_name']?.toString() ?? 'Kitchen';
                            final rName = d['recipient_name']?.toString() ?? 'NGO';

                            return [
                              Marker(
                                point: LatLng(kLat, kLng),
                                width: 110,
                                height: 54,
                                child: Column(
                                  mainAxisSize: MainAxisSize.min,
                                  children: [
                                    Container(
                                      padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                                      decoration: BoxDecoration(
                                        color: Colors.blue.shade800,
                                        borderRadius: BorderRadius.circular(4),
                                        boxShadow: const [BoxShadow(color: Colors.black26, blurRadius: 4)],
                                      ),
                                      child: Text(
                                        'From: $kName',
                                        style: const TextStyle(color: Colors.white, fontSize: 10, fontWeight: FontWeight.bold),
                                        overflow: TextOverflow.ellipsis,
                                        maxLines: 1,
                                      ),
                                    ),
                                    const Icon(Icons.location_on, color: Colors.blue, size: 28),
                                  ],
                                ),
                              ),
                              Marker(
                                point: LatLng(rLat, rLng),
                                width: 110,
                                height: 54,
                                child: Column(
                                  mainAxisSize: MainAxisSize.min,
                                  children: [
                                    Container(
                                      padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                                      decoration: BoxDecoration(
                                        color: Colors.green.shade800,
                                        borderRadius: BorderRadius.circular(4),
                                        boxShadow: const [BoxShadow(color: Colors.black26, blurRadius: 4)],
                                      ),
                                      child: Text(
                                        'To: $rName',
                                        style: const TextStyle(color: Colors.white, fontSize: 10, fontWeight: FontWeight.bold),
                                        overflow: TextOverflow.ellipsis,
                                        maxLines: 1,
                                      ),
                                    ),
                                    const Icon(Icons.location_on, color: Colors.green, size: 28),
                                  ],
                                ),
                              ),
                            ];
                          }).toList(),
                        ),
                      ],
                    ),
                    Positioned(
                      top: 10,
                      left: 10,
                      child: Container(
                        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                        decoration: BoxDecoration(
                          color: Colors.black.withOpacity(0.65),
                          borderRadius: BorderRadius.circular(16),
                        ),
                        child: const Row(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            Icon(Icons.map, size: 14, color: Colors.white),
                            SizedBox(width: 6),
                            Text(
                              'OpenStreetMap (Free)',
                              style: TextStyle(color: Colors.white, fontSize: 11, fontWeight: FontWeight.w500),
                            ),
                          ],
                        ),
                      ),
                    ),
                    Positioned(
                      right: 12,
                      bottom: 12,
                      child: Column(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          FloatingActionButton.small(
                            heroTag: 'map_zoom_in',
                            backgroundColor: AppColors.surfaceElevated,
                            foregroundColor: AppColors.textPrimary,
                            onPressed: () {
                              final z = _mapController.camera.zoom;
                              _mapController.move(_mapController.camera.center, z + 1);
                            },
                            child: const Icon(Icons.add, size: 20),
                          ),
                          const SizedBox(height: 6),
                          FloatingActionButton.small(
                            heroTag: 'map_zoom_out',
                            backgroundColor: AppColors.surfaceElevated,
                            foregroundColor: AppColors.textPrimary,
                            onPressed: () {
                              final z = _mapController.camera.zoom;
                              _mapController.move(_mapController.camera.center, z - 1);
                            },
                            child: const Icon(Icons.remove, size: 20),
                          ),
                          const SizedBox(height: 6),
                          FloatingActionButton.small(
                            heroTag: 'map_recenter',
                            backgroundColor: AppColors.surfaceElevated,
                            foregroundColor: AppColors.primary,
                            onPressed: () {
                              _mapController.move(center, 12.0);
                            },
                            child: const Icon(Icons.my_location, size: 20),
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
              Expanded(
                flex: 3,
                child: RefreshIndicator(
                  onRefresh: () async => ref.refresh(availableRoutesProvider.future),
                  child: ListView.separated(
                    padding: const EdgeInsets.all(16),
                    itemCount: deliveries.length,
                    separatorBuilder: (_, __) => const SizedBox(height: 16),
                    itemBuilder: (context, index) {
                      final delivery = deliveries[index];
                      return _DeliveryCard(
                        delivery: delivery,
                        onClaim: () => _claimRoute(delivery['route_id']),
                        onFocus: () {
                          final kLat = (delivery['kitchen_lat'] as num?)?.toDouble() ?? 28.6139;
                          final kLng = (delivery['kitchen_lng'] as num?)?.toDouble() ?? 77.2090;
                          _mapController.move(LatLng(kLat, kLng), 13.0);
                        },
                      );
                    },
                  ),
                ),
              ),
            ],
          );
        },
    );
  }

  Widget _buildAssignedTab() {
    final routesAsync = ref.watch(assignedRoutesProvider);
    return routesAsync.when(
        loading: () => const Center(child: CircularProgressIndicator()),
        error: (err, _) => ErrorState(
          message: err.toString(),
          onRetry: () => ref.refresh(assignedRoutesProvider),
        ),
        data: (deliveries) {
          if (deliveries.isEmpty) {
            return const EmptyState(
              icon: Icons.done_all,
              title: 'No assigned deliveries',
              subtitle: 'You have no deliveries to make right now.',
            );
          }
          
          double defaultLat = 28.6139;
          double defaultLng = 77.2090;
          if (deliveries.isNotEmpty) {
            final firstKLat = (deliveries.first['kitchen_lat'] as num?)?.toDouble();
            final firstKLng = (deliveries.first['kitchen_lng'] as num?)?.toDouble();
            if (firstKLat != null && firstKLng != null && firstKLat != 0.0) {
              defaultLat = firstKLat;
              defaultLng = firstKLng;
            }
          }
          final center = LatLng(defaultLat, defaultLng);

          return Column(
            children: [
              Expanded(
                flex: 2,
                child: Stack(
                  children: [
                    FlutterMap(
                      options: MapOptions(
                        initialCenter: center,
                        initialZoom: 12.0,
                        minZoom: 4.0,
                        maxZoom: 18.0,
                      ),
                      children: [
                        TileLayer(
                          urlTemplate: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
                          fallbackUrl: 'https://a.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png',
                          userAgentPackageName: 'com.annapurna.foodrescue',
                          maxZoom: 19,
                        ),
                        PolylineLayer(
                          polylines: deliveries.map<Polyline>((d) {
                            final kLat = (d['kitchen_lat'] as num?)?.toDouble() ?? 28.6139;
                            final kLng = (d['kitchen_lng'] as num?)?.toDouble() ?? 77.2090;
                            final rLat = (d['recipient_lat'] as num?)?.toDouble() ?? 28.6200;
                            final rLng = (d['recipient_lng'] as num?)?.toDouble() ?? 77.2150;
                            return Polyline(
                              points: [LatLng(kLat, kLng), LatLng(rLat, rLng)],
                              color: AppColors.good,
                              strokeWidth: 4.0,
                            );
                          }).toList(),
                        ),
                        MarkerLayer(
                          markers: deliveries.expand<Marker>((d) {
                            final kLat = (d['kitchen_lat'] as num?)?.toDouble() ?? 28.6139;
                            final kLng = (d['kitchen_lng'] as num?)?.toDouble() ?? 77.2090;
                            final rLat = (d['recipient_lat'] as num?)?.toDouble() ?? 28.6200;
                            final rLng = (d['recipient_lng'] as num?)?.toDouble() ?? 77.2150;
                            final kName = d['kitchen_name']?.toString() ?? 'Kitchen';
                            final rName = d['recipient_name']?.toString() ?? 'NGO';

                            return [
                              Marker(
                                point: LatLng(kLat, kLng),
                                width: 110,
                                height: 54,
                                child: Column(
                                  mainAxisSize: MainAxisSize.min,
                                  children: [
                                    Container(
                                      padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                                      decoration: BoxDecoration(
                                        color: Colors.blue.shade800,
                                        borderRadius: BorderRadius.circular(4),
                                      ),
                                      child: Text(
                                        'Pickup: $kName',
                                        style: const TextStyle(color: Colors.white, fontSize: 10, fontWeight: FontWeight.bold),
                                        overflow: TextOverflow.ellipsis,
                                        maxLines: 1,
                                      ),
                                    ),
                                    const Icon(Icons.location_on, color: Colors.blue, size: 28),
                                  ],
                                ),
                              ),
                              Marker(
                                point: LatLng(rLat, rLng),
                                width: 110,
                                height: 54,
                                child: Column(
                                  mainAxisSize: MainAxisSize.min,
                                  children: [
                                    Container(
                                      padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                                      decoration: BoxDecoration(
                                        color: Colors.green.shade800,
                                        borderRadius: BorderRadius.circular(4),
                                      ),
                                      child: Text(
                                        'Drop: $rName',
                                        style: const TextStyle(color: Colors.white, fontSize: 10, fontWeight: FontWeight.bold),
                                        overflow: TextOverflow.ellipsis,
                                        maxLines: 1,
                                      ),
                                    ),
                                    const Icon(Icons.location_on, color: Colors.green, size: 28),
                                  ],
                                ),
                              ),
                            ];
                          }).toList(),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
              Expanded(
                flex: 3,
                child: RefreshIndicator(
                  onRefresh: () async => ref.refresh(assignedRoutesProvider.future),
                  child: ListView.separated(
                    padding: const EdgeInsets.all(16),
                    itemCount: deliveries.length,
                    separatorBuilder: (_, __) => const SizedBox(height: 16),
                    itemBuilder: (context, index) {
                      final delivery = deliveries[index];
                      return _DeliveryCard(
                        delivery: delivery,
                        isAssigned: true,
                        onClaim: () {},
                      );
                    },
                  ),
                ),
              ),
            ],
          );
        },
    );
  }
}

class _DeliveryCard extends StatelessWidget {
  final dynamic delivery;
  final bool isAssigned;
  final VoidCallback onClaim;
  final VoidCallback? onFocus;
  
  const _DeliveryCard({
    required this.delivery,
    required this.onClaim,
    this.isAssigned = false,
    this.onFocus,
  });

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
                Expanded(
                  child: Text(
                    '${delivery['quantity_kg']} ${delivery['quantity_unit'] ?? 'kg'} ${delivery['food_name']}',
                    style: Theme.of(context).textTheme.titleMedium,
                    overflow: TextOverflow.ellipsis,
                  ),
                ),
                if (onFocus != null)
                  IconButton(
                    icon: const Icon(Icons.pin_drop, size: 20, color: AppColors.accent),
                    tooltip: 'Focus on Map',
                    onPressed: onFocus,
                    constraints: const BoxConstraints(),
                    padding: const EdgeInsets.only(right: 8),
                  ),
                StatusBadge(label: isAssigned ? 'Assigned' : 'Available', color: isAssigned ? AppColors.accent : AppColors.primary),
              ],
            ),
          ),
          Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceAround,
                  children: [
                    _Metric(icon: Icons.social_distance, label: '${delivery['distance_km'] ?? '??'} km'),
                    _Metric(icon: Icons.timer, label: 'Optimal Route'),
                  ],
                ),
                const SizedBox(height: 16),
                const Divider(),
                const SizedBox(height: 8),
                Row(
                  children: [
                    const Icon(Icons.restaurant, color: Colors.blue, size: 20),
                    const SizedBox(width: 8),
                    Expanded(child: Text('From: ${delivery['kitchen_name']}')),
                  ],
                ),
                const SizedBox(height: 8),
                Row(
                  children: [
                    const Icon(Icons.home, color: Colors.green, size: 20),
                    const SizedBox(width: 8),
                    Expanded(child: Text('To: ${delivery['recipient_name']}')),
                  ],
                ),
                const SizedBox(height: 16),
                if (!isAssigned)
                  ElevatedButton(
                    onPressed: onClaim,
                    style: ElevatedButton.styleFrom(
                      minimumSize: const Size(double.infinity, 48),
                      backgroundColor: AppColors.primary,
                    ),
                    child: const Text('Claim Delivery', style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
                  )
                else
                  ElevatedButton.icon(
                    onPressed: () {
                      context.push('/logistics/scan');
                    },
                    icon: const Icon(Icons.qr_code_scanner, color: Colors.white),
                    label: const Text('Go to Scan', style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
                    style: ElevatedButton.styleFrom(
                      minimumSize: const Size(double.infinity, 48),
                      backgroundColor: AppColors.accent,
                    ),
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
        Text(label, style: Theme.of(context).textTheme.bodyMedium),
      ],
    );
  }
}
