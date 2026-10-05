import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../core/api/api_client.dart';
import '../../core/result/result.dart';
import '../dtos/models.dart';
import '../mocks/mock_data.dart';

final logisticsServiceProvider = Provider<LogisticsService>((ref) {
  return LogisticsService(ref.read(apiClientProvider));
});

/// Stop within a driver route.
/// Matches the `stops[]` shape from the contract's `POST /route` response (#21).
class DriverStop {
  final int sequence;
  final String recipientId;
  final String name;
  final double lat;
  final double lng;
  final String eta;
  final bool deadlineOk;
  final String status;

  const DriverStop({
    required this.sequence,
    required this.recipientId,
    required this.name,
    required this.lat,
    required this.lng,
    required this.eta,
    required this.deadlineOk,
    required this.status,
  });

  factory DriverStop.fromJson(Map<String, dynamic> json) => DriverStop(
    sequence: json['sequence'] as int,
    recipientId: json['recipient_id'] as String,
    name: json['name'] as String? ?? 'Recipient',
    lat: (json['lat'] as num).toDouble(),
    lng: (json['lng'] as num).toDouble(),
    eta: json['eta'] as String,
    deadlineOk: json['deadline_ok'] as bool? ?? true,
    status: json['status'] as String? ?? 'PENDING',
  );
}

/// Route result from `POST /route` (#21).
/// Used for both creating routes and displaying them.
class DriverRoute {
  final String routeId;
  final double distanceKm;
  final int etaMinutes;
  final String solver;
  final List<DriverStop> stops;

  const DriverRoute({
    required this.routeId,
    required this.distanceKm,
    required this.etaMinutes,
    required this.solver,
    required this.stops,
  });

  int get completedStops => stops.where((s) => s.status == 'DELIVERED' || s.status == 'HANDED_OFF').length;
  int get totalStops => stops.length;

  factory DriverRoute.fromJson(Map<String, dynamic> json) => DriverRoute(
    routeId: json['route_id'] as String,
    distanceKm: (json['distance_km'] as num).toDouble(),
    etaMinutes: json['eta_minutes'] as int,
    solver: json['solver'] as String? ?? 'haversine',
    stops: extractList<DriverStop>(json['stops'], DriverStop.fromJson),
  );
}

class LogisticsService {
  final ApiClient _api;
  LogisticsService(this._api);

  /// Contract #21: POST /route — generate/retrieve an optimised route.
  /// The contract only defines route creation, not retrieval.
  /// Logistics drivers use this to get their assigned route.
  Future<Result<List<DriverRoute>>> getTodayRoutes() async {
    // Per contract, there's no GET for routes. We query surplus list
    // filtered to IN_TRANSIT and build context from the route that was
    // generated.
    // For production, the driver receives route data via push notification
    // (contract D23) and caches it locally.
    final response = await _api.get('/surplus', queryParameters: {
      'status': 'IN_TRANSIT',
    }, parser: (data) {
      if (data is Map<String, dynamic> && data.containsKey('items')) {
        return extractList<DriverRoute>(data, DriverRoute.fromJson);
      }
      if (data is List) {
         return data.map((e) => DriverRoute.fromJson(e)).toList();
      }
      return <DriverRoute>[]; 
    });
    return response.when(
      success: (data) => Success(data),
      failure: (_) => const Success([]),
    );
  }

  /// Contract #22: POST /qr/{batch_id}/event — record QR chain event.
  /// Logistics drivers use PICKED_UP and HANDED_OFF events.
  Future<Result<QrEvent>> scanEvent(String batchId, {
    required String eventType,
    double? lat,
    double? lng,
    String? clientEventId,
  }) async {
    return _api.post('/qr/$batchId/event', idempotencyKey: clientEventId, data: {
      'event_type': eventType,
      if (lat != null && lng != null) 'location': {'lat': lat, 'lng': lng},
      if (clientEventId != null) 'client_event_id': clientEventId,
      'client_ts': DateTime.now().toIso8601String(),
    }, parser: (data) => QrEvent.fromJson(data));
  }

  /// Stop status updates are performed via QR events (contract #22),
  /// not via a PATCH endpoint. The old PATCH /logistics/routes/{routeId}/stops/{recipientId}
  /// was fabricated and not in the contract.
  /// Instead, use scanEvent() with the appropriate event_type.
  Future<Result<QrEvent>> updateStopStatus(String batchId, String eventType, {String? clientEventId}) async {
    return scanEvent(batchId, eventType: eventType, clientEventId: clientEventId);
  }

  static final List<DriverRoute> _mockRoutes = [
    DriverRoute(
      routeId: 'route-001',
      distanceKm: 12.4,
      etaMinutes: 38,
      solver: 'ortools-vrptw',
      stops: const [
        DriverStop(
          sequence: 1, recipientId: 'r-001', name: 'Helping Hands Foundation',
          lat: 28.6139, lng: 77.2090,
          eta: '14:15', deadlineOk: true, status: 'HANDED_OFF',
        ),
        DriverStop(
          sequence: 2, recipientId: 'r-002', name: 'Annadaata Seva',
          lat: 28.6353, lng: 77.2250,
          eta: '14:42', deadlineOk: true, status: 'IN_TRANSIT',
        ),
        DriverStop(
          sequence: 3, recipientId: 'r-003', name: 'Roti Bank Delhi',
          lat: 28.6508, lng: 77.2373,
          eta: '15:08', deadlineOk: true, status: 'PENDING',
        ),
      ],
    ),
    DriverRoute(
      routeId: 'route-002',
      distanceKm: 8.1,
      etaMinutes: 22,
      solver: 'haversine',
      stops: const [
        DriverStop(
          sequence: 1, recipientId: 'r-004', name: 'Akshaya Patra',
          lat: 28.5918, lng: 77.0471,
          eta: '16:00', deadlineOk: true, status: 'PENDING',
        ),
        DriverStop(
          sequence: 2, recipientId: 'r-005', name: 'Robin Hood Army',
          lat: 28.6215, lng: 77.0855,
          eta: '16:30', deadlineOk: false, status: 'PENDING',
        ),
      ],
    ),
  ];
}
