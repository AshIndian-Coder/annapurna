import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../core/api/api_client.dart';
import '../../core/result/result.dart';
import '../dtos/models.dart';
import '../mocks/mock_data.dart';

final ngoServiceProvider = Provider<NgoService>((ref) {
  return NgoService(ref.read(apiClientProvider));
});

/// Data class representing an available surplus offer visible to NGOs.
/// Backed by the standard Surplus JSON shape returned from `GET /surplus`.
class NgoOffer {
  final String batchId;
  final String batchCode;
  final String foodName;
  final double quantityKg;
  final String foodCategory;
  final String status;
  final DateTime expiryAt;

  const NgoOffer({
    required this.batchId,
    required this.batchCode,
    required this.foodName,
    required this.quantityKg,
    required this.foodCategory,
    required this.status,
    required this.expiryAt,
  });

  factory NgoOffer.fromJson(Map<String, dynamic> json) => NgoOffer(
    batchId: json['batch_id'] as String,
    batchCode: json['batch_code'] as String? ?? '',
    foodName: json['food_name'] as String? ?? 'Unknown',
    quantityKg: (json['quantity_kg'] as num?)?.toDouble() ?? 0,
    foodCategory: json['food_category'] as String? ?? 'General',
    status: json['status'] as String? ?? 'MATCHED',
    expiryAt: DateTime.parse(json['expiry_at'] as String),
  );

  Duration get timeToExpiry => expiryAt.difference(DateTime.now());
  bool get isUrgent => timeToExpiry.inHours < 3 && !timeToExpiry.isNegative;
}

/// Data class representing a past delivery visible to NGOs.
/// Backed by the standard Surplus JSON shape filtered to DELIVERED status.
class NgoHistory {
  final String batchId;
  final String batchCode;
  final String foodName;
  final double quantityKg;
  final String status;
  final DateTime expiryAt;

  const NgoHistory({
    required this.batchId,
    required this.batchCode,
    required this.foodName,
    required this.quantityKg,
    required this.status,
    required this.expiryAt,
  });

  factory NgoHistory.fromJson(Map<String, dynamic> json) => NgoHistory(
    batchId: json['batch_id'] as String,
    batchCode: json['batch_code'] as String? ?? '',
    foodName: json['food_name'] as String? ?? 'Unknown',
    quantityKg: (json['quantity_kg'] as num?)?.toDouble() ?? 0,
    status: json['status'] as String,
    expiryAt: DateTime.parse(json['expiry_at'] as String),
  );
}

class NgoService {
  final ApiClient _api;
  NgoService(this._api);

  /// Contract #13: GET /surplus?status=MATCHED
  /// NGOs see matched batches offered to them via the standard surplus list.
  Future<Result<List<NgoOffer>>> getOffers() async {
    final response = await _api.get('/surplus', queryParameters: {
      'status': 'MATCHED',
    }, parser: (data) =>
      extractList<NgoOffer>(data, NgoOffer.fromJson),
    );
    return response.when(
      success: (data) => Success(data),
      failure: (_) => const Success([]),
    );
  }

  /// Contract #19: POST /matches/{id}/respond — accept the match.
  Future<Result<Map<String, dynamic>>> acceptOffer(String matchId) async {
    return _api.post('/matches/$matchId/respond', data: {
      'action': 'accept',
    }, parser: (data) => extractMap(data));
  }

  /// Contract #19: POST /matches/{id}/respond — decline the match.
  Future<Result<Map<String, dynamic>>> declineOffer(String matchId) async {
    return _api.post('/matches/$matchId/respond', data: {
      'action': 'decline',
    }, parser: (data) => extractMap(data));
  }

  /// Contract #13: GET /surplus?status=DELIVERED
  /// Past deliveries are DELIVERED surplus batches.
  Future<Result<List<NgoHistory>>> getHistory() async {
    final response = await _api.get('/surplus', queryParameters: {
      'status': 'DELIVERED',
    }, parser: (data) =>
      extractList<NgoHistory>(data, NgoHistory.fromJson),
    );
    return response.when(
      success: (data) => Success(data),
      failure: (_) => const Success([]),
    );
  }

  /// Contract #22: POST /qr/{batch_id}/event — record RECEIVED event.
  /// This was already correct in the original code.
  Future<Result<QrEvent>> receiveDelivery(String batchId, {String? clientEventId}) async {
    return _api.post('/qr/$batchId/event', idempotencyKey: clientEventId, data: {
      'event_type': 'RECEIVED',
      if (clientEventId != null) 'client_event_id': clientEventId,
      'client_ts': DateTime.now().toIso8601String(),
    }, parser: (data) => QrEvent.fromJson(data));
  }

  static final List<NgoOffer> _mockOffers = [
    NgoOffer(
      batchId: 'b-001',
      batchCode: 'B-10291',
      foodName: 'Dal Makhani',
      quantityKg: 15.0,
      foodCategory: 'Lentils',
      status: 'MATCHED',
      expiryAt: DateTime.now().add(const Duration(hours: 5)),
    ),
    NgoOffer(
      batchId: 'b-003',
      batchCode: 'B-10293',
      foodName: 'Steamed Rice',
      quantityKg: 22.0,
      foodCategory: 'Grains',
      status: 'MATCHED',
      expiryAt: DateTime.now().add(const Duration(hours: 4)),
    ),
    NgoOffer(
      batchId: 'b-006',
      batchCode: 'B-10296',
      foodName: 'Rajma Chawal',
      quantityKg: 10.0,
      foodCategory: 'Lentils',
      status: 'MATCHED',
      expiryAt: DateTime.now().add(const Duration(hours: 2, minutes: 30)),
    ),
  ];

  static final List<NgoHistory> _mockHistory = [
    NgoHistory(
      batchId: 'b-005',
      batchCode: 'B-10295',
      foodName: 'Chole Bhature',
      quantityKg: 5.0,
      status: 'DELIVERED',
      expiryAt: DateTime.now().subtract(const Duration(hours: 3)),
    ),
    NgoHistory(
      batchId: 'b-010',
      batchCode: 'B-10300',
      foodName: 'Mixed Vegetables',
      quantityKg: 12.0,
      status: 'DELIVERED',
      expiryAt: DateTime.now().subtract(const Duration(days: 1)),
    ),
    NgoHistory(
      batchId: 'b-011',
      batchCode: 'B-10301',
      foodName: 'Puri Sabji',
      quantityKg: 8.5,
      status: 'DELIVERED',
      expiryAt: DateTime.now().subtract(const Duration(days: 2)),
    ),
  ];
}
