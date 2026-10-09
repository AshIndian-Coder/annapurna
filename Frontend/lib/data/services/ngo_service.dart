import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../core/api/api_client.dart';
import '../../core/result/result.dart';
import '../dtos/models.dart';

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
  final String quantityUnit;
  final String foodCategory;
  final String status;
  final DateTime expiryAt;

  const NgoOffer({
    required this.batchId,
    required this.batchCode,
    required this.foodName,
    required this.quantityKg,
    this.quantityUnit = 'kg',
    required this.foodCategory,
    required this.status,
    required this.expiryAt,
  });

  factory NgoOffer.fromJson(Map<String, dynamic> json) {
    final batchId = (json['batch_id'] ?? json['id'] ?? json['match_id'] ?? '') as String;
    String batchCode = json['batch_code'] as String? ?? '';
    if (batchCode.isEmpty && batchId.isNotEmpty) {
      batchCode = batchId.length > 8 ? 'B-${batchId.substring(0, 5).toUpperCase()}' : batchId;
    }
    return NgoOffer(
      batchId: batchId,
      batchCode: batchCode,
      foodName: json['food_name'] as String? ?? 'Unknown',
      quantityKg: (json['quantity_kg'] as num?)?.toDouble() ?? 0,
      quantityUnit: json['quantity_unit'] as String? ?? 'kg',
      foodCategory: json['food_category'] as String? ?? 'General',
      status: json['status'] as String? ?? 'OFFERED',
      expiryAt: json['expiry_at'] != null ? DateTime.parse(json['expiry_at'] as String) : DateTime.now().add(const Duration(hours: 4)),
    );
  }

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
  final String quantityUnit;
  final String status;
  final DateTime expiryAt;

  const NgoHistory({
    required this.batchId,
    required this.batchCode,
    required this.foodName,
    required this.quantityKg,
    this.quantityUnit = 'kg',
    required this.status,
    required this.expiryAt,
  });

  factory NgoHistory.fromJson(Map<String, dynamic> json) {
    final batchId = (json['batch_id'] ?? json['id'] ?? json['match_id'] ?? '') as String;
    String batchCode = json['batch_code'] as String? ?? '';
    if (batchCode.isEmpty && batchId.isNotEmpty) {
      batchCode = batchId.length > 8 ? 'B-${batchId.substring(0, 5).toUpperCase()}' : batchId;
    }
    return NgoHistory(
      batchId: batchId,
      batchCode: batchCode,
      foodName: json['food_name'] as String? ?? 'Unknown',
      quantityKg: (json['quantity_kg'] as num?)?.toDouble() ?? 0,
      quantityUnit: json['quantity_unit'] as String? ?? 'kg',
      status: json['status'] as String? ?? 'DELIVERED',
      expiryAt: json['expiry_at'] != null ? DateTime.parse(json['expiry_at'] as String) : DateTime.now().add(const Duration(hours: 4)),
    );
  }
}

class NgoService {
  final ApiClient _api;
  NgoService(this._api);

  /// Retrieves available offers for the NGO. First checks direct matches via /matches,
  /// then falls back to available surplus batches via /surplus.
  Future<Result<List<NgoOffer>>> getOffers() async {
    final matchResponse = await _api.get('/matches', parser: (data) =>
      extractList<NgoOffer>(data, NgoOffer.fromJson),
    );
    if (matchResponse is Success<List<NgoOffer>> && matchResponse.data.isNotEmpty) {
      return matchResponse;
    }

    final surplusResponse = await _api.get('/surplus', queryParameters: {
      'status': 'AVAILABLE',
    }, parser: (data) =>
      extractList<NgoOffer>(data, NgoOffer.fromJson),
    );
    return surplusResponse.when(
      success: (data) => Success(data),
      failure: (_) => matchResponse,
    );
  }

  /// Contract #19: POST /matches/{id}/respond — accept the match.
  Future<Result<Map<String, dynamic>>> acceptOffer(String matchId) async {
    return _api.post('/matches/$matchId/respond', data: {
      'action': 'accept',
      'decision': 'ACCEPTED',
    }, parser: (data) => extractMap(data));
  }

  /// Contract #19: POST /matches/{id}/respond — decline the match.
  Future<Result<Map<String, dynamic>>> declineOffer(String matchId) async {
    return _api.post('/matches/$matchId/respond', data: {
      'action': 'decline',
      'decision': 'DECLINED',
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
      'client_ts': DateTime.now().toUtc().toIso8601String(),
    }, parser: (data) => QrEvent.fromJson(data));
  }

  static final List<NgoOffer> _mockOffers = [
    NgoOffer(
      batchId: 'b-001',
      batchCode: 'B-10291',
      foodName: 'Dal Makhani',
      quantityKg: 15.0,
      quantityUnit: 'kg',
      foodCategory: 'Lentils',
      status: 'MATCHED',
      expiryAt: DateTime.now().add(const Duration(hours: 5)),
    ),
    NgoOffer(
      batchId: 'b-003',
      batchCode: 'B-10293',
      foodName: 'Steamed Rice',
      quantityKg: 22.0,
      quantityUnit: 'kg',
      foodCategory: 'Grains',
      status: 'MATCHED',
      expiryAt: DateTime.now().add(const Duration(hours: 4)),
    ),
    NgoOffer(
      batchId: 'b-006',
      batchCode: 'B-10296',
      foodName: 'Rajma Chawal',
      quantityKg: 10.0,
      quantityUnit: 'kg',
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
      quantityUnit: 'kg',
      status: 'DELIVERED',
      expiryAt: DateTime.now().subtract(const Duration(hours: 3)),
    ),
    NgoHistory(
      batchId: 'b-010',
      batchCode: 'B-10300',
      foodName: 'Mixed Vegetables',
      quantityKg: 12.0,
      quantityUnit: 'kg',
      status: 'DELIVERED',
      expiryAt: DateTime.now().subtract(const Duration(days: 1)),
    ),
    NgoHistory(
      batchId: 'b-011',
      batchCode: 'B-10301',
      foodName: 'Puri Sabji',
      quantityKg: 8.5,
      quantityUnit: 'kg',
      status: 'DELIVERED',
      expiryAt: DateTime.now().subtract(const Duration(days: 2)),
    ),
  ];
}
