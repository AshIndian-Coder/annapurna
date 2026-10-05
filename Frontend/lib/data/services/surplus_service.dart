import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../core/api/api_client.dart';
import '../../core/enums.dart';
import '../../core/result/result.dart';
import '../dtos/models.dart';
import '../mocks/mock_data.dart';

final surplusServiceProvider = Provider<SurplusService>((ref) {
  return SurplusService(ref.read(apiClientProvider));
});

class SurplusService {
  final ApiClient _api;
  SurplusService(this._api);

  Future<Result<List<Surplus>>> getSurplusList({String? status, String? cursor}) async {
    if (useMocks) {
      await Future.delayed(const Duration(milliseconds: 400));
      if (status != null) {
        return Success(MockData.surplusList.where((s) => s.status.apiValue == status).toList());
      }
      return Success(MockData.surplusList);
    }
    return _api.get('/surplus', queryParameters: {
      if (status != null) 'status': status,
      if (cursor != null) 'cursor': cursor,
    }, parser: (data) => extractList<Surplus>(data, Surplus.fromJson));
  }

  Future<Result<Surplus>> getSurplus(String id) async {
    if (useMocks) {
      await Future.delayed(const Duration(milliseconds: 200));
      return Success(MockData.surplusList.firstWhere((s) => s.batchId == id));
    }
    return _api.get('/surplus/$id', parser: (data) => Surplus.fromJson(data));
  }

  Future<Result<Surplus>> createSurplus({
    required String foodName,
    required double quantityKg,
    required String quantityUnit,
    required String preparedAt,
    required String expiryAt,
    String? mealId,
    String? foodCategory,
    String? clientEventId,
  }) async {
    if (useMocks) {
      await Future.delayed(const Duration(milliseconds: 500));
      return Success(MockData.surplusList.first);
    }
    return _api.post('/surplus', idempotencyKey: clientEventId, data: {
      'food_name': foodName,
      'quantity_kg': quantityKg,
      'quantity_unit': quantityUnit,
      'prepared_at': preparedAt,
      'expiry_at': expiryAt,
      if (mealId != null) 'meal_id': mealId,
      if (foodCategory != null) 'food_category': foodCategory,
      if (clientEventId != null) 'client_event_id': clientEventId,
    }, parser: (data) => Surplus.fromJson(data));
  }

  /// Retrieves quality check result for a batch.
  /// Contract #15: POST /quality/check is multipart (handled by the CV capture flow).
  /// Contract #14: GET /surplus/{id} returns the batch with embedded quality info.
  /// This method fetches the surplus to display the already-stored quality result.
  Future<Result<QualityResult>> checkQuality(String batchId) async {
    if (useMocks) {
      await Future.delayed(const Duration(milliseconds: 1200));
      return Success(MockData.qualityResult);
    }
    return _api.get('/surplus/$batchId', parser: (data) {
      // The surplus response may contain embedded quality fields.
      // If quality data is present, extract it. Otherwise, return a
      // pending-state result based on the batch's safety_status.
      if (data.containsKey('visual') && data['visual'] != null) {
        return QualityResult.fromJson(data);
      }
      // Fallback: construct from surplus-level fields.
      return QualityResult(
        batchId: data['batch_id'] as String,
        visual: VisualBlock(
          status: data['quality_status'] as String? ?? 'PENDING',
          riskLevel: data['risk_level'] as String? ?? 'LOW',
          confidence: 0.0,
          reason: data['reason'] as String? ?? '',
          modelVersion: data['model_version'] as String? ?? '',
        ),
        safetyDecision: SafetyBlock(
          status: data['safety_status'] as String? ?? 'PENDING',
          reasons: const [],
          dangerZoneMinutes: 0,
          hoursToExpiry: 0,
          modelVersion: '',
        ),
        requiresHumanApproval: data['safety_status'] == 'HOLD',
      );
    });
  }

  Future<Result<Map<String, dynamic>>> approveSurplus(String batchId, {required String decision, String? note, String? overrideReason}) async {
    if (useMocks) {
      await Future.delayed(const Duration(milliseconds: 400));
      return Success({'batch_id': batchId, 'status': 'AVAILABLE'});
    }
    return _api.post('/surplus/$batchId/approve', data: {
      'decision': decision,
      if (note != null) 'note': note,
      if (overrideReason != null) 'override_reason': overrideReason,
    }, parser: (data) => extractMap(data));
  }

  Future<Result<List<MatchResult>>> matchSurplus(String batchId, {int maxResults = 5}) async {
    if (useMocks) {
      await Future.delayed(const Duration(milliseconds: 700));
      return Success(MockData.matchResults);
    }
    return _api.post('/match', data: {
      'batch_id': batchId,
      'max_results': maxResults,
    }, parser: (data) => extractList<MatchResult>(data, MatchResult.fromJson));
  }

  Future<Result<RouteResult>> buildRoute(String batchId, List<String> recipientIds) async {
    if (useMocks) {
      await Future.delayed(const Duration(milliseconds: 800));
      return Success(MockData.routeResult);
    }
    return _api.post('/route', data: {
      'batch_id': batchId,
      'recipient_ids': recipientIds,
    }, parser: (data) => RouteResult.fromJson(data));
  }
}
