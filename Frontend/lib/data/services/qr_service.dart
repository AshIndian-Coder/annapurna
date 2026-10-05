import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../core/api/api_client.dart';
import '../../core/result/result.dart';
import '../dtos/models.dart';
import '../mocks/mock_data.dart';

final qrServiceProvider = Provider<QrService>((ref) {
  return QrService(ref.read(apiClientProvider));
});

class QrService {
  final ApiClient _api;
  QrService(this._api);

  Future<Result<QrEvent>> postEvent(String batchId, {
    required String eventType,
    double? lat,
    double? lng,
    String? clientEventId,
    String? clientTs,
  }) async {
    if (useMocks) {
      await Future.delayed(const Duration(milliseconds: 300));
      return Success(QrEvent(
        eventType: eventType,
        timestamp: DateTime.now().toIso8601String(),
        hash: 'mock_hash_${DateTime.now().millisecondsSinceEpoch}',
        prevHash: 'mock_prev_hash',
        clientTs: clientTs,
      ));
    }
    return _api.post('/qr/$batchId/event', idempotencyKey: clientEventId, data: {
      'event_type': eventType,
      if (lat != null && lng != null) 'location': {'lat': lat, 'lng': lng},
      if (clientEventId != null) 'client_event_id': clientEventId,
      if (clientTs != null) 'client_ts': clientTs,
    }, parser: (data) => QrEvent.fromJson(data));
  }

  Future<Result<QrTimeline>> getTimeline(String batchId) async {
    if (useMocks) {
      await Future.delayed(const Duration(milliseconds: 300));
      return Success(QrTimeline(
        batchId: batchId,
        batchCode: 'B-10291',
        valid: true,
        length: 4,
        events: [
          QrEvent(eventType: 'CREATED', timestamp: DateTime.now().subtract(const Duration(hours: 3)).toIso8601String(), hash: 'h1', prevHash: ''),
          QrEvent(eventType: 'APPROVED', timestamp: DateTime.now().subtract(const Duration(hours: 2)).toIso8601String(), hash: 'h2', prevHash: 'h1'),
          QrEvent(eventType: 'MATCHED', timestamp: DateTime.now().subtract(const Duration(hours: 1)).toIso8601String(), hash: 'h3', prevHash: 'h2'),
          QrEvent(eventType: 'PICKED_UP', timestamp: DateTime.now().subtract(const Duration(minutes: 30)).toIso8601String(), hash: 'h4', prevHash: 'h3'),
        ],
      ));
    }
    return _api.get('/qr/$batchId', parser: (data) => QrTimeline.fromJson(data));
  }

  Future<Result<Map<String, dynamic>>> verify(String batchId) async {
    if (useMocks) {
      await Future.delayed(const Duration(milliseconds: 200));
      return Success({'valid': true, 'length': 4, 'broken_at': null});
    }
    return _api.get('/qr/$batchId/verify', parser: (data) => extractMap(data));
  }
}
