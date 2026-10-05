import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../core/api/api_client.dart';
import '../../core/result/result.dart';
import '../dtos/models.dart';
import '../mocks/mock_data.dart';

final wasteServiceProvider = Provider<WasteService>((ref) {
  return WasteService(ref.read(apiClientProvider));
});

class WasteService {
  final ApiClient _api;
  WasteService(this._api);

  Future<Result<Map<String, dynamic>>> logWaste({
    required double quantityKg,
    required String cause,
    required String mealType,
    String? clientEventId,
  }) async {
    if (useMocks) {
      await Future.delayed(const Duration(milliseconds: 400));
      return Success({'id': 'w-new', 'quantity_kg': quantityKg});
    }
    return _api.post('/waste', idempotencyKey: clientEventId, data: {
      'quantity_kg': quantityKg,
      'cause': cause,
      'meal_type': mealType,
    }, parser: (data) => extractMap(data));
  }

  Future<Result<Map<String, dynamic>>> getAnalytics() async {
    if (useMocks) {
      await Future.delayed(const Duration(milliseconds: 300));
      return Success({
        'total_kg': 14.6,
        'by_cause': {
          'OVERPRODUCTION': 9.7,
          'LOW_ATTENDANCE': 3.1,
          'SPOILAGE': 1.8,
        },
        'trend': List.generate(7, (i) => {
          'date': DateTime.now().subtract(Duration(days: 6 - i)).toIso8601String().substring(0, 10),
          'kg': 8.0 + (i * 1.2) - (i % 2 == 0 ? 3.0 : 0),
        }),
        'entries': MockData.wasteEntries.map((e) => {
          'id': e.id, 'quantity_kg': e.quantityKg, 'cause': e.cause,
          'meal_type': e.mealType, 'date': e.date.toIso8601String(),
        }).toList(),
      });
    }
    return _api.get('/waste/analytics', parser: (data) => extractMap(data));
  }
}
