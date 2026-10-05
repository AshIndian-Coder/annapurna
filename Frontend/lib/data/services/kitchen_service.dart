import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../core/api/api_client.dart';
import '../../core/result/result.dart';
import '../dtos/models.dart';
import '../mocks/mock_data.dart';

final kitchenServiceProvider = Provider<KitchenService>((ref) {
  return KitchenService(ref.read(apiClientProvider));
});

class KitchenService {
  final ApiClient _api;
  KitchenService(this._api);

  Future<Result<KitchenOverview>> getOverview() async {
    final response = await _api.get('/kitchen/overview', parser: (data) => KitchenOverview.fromJson(data));
    return response.when(
      success: (data) => Success(data),
      failure: (_) => const Success(KitchenOverview(
        expectedDiners: 0, forecastP50: 0, recommendedProduction: 0, 
        surplusRisk: 'LOW', wasteToday: 0, availableKg: 0, redistributedToday: 0
      )),
    );
  }

  Future<Result<Prediction>> predictDemand({
    required int attendance,
    required String mealType,
    required List<String> menu,
    required int dayOfWeek,
    required String date,
    String? datasetId,
  }) async {
    if (useMocks) {
      await Future.delayed(const Duration(milliseconds: 800));
      return Success(MockData.prediction);
    }
    return _api.post('/predict-demand', data: {
      // A blank attendance means "predict from my uploaded history alone", so
      // the field is omitted rather than sent as 0.
      if (attendance > 0) 'attendance': attendance,
      'meal_type': mealType,
      'menu': menu,
      'day_of_week': dayOfWeek,
      'date': date,
      if (datasetId != null && datasetId.isNotEmpty) 'dataset_id': datasetId,
    }, parser: (data) => Prediction.fromJson(extractMap(data)));
  }

  Future<Result<Prediction>> whatIf({
    required int attendance,
    required List<String> menu,
    required int dayOfWeek,
  }) async {
    if (useMocks) {
      await Future.delayed(const Duration(milliseconds: 500));
      return Success(MockData.prediction);
    }
    return _api.post('/planning/what-if', data: {
      'attendance': attendance,
      'menu': menu,
      'day_of_week': dayOfWeek,
    }, parser: (data) => Prediction.fromJson(extractMap(data)));
  }

  Future<Result<List<Alert>>> getAlerts() async {
    final response = await _api.get(
      '/alerts',
      parser: (data) => extractList<Alert>(data, Alert.fromJson),
    );
    return response.when(
      success: (data) => Success(data),
      failure: (_) => const Success([]),
    );
  }

  Future<Result<ImpactData>> getImpact() async {
    final response = await _api.get('/analytics/impact', parser: (data) => ImpactData.fromJson(data));
    return response.when(
      success: (data) => Success(data),
      failure: (_) => const Success(ImpactData(
        wasteAvoidedKg: 0, redistributedKg: 0, divertedKg: 0, estimatedCarbonSavedKg: 0,
        carbonFactorUsed: 0, carbonFactorSource: '', successfulRedistributions: 0, mealsEquivalent: 0, trend: []
      )),
    );
  }

  Future<Result<List<SensorReading>>> getSensorReadings() async {
    final response = await _api.get(
      '/sensors/latest',
      parser: (data) => extractList<SensorReading>(data, SensorReading.fromJson),
    );
    return response.when(
      success: (data) => Success(data),
      failure: (_) => const Success([]),
    );
  }

  Future<Result<ProcessingResult>> submitProcessing({
    required String date,
    required double rawMaterialKg,
    required double outputKg,
    required double wasteKg,
    required int runtimeMinutes,
    required int downtimeMinutes,
    required double energyKwh,
  }) async {
    if (useMocks) {
      await Future.delayed(const Duration(milliseconds: 500));
      return const Success(MockData.processing);
    }
    return _api.post('/processing/metrics', data: {
      'date': date,
      'raw_material_kg': rawMaterialKg,
      'output_kg': outputKg,
      'waste_kg': wasteKg,
      'runtime_minutes': runtimeMinutes,
      'downtime_minutes': downtimeMinutes,
      'energy_kwh': energyKwh,
    }, parser: (data) => ProcessingResult.fromJson(extractMap(data)));
  }
}
