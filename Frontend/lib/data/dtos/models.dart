import '../../core/api/api_client.dart';
import '../../core/enums.dart';

class User {
  final String id;
  final String name;
  final String email;
  final UserRole role;

  const User({required this.id, required this.name, required this.email, required this.role});

  /// `name` and `email` are coerced rather than cast: the API may omit them
  /// (an account registered without a display name) and a hard cast here used
  /// to throw, which surfaced as an opaque parse failure on the login screen.
  factory User.fromJson(Map<String, dynamic> json) => User(
    id: json['id'] as String? ?? '',
    name: json['name'] as String? ?? '',
    email: json['email'] as String? ?? '',
    role: UserRoleX.fromApi(json['role'] as String? ?? ''),
  );

  /// Falls back to the email local-part, then the role, so the shell header
  /// always has something to display.
  String get displayName {
    final trimmed = name.trim();
    if (trimmed.isNotEmpty) return trimmed;
    final local = email.split('@').first.trim();
    if (local.isNotEmpty) return local;
    return role.apiValue;
  }
}

class AuthTokens {
  final String accessToken;
  final String refreshToken;
  final int expiresIn;
  final User user;

  const AuthTokens({
    required this.accessToken,
    required this.refreshToken,
    required this.expiresIn,
    required this.user,
  });

  factory AuthTokens.fromJson(Map<String, dynamic> json) => AuthTokens(
    accessToken: json['access_token'] as String? ?? '',
    refreshToken: json['refresh_token'] as String? ?? '',
    expiresIn: (json['expires_in'] as num?)?.toInt() ?? 3600,
    user: User.fromJson((json['user'] as Map<String, dynamic>?) ?? const {}),
  );

  /// The instant the access token stops being accepted. Null when the server
  /// did not report a lifetime, which disables the proactive refresh timer.
  DateTime? get expiresAt =>
      expiresIn > 0 ? DateTime.now().add(Duration(seconds: expiresIn)) : null;
}

class Prediction {
  final double predictedConsumption;
  final double recommendedProduction;
  final double expectedSurplus;
  final String surplusRisk;
  final PredictionInterval interval;
  final double recommendedQuantile;
  final List<Driver> topDrivers;
  final double confidence;
  final String modelVersion;
  final String dataSource;

  const Prediction({
    required this.predictedConsumption,
    required this.recommendedProduction,
    required this.expectedSurplus,
    required this.surplusRisk,
    required this.interval,
    required this.recommendedQuantile,
    required this.topDrivers,
    required this.confidence,
    required this.modelVersion,
    required this.dataSource,
  });

  factory Prediction.fromJson(Map<String, dynamic> json) => Prediction(
    predictedConsumption: (json['predicted_consumption'] as num).toDouble(),
    recommendedProduction: (json['recommended_production'] as num).toDouble(),
    expectedSurplus: (json['expected_surplus'] as num).toDouble(),
    surplusRisk: json['surplus_risk'] as String,
    interval: PredictionInterval.fromJson(json['prediction_interval'] as Map<String, dynamic>),
    recommendedQuantile: (json['recommended_quantile'] as num).toDouble(),
    topDrivers: extractList<Driver>(json['top_drivers'], Driver.fromJson),
    confidence: (json['confidence'] as num).toDouble(),
    modelVersion: json['model_version'] as String,
    dataSource: json['data_source'] as String,
  );
}

class PredictionInterval {
  final double p10;
  final double p50;
  final double p90;
  final double coverageTarget;

  const PredictionInterval({
    required this.p10,
    required this.p50,
    required this.p90,
    required this.coverageTarget,
  });

  factory PredictionInterval.fromJson(Map<String, dynamic> json) => PredictionInterval(
    p10: (json['p10'] as num).toDouble(),
    p50: (json['p50'] as num).toDouble(),
    p90: (json['p90'] as num).toDouble(),
    coverageTarget: (json['coverage_target'] as num).toDouble(),
  );
}

class Driver {
  final String feature;
  final double effectKg;

  const Driver({required this.feature, required this.effectKg});

  factory Driver.fromJson(Map<String, dynamic> json) => Driver(
    feature: json['feature'] as String,
    effectKg: (json['effect_kg'] as num).toDouble(),
  );
}

class Surplus {
  final String batchId;
  final String batchCode;
  final String foodName;
  final double quantityKg;
  final String quantityUnit;
  final SurplusStatus status;
  final String? safetyStatus;
  final DateTime preparedAt;
  final DateTime expiryAt;
  final String? mealId;
  final String? foodCategory;

  const Surplus({
    required this.batchId,
    required this.batchCode,
    required this.foodName,
    required this.quantityKg,
    required this.quantityUnit,
    required this.status,
    this.safetyStatus,
    required this.preparedAt,
    required this.expiryAt,
    this.mealId,
    this.foodCategory,
  });

  factory Surplus.fromJson(Map<String, dynamic> json) => Surplus(
    batchId: (json['id'] ?? json['batch_id']) as String,
    batchCode: json['batch_code'] as String,
    foodName: json['food_name'] as String? ?? 'Unknown',
    quantityKg: (json['quantity_kg'] as num?)?.toDouble() ?? 0,
    quantityUnit: json['quantity_unit'] as String? ?? 'kg',
    status: SurplusStatusX.fromApi(json['status'] as String),
    safetyStatus: json['safety_status'] as String?,
    preparedAt: DateTime.parse(json['prepared_at'] as String),
    expiryAt: DateTime.parse(json['expiry_at'] as String),
    mealId: json['meal_id'] as String?,
    foodCategory: json['food_category'] as String?,
  );

  Duration get timeToExpiry => expiryAt.difference(DateTime.now());
  bool get isExpiringSoon => timeToExpiry.inHours < 4 && timeToExpiry.isNegative == false;
}

class QualityResult {
  final String batchId;
  final VisualBlock visual;
  final SafetyBlock safetyDecision;
  final bool requiresHumanApproval;

  const QualityResult({
    required this.batchId,
    required this.visual,
    required this.safetyDecision,
    required this.requiresHumanApproval,
  });

  factory QualityResult.fromJson(Map<String, dynamic> json) => QualityResult(
    batchId: json['batch_id'] as String,
    visual: VisualBlock.fromJson(json['visual'] as Map<String, dynamic>),
    safetyDecision: SafetyBlock.fromJson(json['safety_decision'] as Map<String, dynamic>),
    requiresHumanApproval: json['requires_human_approval'] as bool,
  );
}

class VisualBlock {
  final String status;
  final String riskLevel;
  final double confidence;
  final String reason;
  final String? heatmapUrl;
  final String modelVersion;

  const VisualBlock({
    required this.status,
    required this.riskLevel,
    required this.confidence,
    required this.reason,
    this.heatmapUrl,
    required this.modelVersion,
  });

  factory VisualBlock.fromJson(Map<String, dynamic> json) => VisualBlock(
    status: json['status'] as String,
    riskLevel: json['risk_level'] as String,
    confidence: (json['confidence'] as num).toDouble(),
    reason: json['reason'] as String? ?? '',
    heatmapUrl: json['heatmap_url'] as String?,
    modelVersion: json['model_version'] as String,
  );
}

class SafetyBlock {
  final String status;
  final List<String> reasons;
  final int dangerZoneMinutes;
  final double hoursToExpiry;
  final String modelVersion;

  const SafetyBlock({
    required this.status,
    required this.reasons,
    required this.dangerZoneMinutes,
    required this.hoursToExpiry,
    required this.modelVersion,
  });

  factory SafetyBlock.fromJson(Map<String, dynamic> json) => SafetyBlock(
    status: json['status'] as String,
    reasons: (json['reasons'] as List?)?.cast<String>() ?? [],
    dangerZoneMinutes: json['danger_zone_minutes'] as int? ?? 0,
    hoursToExpiry: (json['hours_to_expiry'] as num?)?.toDouble() ?? 0,
    modelVersion: json['model_version'] as String? ?? '',
  );
}

class MatchResult {
  final String recipientId;
  final String name;
  final String type;
  final double capacityKg;
  final double distanceKm;
  final String pickupWindow;
  final double score;
  final Map<String, double> scoreBreakdown;
  final List<String> reasons;

  const MatchResult({
    required this.recipientId,
    required this.name,
    required this.type,
    required this.capacityKg,
    required this.distanceKm,
    required this.pickupWindow,
    required this.score,
    required this.scoreBreakdown,
    required this.reasons,
  });

  factory MatchResult.fromJson(Map<String, dynamic> json) => MatchResult(
    recipientId: json['recipient_id'] as String,
    name: json['name'] as String,
    type: json['type'] as String? ?? 'NGO',
    capacityKg: (json['capacity_kg'] as num).toDouble(),
    distanceKm: (json['distance_km'] as num).toDouble(),
    pickupWindow: json['pickup_window'] as String? ?? '',
    score: (json['score'] as num).toDouble(),
    scoreBreakdown: (json['score_breakdown'] as Map<String, dynamic>?)
        ?.map((k, v) => MapEntry(k, (v as num).toDouble())) ?? {},
    reasons: (json['reasons'] as List?)?.cast<String>() ?? [],
  );
}

class RouteResult {
  final String routeId;
  final double distanceKm;
  final int etaMinutes;
  final String solver;
  final List<RouteStop> stops;

  const RouteResult({
    required this.routeId,
    required this.distanceKm,
    required this.etaMinutes,
    required this.solver,
    required this.stops,
  });

  factory RouteResult.fromJson(Map<String, dynamic> json) => RouteResult(
    routeId: json['route_id'] as String,
    distanceKm: (json['distance_km'] as num).toDouble(),
    etaMinutes: json['eta_minutes'] as int,
    solver: json['solver'] as String,
    stops: extractList<RouteStop>(json['stops'], RouteStop.fromJson),
  );
}

class RouteStop {
  final int sequence;
  final String recipientId;
  final String name;
  final double lat;
  final double lng;
  final String eta;
  final bool deadlineOk;

  const RouteStop({
    required this.sequence,
    required this.recipientId,
    required this.name,
    required this.lat,
    required this.lng,
    required this.eta,
    required this.deadlineOk,
  });

  factory RouteStop.fromJson(Map<String, dynamic> json) => RouteStop(
    sequence: json['sequence'] as int,
    recipientId: json['recipient_id'] as String,
    name: json['name'] as String,
    lat: (json['lat'] as num).toDouble(),
    lng: (json['lng'] as num).toDouble(),
    eta: json['eta'] as String,
    deadlineOk: json['deadline_ok'] as bool,
  );
}

class WasteEntry {
  final String? id;
  final double quantityKg;
  final String cause;
  final String mealType;
  final DateTime date;

  const WasteEntry({
    this.id,
    required this.quantityKg,
    required this.cause,
    required this.mealType,
    required this.date,
  });

  factory WasteEntry.fromJson(Map<String, dynamic> json) => WasteEntry(
    id: json['id'] as String?,
    quantityKg: (json['quantity_kg'] as num).toDouble(),
    cause: json['cause'] as String,
    mealType: json['meal_type'] as String,
    date: DateTime.parse(json['date'] as String),
  );
}

class SensorReading {
  final String sensorId;
  final String locationId;
  final double temperatureC;
  final double humidityPct;
  final double energyKwh;
  final String? batchId;
  final DateTime timestamp;

  const SensorReading({
    required this.sensorId,
    required this.locationId,
    required this.temperatureC,
    required this.humidityPct,
    required this.energyKwh,
    this.batchId,
    required this.timestamp,
  });

  factory SensorReading.fromJson(Map<String, dynamic> json) => SensorReading(
    sensorId: json['sensor_id'] as String,
    locationId: json['location_id'] as String,
    temperatureC: (json['temperature_c'] as num).toDouble(),
    humidityPct: (json['humidity_pct'] as num).toDouble(),
    energyKwh: (json['energy_kwh'] as num).toDouble(),
    batchId: json['batch_id'] as String?,
    timestamp: DateTime.parse(json['timestamp'] as String),
  );

  bool get isTempDangerZone => temperatureC >= 5 && temperatureC <= 60;
}

class Alert {
  final String id;
  final String type;
  final String severity;
  final String message;
  final bool acknowledged;
  final DateTime createdAt;

  const Alert({
    required this.id,
    required this.type,
    required this.severity,
    required this.message,
    required this.acknowledged,
    required this.createdAt,
  });

  factory Alert.fromJson(Map<String, dynamic> json) => Alert(
    id: json['id'] as String,
    type: json['type'] as String,
    severity: json['severity'] as String,
    message: json['message'] as String,
    acknowledged: json['acknowledged'] as bool? ?? false,
    createdAt: DateTime.parse(json['created_at'] as String),
  );
}

class ImpactData {
  final double wasteAvoidedKg;
  final double redistributedKg;
  final double divertedKg;
  final double estimatedCarbonSavedKg;
  final double carbonFactorUsed;
  final String carbonFactorSource;
  final int successfulRedistributions;
  final int mealsEquivalent;
  final List<ImpactTrend> trend;

  const ImpactData({
    required this.wasteAvoidedKg,
    required this.redistributedKg,
    required this.divertedKg,
    required this.estimatedCarbonSavedKg,
    required this.carbonFactorUsed,
    required this.carbonFactorSource,
    required this.successfulRedistributions,
    required this.mealsEquivalent,
    required this.trend,
  });

  factory ImpactData.fromJson(Map<String, dynamic> json) => ImpactData(
    wasteAvoidedKg: (json['waste_avoided_kg'] as num).toDouble(),
    redistributedKg: (json['redistributed_kg'] as num).toDouble(),
    divertedKg: (json['diverted_kg'] as num).toDouble(),
    estimatedCarbonSavedKg: (json['estimated_carbon_saved_kg'] as num).toDouble(),
    carbonFactorUsed: (json['carbon_factor_used'] as num).toDouble(),
    carbonFactorSource: json['carbon_factor_source'] as String,
    successfulRedistributions: json['successful_redistributions'] as int,
    mealsEquivalent: json['meals_equivalent'] as int,
    trend: (json['trend'] as List?)?.map((e) => ImpactTrend.fromJson(e)).toList() ?? [],
  );
}

class ImpactTrend {
  final String date;
  final double value;

  const ImpactTrend({required this.date, required this.value});

  factory ImpactTrend.fromJson(Map<String, dynamic> json) => ImpactTrend(
    date: json['date'] as String,
    value: (json['value'] as num).toDouble(),
  );
}

class ProcessingResult {
  final double materialLossPct;
  final double productionEfficiency;
  final double downtimePct;
  final double energyPerKg;
  final double efficiencyScore;
  final List<dynamic> anomalies;

  const ProcessingResult({
    required this.materialLossPct,
    required this.productionEfficiency,
    required this.downtimePct,
    required this.energyPerKg,
    required this.efficiencyScore,
    required this.anomalies,
  });

  factory ProcessingResult.fromJson(Map<String, dynamic> json) => ProcessingResult(
    materialLossPct: (json['material_loss_pct'] as num).toDouble(),
    productionEfficiency: (json['production_efficiency'] as num).toDouble(),
    downtimePct: (json['downtime_pct'] as num).toDouble(),
    energyPerKg: (json['energy_per_kg'] as num).toDouble(),
    efficiencyScore: (json['efficiency_score'] as num).toDouble(),
    anomalies: json['anomalies'] as List? ?? [],
  );
}

class KitchenOverview {
  final int expectedDiners;
  final double forecastP50;
  final double recommendedProduction;
  final String surplusRisk;
  final double wasteToday;
  final double availableKg;
  final double redistributedToday;

  const KitchenOverview({
    required this.expectedDiners,
    required this.forecastP50,
    required this.recommendedProduction,
    required this.surplusRisk,
    required this.wasteToday,
    required this.availableKg,
    required this.redistributedToday,
  });

  factory KitchenOverview.fromJson(Map<String, dynamic> json) => KitchenOverview(
    expectedDiners: json['expected_diners'] as int? ?? 0,
    forecastP50: (json['forecast_p50'] as num?)?.toDouble() ?? 0,
    recommendedProduction: (json['recommended_production'] as num?)?.toDouble() ?? 0,
    surplusRisk: json['surplus_risk'] as String? ?? 'LOW',
    wasteToday: (json['waste_today'] as num?)?.toDouble() ?? 0,
    availableKg: (json['available_kg'] as num?)?.toDouble() ?? 0,
    redistributedToday: (json['redistributed_today'] as num?)?.toDouble() ?? 0,
  );
}

class QrTimeline {
  final String batchId;
  final String batchCode;
  final bool valid;
  final int length;
  final int? brokenAt;
  final List<QrEvent> events;

  const QrTimeline({
    required this.batchId,
    required this.batchCode,
    required this.valid,
    required this.length,
    this.brokenAt,
    required this.events,
  });

  factory QrTimeline.fromJson(Map<String, dynamic> json) => QrTimeline(
    batchId: json['batch_id'] as String? ?? '',
    batchCode: json['batch_code'] as String? ?? '',
    valid: json['valid'] as bool? ?? true,
    length: json['length'] as int? ?? 0,
    brokenAt: json['broken_at'] as int?,
    events: (json['events'] as List?)?.map((e) => QrEvent.fromJson(e)).toList() ?? [],
  );
}

class QrEvent {
  final String eventType;
  final String timestamp;
  final String hash;
  final String prevHash;
  final String? clientTs;

  const QrEvent({
    required this.eventType,
    required this.timestamp,
    required this.hash,
    required this.prevHash,
    this.clientTs,
  });

  factory QrEvent.fromJson(Map<String, dynamic> json) => QrEvent(
    eventType: json['event_type'] as String,
    timestamp: json['timestamp'] as String,
    hash: json['hash'] as String? ?? '',
    prevHash: json['prev_hash'] as String? ?? '',
    clientTs: json['client_ts'] as String?,
  );
}
