import '../dtos/models.dart';
import '../../core/enums.dart';

class MockData {
  MockData._();

  static final User kitchenUser = User(
    id: 'u-kitchen-001',
    name: 'Rajesh Kumar',
    email: 'rajesh@annapurna.in',
    role: UserRole.kitchen,
  );

  static final User ngoUser = User(
    id: 'u-ngo-001',
    name: 'Priya Sharma',
    email: 'priya@helpinghands.org',
    role: UserRole.ngo,
  );

  static final User driverUser = User(
    id: 'u-driver-001',
    name: 'Amit Singh',
    email: 'amit@logistics.in',
    role: UserRole.logistics,
  );

  static final User adminUser = User(
    id: 'u-admin-001',
    name: 'Sunita Verma',
    email: 'sunita@annapurna.in',
    role: UserRole.admin,
  );

  static AuthTokens loginAs(UserRole role) {
    final user = switch (role) {
      UserRole.kitchen => kitchenUser,
      UserRole.ngo => ngoUser,
      UserRole.logistics => driverUser,
      UserRole.admin => adminUser,
      _ => kitchenUser,
    };
    return AuthTokens(
      accessToken: 'mock_access_${role.name}',
      refreshToken: 'mock_refresh_${role.name}',
      expiresIn: 3600,
      user: user,
    );
  }

  static final KitchenOverview overview = KitchenOverview(
    expectedDiners: 450,
    forecastP50: 420,
    recommendedProduction: 435,
    surplusRisk: 'LOW',
    wasteToday: 12.5,
    availableKg: 28.0,
    redistributedToday: 45.0,
  );

  static final Prediction prediction = Prediction(
    predictedConsumption: 420,
    recommendedProduction: 435,
    expectedSurplus: 15,
    surplusRisk: 'LOW',
    interval: const PredictionInterval(p10: 401, p50: 420, p90: 441, coverageTarget: 0.8),
    recommendedQuantile: 0.7,
    topDrivers: const [
      Driver(feature: 'attendance', effectKg: 38.2),
      Driver(feature: 'weekday', effectKg: -6.1),
      Driver(feature: 'menu_complexity', effectKg: 12.4),
      Driver(feature: 'temperature', effectKg: -3.8),
    ],
    confidence: 0.87,
    modelVersion: 'demand-v1',
    dataSource: 'SYNTHETIC',
  );

  static final List<Surplus> surplusList = [
    Surplus(
      batchId: 'b-001',
      batchCode: 'B-10291',
      foodName: 'Dal Makhani',
      quantityKg: 15.0,
      quantityUnit: 'kg',
      status: SurplusStatus.available,
      safetyStatus: 'ELIGIBLE',
      preparedAt: DateTime.now().subtract(const Duration(hours: 3)),
      expiryAt: DateTime.now().add(const Duration(hours: 5)),
      foodCategory: 'Lentils',
    ),
    Surplus(
      batchId: 'b-002',
      batchCode: 'B-10292',
      foodName: 'Paneer Butter Masala',
      quantityKg: 8.5,
      quantityUnit: 'kg',
      status: SurplusStatus.pendingSafety,
      safetyStatus: 'PENDING',
      preparedAt: DateTime.now().subtract(const Duration(hours: 2)),
      expiryAt: DateTime.now().add(const Duration(hours: 6)),
      foodCategory: 'Dairy',
    ),
    Surplus(
      batchId: 'b-003',
      batchCode: 'B-10293',
      foodName: 'Steamed Rice',
      quantityKg: 22.0,
      quantityUnit: 'kg',
      status: SurplusStatus.matched,
      safetyStatus: 'ELIGIBLE',
      preparedAt: DateTime.now().subtract(const Duration(hours: 4)),
      expiryAt: DateTime.now().add(const Duration(hours: 4)),
      foodCategory: 'Grains',
    ),
    Surplus(
      batchId: 'b-004',
      batchCode: 'B-10294',
      foodName: 'Aloo Gobi',
      quantityKg: 10.0,
      quantityUnit: 'kg',
      status: SurplusStatus.inTransit,
      safetyStatus: 'ELIGIBLE',
      preparedAt: DateTime.now().subtract(const Duration(hours: 5)),
      expiryAt: DateTime.now().add(const Duration(hours: 3)),
      foodCategory: 'Vegetables',
    ),
    Surplus(
      batchId: 'b-005',
      batchCode: 'B-10295',
      foodName: 'Chole Bhature',
      quantityKg: 5.0,
      quantityUnit: 'kg',
      status: SurplusStatus.delivered,
      safetyStatus: 'ELIGIBLE',
      preparedAt: DateTime.now().subtract(const Duration(hours: 6)),
      expiryAt: DateTime.now().add(const Duration(hours: 2)),
      foodCategory: 'Lentils',
    ),
  ];

  static final QualityResult qualityResult = QualityResult(
    batchId: 'b-002',
    visual: const VisualBlock(
      status: 'RISK',
      riskLevel: 'MEDIUM',
      confidence: 0.84,
      reason: 'Visible discoloration detected',
      modelVersion: 'cv-v1',
    ),
    safetyDecision: const SafetyBlock(
      status: 'HOLD',
      reasons: ['CV_RISK', 'DANGER_MINUTES_42'],
      dangerZoneMinutes: 42,
      hoursToExpiry: 2.5,
      modelVersion: 'fusion-v1',
    ),
    requiresHumanApproval: true,
  );

  static final List<MatchResult> matchResults = [
    const MatchResult(
      recipientId: 'r-001',
      name: 'Helping Hands Foundation',
      type: 'NGO',
      capacityKg: 50,
      distanceKm: 3.2,
      pickupWindow: '14:00-16:00',
      score: 0.92,
      scoreBreakdown: {'need': 0.95, 'capacity': 0.90, 'distance': 0.88, 'fairness': 0.94},
      reasons: ['High need area', 'Within pickup window', 'Good capacity match'],
    ),
    const MatchResult(
      recipientId: 'r-002',
      name: 'Annadaata Seva',
      type: 'NGO',
      capacityKg: 30,
      distanceKm: 5.7,
      pickupWindow: '13:00-17:00',
      score: 0.85,
      scoreBreakdown: {'need': 0.88, 'capacity': 0.75, 'distance': 0.72, 'fairness': 0.98},
      reasons: ['High fairness score', 'Flexible window'],
    ),
    const MatchResult(
      recipientId: 'r-003',
      name: 'Roti Bank Delhi',
      type: 'NGO',
      capacityKg: 100,
      distanceKm: 8.1,
      pickupWindow: '12:00-18:00',
      score: 0.78,
      scoreBreakdown: {'need': 0.82, 'capacity': 0.95, 'distance': 0.55, 'fairness': 0.80},
      reasons: ['Very high capacity', 'Longer distance'],
    ),
  ];

  static final RouteResult routeResult = RouteResult(
    routeId: 'route-001',
    distanceKm: 12.4,
    etaMinutes: 38,
    solver: 'ortools-vrptw',
    stops: const [
      RouteStop(sequence: 1, recipientId: 'r-001', name: 'Helping Hands Foundation', lat: 28.6139, lng: 77.2090, eta: '14:15', deadlineOk: true),
      RouteStop(sequence: 2, recipientId: 'r-002', name: 'Annadaata Seva', lat: 28.6353, lng: 77.2250, eta: '14:42', deadlineOk: true),
      RouteStop(sequence: 3, recipientId: 'r-003', name: 'Roti Bank Delhi', lat: 28.6508, lng: 77.2373, eta: '15:08', deadlineOk: true),
    ],
  );

  static final List<WasteEntry> wasteEntries = [
    WasteEntry(id: 'w-001', quantityKg: 5.2, cause: 'OVERPRODUCTION', mealType: 'LUNCH', date: DateTime.now()),
    WasteEntry(id: 'w-002', quantityKg: 3.1, cause: 'LOW_ATTENDANCE', mealType: 'DINNER', date: DateTime.now().subtract(const Duration(days: 1))),
    WasteEntry(id: 'w-003', quantityKg: 1.8, cause: 'SPOILAGE', mealType: 'BREAKFAST', date: DateTime.now().subtract(const Duration(days: 1))),
    WasteEntry(id: 'w-004', quantityKg: 4.5, cause: 'OVERPRODUCTION', mealType: 'LUNCH', date: DateTime.now().subtract(const Duration(days: 2))),
  ];

  static final List<SensorReading> sensorReadings = [
    SensorReading(sensorId: 's-001', locationId: 'cold-store-1', temperatureC: 4.2, humidityPct: 65, energyKwh: 2.1, timestamp: DateTime.now()),
    SensorReading(sensorId: 's-002', locationId: 'kitchen-main', temperatureC: 24.5, humidityPct: 55, energyKwh: 5.8, timestamp: DateTime.now()),
    SensorReading(sensorId: 's-003', locationId: 'prep-area', temperatureC: 22.0, humidityPct: 48, energyKwh: 1.2, timestamp: DateTime.now()),
  ];

  static final List<Alert> alerts = [
    Alert(id: 'a-001', type: 'TEMP_EXCURSION', severity: 'CRITICAL', message: 'Cold store temperature exceeded 8°C for 15 minutes', acknowledged: false, createdAt: DateTime.now().subtract(const Duration(minutes: 5))),
    Alert(id: 'a-002', type: 'EXPIRY_SOON', severity: 'WARN', message: 'Batch B-10291 expires in 2 hours', acknowledged: false, createdAt: DateTime.now().subtract(const Duration(minutes: 30))),
    Alert(id: 'a-003', type: 'ENERGY_SPIKE', severity: 'INFO', message: 'Energy consumption 20% above average', acknowledged: true, createdAt: DateTime.now().subtract(const Duration(hours: 2))),
  ];

  static final ImpactData impact = ImpactData(
    wasteAvoidedKg: 1250,
    redistributedKg: 980,
    divertedKg: 270,
    estimatedCarbonSavedKg: 3120,
    carbonFactorUsed: 2.5,
    carbonFactorSource: 'WRAP UK Food Waste Reduction Roadmap (estimate)',
    successfulRedistributions: 142,
    mealsEquivalent: 3920,
    trend: List.generate(14, (i) => ImpactTrend(
      date: DateTime.now().subtract(Duration(days: 13 - i)).toIso8601String().substring(0, 10),
      value: 60 + (i * 5.2) + (i % 3 == 0 ? 15 : 0),
    )),
  );

  static const ProcessingResult processing = ProcessingResult(
    materialLossPct: 8.0,
    productionEfficiency: 0.92,
    downtimePct: 8.33,
    energyPerKg: 0.272,
    efficiencyScore: 77.9,
    anomalies: [],
  );
}
