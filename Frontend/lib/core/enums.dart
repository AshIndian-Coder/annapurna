enum UserRole { kitchen, ngo, logistics, admin, system }

enum MealType { breakfast, lunch, snack, dinner }

enum WasteCause {
  overproduction,
  lowAttendance,
  spoilage,
  expiry,
  storageIssue,
  preparationError,
  other,
}

enum SurplusStatus {
  pendingSafety,
  available,
  hold,
  matched,
  inTransit,
  delivered,
  expired,
  diverted,
}

enum VisualStatus { good, risk, rejected }

enum SafetyDecision { eligible, hold, rejected }

enum RiskLevel { low, medium, high }

enum QrEventType { created, approved, matched, pickedUp, handedOff, received }

enum DiversionType { animalFeed, compost, biogas }

enum AlertType {
  tempExcursion,
  humidityHigh,
  energySpike,
  downtimeHigh,
  materialLossHigh,
  expirySoon,
  sensorStale,
}

enum Severity { info, warn, critical }

enum SyncItemStatus { accepted, duplicate, rejected }

extension UserRoleX on UserRole {
  String get apiValue => switch (this) {
    UserRole.kitchen => 'KITCHEN',
    UserRole.ngo => 'NGO',
    UserRole.logistics => 'LOGISTICS',
    UserRole.admin => 'ADMIN',
    UserRole.system => 'SYSTEM',
  };

  static UserRole fromApi(String value) => switch (value.toUpperCase()) {
    'KITCHEN' => UserRole.kitchen,
    'NGO' => UserRole.ngo,
    'LOGISTICS' => UserRole.logistics,
    'ADMIN' => UserRole.admin,
    _ => UserRole.system,
  };
}

extension MealTypeX on MealType {
  String get apiValue => name.toUpperCase();
  String get label => switch (this) {
    MealType.breakfast => 'Breakfast',
    MealType.lunch => 'Lunch',
    MealType.snack => 'Snack',
    MealType.dinner => 'Dinner',
  };
}

extension SurplusStatusX on SurplusStatus {
  String get apiValue => switch (this) {
    SurplusStatus.pendingSafety => 'PENDING_SAFETY',
    SurplusStatus.available => 'AVAILABLE',
    SurplusStatus.hold => 'HOLD',
    SurplusStatus.matched => 'MATCHED',
    SurplusStatus.inTransit => 'IN_TRANSIT',
    SurplusStatus.delivered => 'DELIVERED',
    SurplusStatus.expired => 'EXPIRED',
    SurplusStatus.diverted => 'DIVERTED',
  };

  String get label => switch (this) {
    SurplusStatus.pendingSafety => 'Pending Safety',
    SurplusStatus.available => 'Available',
    SurplusStatus.hold => 'On Hold',
    SurplusStatus.matched => 'Matched',
    SurplusStatus.inTransit => 'In Transit',
    SurplusStatus.delivered => 'Delivered',
    SurplusStatus.expired => 'Expired',
    SurplusStatus.diverted => 'Diverted',
  };

  static SurplusStatus fromApi(String value) => switch (value) {
    'PENDING_SAFETY' => SurplusStatus.pendingSafety,
    'AVAILABLE' => SurplusStatus.available,
    'HOLD' => SurplusStatus.hold,
    'MATCHED' => SurplusStatus.matched,
    'IN_TRANSIT' => SurplusStatus.inTransit,
    'DELIVERED' => SurplusStatus.delivered,
    'EXPIRED' => SurplusStatus.expired,
    'DIVERTED' => SurplusStatus.diverted,
    _ => SurplusStatus.pendingSafety,
  };
}

extension RiskLevelX on RiskLevel {
  String get apiValue => name.toUpperCase();
  static RiskLevel fromApi(String value) => switch (value.toUpperCase()) {
    'LOW' => RiskLevel.low,
    'MEDIUM' => RiskLevel.medium,
    'HIGH' => RiskLevel.high,
    _ => RiskLevel.low,
  };
}

extension SeverityX on Severity {
  String get apiValue => name.toUpperCase();
  static Severity fromApi(String value) => switch (value.toUpperCase()) {
    'INFO' => Severity.info,
    'WARN' => Severity.warn,
    'CRITICAL' => Severity.critical,
    _ => Severity.info,
  };
}

extension QrEventTypeX on QrEventType {
  String get apiValue => switch (this) {
    QrEventType.created => 'CREATED',
    QrEventType.approved => 'APPROVED',
    QrEventType.matched => 'MATCHED',
    QrEventType.pickedUp => 'PICKED_UP',
    QrEventType.handedOff => 'HANDED_OFF',
    QrEventType.received => 'RECEIVED',
  };

  String get label => switch (this) {
    QrEventType.created => 'Created',
    QrEventType.approved => 'Approved',
    QrEventType.matched => 'Matched',
    QrEventType.pickedUp => 'Picked Up',
    QrEventType.handedOff => 'Handed Off',
    QrEventType.received => 'Received',
  };
}

extension WasteCauseX on WasteCause {
  String get apiValue => switch (this) {
    WasteCause.overproduction => 'OVERPRODUCTION',
    WasteCause.lowAttendance => 'LOW_ATTENDANCE',
    WasteCause.spoilage => 'SPOILAGE',
    WasteCause.expiry => 'EXPIRY',
    WasteCause.storageIssue => 'STORAGE_ISSUE',
    WasteCause.preparationError => 'PREPARATION_ERROR',
    WasteCause.other => 'OTHER',
  };

  String get label => switch (this) {
    WasteCause.overproduction => 'Overproduction',
    WasteCause.lowAttendance => 'Low Attendance',
    WasteCause.spoilage => 'Spoilage',
    WasteCause.expiry => 'Expiry',
    WasteCause.storageIssue => 'Storage Issue',
    WasteCause.preparationError => 'Preparation Error',
    WasteCause.other => 'Other',
  };
}
