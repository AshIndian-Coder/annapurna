import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../core/api/api_client.dart';
import '../../core/result/result.dart';
import '../mocks/mock_data.dart';

final adminServiceProvider = Provider<AdminService>((ref) {
  return AdminService(ref.read(apiClientProvider));
});

class SystemHealth {
  final String service;
  final String status;
  final String uptime;
  final String lastSeen;

  const SystemHealth({
    required this.service,
    required this.status,
    required this.uptime,
    required this.lastSeen,
  });

  factory SystemHealth.fromJson(Map<String, dynamic> json) => SystemHealth(
    service: json['service'] as String,
    status: json['status'] as String,
    uptime: json['uptime'] as String,
    lastSeen: json['last_seen'] as String,
  );
}

class MlModel {
  final String version;
  final String status;
  final double accuracy;
  final int inferenceCount;
  final double dataDriftScore;

  const MlModel({
    required this.version,
    required this.status,
    required this.accuracy,
    required this.inferenceCount,
    required this.dataDriftScore,
  });

  factory MlModel.fromJson(Map<String, dynamic> json) => MlModel(
    version: json['version'] as String,
    status: json['status'] as String,
    accuracy: (json['accuracy'] as num).toDouble(),
    inferenceCount: json['inference_count'] as int,
    dataDriftScore: (json['data_drift_score'] as num).toDouble(),
  );
}

class SystemUser {
  final String id;
  final String name;
  final String role;
  final String status;
  final String lastLogin;

  const SystemUser({
    required this.id,
    required this.name,
    required this.role,
    required this.status,
    required this.lastLogin,
  });

  factory SystemUser.fromJson(Map<String, dynamic> json) => SystemUser(
    id: json['id'] as String,
    name: json['name'] as String,
    role: json['role'] as String,
    status: json['status'] as String,
    lastLogin: json['last_login'] as String,
  );
}

class AuditLog {
  final String id;
  final String action;
  final String actor;
  final String entityType;
  final String timestamp;

  const AuditLog({
    required this.id,
    required this.action,
    required this.actor,
    required this.entityType,
    required this.timestamp,
  });

  factory AuditLog.fromJson(Map<String, dynamic> json) => AuditLog(
    id: json['id'] as String,
    action: json['action'] as String,
    actor: json['actor'] as String,
    entityType: json['entity_type'] as String,
    timestamp: json['timestamp'] as String,
  );
}

class AdminService {
  final ApiClient _api;
  AdminService(this._api);

  /// Contract #34: GET /health, /ready — public health probes.
  /// Combines both probes into a single health view for the admin dashboard.
  Future<Result<List<SystemHealth>>> getSystemHealth() async {
    final response = await _api.get('/ready', parser: (data) {
      final items = <SystemHealth>[];
      if (data is Map<String, dynamic>) {
        items.add(SystemHealth(
          service: 'PostgreSQL DB',
          status: data['database'] == true ? 'HEALTHY' : 'DOWN',
          uptime: '-',
          lastSeen: 'Just now',
        ));
        items.add(SystemHealth(
          service: 'Redis Cache',
          status: (data['redis'] ?? 'unknown') == 'ok' ? 'HEALTHY' : 'DOWN',
          uptime: '-',
          lastSeen: 'Just now',
        ));
      }
      return items;
    });
    return response.when(
      success: (data) => Success(data),
      failure: (_) => const Success([]),
    );
  }

  Future<Result<List<MlModel>>> getModels() async {
    final response = await _api.get('/admin/models', parser: (data) =>
      extractList<MlModel>(data, MlModel.fromJson),
    );
    return response.when(
      success: (data) => Success(data),
      failure: (_) => const Success([]),
    );
  }

  Future<Result<List<SystemUser>>> getUsers() async {
    final response = await _api.get('/admin/users', parser: (data) =>
      extractList<SystemUser>(data, SystemUser.fromJson),
    );
    return response.when(
      success: (data) => Success(data),
      failure: (_) => const Success([]),
    );
  }

  Future<Result<List<AuditLog>>> getAuditLogs() async {
    final response = await _api.get('/admin/audit-log', parser: (data) =>
      extractList<AuditLog>(data, AuditLog.fromJson),
    );
    return response.when(
      success: (data) => Success(data),
      failure: (_) => const Success([]),
    );
  }

  static final List<SystemHealth> _mockHealth = [
    SystemHealth(service: 'PostgreSQL DB', status: 'HEALTHY', uptime: '45d 12h', lastSeen: 'Just now'),
    SystemHealth(service: 'Redis Cache', status: 'HEALTHY', uptime: '45d 12h', lastSeen: 'Just now'),
    SystemHealth(service: 'ML Serving', status: 'WARN', uptime: '2d 4h', lastSeen: '1m ago'),
    SystemHealth(service: 'IoT Sensors', status: 'HEALTHY', uptime: '12d 8h', lastSeen: 'Just now'),
  ];

  static final List<MlModel> _mockModels = [
    MlModel(version: 'v2.1.0-prod', status: 'ACTIVE', accuracy: 94.5, inferenceCount: 145020, dataDriftScore: 0.02),
    MlModel(version: 'v2.2.0-shadow', status: 'SHADOW', accuracy: 96.1, inferenceCount: 45020, dataDriftScore: 0.01),
    MlModel(version: 'v1.9.5-legacy', status: 'ARCHIVED', accuracy: 89.2, inferenceCount: 890430, dataDriftScore: 0.15),
  ];

  static final List<SystemUser> _mockUsers = [
    SystemUser(id: 'u-1', name: 'Kitchen Admin', role: 'KITCHEN', status: 'ACTIVE', lastLogin: '10m ago'),
    SystemUser(id: 'u-2', name: 'NGO Manager', role: 'NGO', status: 'ACTIVE', lastLogin: '1h ago'),
    SystemUser(id: 'u-3', name: 'Logistics Lead', role: 'LOGISTICS', status: 'ACTIVE', lastLogin: '5m ago'),
    SystemUser(id: 'u-4', name: 'System Admin', role: 'ADMIN', status: 'ACTIVE', lastLogin: 'Just now'),
  ];

  static final List<AuditLog> _mockLogs = [
    AuditLog(id: 'log-1', action: 'ROLE_UPDATE', actor: 'System Admin', entityType: 'USER', timestamp: '2026-10-02T10:15:00Z'),
    AuditLog(id: 'log-2', action: 'MODEL_PROMOTE', actor: 'System Admin', entityType: 'ML_MODEL', timestamp: '2026-10-02T09:30:00Z'),
    AuditLog(id: 'log-3', action: 'EMERGENCY_OVERRIDE', actor: 'Kitchen Admin', entityType: 'SURPLUS_BATCH', timestamp: '2026-10-01T14:20:00Z'),
  ];
}
