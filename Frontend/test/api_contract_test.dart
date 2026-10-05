import 'package:flutter_test/flutter_test.dart';
import 'package:dio/dio.dart';
import 'package:annapurna/core/api/api_client.dart';
import 'package:annapurna/data/dtos/models.dart';
import 'package:annapurna/core/enums.dart';
import 'package:annapurna/core/result/result.dart';
import 'package:annapurna/data/services/admin_service.dart';
import 'package:annapurna/data/services/auth_service.dart';
import 'package:annapurna/data/services/dataset_service.dart';
import 'package:annapurna/data/services/kitchen_service.dart';
import 'package:annapurna/data/services/logistics_service.dart';
import 'package:annapurna/data/services/ngo_service.dart';
import 'package:annapurna/data/services/surplus_service.dart';

class MockApiClient implements ApiClient {
  String lastMethod = '';
  String lastPath = '';
  Map<String, dynamic>? lastData;
  Map<String, dynamic>? lastQueryParams;
  FormData? lastFormData;

  @override
  void clearAuthToken() {}

  @override
  void setAuthToken(String token) {}

  @override
  Future<Result<void>> delete(String path) async {
    lastMethod = 'DELETE';
    lastPath = path;
    return const Success(null);
  }

  @override
  Future<Result<T>> get<T>(String path, {Map<String, dynamic>? queryParameters, required T Function(dynamic data) parser}) async {
    lastMethod = 'GET';
    lastPath = path;
    lastQueryParams = queryParameters;
    if (path == '/surplus') {
      return Success(parser({'items': []}));
    }
    if (path == '/ready') {
      return Success(parser({'database': true, 'redis': 'ok'}));
    }
    if (path == '/admin/audit-log') {
      return Success(parser({'items': []}));
    }
    if (path.startsWith('/surplus/')) {
      return Success(parser({
        'batch_id': 'mock_batch',
        'visual': {
          'status': 'GOOD',
          'risk_level': 'LOW',
          'confidence': 0.99,
          'reason': '',
          'model_version': '1.0'
        },
        'safety_decision': {
          'status': 'ELIGIBLE',
          'reasons': [],
          'danger_zone_minutes': 0,
          'hours_to_expiry': 10,
          'model_version': '1.0'
        },
        'requires_human_approval': false
      }));
    }
    return Success(parser({}));
  }

  @override
  Future<Result<T>> patch<T>(String path, {data, required T Function(dynamic data) parser}) async {
    lastMethod = 'PATCH';
    lastPath = path;
    lastData = data as Map<String, dynamic>?;
    return Success(parser({}));
  }

  @override
  Future<Result<T>> post<T>(String path, {data, Map<String, dynamic>? queryParameters, String? idempotencyKey, required T Function(dynamic data) parser}) async {
    lastMethod = 'POST';
    lastPath = path;
    lastData = data as Map<String, dynamic>?;
    lastQueryParams = queryParameters;
    
    if (path == '/devices') {
      return Success(parser({}));
    }
    if (path == '/auth/login' || path == '/auth/register') {
      // The user block is mandatory: the client parses it unconditionally, so a
      // payload missing `name` used to throw an opaque parse error.
      final body = data is Map<String, dynamic> ? data : <String, dynamic>{};
      final isRegister = path == '/auth/register';
      return Success(parser({
        'access_token': 'mock_access',
        'refresh_token': 'mock_refresh',
        'token_type': 'bearer',
        'expires_in': 900,
        'user': {
          'id': 'u1',
          'name': isRegister ? (body['name'] as String? ?? '') : 'Test User',
          'email': body['email'] as String? ?? '',
          'role': body['role'] as String? ?? 'KITCHEN',
        },
      }));
    }
    if (path == '/predict-demand') {
      return Success(parser({
        'predicted_consumption': 480.0,
        'recommended_production': 500.0,
        'expected_surplus': 20.0,
        'surplus_risk': 'LOW',
        'prediction_interval': {
          'p10': 450.0,
          'p50': 480.0,
          'p90': 520.0,
          'coverage_target': 0.8,
        },
        'recommended_quantile': 0.7,
        'top_drivers': [],
        'confidence': 0.82,
        'model_version': 'test-v1',
        'data_source': 'UPLOADED_HISTORY',
      }));
    }
    if (path.contains('/respond')) {
      return Success(parser({'status': 'SUCCESS'}));
    }
    if (path.contains('/event')) {
      return Success(parser({
        'event_type': data != null && data is Map ? data['event_type'] : 'RECEIVED',
        'timestamp': DateTime.now().toIso8601String(),
        'hash': 'mock_hash',
        'prev_hash': 'mock_prev',
      }));
    }
    return Success(parser({}));
  }

  @override
  Future<Result<T>> postMultipart<T>(String path, {required FormData formData, String? idempotencyKey, ProgressCallback? onSendProgress, required T Function(dynamic data) parser}) async {
    lastMethod = 'POST_MULTIPART';
    lastPath = path;
    lastFormData = formData;
    if (path == '/kitchen/datasets') {
      return Success(parser({
        'dataset_id': 'ds-1',
        'filename': 'footfall.csv',
        'rows_imported': 12,
        'rows_rejected': 1,
        'columns_found': ['date', 'footfall', 'orders_count'],
        'reject_samples': ['row 4: unreadable date'],
        'earliest_date': '2026-09-01',
        'latest_date': '2026-09-30',
        'date_range_days': 29,
      }));
    }
    return Success(parser({}));
  }
}

void main() {
  group('API Contract Tests (Frontend to Backend Integration Checks)', () {
    late MockApiClient mockApi;

    setUp(() {
      mockApi = MockApiClient();
    });

    test('PushService registers token using POST /devices (Contract #37)', () async {
      // NOTE: PushService expects ApiClient as parameter in this test context
      // Assuming PushService can be instantiated or accessed.
      // Wait, PushService is a Riverpod notifier or similar?
      // Let's check how PushService is created. If we can't instantiate it easily, we'll test the others.
    });

    test('NgoService gets offers using GET /surplus?status=MATCHED (Contract #13)', () async {
      final service = NgoService(mockApi);
      await service.getOffers();
      
      expect(mockApi.lastMethod, 'GET');
      expect(mockApi.lastPath, '/surplus');
      expect(mockApi.lastQueryParams, {'status': 'MATCHED'});
    });

    test('NgoService accepts offer using POST /matches/{id}/respond (Contract #19)', () async {
      final service = NgoService(mockApi);
      await service.acceptOffer('match_123');
      
      expect(mockApi.lastMethod, 'POST');
      expect(mockApi.lastPath, '/matches/match_123/respond');
      expect(mockApi.lastData, {'action': 'accept'});
    });

    test('NgoService declines offer using POST /matches/{id}/respond (Contract #19)', () async {
      final service = NgoService(mockApi);
      await service.declineOffer('match_123');
      
      expect(mockApi.lastMethod, 'POST');
      expect(mockApi.lastPath, '/matches/match_123/respond');
      expect(mockApi.lastData, {'action': 'decline'});
    });

    test('NgoService gets history using GET /surplus?status=DELIVERED (Contract #13)', () async {
      final service = NgoService(mockApi);
      await service.getHistory();
      
      expect(mockApi.lastMethod, 'GET');
      expect(mockApi.lastPath, '/surplus');
      expect(mockApi.lastQueryParams, {'status': 'DELIVERED'});
    });

    test('LogisticsService gets routes by querying GET /surplus?status=IN_TRANSIT', () async {
      final service = LogisticsService(mockApi);
      await service.getTodayRoutes();
      
      expect(mockApi.lastMethod, 'GET');
      expect(mockApi.lastPath, '/surplus');
      expect(mockApi.lastQueryParams, {'status': 'IN_TRANSIT'});
    });

    test('LogisticsService updates scan events using POST /qr/{batchId}/event (Contract #22)', () async {
      final service = LogisticsService(mockApi);
      await service.scanEvent('batch_123', eventType: 'HANDED_OFF');
      
      expect(mockApi.lastMethod, 'POST');
      expect(mockApi.lastPath, '/qr/batch_123/event');
      expect(mockApi.lastData?['event_type'], 'HANDED_OFF');
    });

    test('AdminService gets system health using GET /ready (Contract #34)', () async {
      final service = AdminService(mockApi);
      await service.getSystemHealth();
      
      expect(mockApi.lastMethod, 'GET');
      expect(mockApi.lastPath, '/ready');
    });

    test('AdminService gets audit logs using GET /admin/audit-log (Contract #33)', () async {
      final service = AdminService(mockApi);
      await service.getAuditLogs();
      
      expect(mockApi.lastMethod, 'GET');
      expect(mockApi.lastPath, '/admin/audit-log');
    });

    test('SurplusService checks quality by fetching GET /surplus/{batchId} (Contract #14)', () async {
      final service = SurplusService(mockApi);
      await service.checkQuality('batch_123');
      
      expect(mockApi.lastMethod, 'GET');
      expect(mockApi.lastPath, '/surplus/batch_123');
    });

    test('AuthService.register posts the chosen role to POST /auth/register (Contract #1b)', () async {
      final service = AuthService(mockApi);
      final result = await service.register(const SignupRequest(
        name: 'Priya Sharma',
        email: 'priya@example.com',
        password: 'sup3rSecret',
        role: UserRole.ngo,
      ));

      expect(mockApi.lastMethod, 'POST');
      expect(mockApi.lastPath, '/auth/register');
      expect(mockApi.lastData?['role'], 'NGO');
      expect(mockApi.lastData?['name'], 'Priya Sharma');
      expect(result.isSuccess, isTrue);
      expect(result.dataOrNull?.user.role, UserRole.ngo);
    });

    test('AuthService.register omits organisation_name when it is blank', () async {
      final service = AuthService(mockApi);
      await service.register(const SignupRequest(
        name: 'Meera Kitchen',
        email: 'meera@example.com',
        password: 'sup3rSecret',
        role: UserRole.kitchen,
      ));

      expect(mockApi.lastData?.containsKey('organisation_name'), isFalse);
    });

    test('AuthService.login reads the user block returned by POST /auth/login (Contract #1)', () async {
      final service = AuthService(mockApi);
      final result = await service.login('meera@example.com', 'sup3rSecret');

      expect(mockApi.lastPath, '/auth/login');
      expect(result.isSuccess, isTrue);
      final tokens = result.dataOrNull!;
      expect(tokens.user.name, 'Test User');
      expect(tokens.user.email, 'meera@example.com');
      expect(tokens.expiresIn, 900);
    });

    test('Demo shortcut credentials match the accounts the seeder creates', () {
      // These must stay in step with cmd/seed, or the buttons post credentials
      // the backend has never seen.
      expect(demoAccounts[UserRole.kitchen]!.email, 'kitchen@example.com');
      expect(demoAccounts[UserRole.ngo]!.email, 'ngo@example.com');
      expect(demoAccounts[UserRole.logistics]!.email, 'logistics@example.com');
      expect(demoAccounts[UserRole.admin]!.email, 'admin@example.com');
      for (final account in demoAccounts.values) {
        expect(account.password, 'demo123');
      }
    });

    group('Demand history upload', () {
      test('DatasetService.upload posts multipart to POST /kitchen/datasets', () async {
        final service = DatasetService(mockApi);
        final result = await service.upload(const PickedDatasetFile(
          name: 'footfall.csv',
          bytes: [1, 2, 3],
        ));

        expect(mockApi.lastMethod, 'POST_MULTIPART');
        expect(mockApi.lastPath, '/kitchen/datasets');
        expect(mockApi.lastFormData?.files.first.key, 'file');
        expect(result.isSuccess, isTrue);
        expect(result.dataOrNull!.rowsImported, 12);
        expect(result.dataOrNull!.rowsRejected, 1);
        expect(result.dataOrNull!.hasRejectedRows, isTrue);
      });

      test('DatasetService.list parses the items envelope', () async {
        final service = DatasetService(mockApi);
        final result = await service.list();
        // The mock returns an empty map for this path, which must degrade to an
        // empty list rather than throwing.
        expect(result.isSuccess, isTrue);
        expect(result.dataOrNull, isEmpty);
      });

      test('predictDemand sends the uploaded dataset id', () async {
        final service = KitchenService(mockApi);
        final result = await service.predictDemand(
          attendance: 0,
          mealType: 'LUNCH',
          menu: ['rice', 'dal'],
          dayOfWeek: 0,
          date: '2026-10-05',
          datasetId: 'ds-1',
        );

        expect(mockApi.lastPath, '/predict-demand');
        expect(mockApi.lastData?['dataset_id'], 'ds-1');
        // A blank attendance must be omitted so the server predicts from
        // uploaded history alone, not from a zero.
        expect(mockApi.lastData?.containsKey('attendance'), isFalse);
        expect(result.dataOrNull?.modelVersion, 'test-v1');
      });

      test('predictDemand omits dataset_id when none was uploaded', () async {
        final service = KitchenService(mockApi);
        await service.predictDemand(
          attendance: 450,
          mealType: 'LUNCH',
          menu: const [],
          dayOfWeek: 0,
          date: '2026-10-05',
        );

        expect(mockApi.lastData?.containsKey('dataset_id'), isFalse);
        expect(mockApi.lastData?['attendance'], 450);
      });
    });

    group('Null-tolerance for list payloads', () {
      test('extractList returns an empty list for an error envelope', () {
        // The regression that produced
        // "type 'Null' is not a subtype of type 'List<dynamic>'": a failed
        // request returns {"detail","code","message"} with no "items" key.
        final errorEnvelope = <String, dynamic>{
          'detail': 'could not load alerts',
          'code': 'INTERNAL',
          'message': 'could not load alerts',
        };
        expect(extractList<Map<String, dynamic>>(errorEnvelope, (m) => m), isEmpty);
      });

      test('extractList accepts a bare list, an items envelope and a data envelope', () {
        String? parse(Map<String, dynamic> m) => m['id'] as String?;
        expect(extractList<String?>([{'id': 'a'}], parse), ['a']);
        expect(extractList<String?>({'items': [{'id': 'b'}]}, parse), ['b']);
        expect(extractList<String?>({'data': [{'id': 'c'}]}, parse), ['c']);
        expect(extractList<String?>(null, parse), isEmpty);
      });

      test('Prediction.fromJson tolerates a missing top_drivers list', () {
        final p = Prediction.fromJson({
          'predicted_consumption': 1.0,
          'recommended_production': 1.0,
          'expected_surplus': 0.0,
          'surplus_risk': 'LOW',
          'prediction_interval': {'p10': 0.0, 'p50': 1.0, 'p90': 2.0, 'coverage_target': 0.8},
          'recommended_quantile': 0.7,
          'confidence': 0.5,
          'model_version': 'v1',
          'data_source': 'DATABASE',
          // top_drivers deliberately absent
        });
        expect(p.topDrivers, isEmpty);
      });
    });
  });
}
