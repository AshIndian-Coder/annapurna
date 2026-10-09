import 'package:dio/dio.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import '../errors/app_error.dart';
import '../result/result.dart';

/// Reads a list out of an API payload without throwing.
///
/// Collection endpoints are not consistent: some return a bare array, some
/// `{"items": [...]}`, and an **error envelope** (`{"detail","code","message"}`)
/// carries no list field at all. A direct `data['items'] as List` therefore
/// threw `type 'Null' is not a subtype of type 'List<dynamic>'` whenever the
/// request failed — surfacing as a red screen instead of an error message.
List<T> extractList<T>(dynamic data, T Function(Map<String, dynamic>) parse) {
  final list = switch (data) {
    List<dynamic> l => l,
    Map<String, dynamic> m =>
      (m['items'] ?? m['data'] ?? m['results'] ?? m['matches']) is List
          ? (m['items'] ?? m['data'] ?? m['results'] ?? m['matches']) as List
          : const <dynamic>[],
    _ => const <dynamic>[],
  };
  return list
      .whereType<Map<String, dynamic>>()
      .map(parse)
      .toList(growable: false);
}

/// Reads an object payload, tolerating a bare map under an `items`/`data` key.
Map<String, dynamic> extractMap(dynamic data) {
  if (data is Map<String, dynamic>) return data;
  return const {};
}

/// Reads a numeric field without throwing on a missing or non-numeric value.
double asDouble(dynamic value, [double fallback = 0]) {
  if (value is num) return value.toDouble();
  if (value is String) return double.tryParse(value) ?? fallback;
  return fallback;
}

const String _baseUrl = String.fromEnvironment('API_URL', defaultValue: 'http://127.0.0.1:8000/api/v1');

/// Whether the app talks to the real backend.
///
/// This defaults to **false**: with mocks on, [AuthService.login] ignored the
/// credentials entirely and returned a hardcoded Kitchen user, so every account
/// landed on a demo page. Pass `--dart-define=USE_MOCKS=true` to run against
/// the bundled fixtures instead (the login screen only offers the demo-role
/// shortcuts in that mode).
const bool useMocks = bool.fromEnvironment('USE_MOCKS', defaultValue: false);
const String appVersion = '1.0.0';

final apiClientProvider = Provider<ApiClient>((ref) => ApiClient());

final secureStorageProvider = Provider<FlutterSecureStorage>(
  (ref) => const FlutterSecureStorage(),
);

class ApiClient {
  late final Dio _dio;

  ApiClient() {
    _dio = Dio(BaseOptions(
      baseUrl: _baseUrl,
      connectTimeout: const Duration(seconds: 20),
      receiveTimeout: const Duration(seconds: 20),
      headers: {
        'Content-Type': 'application/json',
        'X-App-Version': appVersion,
      },
    ));

    _dio.interceptors.add(_AuthInterceptor());
    _dio.interceptors.add(_RetryInterceptor(_dio));
  }

  Future<Result<T>> get<T>(
    String path, {
    Map<String, dynamic>? queryParameters,
    required T Function(dynamic data) parser,
  }) async {
    try {
      final response = await _dio.get(path, queryParameters: queryParameters);
      return Success(parser(response.data));
    } catch (e) {
      return Failure(_handleError(e));
    }
  }

  Future<Result<T>> post<T>(
    String path, {
    dynamic data,
    Map<String, dynamic>? queryParameters,
    String? idempotencyKey,
    required T Function(dynamic data) parser,
  }) async {
    try {
      final options = Options();
      if (idempotencyKey != null) {
        options.headers = {'Idempotency-Key': idempotencyKey};
      }
      final response = await _dio.post(
        path,
        data: data,
        queryParameters: queryParameters,
        options: options,
      );
      return Success(parser(response.data));
    } catch (e) {
      return Failure(_handleError(e));
    }
  }

  Future<Result<T>> postMultipart<T>(
    String path, {
    required FormData formData,
    String? idempotencyKey,
    ProgressCallback? onSendProgress,
    required T Function(dynamic data) parser,
  }) async {
    try {
      final options = Options(contentType: 'multipart/form-data');
      if (idempotencyKey != null) {
        options.headers = {'Idempotency-Key': idempotencyKey};
      }
      final response = await _dio.post(
        path,
        data: formData,
        options: options,
        onSendProgress: onSendProgress,
      );
      return Success(parser(response.data));
    } catch (e) {
      return Failure(_handleError(e));
    }
  }

  Future<Result<T>> patch<T>(
    String path, {
    dynamic data,
    required T Function(dynamic data) parser,
  }) async {
    try {
      final response = await _dio.patch(path, data: data);
      return Success(parser(response.data));
    } catch (e) {
      return Failure(_handleError(e));
    }
  }

  Future<Result<void>> delete(String path) async {
    try {
      await _dio.delete(path);
      return const Success(null);
    } catch (e) {
      return Failure(_handleError(e));
    }
  }

  void setAuthToken(String token) {
    _dio.options.headers['Authorization'] = 'Bearer $token';
  }

  void clearAuthToken() {
    _dio.options.headers.remove('Authorization');
  }

  AppError _handleError(Object error) {
    if (error is DioException) {
      if (error.type == DioExceptionType.connectionTimeout ||
          error.type == DioExceptionType.receiveTimeout ||
          error.type == DioExceptionType.sendTimeout) {
        return AppError.timeout();
      }
      if (error.type == DioExceptionType.connectionError) {
        return AppError.network();
      }
      final response = error.response;
      if (response != null && response.data is Map<String, dynamic>) {
        return AppError.fromJson({
          ...response.data as Map<String, dynamic>,
          'status': response.statusCode,
        });
      }
      if (response != null) {
        return AppError(
          status: response.statusCode,
          code: 'HTTP_${response.statusCode}',
          message: response.statusMessage ?? 'Request failed',
        );
      }
      return AppError.network();
    }
    return AppError.unknown(error.toString());
  }
}

class _AuthInterceptor extends Interceptor {
  final _storage = const FlutterSecureStorage();

  @override
  void onRequest(RequestOptions options, RequestInterceptorHandler handler) async {
    if (!options.headers.containsKey('Authorization')) {
      final token = await _storage.read(key: 'access_token');
      if (token != null) {
        options.headers['Authorization'] = 'Bearer $token';
      }
    }
    handler.next(options);
  }
}

class _RetryInterceptor extends Interceptor {
  final Dio _dio;
  _RetryInterceptor(this._dio);

  @override
  void onError(DioException err, ErrorInterceptorHandler handler) async {
    if (err.requestOptions.method == 'GET' &&
        (err.type == DioExceptionType.connectionError ||
         err.type == DioExceptionType.connectionTimeout) &&
        (err.requestOptions.extra['retryCount'] ?? 0) < 2) {
      final retryCount = ((err.requestOptions.extra['retryCount'] ?? 0) as int) + 1;
      err.requestOptions.extra['retryCount'] = retryCount;
      await Future.delayed(Duration(milliseconds: 500 * retryCount));
      try {
        final response = await _dio.fetch(err.requestOptions);
        handler.resolve(response);
        return;
      } catch (_) {}
    }
    handler.next(err);
  }
}
