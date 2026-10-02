class AppError {
  final int? status;
  final String code;
  final String message;

  const AppError({
    this.status,
    required this.code,
    required this.message,
  });

  factory AppError.fromJson(Map<String, dynamic> json) {
    return AppError(
      status: json['status'] as int?,
      code: (json['code'] ?? 'UNKNOWN') as String,
      message: (json['message'] ?? json['detail'] ?? 'An error occurred') as String,
    );
  }

  factory AppError.network() =>
      const AppError(code: 'NETWORK_ERROR', message: 'No internet connection');

  factory AppError.timeout() =>
      const AppError(code: 'TIMEOUT', message: 'Request timed out');

  factory AppError.unknown([String? msg]) =>
      AppError(code: 'UNKNOWN', message: msg ?? 'Something went wrong');

  factory AppError.appVersionUnsupported() =>
      const AppError(status: 426, code: 'APP_VERSION_UNSUPPORTED', message: 'Please update the app');

  bool get isNetworkError => code == 'NETWORK_ERROR';
  bool get isTimeout => code == 'TIMEOUT';
  bool get isUnauthorized => status == 401;
  bool get isForbidden => status == 403;
  bool get isConflict => status == 409;
  bool get isRateLimited => status == 429;

  @override
  String toString() => 'AppError($code: $message)';
}
