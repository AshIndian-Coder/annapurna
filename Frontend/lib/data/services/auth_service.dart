import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../core/enums.dart';
import '../../core/api/api_client.dart';
import '../../core/result/result.dart';
import '../dtos/models.dart';
import '../mocks/mock_data.dart';

final authServiceProvider = Provider<AuthService>((ref) {
  return AuthService(ref.read(apiClientProvider));
});

/// Credentials created by `go run ./cmd/seed`.
///
/// The demo shortcuts must match the seeded rows exactly, otherwise the button
/// posts credentials the backend has never heard of.
const Map<UserRole, ({String email, String password})> demoAccounts = {
  UserRole.kitchen: (email: 'kitchen@example.com', password: 'demo123'),
  UserRole.ngo: (email: 'ngo@example.com', password: 'demo123'),
  UserRole.logistics: (email: 'logistics@example.com', password: 'demo123'),
  UserRole.admin: (email: 'admin@example.com', password: 'demo123'),
};

class SignupRequest {
  final String name;
  final String email;
  final String password;
  final UserRole role;

  /// Only used by KITCHEN and NGO signups; the backend provisions the
  /// organisation plus the kitchen or recipient row from these.
  final String? organisationName;

  const SignupRequest({
    required this.name,
    required this.email,
    required this.password,
    required this.role,
    this.organisationName,
  });

  Map<String, dynamic> toJson() => {
    'name': name,
    'email': email,
    'password': password,
    'role': role.apiValue,
    if (organisationName != null && organisationName!.isNotEmpty)
      'organisation_name': organisationName,
  };
}

class AuthService {
  final ApiClient _api;
  AuthService(this._api);

  Future<Result<AuthTokens>> login(String email, String password) async {
    if (useMocks) {
      await Future.delayed(const Duration(milliseconds: 600));
      // In mock mode there is no server to check the password against, so the
      // role is inferred from the address rather than always being Kitchen.
      return Success(MockData.loginAs(_roleFromEmail(email)));
    }
    return _api.post(
      '/auth/login',
      data: {'email': email.trim(), 'password': password},
      parser: (data) => AuthTokens.fromJson(extractMap(data)),
    );
  }

  /// Creates an account and returns a token pair, so the caller is signed in
  /// immediately and lands on the dashboard for the chosen role.
  Future<Result<AuthTokens>> register(SignupRequest request) async {
    if (useMocks) {
      await Future.delayed(const Duration(milliseconds: 700));
      return Success(MockData.loginAs(request.role));
    }
    return _api.post(
      '/auth/register',
      data: request.toJson(),
      parser: (data) => AuthTokens.fromJson(extractMap(data)),
    );
  }

  Future<Result<AuthTokens>> loginAsRole(UserRole role) async {
    final account = demoAccounts[role]!;
    if (useMocks) {
      await Future.delayed(const Duration(milliseconds: 400));
      return Success(MockData.loginAs(role));
    }
    return _api.post(
      '/auth/login',
      data: {'email': account.email, 'password': account.password},
      parser: (data) => AuthTokens.fromJson(extractMap(data)),
    );
  }

  Future<Result<User>> getMe() async {
    if (useMocks) {
      await Future.delayed(const Duration(milliseconds: 200));
      return Success(MockData.kitchenUser);
    }
    return _api.get(
      '/auth/me',
      parser: (data) => User.fromJson(extractMap(data)),
    );
  }

  Future<Result<AuthTokens>> refresh(String refreshToken) async {
    return _api.post(
      '/auth/refresh',
      data: {'refresh_token': refreshToken},
      parser: (data) => AuthTokens.fromJson(extractMap(data)),
    );
  }

  Future<Result<void>> logout(String? refreshToken) async {
    if (useMocks) return const Success(null);
    return _api.post(
      '/auth/logout',
      data: refreshToken == null ? null : {'refresh_token': refreshToken},
      parser: (_) {},
    );
  }

  /// Best-effort role guess for mock logins, so typing a seeded address selects
  /// the matching demo dashboard.
  static UserRole _roleFromEmail(String email) {
    final normalised = email.trim().toLowerCase();
    for (final entry in demoAccounts.entries) {
      if (normalised == entry.value.email || normalised.startsWith(entry.key.name)) {
        return entry.key;
      }
    }
    return UserRole.kitchen;
  }
}
