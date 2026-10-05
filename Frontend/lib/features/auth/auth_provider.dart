import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import '../../core/enums.dart';
import '../../core/result/result.dart';
import '../../data/dtos/models.dart';
import '../../data/services/auth_service.dart';

final authProvider = StateNotifierProvider<AuthNotifier, AuthState>((ref) {
  final notifier = AuthNotifier(ref.read(authServiceProvider));
  // Restore a previous session so a relaunch does not force a re-login.
  notifier.restoreSession();
  return notifier;
});

final currentUserProvider = Provider<User?>((ref) {
  return ref.watch(authProvider).user;
});

final currentRoleProvider = Provider<UserRole?>((ref) {
  return ref.watch(currentUserProvider)?.role;
});

class AuthState {
  final User? user;
  final bool isLoading;

  /// True while the stored session is being restored. The router must treat
  /// this differently from "not signed in", otherwise a valid session looks
  /// logged out for one frame and the user is bounced to /login.
  final bool isRestoring;
  final String? error;
  final bool isAuthenticated;

  const AuthState({
    this.user,
    this.isLoading = false,
    this.isRestoring = true,
    this.error,
    this.isAuthenticated = false,
  });

  AuthState copyWith({
    User? user,
    bool? isLoading,
    bool? isRestoring,
    String? error,
    bool clearError = false,
    bool? isAuthenticated,
  }) {
    return AuthState(
      user: user ?? this.user,
      isLoading: isLoading ?? this.isLoading,
      isRestoring: isRestoring ?? this.isRestoring,
      error: clearError ? null : (error ?? this.error),
      isAuthenticated: isAuthenticated ?? this.isAuthenticated,
    );
  }
}

class AuthNotifier extends StateNotifier<AuthState> {
  final AuthService _authService;
  final FlutterSecureStorage _storage;

  static const _accessTokenKey = 'access_token';
  static const _refreshTokenKey = 'refresh_token';
  static const _userKey = 'user_json';

  AuthNotifier(this._authService, {FlutterSecureStorage? storage})
      : _storage = storage ?? const FlutterSecureStorage(),
        super(const AuthState());

  Future<bool> login(String email, String password) async {
    state = state.copyWith(isLoading: true, isRestoring: false, clearError: true);
    final result = await _authService.login(email, password);
    return result.when(
      success: (tokens) => _onAuthenticated(tokens),
      failure: (error) => _onFailure(error.message),
    );
  }

  /// Creates an account and signs the user straight in.
  Future<bool> register(SignupRequest request) async {
    state = state.copyWith(isLoading: true, isRestoring: false, clearError: true);
    final result = await _authService.register(request);
    return result.when(
      success: (tokens) => _onAuthenticated(tokens),
      failure: (error) => _onFailure(error.message),
    );
  }

  Future<bool> loginAsRole(UserRole role) async {
    state = state.copyWith(isLoading: true, isRestoring: false, clearError: true);
    final result = await _authService.loginAsRole(role);
    return result.when(
      success: (tokens) => _onAuthenticated(tokens),
      failure: (error) => _onFailure(error.message),
    );
  }

  Future<bool> _onAuthenticated(AuthTokens tokens) async {
    await _persist(tokens);
    state = state.copyWith(
      user: tokens.user,
      isLoading: false,
      isRestoring: false,
      isAuthenticated: true,
      clearError: true,
    );
    return true;
  }

  bool _onFailure(String message) {
    state = state.copyWith(
      isLoading: false,
      isRestoring: false,
      error: message,
      isAuthenticated: false,
    );
    return false;
  }

  /// Re-establishes the session from secure storage.
  ///
  /// The stored access token may already have expired, so a failure here is not
  /// necessarily fatal: [refresh] is attempted once before giving up.
  Future<void> restoreSession() async {
    try {
      final access = await _storage.read(key: _accessTokenKey);
      final refreshToken = await _storage.read(key: _refreshTokenKey);
      final userJson = await _storage.read(key: _userKey);
      if (access == null || refreshToken == null || userJson == null) {
        state = state.copyWith(isRestoring: false);
        return;
      }

      final me = await _authService.getMe();
      if (me is Success<User>) {
        await _persist(AuthTokens(
          accessToken: access,
          refreshToken: refreshToken,
          expiresIn: 0,
          user: me.data,
        ));
        state = state.copyWith(
          user: me.data,
          isRestoring: false,
          isAuthenticated: true,
        );
        return;
      }

      final refreshed = await _authService.refresh(refreshToken);
      if (refreshed is Success<AuthTokens>) {
        await _persist(refreshed.data);
        state = state.copyWith(
          user: refreshed.data.user,
          isRestoring: false,
          isAuthenticated: true,
        );
        return;
      }

      await _clear();
      state = state.copyWith(isRestoring: false);
    } catch (_) {
      await _clear();
      state = state.copyWith(isRestoring: false);
    }
  }

  Future<void> logout() async {
    final refreshToken = await _storage.read(key: _refreshTokenKey);
    // The refresh token is what the server needs in order to revoke the
    // family; without it logout is only local.
    await _authService.logout(refreshToken);
    await _clear();
    state = const AuthState(isRestoring: false);
  }

  Future<void> _persist(AuthTokens tokens) async {
    await _storage.write(key: _accessTokenKey, value: tokens.accessToken);
    await _storage.write(key: _refreshTokenKey, value: tokens.refreshToken);
    await _storage.write(
      key: _userKey,
      value: '{"id":${_quote(tokens.user.id)},"name":${_quote(tokens.user.name)},'
          '"email":${_quote(tokens.user.email)},"role":"${tokens.user.role.apiValue}"}',
    );
  }

  Future<void> _clear() => _storage.deleteAll();

  static String _quote(String value) {
    final escaped = value
        .replaceAll(r'\', r'\\')
        .replaceAll('"', r'\"');
    return '"$escaped"';
  }
}