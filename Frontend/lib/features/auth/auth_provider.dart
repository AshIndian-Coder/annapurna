import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import '../../core/enums.dart';
import '../../data/dtos/models.dart';
import '../../data/services/auth_service.dart';

final authProvider = StateNotifierProvider<AuthNotifier, AuthState>((ref) {
  return AuthNotifier(ref.read(authServiceProvider));
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
  final String? error;
  final bool isAuthenticated;

  const AuthState({
    this.user,
    this.isLoading = false,
    this.error,
    this.isAuthenticated = false,
  });

  AuthState copyWith({User? user, bool? isLoading, String? error, bool? isAuthenticated}) {
    return AuthState(
      user: user ?? this.user,
      isLoading: isLoading ?? this.isLoading,
      error: error,
      isAuthenticated: isAuthenticated ?? this.isAuthenticated,
    );
  }
}

class AuthNotifier extends StateNotifier<AuthState> {
  final AuthService _authService;
  final _storage = const FlutterSecureStorage();

  AuthNotifier(this._authService) : super(const AuthState());

  Future<bool> login(String email, String password) async {
    state = state.copyWith(isLoading: true, error: null);
    final result = await _authService.login(email, password);
    return result.when(
      success: (tokens) async {
        await _saveTokens(tokens);
        state = state.copyWith(
          user: tokens.user,
          isLoading: false,
          isAuthenticated: true,
        );
        return true;
      },
      failure: (error) {
        state = state.copyWith(isLoading: false, error: error.message);
        return false;
      },
    );
  }

  Future<bool> loginAsRole(UserRole role) async {
    state = state.copyWith(isLoading: true, error: null);
    final result = await _authService.loginAsRole(role.apiValue);
    return result.when(
      success: (tokens) async {
        await _saveTokens(tokens);
        state = state.copyWith(
          user: tokens.user,
          isLoading: false,
          isAuthenticated: true,
        );
        return true;
      },
      failure: (error) {
        state = state.copyWith(isLoading: false, error: error.message);
        return false;
      },
    );
  }

  Future<void> logout() async {
    await _authService.logout();
    await _storage.deleteAll();
    state = const AuthState();
  }

  Future<void> _saveTokens(AuthTokens tokens) async {
    await _storage.write(key: 'access_token', value: tokens.accessToken);
    await _storage.write(key: 'refresh_token', value: tokens.refreshToken);
  }
}
